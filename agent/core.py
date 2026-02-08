"""
Core agent logic - orchestrates Gemini calls and tool execution.

The OrchestratorAgent routes requests to:
- MissionAgent sub-agents for data operations (per spacecraft)
- VisualizationAgent sub-agent for all visualization
"""

from typing import Optional

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_SUB_AGENT_MODEL
from .tools import get_tool_schemas
from .prompts import get_system_prompt, format_tool_result
from .time_utils import parse_time_range, TimeRangeError
from .tasks import (
    Task, TaskPlan, TaskStatus, PlanStatus,
    get_task_store, create_task, create_plan,
)
from .planner import create_plan_from_request, format_plan_for_display
from .mission_agent import MissionAgent
from .visualization_agent import VisualizationAgent
from .data_ops_agent import DataOpsAgent
from .logging import (
    setup_logging, get_logger, log_error, log_tool_call,
    log_tool_result, log_plan_event, log_session_end,
)
from rendering.registry import get_method, validate_args
from rendering.plotly_renderer import PlotlyRenderer
from knowledge.catalog import search_by_keywords
from knowledge.cdaweb_catalog import search_catalog as search_full_cdaweb_catalog
from knowledge.hapi_client import list_parameters as hapi_list_parameters, get_dataset_time_range
from data_ops.store import get_store, DataEntry
from data_ops.fetch import fetch_hapi_data
from data_ops.custom_ops import run_custom_operation
from rendering.custom_viz_ops import run_custom_visualization

# Orchestrator sees discovery, conversation, and routing tools
# (NOT data fetching or data_ops — handled by sub-agents)
ORCHESTRATOR_CATEGORIES = ["discovery", "conversation", "routing"]
ORCHESTRATOR_EXTRA_TOOLS = ["list_fetched_data"]

DEFAULT_MODEL = GEMINI_MODEL
SUB_AGENT_MODEL = GEMINI_SUB_AGENT_MODEL


class OrchestratorAgent:
    """Main orchestrator agent that routes to mission and visualization sub-agents."""

    # Class-level fallback so tests using __new__ don't crash on self.logger
    logger = get_logger()

    def __init__(self, verbose: bool = False, gui_mode: bool = False, model: str | None = None):
        """Initialize the orchestrator agent.

        Args:
            verbose: If True, print debug info about tool calls.
            gui_mode: If True, launch with visible GUI window.
            model: Gemini model name (default: DEFAULT_MODEL).
        """
        self.verbose = verbose
        self.gui_mode = gui_mode
        self.web_mode = False  # Set True by gradio_app.py to suppress auto-open

        # Initialize logging
        self.logger = setup_logging(verbose=verbose)
        self.logger.info("Initializing OrchestratorAgent")

        # Initialize Gemini client
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

        # Build function declarations for Gemini (orchestrator tools only)
        function_declarations = []
        for tool_schema in get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS):
            fd = types.FunctionDeclaration(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            )
            function_declarations.append(fd)

        # Create tool object with all function declarations
        # Note: google_search is NOT combined here — the Gemini generateContent API
        # does not support multi-tool use (function calling + google_search together).
        # Instead, google_search is a custom function that makes a separate API call.
        tool = types.Tool(function_declarations=function_declarations)

        # Store model name and config
        self.model_name = model or DEFAULT_MODEL
        self.config = types.GenerateContentConfig(
            system_instruction=get_system_prompt(gui_mode=gui_mode),
            tools=[tool],
        )

        # Create chat session
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=self.config
        )

        # Plotly renderer for visualization
        self._renderer = PlotlyRenderer(verbose=self.verbose, gui_mode=self.gui_mode)

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._api_calls = 0

        # Current plan being executed (if any)
        self._current_plan: Optional[TaskPlan] = None

        # Cache of mission sub-agents, reused across requests in the session
        self._mission_agents: dict[str, MissionAgent] = {}

        # Cached visualization sub-agent
        self._viz_agent: Optional[VisualizationAgent] = None

        # Cached data ops sub-agent
        self._dataops_agent: Optional[DataOpsAgent] = None

    def get_plotly_figure(self):
        """Return the current Plotly figure (or None)."""
        return self._renderer.get_figure()

    def _track_usage(self, response):
        """Accumulate token usage from a Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            self._total_input_tokens += getattr(meta, "prompt_token_count", 0) or 0
            self._total_output_tokens += getattr(meta, "candidates_token_count", 0) or 0
        self._api_calls += 1

    def _extract_grounding_sources(self, response) -> str:
        """Extract source citations from Google Search grounding metadata."""
        if not response.candidates:
            return ""
        candidate = response.candidates[0]
        meta = getattr(candidate, "grounding_metadata", None)
        if not meta:
            return ""
        chunks = getattr(meta, "grounding_chunks", None) or []
        sources = []
        seen = set()
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None)
                if uri and uri not in seen:
                    seen.add(uri)
                    sources.append(f"- [{title or uri}]({uri})")
        if not sources:
            return ""
        return "\n\nSources:\n" + "\n".join(sources)

    def _google_search(self, query: str) -> dict:
        """Execute a Google Search query via a separate Gemini API call.

        The generateContent API does not support combining google_search with
        function_declarations in the same request.  This method makes an
        isolated call with only the GoogleSearch tool so Gemini can ground its
        response in real web results.

        Args:
            query: The search query string.

        Returns:
            Dict with status, answer text, and source URLs.
        """
        try:
            search_config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=query,
                config=search_config,
            )
            self._track_usage(response)

            # Extract text
            text = response.text or ""

            # Extract sources from grounding metadata
            sources_text = self._extract_grounding_sources(response)

            if response.candidates:
                meta = getattr(response.candidates[0], "grounding_metadata", None)
                if meta and getattr(meta, "web_search_queries", None):
                    self.logger.debug(f"[Search] Queries: {meta.web_search_queries}")

            return {
                "status": "success",
                "answer": text + sources_text,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Google Search failed: {e}",
            }

    def get_token_usage(self) -> dict:
        """Return cumulative token usage for this session (including sub-agents)."""
        input_tokens = self._total_input_tokens
        output_tokens = self._total_output_tokens
        api_calls = self._api_calls

        # Include usage from cached mission agents
        for agent in self._mission_agents.values():
            usage = agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            api_calls += usage["api_calls"]

        # Include usage from visualization agent
        if self._viz_agent:
            usage = self._viz_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            api_calls += usage["api_calls"]

        # Include usage from data ops agent
        if self._dataops_agent:
            usage = self._dataops_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            api_calls += usage["api_calls"]

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "api_calls": api_calls,
        }

    def _validate_time_range(self, dataset_id: str, start, end) -> dict | None:
        """Check and clamp a requested time range to a dataset's availability.

        Auto-adjusts the time range to fit the available data window:
        - Fully within range → returns None (no adjustment needed)
        - Partial overlap → clamps to available window
        - No overlap (after stop) → shifts window to end at available stop
        - No overlap (before start) → shifts window to start at available start

        Args:
            dataset_id: CDAWeb dataset ID
            start: Requested start datetime (timezone-aware)
            end: Requested end datetime (timezone-aware)

        Returns:
            None if fully valid, or a dict with:
                - "start": clamped start datetime
                - "end": clamped end datetime
                - "note": human-readable note about the adjustment
            Returns None if the HAPI call fails (fail-open).
        """
        from datetime import datetime, timezone

        time_range = get_dataset_time_range(dataset_id)
        if time_range is None:
            return None  # fail-open

        try:
            avail_start_str = time_range.get("start")
            avail_stop_str = time_range.get("stop")
            if not avail_start_str or not avail_stop_str:
                return None

            # Parse HAPI date strings (may or may not have timezone)
            avail_start = datetime.fromisoformat(avail_start_str)
            avail_stop = datetime.fromisoformat(avail_stop_str)

            # Normalize to UTC-aware
            if avail_start.tzinfo is None:
                avail_start = avail_start.replace(tzinfo=timezone.utc)
            if avail_stop.tzinfo is None:
                avail_stop = avail_stop.replace(tzinfo=timezone.utc)

            req_start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
            req_end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)

            avail_range_str = (
                f"{avail_start.strftime('%Y-%m-%d')} to "
                f"{avail_stop.strftime('%Y-%m-%d')}"
            )
            duration = req_end - req_start

            # No overlap — request is entirely after available data
            if req_start >= avail_stop:
                new_end = avail_stop
                new_start = max(avail_start, avail_stop - duration)
                return {
                    "start": new_start,
                    "end": new_end,
                    "note": (
                        f"Requested dates are after the latest available data for "
                        f"'{dataset_id}' (available: {avail_range_str}). "
                        f"Auto-adjusted to {new_start.strftime('%Y-%m-%d')} to "
                        f"{new_end.strftime('%Y-%m-%d')}."
                    ),
                }

            # No overlap — request is entirely before available data
            if req_end <= avail_start:
                new_start = avail_start
                new_end = min(avail_stop, avail_start + duration)
                return {
                    "start": new_start,
                    "end": new_end,
                    "note": (
                        f"Requested dates are before the earliest available data for "
                        f"'{dataset_id}' (available: {avail_range_str}). "
                        f"Auto-adjusted to {new_start.strftime('%Y-%m-%d')} to "
                        f"{new_end.strftime('%Y-%m-%d')}."
                    ),
                }

            # Partial overlap — clamp to available window
            if req_start < avail_start or req_end > avail_stop:
                new_start = max(req_start, avail_start)
                new_end = min(req_end, avail_stop)
                return {
                    "start": new_start,
                    "end": new_end,
                    "note": (
                        f"Requested range partially outside available data for "
                        f"'{dataset_id}' (available: {avail_range_str}). "
                        f"Clamped to {new_start.strftime('%Y-%m-%d')} to "
                        f"{new_end.strftime('%Y-%m-%d')}."
                    ),
                }

        except (ValueError, TypeError):
            return None  # fail-open on parse errors

        return None  # fully valid

    def _dispatch_viz_method(self, method: str, args: dict) -> dict:
        """Dispatch an execute_visualization call to the appropriate renderer method.

        Core methods (plot_stored_data, set_time_range, export, reset,
        get_plot_state) are dispatched here.  Thin wrappers (title, axis
        labels, log scale, canvas size, render type, etc.) have been
        replaced by the ``custom_visualization`` tool.

        Args:
            method: Method name from the registry
            args: Arguments dict

        Returns:
            Result dict from the bridge method
        """
        # Validate against registry
        errors = validate_args(method, args)
        if errors:
            return {"status": "error", "message": "; ".join(errors)}

        if method == "reset":
            return self._renderer.reset()

        elif method == "get_plot_state":
            return self._renderer.get_current_state()

        elif method == "plot_stored_data":
            store = get_store()
            labels = [l.strip() for l in args["labels"].split(",")]
            entries = []
            for label in labels:
                entry = store.get(label)
                if entry is None:
                    return {"status": "error", "message": f"Label '{label}' not found in memory"}
                entries.append(entry)
            try:
                result = self._renderer.plot_dataset(
                    entries=entries,
                    title=args.get("title", ""),
                    filename=args.get("filename", ""),
                    index=int(args.get("index", -1)),
                )
            except Exception as e:
                return {"status": "error", "message": str(e)}
            return result

        elif method == "set_time_range":
            try:
                time_range = parse_time_range(args["time_range"])
            except TimeRangeError as e:
                return {"status": "error", "message": str(e)}
            return self._renderer.set_time_range(time_range)

        elif method == "export":
            fmt = args.get("format", "png")
            filename = args["filename"]
            result = self._renderer.export(filename, format=fmt)

            # Auto-open the exported file in default viewer (skip in GUI mode)
            if result.get("status") == "success" and not self.gui_mode and not self.web_mode:
                try:
                    import os
                    import platform
                    filepath = result["filepath"]
                    if platform.system() == "Windows":
                        os.startfile(filepath)
                    elif platform.system() == "Darwin":
                        import subprocess
                        subprocess.Popen(["open", filepath])
                    else:
                        import subprocess
                        subprocess.Popen(["xdg-open", filepath])
                    result["auto_opened"] = True
                except Exception as e:
                    self.logger.debug(f"[Export] Could not auto-open: {e}")
                    result["auto_opened"] = False

            return result

        else:
            return {"status": "error", "message": f"Unknown visualization method: {method}"}

    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Dict with result data (varies by tool)
        """
        # Log the tool call
        log_tool_call(tool_name, tool_args)

        self.logger.debug(f"[Tool: {tool_name}({tool_args})]")

        if tool_name == "search_datasets":
            self.logger.debug(f"[Catalog] Searching for: {tool_args['query']}")
            result = search_by_keywords(tool_args["query"])
            if result:
                self.logger.debug("[Catalog] Found matches.")
                return {"status": "success", **result}
            else:
                self.logger.debug("[Catalog] No matches found.")
                return {"status": "success", "message": "No matching datasets found."}

        elif tool_name == "list_parameters":
            self.logger.debug(f"[HAPI] Fetching parameters for {tool_args['dataset_id']}...")
            params = hapi_list_parameters(tool_args["dataset_id"])
            self.logger.debug(f"[HAPI] Got {len(params)} parameters.")
            return {"status": "success", "parameters": params}

        elif tool_name == "get_data_availability":
            dataset_id = tool_args["dataset_id"]
            time_range = get_dataset_time_range(dataset_id)
            if time_range is None:
                return {"status": "error", "message": f"Could not fetch availability for '{dataset_id}'."}
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "start": time_range["start"],
                "stop": time_range["stop"],
            }

        elif tool_name == "browse_datasets":
            from knowledge.hapi_client import browse_datasets as hapi_browse
            datasets = hapi_browse(tool_args["mission_id"])
            if datasets is None:
                return {"status": "error", "message": f"No dataset index for '{tool_args['mission_id']}'."}
            return {"status": "success", "mission_id": tool_args["mission_id"],
                    "dataset_count": len(datasets), "datasets": datasets}

        elif tool_name == "get_dataset_docs":
            from knowledge.hapi_client import get_dataset_docs
            docs = get_dataset_docs(tool_args["dataset_id"])
            if docs.get("documentation"):
                return {"status": "success", **docs}
            else:
                result = {"status": "partial" if docs.get("contact") else "error",
                          "dataset_id": docs["dataset_id"],
                          "message": "Could not fetch documentation."}
                if docs.get("contact"):
                    result["contact"] = docs["contact"]
                if docs.get("resource_url"):
                    result["resource_url"] = docs["resource_url"]
                return result

        elif tool_name == "search_full_catalog":
            query = tool_args["query"]
            max_results = int(tool_args.get("max_results", 20))
            self.logger.debug(f"[Catalog] Full catalog search: {query}")
            results = search_full_cdaweb_catalog(query, max_results=max_results)
            if results:
                return {
                    "status": "success",
                    "query": query,
                    "count": len(results),
                    "datasets": results,
                    "note": "Use fetch_data with any dataset ID above. Use list_parameters to see available parameters.",
                }
            else:
                return {
                    "status": "success",
                    "query": query,
                    "count": 0,
                    "datasets": [],
                    "message": f"No datasets found matching '{query}'. Try broader search terms.",
                }

        elif tool_name == "google_search":
            self.logger.debug(f"[Search] Query: {tool_args['query']}")
            return self._google_search(tool_args["query"])

        elif tool_name == "ask_clarification":
            # Return the question to show to user
            return {
                "status": "clarification_needed",
                "question": tool_args["question"],
                "options": tool_args.get("options", []),
                "context": tool_args.get("context", ""),
            }

        # --- Visualization (registry-driven dispatch) ---

        elif tool_name == "execute_visualization":
            method = tool_args["method"]
            args = tool_args.get("args", {})
            return self._dispatch_viz_method(method, args)

        elif tool_name == "custom_visualization":
            fig = self._renderer.get_figure()
            if fig is None:
                self._renderer._ensure_figure()
                fig = self._renderer.get_figure()
            try:
                run_custom_visualization(fig, tool_args["plotly_code"])
            except ValueError as e:
                return {"status": "error", "message": f"Validation error: {e}"}
            except RuntimeError as e:
                return {"status": "error", "message": f"Execution error: {e}"}
            return {"status": "success", "message": "Figure updated.", "display": "plotly"}

        # --- Data Operations Tools ---

        elif tool_name == "fetch_data":
            try:
                time_range = parse_time_range(tool_args["time_range"])
            except TimeRangeError as e:
                return {"status": "error", "message": str(e)}

            # Auto-clamp to available data window
            fetch_start = time_range.start
            fetch_end = time_range.end
            adjustment_note = None

            validation = self._validate_time_range(
                tool_args["dataset_id"], time_range.start, time_range.end
            )
            if validation is not None:
                fetch_start = validation["start"]
                fetch_end = validation["end"]
                adjustment_note = validation["note"]
                self.logger.debug(
                    f"[DataOps] Time range adjusted for {tool_args['dataset_id']}: "
                    f"{adjustment_note}"
                )

            try:
                result = fetch_hapi_data(
                    dataset_id=tool_args["dataset_id"],
                    parameter_id=tool_args["parameter_id"],
                    time_min=fetch_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    time_max=fetch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            except Exception as e:
                return {"status": "error", "message": str(e)}
            label = f"{tool_args['dataset_id']}.{tool_args['parameter_id']}"
            entry = DataEntry(
                label=label,
                data=result["data"],
                units=result["units"],
                description=result["description"],
                source="hapi",
            )
            store = get_store()
            store.put(entry)
            self.logger.debug(f"[DataOps] Stored '{label}' ({len(entry.time)} points)")
            response = {"status": "success", **entry.summary()}
            if adjustment_note:
                response["time_range_note"] = adjustment_note
            return response

        elif tool_name == "list_fetched_data":
            store = get_store()
            entries = store.list_entries()
            return {"status": "success", "entries": entries, "count": len(entries)}

        elif tool_name == "custom_operation":
            store = get_store()
            source = store.get(tool_args["source_label"])
            if source is None:
                return {"status": "error", "message": f"Label '{tool_args['source_label']}' not found"}
            try:
                result_df = run_custom_operation(source.data, tool_args["pandas_code"])
            except ValueError as e:
                return {"status": "error", "message": f"Validation error: {e}"}
            except RuntimeError as e:
                return {"status": "error", "message": f"Execution error: {e}"}
            desc = tool_args.get("description", f"Custom operation on {source.label}")
            entry = DataEntry(
                label=tool_args["output_label"],
                data=result_df,
                units=source.units,
                description=desc,
                source="computed",
            )
            store.put(entry)
            self.logger.debug(f"[DataOps] Custom operation -> '{tool_args['output_label']}' ({len(result_df)} points)")
            return {"status": "success", **entry.summary()}

        # --- Describe & Export Tools ---

        elif tool_name == "describe_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            df = entry.data
            stats = {}

            # Per-column statistics
            desc = df.describe(percentiles=[0.25, 0.5, 0.75])
            for col in df.columns:
                col_stats = {
                    "min": float(desc.loc["min", col]),
                    "max": float(desc.loc["max", col]),
                    "mean": float(desc.loc["mean", col]),
                    "std": float(desc.loc["std", col]),
                    "25%": float(desc.loc["25%", col]),
                    "50%": float(desc.loc["50%", col]),
                    "75%": float(desc.loc["75%", col]),
                }
                stats[col] = col_stats

            # Global metadata
            nan_count = int(df.isna().sum().sum())
            total_points = len(df)
            time_span = str(df.index[-1] - df.index[0]) if total_points > 1 else "single point"

            # Cadence estimate (median time step)
            if total_points > 1:
                dt = df.index.to_series().diff().dropna()
                median_cadence = str(dt.median())
            else:
                median_cadence = "N/A"

            return {
                "status": "success",
                "label": entry.label,
                "units": entry.units,
                "num_points": total_points,
                "num_columns": len(df.columns),
                "columns": list(df.columns),
                "time_start": str(df.index[0]),
                "time_end": str(df.index[-1]),
                "time_span": time_span,
                "median_cadence": median_cadence,
                "nan_count": nan_count,
                "nan_percentage": round(nan_count / (total_points * len(df.columns)) * 100, 1) if total_points > 0 else 0,
                "statistics": stats,
            }

        elif tool_name == "save_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            from pathlib import Path

            # Generate filename if not provided
            filename = tool_args.get("filename", "")
            if not filename:
                safe_label = entry.label.replace(".", "_").replace("/", "_")
                filename = f"{safe_label}.csv"
            if not filename.endswith(".csv"):
                filename += ".csv"

            # Ensure parent directory exists
            parent = Path(filename).parent
            if parent and str(parent) != "." and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

            # Export with ISO 8601 timestamps
            df = entry.data.copy()
            df.index.name = "timestamp"
            df.to_csv(filename, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

            filepath = str(Path(filename).resolve())
            file_size = Path(filename).stat().st_size

            self.logger.debug(f"[DataOps] Exported '{entry.label}' to {filepath} ({file_size:,} bytes)")

            return {
                "status": "success",
                "label": entry.label,
                "filepath": filepath,
                "num_points": len(df),
                "num_columns": len(df.columns),
                "file_size_bytes": file_size,
            }

        # --- Routing ---

        elif tool_name == "delegate_to_mission":
            mission_id = tool_args["mission_id"]
            request = tool_args["request"]
            self.logger.debug(f"[Router] Delegating to {mission_id} specialist")
            try:
                agent = self._get_or_create_mission_agent(mission_id)
                sub_result = agent.process_request(request)
                return {
                    "status": "success",
                    "mission": mission_id,
                    "result": sub_result,
                }
            except (KeyError, FileNotFoundError):
                return {
                    "status": "error",
                    "message": f"Unknown mission '{mission_id}'. Check the supported missions table.",
                }

        elif tool_name == "delegate_to_visualization":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to Visualization specialist")
            agent = self._get_or_create_viz_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            return {
                "status": "success",
                "result": sub_result,
            }

        elif tool_name == "delegate_to_data_ops":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to DataOps specialist")
            agent = self._get_or_create_dataops_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            return {
                "status": "success",
                "result": sub_result,
            }

        else:
            result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
            log_error(
                f"Unknown tool called: {tool_name}",
                context={"tool_name": tool_name, "tool_args": tool_args}
            )
            return result

    def _execute_tool_safe(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a tool with error handling and logging.

        Wraps _execute_tool to catch unexpected exceptions and log them.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Dict with result data (varies by tool)
        """
        try:
            result = self._execute_tool(tool_name, tool_args)

            # Log the result
            is_success = result.get("status") != "error"
            log_tool_result(tool_name, result, is_success)

            # If error, log with more detail
            if not is_success:
                log_error(
                    f"Tool {tool_name} returned error: {result.get('message', 'Unknown')}",
                    context={"tool_name": tool_name, "tool_args": tool_args, "result": result}
                )

            return result

        except Exception as e:
            # Unexpected exception - log with full stack trace
            log_error(
                f"Unexpected exception in tool {tool_name}",
                exc=e,
                context={"tool_name": tool_name, "tool_args": tool_args}
            )
            return {"status": "error", "message": f"Internal error: {e}"}

    def _execute_task(self, task: Task) -> str:
        """Execute a single task and return the result.

        Sends the task instruction to Gemini and handles tool calls.
        Updates the task status and records tool calls made.

        Args:
            task: The task to execute

        Returns:
            The text response from Gemini after completing the task
        """
        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []

        self.logger.debug(f"[Task] Executing: {task.description}")
        self.logger.debug("[Gemini] Sending task instruction...")

        try:
            # Create a fresh chat session for task execution with forced function calling
            task_config = types.GenerateContentConfig(
                system_instruction=get_system_prompt(gui_mode=self.gui_mode),
                tools=[types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                    ) for t in get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS)
                ])],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                ),
            )
            task_chat = self.client.chats.create(
                model=self.model_name,
                config=task_config,
            )
            response = task_chat.send_message(f"Execute this task: {task.instruction}")
            self._track_usage(response)

            # Process tool calls (limit to 3 iterations for task execution)
            max_iterations = 3
            iteration = 0
            previous_calls = set()  # Track (tool_name, args_key) to detect duplicates

            while iteration < max_iterations:
                iteration += 1

                if not response.candidates or not response.candidates[0].content.parts:
                    break

                function_calls = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                        function_calls.append(part.function_call)

                if not function_calls:
                    break

                # Break if Gemini is trying to ask for clarification (not supported in task execution)
                if any(fc.name == "ask_clarification" for fc in function_calls):
                    self.logger.debug("[Task] Skipping clarification request")
                    break

                # Detect duplicate tool calls (mode="ANY" forces repeated calls)
                call_keys = set()
                for fc in function_calls:
                    args_str = str(sorted(dict(fc.args).items())) if fc.args else ""
                    call_keys.add((fc.name, args_str))
                if call_keys and call_keys.issubset(previous_calls):
                    self.logger.debug("[Task] Duplicate tool call detected, stopping")
                    break

                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    task.tool_calls.append(tool_name)
                    result = self._execute_tool_safe(tool_name, tool_args)

                    if result.get("status") == "error":
                        self.logger.warning(f"[Tool Result: ERROR] {result.get('message', '')}")

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    )

                    # Track this call
                    args_str = str(sorted(tool_args.items()))
                    previous_calls.add((tool_name, args_str))

                self.logger.debug(f"[Gemini] Sending {len(function_responses)} tool result(s) back...")
                response = task_chat.send_message(message=function_responses)
                self._track_usage(response)

            # Warn if no tools were called (Gemini just responded with text)
            if not task.tool_calls:
                log_error(
                    f"Task completed without any tool calls: {task.description}",
                    context={"task_instruction": task.instruction}
                )
                self.logger.warning("[WARNING] No tools were called for this task")

            # Extract text response
            text_parts = []
            parts = response.candidates[0].content.parts if response.candidates and response.candidates[0].content else None
            if parts:
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

            result_text = "\n".join(text_parts) if text_parts else "Done."
            task.status = TaskStatus.COMPLETED
            task.result = result_text

            self.logger.debug(f"[Task] Completed: {task.description}")

            return result_text

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.logger.warning(f"[Task] Failed: {task.description} - {e}")
            return f"Error: {e}"

    def _summarize_plan_execution(self, plan: TaskPlan) -> str:
        """Generate a summary of the completed plan execution."""
        # Build context from completed tasks
        summary_parts = [f"I just executed a multi-step plan for: \"{plan.user_request}\""]
        summary_parts.append("")

        completed = plan.get_completed_tasks()
        failed = plan.get_failed_tasks()

        if completed:
            summary_parts.append("Completed tasks:")
            for task in completed:
                summary_parts.append(f"  - {task.description}")
                if task.result:
                    result_preview = task.result[:100] + "..." if len(task.result) > 100 else task.result
                    summary_parts.append(f"    Result: {result_preview}")

        if failed:
            summary_parts.append("")
            summary_parts.append("Failed tasks:")
            for task in failed:
                summary_parts.append(f"  - {task.description}")
                if task.error:
                    summary_parts.append(f"    Error: {task.error}")

        summary_parts.append("")
        summary_parts.append("Please provide a brief summary of what was accomplished for the user.")

        prompt = "\n".join(summary_parts)

        self.logger.debug("[Gemini] Generating execution summary...")

        try:
            response = self.chat.send_message(message=prompt)
            self._track_usage(response)

            text_parts = []
            parts = (
                response.candidates[0].content.parts
                if response.candidates and response.candidates[0].content
                else None
            )
            if parts:
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

            text = "\n".join(text_parts) if text_parts else plan.progress_summary()
            text += self._extract_grounding_sources(response)
            return text

        except Exception as e:
            log_error("Error generating plan summary", exc=e, context={"plan_id": plan.id})
            self.logger.warning(f"[Summary] Error generating summary: {e}")
            return plan.progress_summary()

    def _process_complex_request(self, user_message: str) -> str:
        """Process a complex multi-step request."""
        self.logger.debug("[Planner] Detected complex request, creating plan...")

        plan = create_plan_from_request(
            client=self.client,
            model_name=self.model_name,
            user_request=user_message,
            verbose=self.verbose,
        )

        if plan is None:
            self.logger.debug("[Planner] Falling back to direct execution")
            return self._process_single_message(user_message)

        self._current_plan = plan
        plan.status = PlanStatus.EXECUTING
        store = get_task_store()
        store.save(plan)

        log_plan_event("created", plan.id, f"{len(plan.tasks)} tasks for: {user_message[:50]}...")

        self.logger.debug(format_plan_for_display(plan))

        # Get or create agents for each unique mission in the plan
        special_missions = {"__visualization__", "__data_ops__"}
        mission_agents = {}
        for task in plan.tasks:
            if task.mission and task.mission not in special_missions and task.mission not in mission_agents:
                try:
                    mission_agents[task.mission] = self._get_or_create_mission_agent(task.mission)
                except (KeyError, FileNotFoundError):
                    self.logger.debug(f"[Plan] Unknown mission '{task.mission}', will use main agent")

        completed_task_ids = set()

        for i, task in enumerate(plan.tasks):
            plan.current_task_index = i
            store.save(plan)

            # Check dependencies
            unmet_deps = [dep for dep in task.depends_on if dep not in completed_task_ids]
            if unmet_deps:
                dep_tasks = {t.id: t for t in plan.tasks}
                failed_deps = [d for d in unmet_deps if dep_tasks.get(d) and dep_tasks[d].status == TaskStatus.FAILED]
                if failed_deps:
                    task.status = TaskStatus.SKIPPED
                    task.error = "Skipped: dependency failed"
                    self.logger.debug(f"[Plan] Step {i+1}/{len(plan.tasks)}: SKIPPED (dependency failed)")
                    store.save(plan)
                    continue

            mission_tag = f" [{task.mission}]" if task.mission else ""
            self.logger.debug(f"[Plan] Step {i+1}/{len(plan.tasks)}{mission_tag}: {task.description}")

            # Route to appropriate agent
            if task.mission == "__visualization__":
                self._get_or_create_viz_agent().execute_task(task)
            elif task.mission == "__data_ops__":
                self._get_or_create_dataops_agent().execute_task(task)
            elif task.mission and task.mission in mission_agents:
                mission_agents[task.mission].execute_task(task)
            else:
                self._execute_task(task)

            if task.status == TaskStatus.COMPLETED:
                completed_task_ids.add(task.id)

            store.save(plan)

        # Mark plan as complete
        if plan.get_failed_tasks():
            plan.status = PlanStatus.FAILED
            log_plan_event("failed", plan.id, plan.progress_summary())
        else:
            plan.status = PlanStatus.COMPLETED
            log_plan_event("completed", plan.id, plan.progress_summary())
        store.save(plan)

        summary = self._summarize_plan_execution(plan)
        self._current_plan = None

        return summary

    def _process_single_message(self, user_message: str) -> str:
        """Process a single (non-complex) user message."""
        self.logger.debug("[Gemini] Sending message to model...")
        response = self.chat.send_message(message=user_message)
        self._track_usage(response)
        self.logger.debug("[Gemini] Response received.")
        if response.candidates:
            meta = getattr(response.candidates[0], "grounding_metadata", None)
            if meta and getattr(meta, "web_search_queries", None):
                self.logger.debug(f"[Search] Queries: {meta.web_search_queries}")

        max_iterations = 10
        iteration = 0
        consecutive_delegation_errors = 0

        while iteration < max_iterations:
            iteration += 1

            if not response.candidates or not response.candidates[0].content.parts:
                break

            function_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                    function_calls.append(part.function_call)

            if not function_calls:
                break

            function_responses = []
            has_delegation_error = False
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                result = self._execute_tool_safe(tool_name, tool_args)

                if result.get("status") == "error":
                    self.logger.warning(f"[Tool Result: ERROR] {result.get('message', '')}")

                # Track delegation failures (mission agent couldn't fulfill request)
                if tool_name.startswith("delegate_to_") and result.get("status") == "success":
                    sub_result = result.get("result", "")
                    if isinstance(sub_result, str) and (
                        "No data available" in sub_result
                        or "outside the available data period" in sub_result
                        or "Error processing request" in sub_result
                    ):
                        has_delegation_error = True

                # Handle clarification specially - return immediately
                if result.get("status") == "clarification_needed":
                    question = result["question"]
                    if result.get("context"):
                        question = f"{result['context']}\n\n{question}"
                    if result.get("options"):
                        question += "\n\nOptions:\n" + "\n".join(
                            f"  {i+1}. {opt}" for i, opt in enumerate(result["options"])
                        )
                    return question

                function_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result}
                    )
                )

            # Track consecutive delegation failures
            if has_delegation_error:
                consecutive_delegation_errors += 1
            else:
                consecutive_delegation_errors = 0

            if consecutive_delegation_errors >= 2:
                self.logger.debug(f"[Orchestrator] {consecutive_delegation_errors} consecutive delegation failures, stopping retries")
                # Send results back one more time so Gemini can produce a final text answer
                response = self.chat.send_message(message=function_responses)
                self._track_usage(response)
                break

            self.logger.debug(f"[Gemini] Sending {len(function_responses)} tool result(s) back to model...")
            response = self.chat.send_message(message=function_responses)
            self._track_usage(response)
            self.logger.debug("[Gemini] Response received.")
            if response.candidates:
                meta = getattr(response.candidates[0], "grounding_metadata", None)
                if meta and getattr(meta, "web_search_queries", None):
                    self.logger.debug(f"[Search] Queries: {meta.web_search_queries}")

        # Extract text response
        text_parts = []
        parts = (
            response.candidates[0].content.parts
            if response.candidates and response.candidates[0].content
            else None
        )
        if parts:
            for part in parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

        text = "\n".join(text_parts) if text_parts else "Done."
        text += self._extract_grounding_sources(response)
        return text

    def _get_or_create_mission_agent(self, mission_id: str) -> MissionAgent:
        """Get a cached mission agent or create a new one."""
        if mission_id not in self._mission_agents:
            self._mission_agents[mission_id] = MissionAgent(
                mission_id=mission_id,
                client=self.client,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
            )
            self.logger.debug(f"[Router] Created {mission_id} mission agent ({SUB_AGENT_MODEL})")
        return self._mission_agents[mission_id]

    def _get_or_create_viz_agent(self) -> VisualizationAgent:
        """Get the cached visualization agent or create a new one."""
        if self._viz_agent is None:
            self._viz_agent = VisualizationAgent(
                client=self.client,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                gui_mode=self.gui_mode,
            )
            self.logger.debug(f"[Router] Created Visualization agent ({SUB_AGENT_MODEL})")
        return self._viz_agent

    def _get_or_create_dataops_agent(self) -> DataOpsAgent:
        """Get the cached data ops agent or create a new one."""
        if self._dataops_agent is None:
            self._dataops_agent = DataOpsAgent(
                client=self.client,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
            )
            self.logger.debug("[Router] Created DataOps agent")
        return self._dataops_agent

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Every message goes to the orchestrator, which decides what to do:
        - delegate_to_mission for data requests
        - delegate_to_visualization for visualization requests
        - Direct tool calls for discovery, data ops
        - Text response for greetings, questions, summaries
        """
        return self._process_single_message(user_message)

    def reset(self):
        """Reset conversation history, mission agent cache, and sub-agents."""
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=self.config
        )
        self._current_plan = None
        self._mission_agents.clear()
        self._viz_agent = None
        self._dataops_agent = None
        self._renderer.reset()

    def get_current_plan(self) -> Optional[TaskPlan]:
        """Get the currently executing plan, if any."""
        return self._current_plan

    def get_plan_status(self) -> Optional[str]:
        """Get a formatted status of the current plan."""
        if self._current_plan is None:
            store = get_task_store()
            incomplete = store.get_incomplete_plans()
            if incomplete:
                plan = sorted(incomplete, key=lambda p: p.created_at, reverse=True)[0]
                return format_plan_for_display(plan)
            return None
        return format_plan_for_display(self._current_plan)

    def cancel_plan(self) -> str:
        """Cancel the current plan and mark remaining tasks as skipped."""
        if self._current_plan is None:
            return "No active plan to cancel."

        plan = self._current_plan
        skipped_count = 0

        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.SKIPPED
                skipped_count += 1

        plan.status = PlanStatus.CANCELLED
        store = get_task_store()
        store.save(plan)

        completed = len(plan.get_completed_tasks())
        self._current_plan = None

        return f"Plan cancelled. {completed} task(s) completed, {skipped_count} skipped."

    def retry_failed_task(self) -> str:
        """Retry the first failed task in the current plan."""
        if self._current_plan is None:
            store = get_task_store()
            incomplete = store.get_incomplete_plans()
            failed_plans = [p for p in incomplete if p.get_failed_tasks()]
            if not failed_plans:
                return "No failed tasks to retry."
            self._current_plan = sorted(failed_plans, key=lambda p: p.created_at, reverse=True)[0]

        plan = self._current_plan
        failed = plan.get_failed_tasks()
        if not failed:
            return "No failed tasks to retry."

        task = failed[0]
        task.status = TaskStatus.PENDING
        task.error = None
        task.result = None
        task.tool_calls = []

        plan.status = PlanStatus.EXECUTING
        store = get_task_store()
        store.save(plan)

        self.logger.debug(f"[Retry] Retrying task: {task.description}")

        result = self._execute_task(task)
        store.save(plan)

        if plan.is_complete():
            if plan.get_failed_tasks():
                plan.status = PlanStatus.FAILED
            else:
                plan.status = PlanStatus.COMPLETED
            store.save(plan)

        return f"Retried: {task.description}\nResult: {result}"

    def resume_plan(self, plan: TaskPlan) -> str:
        """Resume an incomplete plan from storage."""
        self._current_plan = plan
        plan.status = PlanStatus.EXECUTING
        store = get_task_store()

        self.logger.debug(f"[Resume] Resuming plan: {plan.user_request[:50]}...")
        self.logger.debug(format_plan_for_display(plan))

        pending = plan.get_pending_tasks()
        if not pending:
            plan.status = PlanStatus.COMPLETED if not plan.get_failed_tasks() else PlanStatus.FAILED
            store.save(plan)
            return self._summarize_plan_execution(plan)

        for i, task in enumerate(plan.tasks):
            if task.status != TaskStatus.PENDING:
                continue

            plan.current_task_index = i
            store.save(plan)

            self.logger.debug(f"[Plan] Resuming step {i+1}/{len(plan.tasks)}: {task.description}")

            self._execute_task(task)
            store.save(plan)

        if plan.get_failed_tasks():
            plan.status = PlanStatus.FAILED
        else:
            plan.status = PlanStatus.COMPLETED
        store.save(plan)

        summary = self._summarize_plan_execution(plan)
        self._current_plan = None

        return summary

    def discard_plan(self, plan: TaskPlan) -> str:
        """Discard an incomplete plan."""
        store = get_task_store()
        store.delete(plan.id)
        return f"Discarded plan: {plan.user_request[:50]}..."


def create_agent(verbose: bool = False, gui_mode: bool = False, model: str | None = None) -> OrchestratorAgent:
    """Factory function to create a new agent instance.

    Args:
        verbose: If True, print debug info about tool calls.
        gui_mode: If True, launch with visible GUI window.
        model: Gemini model name (default: gemini-2.5-flash).

    Returns:
        Configured OrchestratorAgent instance.
    """
    return OrchestratorAgent(verbose=verbose, gui_mode=gui_mode, model=model)
