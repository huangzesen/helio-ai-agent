"""
Core agent logic - orchestrates Gemini calls and tool execution.

The OrchestratorAgent routes requests to:
- MissionAgent sub-agents for data operations (per spacecraft)
- VisualizationAgent sub-agent for all visualization
"""

import math
import threading
from typing import Optional

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_SUB_AGENT_MODEL, GEMINI_PLANNER_MODEL, GEMINI_FALLBACK_MODEL
from .tools import get_tool_schemas
from .prompts import get_system_prompt, format_tool_result
from .time_utils import parse_time_range, TimeRangeError
from .tasks import (
    Task, TaskPlan, TaskStatus, PlanStatus,
    get_task_store, create_task, create_plan,
)
from .planner import PlannerAgent, is_complex_request, format_plan_for_display, MAX_ROUNDS
from .session import SessionManager
from .memory import MemoryStore
from .mission_agent import MissionAgent
from .visualization_agent import VisualizationAgent
from .data_ops_agent import DataOpsAgent
from .data_extraction_agent import DataExtractionAgent
from .logging import (
    setup_logging, get_logger, log_error, log_tool_call,
    log_tool_result, log_plan_event, log_session_end,
    set_session_id,
)
from .memory_agent import MemoryAgent
from .loop_guard import LoopGuard, make_call_key
from .model_fallback import activate_fallback, get_active_model, is_quota_error
from rendering.registry import get_method
from rendering.plotly_renderer import PlotlyRenderer
from knowledge.catalog import search_by_keywords
from knowledge.cdaweb_catalog import search_catalog as search_full_cdaweb_catalog
from knowledge.hapi_client import (
    list_parameters as hapi_list_parameters,
    get_dataset_time_range,
    list_missions as hapi_list_missions,
    validate_dataset_id as hapi_validate_dataset_id,
    validate_parameter_id as hapi_validate_parameter_id,
)
from data_ops.store import get_store, DataEntry
from data_ops.fetch import fetch_data
from data_ops.custom_ops import run_custom_operation, run_dataframe_creation, run_spectrogram_computation

# Orchestrator sees discovery, web search, conversation, and routing tools
# (NOT data fetching or data_ops — handled by sub-agents)
ORCHESTRATOR_CATEGORIES = ["discovery", "web_search", "conversation", "routing", "document", "memory"]
ORCHESTRATOR_EXTRA_TOOLS = ["list_fetched_data", "preview_data"]

DEFAULT_MODEL = GEMINI_MODEL
SUB_AGENT_MODEL = GEMINI_SUB_AGENT_MODEL


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON safety.

    Gemini's API rejects function_response containing NaN or Inf values
    (400 INVALID_ARGUMENT). This ensures all tool results are safe.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


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
        self._cancel_event = threading.Event()

        # Initialize logging
        self.logger = setup_logging(verbose=verbose)
        self.logger.info("Initializing OrchestratorAgent")

        # Initialize Gemini client with retry for transient errors (503, 429, etc.)
        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(),
            ),
        )

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
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level="HIGH",
            ),
        )

        # Create chat session (use get_active_model in case fallback was already activated)
        self.chat = self.client.chats.create(
            model=get_active_model(self.model_name),
            config=self.config
        )

        # Plotly renderer for visualization
        self._renderer = PlotlyRenderer(verbose=self.verbose, gui_mode=self.gui_mode)

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._api_calls = 0

        # Current plan being executed (if any)
        self._current_plan: Optional[TaskPlan] = None

        # Cache of mission sub-agents, reused across requests in the session
        self._mission_agents: dict[str, MissionAgent] = {}

        # Cached visualization sub-agent
        self._viz_agent: Optional[VisualizationAgent] = None

        # Cached data ops sub-agent
        self._dataops_agent: Optional[DataOpsAgent] = None

        # Cached data extraction sub-agent
        self._data_extraction_agent: Optional[DataExtractionAgent] = None

        # Cached planner agent
        self._planner_agent: Optional[PlannerAgent] = None

        # Canonical time range for the current plan (reset after plan completes)
        self._plan_time_range: Optional['TimeRange'] = None

        # Session persistence
        self._session_id: Optional[str] = None
        self._session_manager = SessionManager()
        self._auto_save: bool = False

        # Long-term memory
        self._memory_store = MemoryStore()

        # Passive memory agent (lazy-initialized)
        self._memory_agent: Optional[MemoryAgent] = None

    # ---- Cancellation API ----

    def request_cancel(self):
        """Signal the agent to stop after the current atomic operation."""
        self._cancel_event.set()
        self.logger.info("[Cancel] Cancellation requested")

    def clear_cancel(self):
        """Clear the cancellation flag (called at start of process_message)."""
        self._cancel_event.clear()

    def is_cancelled(self) -> bool:
        """Check whether cancellation has been requested."""
        return self._cancel_event.is_set()

    @property
    def memory_store(self) -> MemoryStore:
        """Return the long-term memory store (for Gradio UI access)."""
        return self._memory_store

    def get_plotly_figure(self):
        """Return the current Plotly figure (or None)."""
        return self._renderer.get_figure()

    def _track_usage(self, response):
        """Accumulate token usage from a Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            self._total_input_tokens += getattr(meta, "prompt_token_count", 0) or 0
            self._total_output_tokens += getattr(meta, "candidates_token_count", 0) or 0
            self._total_thinking_tokens += getattr(meta, "thoughts_token_count", 0) or 0
        self._api_calls += 1
        if self.verbose:
            from .thinking import extract_thoughts
            for thought in extract_thoughts(response):
                self.logger.debug(f"[Thinking] {thought}")

    def _send_message(self, message):
        """Send a message on self.chat with automatic model fallback on 429."""
        try:
            return self.chat.send_message(message)
        except Exception as exc:
            if is_quota_error(exc) and GEMINI_FALLBACK_MODEL:
                activate_fallback(GEMINI_FALLBACK_MODEL)
                self.chat = self.client.chats.create(
                    model=GEMINI_FALLBACK_MODEL, config=self.config,
                )
                self.model_name = GEMINI_FALLBACK_MODEL
                return self.chat.send_message(message)
            raise

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
                model=get_active_model(self.model_name),
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
        thinking_tokens = self._total_thinking_tokens
        api_calls = self._api_calls

        # Include usage from cached mission agents
        for agent in self._mission_agents.values():
            usage = agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from visualization agent
        if self._viz_agent:
            usage = self._viz_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from data ops agent
        if self._dataops_agent:
            usage = self._dataops_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from data extraction agent
        if self._data_extraction_agent:
            usage = self._data_extraction_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)
            api_calls += usage["api_calls"]

        # Include usage from planner agent
        if self._planner_agent:
            usage = self._planner_agent.get_token_usage()
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
            thinking_tokens += usage.get("thinking_tokens", 0)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": input_tokens + output_tokens + thinking_tokens,
            "api_calls": api_calls,
        }

    def _validate_time_range(self, dataset_id: str, start, end) -> dict | None:
        """Check a requested time range against a dataset's availability.

        Validates and adjusts the time range:
        - Fully within range → returns None (no adjustment needed)
        - Partial overlap → clamps to available window
        - No overlap → returns error dict (does NOT silently shift)

        Note: This validates dataset-level availability only. Individual
        parameters may still return all-NaN data within a valid range.

        Args:
            dataset_id: CDAWeb dataset ID
            start: Requested start datetime (timezone-aware)
            end: Requested end datetime (timezone-aware)

        Returns:
            None if fully valid, or a dict with:
                - "start": clamped start datetime
                - "end": clamped end datetime
                - "note": human-readable note about the adjustment
            Returns error dict with "status"="error" if no overlap.
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
                return {
                    "error": True,
                    "note": (
                        f"No data available for '{dataset_id}' in the requested period. "
                        f"Dataset covers {avail_range_str}. "
                        f"Try a different dataset or adjust your time range."
                    ),
                }

            # No overlap — request is entirely before available data
            if req_end <= avail_start:
                return {
                    "error": True,
                    "note": (
                        f"No data available for '{dataset_id}' in the requested period. "
                        f"Dataset covers {avail_range_str}. "
                        f"Try a different dataset or adjust your time range."
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

    def _handle_plot_data(self, tool_args: dict) -> dict:
        """Handle the plot_data tool call."""
        store = get_store()
        # Derive labels from panels if labels not explicitly provided
        labels_str = tool_args.get("labels", "")
        if not labels_str and tool_args.get("panels"):
            labels_str = ",".join(
                label for panel in tool_args["panels"] for label in panel
            )
        if not labels_str:
            return {"status": "error", "message": "Missing 'labels' parameter"}
        labels = [l.strip() for l in labels_str.split(",")]
        entries = []
        for label in labels:
            entry = store.get(label)
            if entry is None:
                return {"status": "error", "message": f"Label '{label}' not found in memory"}
            entries.append(entry)

        # Auto-split into panels by units when panels not explicitly provided
        panels = tool_args.get("panels")
        if panels is None and len(entries) > 1:
            unit_groups: dict[str, list[str]] = {}
            for entry in entries:
                unit_key = (entry.units or "").strip() or "_dimensionless_"
                unit_groups.setdefault(unit_key, []).append(entry.label)
            if len(unit_groups) > 1:
                panels = list(unit_groups.values())
                self.logger.debug(
                    f"[PlotReview] Auto-split into {len(panels)} panels by units: "
                    + ", ".join(
                        f"{k}: {v}" for k, v in unit_groups.items()
                    )
                )

        try:
            result = self._renderer.plot_data(
                entries=entries,
                panels=panels,
                title=tool_args.get("title", ""),
                plot_type=tool_args.get("plot_type", "line"),
                colorscale=tool_args.get("colorscale", "Viridis"),
                log_y=tool_args.get("log_y", False),
                log_z=tool_args.get("log_z", False),
                z_min=tool_args.get("z_min"),
                z_max=tool_args.get("z_max"),
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}

        review = result.get("review", {})
        for w in review.get("warnings", []):
            self.logger.debug(f"[PlotReview] {w}")

        return result

    def _handle_style_plot(self, tool_args: dict) -> dict:
        """Handle the style_plot tool call."""
        # Parse y_label dict string: "{1: 'B (nT)', 2: '...'}" -> dict
        y_label = tool_args.get("y_label")
        if isinstance(y_label, str) and y_label.strip().startswith("{"):
            try:
                import ast
                parsed = ast.literal_eval(y_label)
                if isinstance(parsed, dict):
                    tool_args = {**tool_args, "y_label": parsed}
            except (ValueError, SyntaxError):
                pass
        try:
            result = self._renderer.style(**tool_args)
        except Exception as e:
            return {"status": "error", "message": str(e)}
        return result

    def _handle_manage_plot(self, tool_args: dict) -> dict:
        """Handle the manage_plot tool call."""
        action = tool_args.get("action")
        if not action:
            return {"status": "error", "message": "action is required"}

        if action == "reset":
            return self._renderer.reset()

        elif action == "get_state":
            return self._renderer.get_current_state()

        elif action == "set_time_range":
            tr_str = tool_args.get("time_range")
            if not tr_str:
                return {"status": "error", "message": "time_range is required for set_time_range"}
            try:
                time_range = parse_time_range(tr_str)
            except TimeRangeError as e:
                return {"status": "error", "message": str(e)}
            return self._renderer.set_time_range(time_range)

        elif action == "export":
            filename = tool_args.get("filename", "output.png")
            fmt = tool_args.get("format", "png")
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

        elif action == "remove_trace":
            label = tool_args.get("label")
            if not label:
                return {"status": "error", "message": "label is required for remove_trace"}
            return self._renderer.manage("remove_trace", label=label)

        elif action == "add_trace":
            label = tool_args.get("label")
            if not label:
                return {"status": "error", "message": "label is required for add_trace"}
            store = get_store()
            entry = store.get(label)
            if entry is None:
                return {"status": "error", "message": f"Label '{label}' not found in memory"}
            panel = int(tool_args.get("panel", 1))
            return self._renderer.manage("add_trace", entry=entry, panel=panel)

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

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
            result = {"status": "success", "parameters": params}

            # When using CDF backend, also include actual CDF variable names
            from config import DATA_BACKEND
            if DATA_BACKEND == "cdf":
                try:
                    from data_ops.fetch_cdf import list_cdf_variables
                    cdf_vars = list_cdf_variables(tool_args["dataset_id"])
                    result["cdf_variables"] = cdf_vars
                    result["note"] = (
                        "IMPORTANT: Use names from cdf_variables (not parameters) for "
                        "fetch_data calls. HAPI parameter names will NOT work."
                    )
                except Exception as e:
                    self.logger.debug(
                        f"[CDF] Could not list CDF variables for "
                        f"{tool_args['dataset_id']}: {e}"
                    )
            return result

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
            from knowledge.mission_loader import load_mission as _load_mission
            mission_id = tool_args["mission_id"]
            # Ensure HAPI cache exists (triggers download if needed)
            try:
                _load_mission(mission_id)
            except FileNotFoundError:
                pass
            datasets = hapi_browse(mission_id)
            if datasets is None:
                return {"status": "error", "message": f"No dataset index for '{mission_id}'."}
            return {"status": "success", "mission_id": mission_id,
                    "dataset_count": len(datasets), "datasets": datasets}

        elif tool_name == "list_missions":
            missions = hapi_list_missions()
            return {"status": "success", "missions": missions, "count": len(missions)}

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

        # --- Visualization (declarative tools) ---

        elif tool_name == "plot_data":
            return self._handle_plot_data(tool_args)

        elif tool_name == "style_plot":
            return self._handle_style_plot(tool_args)

        elif tool_name == "manage_plot":
            return self._handle_manage_plot(tool_args)

        # --- Data Operations Tools ---

        elif tool_name == "fetch_data":
            # Pre-fetch validation: reject dataset/parameter IDs not in local cache
            ds_validation = hapi_validate_dataset_id(tool_args["dataset_id"])
            if not ds_validation["valid"]:
                return {"status": "error", "message": ds_validation["message"]}

            # Skip HAPI parameter validation for CDF backend — CDF variable
            # names don't always match HAPI parameter names.
            from config import DATA_BACKEND
            if DATA_BACKEND != "cdf":
                param_validation = hapi_validate_parameter_id(
                    tool_args["dataset_id"], tool_args["parameter_id"]
                )
                if not param_validation["valid"]:
                    return {"status": "error", "message": param_validation["message"]}

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
                if validation.get("error"):
                    return {"status": "error", "message": validation["note"]}
                fetch_start = validation["start"]
                fetch_end = validation["end"]
                adjustment_note = validation["note"]
                self.logger.debug(
                    f"[DataOps] Time range adjusted for {tool_args['dataset_id']}: "
                    f"{adjustment_note}"
                )

            # Dedup: skip fetch if identical data already exists in store
            label = f"{tool_args['dataset_id']}.{tool_args['parameter_id']}"
            store = get_store()
            existing = store.get(label)
            if existing is not None and len(existing.data) > 0:
                existing_start = existing.data.index[0].to_pydatetime().replace(tzinfo=None)
                existing_end = existing.data.index[-1].to_pydatetime().replace(tzinfo=None)
                fetch_start_naive = fetch_start.replace(tzinfo=None)
                fetch_end_naive = fetch_end.replace(tzinfo=None)
                if existing_start <= fetch_start_naive and existing_end >= fetch_end_naive:
                    self.logger.debug(
                        f"[DataOps] Dedup: '{label}' already in memory "
                        f"({existing_start} to {existing_end}), skipping fetch"
                    )
                    response = {
                        "status": "success",
                        "already_loaded": True,
                        **existing.summary(),
                    }
                    if adjustment_note:
                        response["time_range_note"] = adjustment_note
                    return response

            try:
                result = fetch_data(
                    dataset_id=tool_args["dataset_id"],
                    parameter_id=tool_args["parameter_id"],
                    time_min=fetch_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    time_max=fetch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            except Exception as e:
                return {"status": "error", "message": str(e)}
            # Detect all-NaN fetches (parameter has no real data in range)
            df = result["data"]
            numeric_cols = df.select_dtypes(include="number")
            if len(df) > 0 and len(numeric_cols.columns) > 0 and numeric_cols.isna().all(axis=None):
                return {
                    "status": "error",
                    "message": (
                        f"Parameter '{tool_args['parameter_id']}' in dataset "
                        f"'{tool_args['dataset_id']}' returned {len(df)} rows "
                        f"but ALL values are fill/NaN — no real data available "
                        f"for this parameter in the requested time range. "
                        f"Try a different parameter or dataset."
                    ),
                }

            # Check NaN percentage before storing
            nan_total = numeric_cols.isna().sum().sum()
            nan_pct = round(100 * nan_total / numeric_cols.size, 1) if numeric_cols.size > 0 else 0.0

            from config import DATA_BACKEND
            entry = DataEntry(
                label=label,
                data=df,
                units=result["units"],
                description=result["description"],
                source=DATA_BACKEND,
            )
            store.put(entry)
            self.logger.debug(f"[DataOps] Stored '{label}' ({len(entry.time)} points)")
            response = {"status": "success", **entry.summary()}
            if adjustment_note:
                response["time_range_note"] = adjustment_note

            # Report NaN percentage for transparency
            if nan_pct > 0:
                response["nan_percentage"] = nan_pct
                if nan_pct >= 25:
                    response["quality_warning"] = (
                        f"High NaN/fill ratio ({nan_pct}%). Data was stored but "
                        f"quality is degraded. Consider trying a different "
                        f"parameter or dataset if one with better coverage exists."
                    )

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
            units = tool_args.get("units", source.units)
            entry = DataEntry(
                label=tool_args["output_label"],
                data=result_df,
                units=units,
                description=desc,
                source="computed",
            )
            store.put(entry)
            self.logger.debug(f"[DataOps] Custom operation -> '{tool_args['output_label']}' ({len(result_df)} points)")
            return {"status": "success", **entry.summary()}

        elif tool_name == "store_dataframe":
            try:
                result_df = run_dataframe_creation(tool_args["pandas_code"])
            except ValueError as e:
                return {"status": "error", "message": f"Validation error: {e}"}
            except RuntimeError as e:
                return {"status": "error", "message": f"Execution error: {e}"}
            entry = DataEntry(
                label=tool_args["output_label"],
                data=result_df,
                units=tool_args.get("units", ""),
                description=tool_args.get("description", "Created from code"),
                source="created",
            )
            store = get_store()
            store.put(entry)
            self.logger.debug(f"[DataOps] Created DataFrame -> '{tool_args['output_label']}' ({len(result_df)} points)")
            return {"status": "success", **entry.summary()}

        elif tool_name == "compute_spectrogram":
            store = get_store()
            source = store.get(tool_args["source_label"])
            if source is None:
                return {"status": "error", "message": f"Label '{tool_args['source_label']}' not found"}
            try:
                result_df = run_spectrogram_computation(source.data, tool_args["python_code"])
            except ValueError as e:
                return {"status": "error", "message": f"Validation error: {e}"}
            except RuntimeError as e:
                return {"status": "error", "message": f"Execution error: {e}"}

            metadata = {
                "type": "spectrogram",
                "bin_label": tool_args.get("bin_label", ""),
                "value_label": tool_args.get("value_label", ""),
            }
            try:
                bin_values = [float(c) for c in result_df.columns]
                metadata["bin_values"] = bin_values
            except (ValueError, TypeError):
                metadata["bin_values"] = list(range(len(result_df.columns)))

            entry = DataEntry(
                label=tool_args["output_label"],
                data=result_df,
                units=source.units,
                description=tool_args.get("description", "Spectrogram"),
                source="computed",
                metadata=metadata,
            )
            store.put(entry)
            self.logger.debug(f"[DataOps] Spectrogram -> '{tool_args['output_label']}' ({result_df.shape})")
            return {"status": "success", **entry.summary()}

        # --- Describe & Export Tools ---

        elif tool_name == "describe_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            df = entry.data
            stats = {}

            # Per-column statistics (numeric columns get full stats, others get count/unique)
            desc = df.describe(percentiles=[0.25, 0.5, 0.75], include="all")
            for col in df.columns:
                if df[col].dtype.kind in ("f", "i", "u"):  # numeric
                    col_stats = {
                        "min": float(desc.loc["min", col]),
                        "max": float(desc.loc["max", col]),
                        "mean": float(desc.loc["mean", col]),
                        "std": float(desc.loc["std", col]),
                        "25%": float(desc.loc["25%", col]),
                        "50%": float(desc.loc["50%", col]),
                        "75%": float(desc.loc["75%", col]),
                    }
                else:  # string/object/categorical columns
                    col_stats = {
                        "type": str(df[col].dtype),
                        "count": int(desc.loc["count", col]),
                        "unique": int(desc.loc["unique", col]) if "unique" in desc.index else None,
                        "top": str(desc.loc["top", col]) if "top" in desc.index else None,
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

        elif tool_name == "preview_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            df = entry.data
            n_rows = min(tool_args.get("n_rows", 5), 50)
            position = tool_args.get("position", "both")

            def _df_to_rows(sub_df):
                rows = []
                for ts, row in sub_df.iterrows():
                    d = {"timestamp": str(ts)}
                    for col in sub_df.columns:
                        v = row[col]
                        d[col] = float(v) if isinstance(v, (int, float)) else str(v)
                    rows.append(d)
                return rows

            result = {
                "status": "success",
                "label": entry.label,
                "units": entry.units,
                "total_rows": len(df),
                "columns": list(df.columns),
            }

            if position in ("head", "both"):
                result["head"] = _df_to_rows(df.head(n_rows))
            if position in ("tail", "both"):
                result["tail"] = _df_to_rows(df.tail(n_rows))

            return result

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

        # --- Document Reading (Gemini multimodal) ---

        elif tool_name == "read_document":
            from pathlib import Path

            file_path = tool_args["file_path"]
            if not Path(file_path).is_file():
                return {"status": "error", "message": f"File not found: {file_path}"}

            # MIME type map for supported formats
            mime_map = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tiff": "image/tiff",
            }
            ext = Path(file_path).suffix.lower()
            mime_type = mime_map.get(ext)
            if not mime_type:
                supported = ", ".join(sorted(mime_map.keys()))
                return {
                    "status": "error",
                    "message": f"Unsupported file format '{ext}'. Supported: {supported}",
                }

            try:
                import shutil

                # Read file bytes
                file_bytes = Path(file_path).read_bytes()

                # Build extraction prompt
                custom_prompt = tool_args.get("prompt", "")
                if custom_prompt:
                    extraction_prompt = custom_prompt
                elif ext == ".pdf":
                    extraction_prompt = (
                        "Extract all text content from this document. "
                        "Preserve the document structure (headings, paragraphs, lists). "
                        "Render tables as markdown tables. "
                        "Describe any figures or charts briefly."
                    )
                else:
                    extraction_prompt = (
                        "Extract all text and data from this image. "
                        "If it contains a table or chart, transcribe the data. "
                        "If it contains text, transcribe it faithfully. "
                        "Describe any visual elements briefly."
                    )

                # Send to Gemini as multimodal content
                doc_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                response = self.client.models.generate_content(
                    model=get_active_model(self.model_name),
                    contents=[doc_part, extraction_prompt],
                )
                self._track_usage(response)

                full_text = response.text or ""

                # Save original + extracted text to ~/.helio-agent/documents/{stem}/
                docs_dir = Path.home() / ".helio-agent" / "documents"
                src = Path(file_path)
                stem = src.stem

                # Find a unique subfolder name
                folder = docs_dir / stem
                counter = 1
                while folder.exists():
                    folder = docs_dir / f"{stem}_{counter}"
                    counter += 1
                folder.mkdir(parents=True, exist_ok=True)

                # Copy original file
                original_copy = folder / src.name
                shutil.copy2(str(src), str(original_copy))

                # Save extracted text
                out_path = folder / f"{stem}.md"
                out_path.write_text(full_text, encoding="utf-8")
                self.logger.debug(f"[Document] Saved to {folder} ({len(full_text)} chars)")

                # Truncate for LLM context
                max_chars = 50_000
                text = full_text
                truncated = len(full_text) > max_chars
                if truncated:
                    text = full_text[:max_chars]

                return {
                    "status": "success",
                    "file": Path(file_path).name,
                    "original_saved_to": str(original_copy),
                    "text_saved_to": str(out_path),
                    "char_count": len(full_text),
                    "truncated": truncated,
                    "content": text,
                }
            except Exception as e:
                return {"status": "error", "message": f"Document reading failed: {e}"}

        # --- Routing ---

        elif tool_name == "delegate_to_mission":
            mission_id = tool_args["mission_id"]
            request = tool_args["request"]
            self.logger.debug(f"[Router] Delegating to {mission_id} specialist")
            try:
                agent = self._get_or_create_mission_agent(mission_id)
                # Inject current data store contents so mission agent knows what's loaded
                store = get_store()
                entries = store.list_entries()
                if entries:
                    labels = [
                        f"  - {e['label']} ({e['num_points']} pts, {e['time_min']} to {e['time_max']})"
                        for e in entries
                    ]
                    request += (
                        "\n\nData currently in memory:\n"
                        + "\n".join(labels)
                        + "\nDo NOT re-fetch data that is already in memory with a matching label and time range."
                    )
                sub_result = agent.process_request(request)
                result = self._wrap_delegation_result(sub_result)
                result["mission"] = mission_id
                return result
            except (KeyError, FileNotFoundError):
                return {
                    "status": "error",
                    "message": f"Unknown mission '{mission_id}'. Check the supported missions table.",
                }

        elif tool_name == "delegate_to_visualization":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to Visualization specialist")

            # Intercept export requests — handle directly, no LLM needed
            req_lower = request.lower()
            if "export" in req_lower or ".png" in req_lower or ".pdf" in req_lower:
                import re as _re
                fn_match = _re.search(r'[\w.-]+\.(?:png|pdf|svg)', request, _re.IGNORECASE)
                filename = fn_match.group(0) if fn_match else "output.png"
                fmt = "pdf" if filename.endswith(".pdf") else "png"
                result = self._renderer.export(filename, format=fmt)
                if result.get("status") == "success" and not self.gui_mode and not self.web_mode:
                    try:
                        import os, platform, subprocess
                        fp = result["filepath"]
                        if platform.system() == "Darwin":
                            subprocess.Popen(["open", fp])
                        elif platform.system() == "Windows":
                            os.startfile(fp)
                        else:
                            subprocess.Popen(["xdg-open", fp])
                    except Exception:
                        pass
                return {"status": "success", "result": f"Exported plot to {result.get('filepath', filename)}"}

            agent = self._get_or_create_viz_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            return self._wrap_delegation_result(sub_result)

        elif tool_name == "delegate_to_data_ops":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to DataOps specialist")
            agent = self._get_or_create_dataops_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            return self._wrap_delegation_result(sub_result)

        elif tool_name == "delegate_to_data_extraction":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to DataExtraction specialist")
            agent = self._get_or_create_data_extraction_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            return self._wrap_delegation_result(sub_result)

        elif tool_name == "recall_memories":
            query = tool_args.get("query", "")
            mem_type = tool_args.get("type")
            limit = tool_args.get("limit", 20)
            if query:
                results = self._memory_store.search_cold(query, mem_type=mem_type, limit=limit)
            else:
                results = self._memory_store.read_cold()
                if mem_type:
                    results = [m for m in results if m.get("type") == mem_type]
                results = results[-limit:]
            return {
                "status": "success",
                "count": len(results),
                "memories": results,
            }

        elif tool_name == "request_planning":
            request = tool_args["request"]
            reasoning = tool_args.get("reasoning", "")
            self.logger.debug(f"[Planner] Planning requested: {reasoning}")
            summary = self._handle_planning_request(request)
            return {"status": "success", "result": summary, "planning_used": True}

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
            result = _sanitize_for_json(result)

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
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level="LOW",
                ),
            )
            task_chat = self.client.chats.create(
                model=get_active_model(self.model_name),
                config=task_config,
            )
            task_prompt = (
                f"Execute this task: {task.instruction}\n\n"
                "CRITICAL: Do ONLY what the instruction says. Do NOT add extra steps.\n"
                "- If the task says 'search', just search and report the results as text.\n"
                "- If the task says 'fetch', just fetch the data.\n"
                "- Do NOT create DataFrames, plots, or visualizations unless the instruction explicitly asks for it.\n"
                "- Do NOT delegate to other agents unless the instruction explicitly asks for it.\n"
                "- Return results as concise text, not as tool calls."
            )
            response = task_chat.send_message(task_prompt)
            self._track_usage(response)

            # Process tool calls with loop guard
            guard = LoopGuard(max_total_calls=10, max_iterations=5)
            last_stop_reason = None

            while True:
                stop_reason = guard.check_iteration()
                if stop_reason:
                    self.logger.debug(f"[Task] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                if self._cancel_event.is_set():
                    self.logger.info("[Cancel] Stopping task execution loop")
                    last_stop_reason = "cancelled by user"
                    break

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

                # Check for loops/duplicates/cycling
                call_keys = set()
                for fc in function_calls:
                    call_keys.add(make_call_key(fc.name, dict(fc.args) if fc.args else {}))
                stop_reason = guard.check_calls(call_keys)
                if stop_reason:
                    self.logger.debug(f"[Task] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
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

                guard.record_calls(call_keys)

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

            if last_stop_reason:
                task.status = TaskStatus.FAILED
                task.error = f"Task stopped by loop guard: {last_stop_reason}"
                result_text += f" [STOPPED: {last_stop_reason}]"
            else:
                task.status = TaskStatus.COMPLETED

            task.result = result_text

            self.logger.debug(f"[Task] {'Failed' if last_stop_reason else 'Completed'}: {task.description}")

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
            response = self._send_message(prompt)
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

    def _get_or_create_planner_agent(self) -> PlannerAgent:
        """Get the cached planner agent or create a new one."""
        if self._planner_agent is None:
            self._planner_agent = PlannerAgent(
                client=self.client,
                model_name=GEMINI_PLANNER_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
            )
            self.logger.debug(f"[Router] Created PlannerAgent ({GEMINI_PLANNER_MODEL})")
        return self._planner_agent

    def _execute_plan_task(self, task: Task, plan: TaskPlan) -> None:
        """Execute a single plan task, routing to the appropriate agent.

        Updates the task status in place.

        Args:
            task: The task to execute
            plan: The parent plan (for logging context)
        """
        mission_tag = f" [{task.mission}]" if task.mission else ""
        self.logger.debug(f"[Plan]{mission_tag}: {task.description}")

        # Inject canonical time range so all tasks use the same dates
        if self._plan_time_range:
            tr_str = self._plan_time_range.to_time_range_string()
            task.instruction += f"\n\nCanonical time range for this plan: {tr_str}"

        # Inject current data-store contents so sub-agents know what's available
        store = get_store()
        entries = store.list_entries()
        if entries:
            labels = [
                f"  - {e['label']} (columns: {e.get('columns', [])}, {e['num_points']} pts)"
                for e in entries
            ]
            task.instruction += "\n\nData currently in memory:\n" + "\n".join(labels)

        special_missions = {"__visualization__", "__data_ops__", "__data_extraction__"}

        if task.mission == "__visualization__":
            # Set renderer time range from plan so plot_data auto-applies it
            if self._plan_time_range:
                self._renderer.set_time_range(self._plan_time_range)

            instr_lower = task.instruction.lower()
            is_export = "export" in instr_lower or ".png" in instr_lower or ".pdf" in instr_lower

            if is_export:
                # Export is a simple dispatch — handle directly, no need for LLM
                self._handle_export_task(task)
            else:
                # Plot tasks: ensure instruction includes actual labels
                has_tool_ref = "plot_data" in instr_lower
                if not has_tool_ref and entries:
                    all_labels = ",".join(e["label"] for e in entries)
                    task.instruction = (
                        f"Use plot_data to plot {all_labels}. "
                        f"Original request: {task.instruction}"
                    )
                self._get_or_create_viz_agent().execute_task(task)
        elif task.mission == "__data_ops__":
            self._get_or_create_dataops_agent().execute_task(task)
        elif task.mission == "__data_extraction__":
            self._get_or_create_data_extraction_agent().execute_task(task)
        elif task.mission and task.mission not in special_missions:
            # Inject candidate datasets into instruction for mission agent
            if task.candidate_datasets:
                ds_list = ", ".join(task.candidate_datasets)
                task.instruction += f"\n\nCandidate datasets to inspect: {ds_list}"
            try:
                agent = self._get_or_create_mission_agent(task.mission)
                agent.execute_task(task)
            except (KeyError, FileNotFoundError):
                self.logger.debug(f"[Plan] Unknown mission '{task.mission}', using main agent")
                self._execute_task(task)
        else:
            self._execute_task(task)

    def _handle_export_task(self, task: Task) -> None:
        """Handle an export task directly without the VisualizationAgent.

        Export is a simple dispatch call — no LLM reasoning needed.
        Extracts the filename from the task instruction and calls the
        renderer's export method directly.

        Args:
            task: The export task to execute
        """
        import re

        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []

        # Extract filename from instruction
        fn_match = re.search(r'[\w.-]+\.(?:png|pdf|svg)', task.instruction, re.IGNORECASE)
        filename = fn_match.group(0) if fn_match else "output.png"

        self.logger.debug(f"[Plan] Direct export: {filename}")
        task.tool_calls.append("export")

        result = self._renderer.export(filename)
        # Auto-open in non-GUI/non-web mode
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
        if result.get("status") == "success":
            task.status = TaskStatus.COMPLETED
            task.result = f"Exported plot to {result.get('filepath', filename)}"
        else:
            task.status = TaskStatus.FAILED
            task.error = result.get("message", "Export failed")
            task.result = f"Export failed: {task.error}"

    def _extract_time_range(self, text: str):
        """Try to extract a resolved TimeRange from a user message.

        Uses parse_time_range() on the full text and common sub-patterns.
        Returns a TimeRange on success, or None if parsing fails.
        """
        import re as _re

        # Try the whole text first (works for "ACE mag for 2024-01-01 to 2024-01-15")
        try:
            return parse_time_range(text)
        except (TimeRangeError, ValueError):
            pass

        # Try to extract a "for <time_expr>" or "from <time_expr>" clause
        for pattern in [
            r'\bfor\s+(.+?)(?:\s*$)',
            r'\bfrom\s+(\d{4}.+?)(?:\s*$)',
            r'\bduring\s+(.+?)(?:\s*$)',
            r'(\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2})',
            r'(\d{4}-\d{2}-\d{2}T[\d:]+\s+to\s+\d{4}-\d{2}-\d{2}T[\d:]+)',
            r'((?:last\s+(?:\d+\s+)?(?:week|day|month|year))s?)',
            r'((?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})',
        ]:
            match = _re.search(pattern, text, _re.IGNORECASE)
            if match:
                try:
                    return parse_time_range(match.group(1).strip())
                except (TimeRangeError, ValueError):
                    continue

        return None

    def _handle_planning_request(self, user_message: str) -> str:
        """Process a complex multi-step request using the plan-execute-replan loop."""
        self.logger.debug("[PlannerAgent] Starting planner for complex request...")

        # Resolve and store the canonical time range for this plan
        self._plan_time_range = self._extract_time_range(user_message)
        if self._plan_time_range:
            self.logger.debug(f"[PlannerAgent] Resolved time range: {self._plan_time_range.to_time_range_string()}")

        planner = self._get_or_create_planner_agent()

        # Build planning message, injecting resolved time range if available
        planning_msg = user_message
        if self._plan_time_range:
            tr_str = self._plan_time_range.to_time_range_string()
            planning_msg = f"{user_message}\n\nResolved time range: {tr_str}. Use this exact range for ALL fetch tasks."

        # Round 1: initial planning
        response = planner.start_planning(planning_msg)
        if response is None:
            self.logger.debug("[PlannerAgent] Planner failed, falling back to direct execution")
            return self._process_single_message(user_message)

        plan = create_plan(user_message, [])
        self._current_plan = plan
        plan.status = PlanStatus.EXECUTING
        store = get_task_store()
        store.save(plan)

        log_plan_event("created", plan.id, f"Dynamic plan for: {user_message[:50]}...")

        round_num = 0
        while round_num < MAX_ROUNDS:
            round_num += 1

            if self._cancel_event.is_set():
                self.logger.info("[Cancel] Stopping plan loop between rounds")
                break

            tasks_data = response.get("tasks", [])

            if not tasks_data and response.get("status") == "done":
                break

            if not tasks_data:
                # Empty tasks with "continue" — treat as done
                break

            # Create Task objects for this batch
            new_tasks = []
            for td in tasks_data:
                mission = td.get("mission")
                if isinstance(mission, str) and mission.lower() in ("null", "none", ""):
                    mission = None
                task = create_task(
                    description=td["description"],
                    instruction=td["instruction"],
                    mission=mission,
                )
                task.round = round_num
                task.candidate_datasets = td.get("candidate_datasets")
                new_tasks.append(task)

            plan.add_tasks(new_tasks)
            store.save(plan)

            self.logger.debug(
                f"[PlannerAgent] Round {round_num}: {len(new_tasks)} tasks "
                f"(status={response['status']})"
            )
            self.logger.debug(format_plan_for_display(plan))

            # Execute batch — snapshot store labels before to detect new data
            round_results = []
            cancelled = False
            for i, task in enumerate(new_tasks):
                if self._cancel_event.is_set():
                    self.logger.info("[Cancel] Stopping plan mid-batch")
                    # Mark remaining tasks as SKIPPED
                    for remaining in new_tasks[i:]:
                        remaining.status = TaskStatus.SKIPPED
                        remaining.error = "Cancelled by user"
                    cancelled = True
                    break
                labels_before = set(e['label'] for e in get_store().list_entries())
                self._execute_plan_task(task, plan)
                labels_after = set(e['label'] for e in get_store().list_entries())

                # Build an informative result summary for the planner
                result_text = (task.result or "")[:500]
                if result_text in ("", "Done.") and task.tool_calls:
                    result_text = f"Tools called: {', '.join(task.tool_calls)}"

                # Append concrete outcome signals
                new_labels = labels_after - labels_before
                if new_labels:
                    result_text += f" | New data labels: {', '.join(sorted(new_labels))}"
                elif task.status == TaskStatus.FAILED:
                    result_text += " | No new data added."

                round_results.append({
                    "description": task.description,
                    "status": task.status.value,
                    "result_summary": result_text,
                    "error": task.error,
                })
                store.save(plan)

            if cancelled:
                store.save(plan)
                break

            # Append current store state so planner knows what data exists
            store_labels = [e['label'] for e in get_store().list_entries()]
            if store_labels:
                for r in round_results:
                    r["data_in_memory"] = store_labels

            if response.get("status") == "done":
                break

            # Replan: send results back to planner with round budget
            response = planner.continue_planning(
                round_results,
                round_num=round_num,
                max_rounds=MAX_ROUNDS,
            )
            if response is None:
                self.logger.debug("[PlannerAgent] Planner error mid-plan, finalizing")
                break

        # Finalize
        if self._cancel_event.is_set():
            plan.status = PlanStatus.CANCELLED
            log_plan_event("cancelled", plan.id, plan.progress_summary())
        elif plan.get_failed_tasks():
            plan.status = PlanStatus.FAILED
            log_plan_event("failed", plan.id, plan.progress_summary())
        else:
            plan.status = PlanStatus.COMPLETED
            log_plan_event("completed", plan.id, plan.progress_summary())
        store.save(plan)
        planner.reset()

        summary = self._summarize_plan_execution(plan)
        self._current_plan = None
        self._plan_time_range = None

        return summary

    def _process_single_message(self, user_message: str) -> str:
        """Process a single (non-complex) user message."""
        self.logger.debug("[Gemini] Sending message to model...")
        response = self._send_message(user_message)
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

            if self._cancel_event.is_set():
                self.logger.info("[Cancel] Stopping orchestrator loop")
                break

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

                # Track delegation failures (sub-agent stopped due to errors)
                if tool_name.startswith("delegate_to_") and result.get("status") == "error":
                    has_delegation_error = True
                    sub_text = result.get("result", "")
                    if sub_text:
                        self.logger.debug(f"[Delegation Failed] {tool_name} sub-agent response: {sub_text}")

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
                response = self._send_message(function_responses)
                self._track_usage(response)
                break

            self.logger.debug(f"[Gemini] Sending {len(function_responses)} tool result(s) back to model...")
            response = self._send_message(function_responses)
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

    @staticmethod
    def _wrap_delegation_result(sub_result) -> dict:
        """Convert a sub-agent process_request result into a tool result dict.

        If the sub-agent reported failure (stopped due to errors/loops),
        return status='error' so the orchestrator knows not to retry.
        """
        if isinstance(sub_result, dict):
            text = sub_result.get("text", "")
            failed = sub_result.get("failed", False)
            errors = sub_result.get("errors", [])
        else:
            # Legacy: plain string (shouldn't happen, but be safe)
            text = str(sub_result)
            failed = False
            errors = []

        if failed and errors:
            error_summary = "; ".join(errors[-3:])  # last 3 errors
            return {
                "status": "error",
                "message": f"Sub-agent failed. Errors: {error_summary}",
                "result": text,
            }
        return {"status": "success", "result": text}

    def _get_or_create_mission_agent(self, mission_id: str) -> MissionAgent:
        """Get a cached mission agent or create a new one."""
        if mission_id not in self._mission_agents:
            pitfalls = self._memory_store.get_scoped_pitfall_texts(f"mission:{mission_id}")
            self._mission_agents[mission_id] = MissionAgent(
                mission_id=mission_id,
                client=self.client,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                pitfalls=pitfalls,
            )
            self.logger.debug(f"[Router] Created {mission_id} mission agent ({SUB_AGENT_MODEL})")
        return self._mission_agents[mission_id]

    def _get_or_create_viz_agent(self) -> VisualizationAgent:
        """Get the cached visualization agent or create a new one."""
        if self._viz_agent is None:
            pitfalls = self._memory_store.get_scoped_pitfall_texts("visualization")
            self._viz_agent = VisualizationAgent(
                client=self.client,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                gui_mode=self.gui_mode,
                cancel_event=self._cancel_event,
                pitfalls=pitfalls,
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
                cancel_event=self._cancel_event,
            )
            self.logger.debug("[Router] Created DataOps agent")
        return self._dataops_agent

    def _get_or_create_data_extraction_agent(self) -> DataExtractionAgent:
        """Get the cached data extraction agent or create a new one."""
        if self._data_extraction_agent is None:
            self._data_extraction_agent = DataExtractionAgent(
                client=self.client,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
            )
            self.logger.debug("[Router] Created DataExtraction agent")
        return self._data_extraction_agent

    # ---- Long-term memory (background) ----

    def _ensure_memory_agent_started(self) -> None:
        """Start the background memory agent if not already running."""
        if self._memory_agent is not None and self._memory_agent.is_running:
            return
        self._memory_agent = MemoryAgent(
            client=self.client,
            model_name=SUB_AGENT_MODEL,
            memory_store=self._memory_store,
            verbose=self.verbose,
            session_id=self._session_id or "",
        )
        self._memory_agent.start()

    def stop_memory_agent(self) -> None:
        """Stop the background memory agent thread."""
        if self._memory_agent is not None:
            self._memory_agent.stop()
            self._memory_agent = None

    def extract_and_save_memories(self) -> None:
        """Run a final memory extraction pass (blocking).

        Called during shutdown to ensure any remaining log content is analyzed.
        Delegates to the MemoryAgent's one-shot analysis.
        """
        try:
            if self._memory_agent is None:
                self._memory_agent = MemoryAgent(
                    client=self.client,
                    model_name=SUB_AGENT_MODEL,
                    memory_store=self._memory_store,
                    verbose=self.verbose,
                    session_id=self._session_id or "",
                )
            self._memory_agent.analyze_once()
        except Exception as e:
            self.logger.debug(f"[Memory] Shutdown extraction failed: {e}")

    def generate_follow_ups(self, max_suggestions: int = 3) -> list[str]:
        """Generate contextual follow-up suggestions based on the conversation.

        Uses a lightweight single-shot Gemini call (Flash model) to produce
        2-3 short, actionable follow-up questions the user might ask next.

        Returns:
            List of suggestion strings, or [] on any failure.
        """
        try:
            history = self.chat.get_history()
        except Exception:
            return []

        # Build context from last 6 turns
        turns = []
        for content in history[-6:]:
            role = getattr(content, "role", "")
            if role not in ("user", "model"):
                continue
            for part in (content.parts or []):
                text = getattr(part, "text", None)
                if text:
                    prefix = "User" if role == "user" else "Agent"
                    turns.append(f"{prefix}: {text[:300]}")
                    break

        if not turns:
            return []

        conversation_text = "\n".join(turns)

        # DataStore context
        store = get_store()
        labels = [e["label"] for e in store.list_entries()]
        data_context = f"Data in memory: {', '.join(labels)}" if labels else "No data in memory yet."

        has_plot = self._renderer.get_figure() is not None
        plot_context = "A plot is currently displayed." if has_plot else "No plot is displayed."

        prompt = f"""Based on this conversation, suggest {max_suggestions} short follow-up questions the user might ask next.

{conversation_text}

{data_context}
{plot_context}

Respond with a JSON array of strings only (no markdown fencing). Each suggestion should be:
- A natural, conversational question (max 12 words)
- Actionable — something the agent can actually do
- Different from what was already asked
- Related to the current context (data, plots, spacecraft)

Example: ["Compare this with solar wind speed", "Zoom in to January 10-15", "Export the plot as PDF"]"""

        try:
            response = self.client.models.generate_content(
                model=get_active_model(SUB_AGENT_MODEL),
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )
            self._track_usage(response)

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3].strip()

            import json
            suggestions = json.loads(text)
            if isinstance(suggestions, list):
                return [s for s in suggestions if isinstance(s, str)][:max_suggestions]
        except Exception as e:
            self.logger.debug(f"[FollowUp] Generation failed: {e}")

        return []

    # ---- Session persistence ----

    def start_session(self) -> str:
        """Create a new session and enable auto-save.

        Also cleans up empty sessions from previous runs (fast no-op
        when there are none).

        Returns:
            The new session_id.
        """
        self._session_manager.cleanup_empty_sessions()
        self._session_id = self._session_manager.create_session(self.model_name)
        self._auto_save = True
        set_session_id(self._session_id)
        self.logger.debug(f"[Session] Started: {self._session_id}")
        return self._session_id

    def save_session(self) -> None:
        """Persist the current chat history and DataStore to disk."""
        if not self._session_id:
            return
        try:
            history_dicts = [
                content.model_dump(exclude_none=True)
                for content in self.chat.get_history()
            ]
        except Exception:
            history_dicts = []

        store = get_store()
        usage = self.get_token_usage()

        # Count user turns in history
        turn_count = sum(1 for h in history_dicts if h.get("role") == "user")

        # Don't persist empty sessions (no user messages, no data)
        if turn_count == 0 and len(store) == 0:
            return

        # Preview from last user message
        last_preview = ""
        for h in reversed(history_dicts):
            if h.get("role") == "user":
                parts = h.get("parts", [])
                for p in parts:
                    text = p.get("text", "") if isinstance(p, dict) else ""
                    if text:
                        last_preview = text[:80]
                        break
                if last_preview:
                    break

        self._session_manager.save_session(
            session_id=self._session_id,
            chat_history=history_dicts,
            data_store=store,
            metadata_updates={
                "turn_count": turn_count,
                "last_message_preview": last_preview,
                "token_usage": usage,
                "model": self.model_name,
            },
            figure_state=self._renderer.save_state(),
        )
        self.logger.debug(f"[Session] Saved ({turn_count} turns, {len(store)} data entries)")

    def load_session(self, session_id: str) -> dict:
        """Restore chat history and DataStore from a saved session.

        Args:
            session_id: The session to load.

        Returns:
            The session metadata dict.
        """
        history_dicts, data_dir, metadata, figure_state = self._session_manager.load_session(session_id)

        # Restore chat with saved history — fall back to fresh chat if
        # the Gemini SDK can't reconstruct function_call/function_response parts
        if history_dicts:
            try:
                self.chat = self.client.chats.create(
                    model=self.model_name,
                    config=self.config,
                    history=history_dicts,
                )
            except Exception as e:
                self.logger.warning(
                    f"[Session] Could not restore chat history: {e}. "
                    "Starting fresh chat (data still restored)."
                )
                self.chat = self.client.chats.create(
                    model=self.model_name,
                    config=self.config,
                )
        else:
            self.chat = self.client.chats.create(
                model=self.model_name,
                config=self.config,
            )

        # Restore DataStore
        store = get_store()
        store.clear()
        if data_dir.exists():
            count = store.load_from_directory(data_dir)
            self.logger.debug(f"[Session] Restored {count} data entries")

        # Clear sub-agent caches (they'll be recreated on next use)
        self._mission_agents.clear()
        self._viz_agent = None
        self._dataops_agent = None
        self._data_extraction_agent = None
        self._planner_agent = None
        self._renderer.reset()

        # Restore the Plotly figure and renderer state
        if figure_state:
            try:
                self._renderer.restore_state(figure_state)
                self.logger.debug("[Session] Restored plot figure")
            except Exception as e:
                self.logger.warning(f"[Session] Could not restore figure: {e}")

        self._session_id = session_id
        self._auto_save = True
        set_session_id(session_id)

        self.logger.debug(f"[Session] Loaded: {session_id}")
        return metadata

    def get_session_id(self) -> Optional[str]:
        """Return the current session ID, or None."""
        return self._session_id

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Hybrid routing: regex pre-filter catches obvious complex cases and routes
        them directly to the planner. All other messages go through the orchestrator
        loop, where the orchestrator (HIGH thinking) can also call request_planning
        for complex cases the regex missed.
        """
        self.clear_cancel()

        # Inject long-term memory context
        memory_section = self._memory_store.build_prompt_section()
        if memory_section:
            augmented = (
                f"[CONTEXT FROM LONG-TERM MEMORY]\n{memory_section}\n"
                f"[END MEMORY CONTEXT]\n\n{user_message}"
            )
        else:
            augmented = user_message

        if is_complex_request(user_message):
            self.logger.debug("[Orchestrator] Heuristic: request appears complex, routing to planner")
            result = self._handle_planning_request(augmented)
        else:
            result = self._process_single_message(augmented)

        # Auto-save after each turn
        if self._auto_save and self._session_id:
            try:
                self.save_session()
            except Exception as e:
                self.logger.warning(f"Auto-save failed: {e}")

        # Ensure background memory agent is running (non-blocking)
        self._ensure_memory_agent_started()

        return result

    def reset(self):
        """Reset conversation history, mission agent cache, and sub-agents."""
        self._cancel_event.clear()
        self.chat = self.client.chats.create(
            model=get_active_model(self.model_name),
            config=self.config
        )
        self._current_plan = None
        self._plan_time_range = None
        self._mission_agents.clear()
        self._viz_agent = None
        self._dataops_agent = None
        self._data_extraction_agent = None
        self._planner_agent = None
        self._renderer.reset()

        # Start a fresh session if auto-save was active
        if self._auto_save:
            self._session_id = self._session_manager.create_session(self.model_name)
            set_session_id(self._session_id)
            self.logger.debug(f"[Session] New session after reset: {self._session_id}")

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
        self._plan_time_range = None

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
        self._plan_time_range = None

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
