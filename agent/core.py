"""
Core agent logic - orchestrates Gemini calls and tool execution.

The OrchestratorAgent routes requests to:
- MissionAgent sub-agents for data operations (per spacecraft)
- VisualizationAgent sub-agent for all visualization
"""

import math
import time
import threading
import pandas as pd
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Optional

from config import GEMINI_MODEL, GEMINI_SUB_AGENT_MODEL, GEMINI_PLANNER_MODEL, GEMINI_FALLBACK_MODEL, get_data_dir
from config import LLM_PROVIDER, LLM_BASE_URL, get_api_key
from .llm import LLMAdapter, GeminiAdapter, OpenAIAdapter, AnthropicAdapter, LLMResponse, FunctionSchema
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
    set_session_id, tagged, log_token_usage, get_token_log_path,
)
from .memory_agent import MemoryAgent

from .loop_guard import LoopGuard, make_call_key
from .model_fallback import activate_fallback, get_active_model, is_quota_error
from .base_agent import _LLM_WARN_INTERVAL, _LLM_RETRY_TIMEOUT, _LLM_MAX_RETRIES
from rendering.registry import get_method
from rendering.plotly_renderer import PlotlyRenderer
from knowledge.catalog import search_by_keywords
from knowledge.cdaweb_catalog import search_catalog as search_full_cdaweb_catalog
from knowledge.metadata_client import (
    list_parameters,
    get_dataset_time_range,
    list_missions,
    validate_dataset_id,
    validate_parameter_id,
)
from data_ops.store import get_store, DataEntry, build_source_map, describe_sources
from data_ops.fetch import fetch_data
from data_ops.custom_ops import run_custom_operation, run_multi_source_operation, run_dataframe_creation
from data_ops.operations_log import get_operations_log

# Orchestrator sees discovery, web search, conversation, and routing tools
# (NOT data fetching or data_ops — handled by sub-agents)
ORCHESTRATOR_CATEGORIES = ["discovery", "web_search", "conversation", "routing", "document", "memory", "data_export"]
ORCHESTRATOR_EXTRA_TOOLS = ["list_fetched_data", "preview_data"]

DEFAULT_MODEL = GEMINI_MODEL
SUB_AGENT_MODEL = GEMINI_SUB_AGENT_MODEL


def _create_adapter() -> LLMAdapter:
    """Create the LLM adapter based on config (llm_provider, llm_base_url, etc.)."""
    provider = LLM_PROVIDER.lower()
    api_key = get_api_key(provider)
    if provider == "openai":
        return OpenAIAdapter(api_key=api_key, base_url=LLM_BASE_URL)
    elif provider == "anthropic":
        return AnthropicAdapter(api_key=api_key)
    else:
        return GeminiAdapter(api_key=api_key)


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
        self._token_log_path = get_token_log_path()  # snapshot before concurrent processes overwrite
        self.logger.info("Initializing OrchestratorAgent")

        # Initialize LLM adapter (wraps all provider SDK calls)
        self.adapter: LLMAdapter = _create_adapter()

        # Build tool schemas for the orchestrator
        self._tool_schemas = [
            FunctionSchema(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS)
        ]

        # Store model name and system prompt for chat creation
        self.model_name = model or DEFAULT_MODEL
        self._system_prompt = get_system_prompt(gui_mode=gui_mode)

        # Create chat session (use get_active_model in case fallback was already activated)
        self.chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            thinking="high",
        )

        # Plotly renderer for visualization
        self._renderer = PlotlyRenderer(verbose=self.verbose, gui_mode=self.gui_mode)

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._api_calls = 0
        self._last_tool_context = "send_message"

        # Current plan being executed (if any)
        self._current_plan: Optional[TaskPlan] = None

        # Cache of mission sub-agents, reused across requests in the session
        self._mission_agents: dict[str, MissionAgent] = {}
        self._mission_agents_lock = threading.Lock()

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

        # Thread pool for timeout-wrapped Gemini calls
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)

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

    # ---- Parallel tool execution ----

    # Tools safe to run concurrently (I/O-bound, no shared mutable state conflicts)
    _PARALLEL_SAFE_TOOLS = {"fetch_data", "delegate_to_mission"}

    def _execute_tools_parallel(
        self, function_calls: list
    ) -> list[tuple[str | None, str, dict, dict]]:
        """Execute a batch of tool calls, parallelizing when safe.

        If all calls are in _PARALLEL_SAFE_TOOLS and len > 1, runs them
        concurrently via ThreadPoolExecutor. Otherwise falls back to serial.

        Returns:
            List of (tool_call_id, tool_name, tool_args, result) tuples in original order.
        """
        parsed = [
            (getattr(fc, "id", None), fc.name, dict(fc.args) if fc.args else {})
            for fc in function_calls
        ]

        from config import PARALLEL_FETCH, PARALLEL_MAX_WORKERS

        # Check if all tools are safe for parallel execution
        all_safe = (
            PARALLEL_FETCH
            and len(parsed) > 1
            and all(name in self._PARALLEL_SAFE_TOOLS for _, name, _ in parsed)
        )

        if not all_safe:
            # Serial fallback
            return [
                (tc_id, name, args, self._execute_tool_safe(name, args))
                for tc_id, name, args in parsed
            ]

        # Parallel execution
        self.logger.debug(
            f"[Parallel] Executing {len(parsed)} tools concurrently: "
            f"{[name for _, name, _ in parsed]}"
        )
        max_workers = min(len(parsed), PARALLEL_MAX_WORKERS)
        results_by_idx: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._execute_tool_safe, name, args): idx
                for idx, (_, name, args) in enumerate(parsed)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_by_idx[idx] = future.result()
                except Exception as e:
                    results_by_idx[idx] = {
                        "status": "error",
                        "message": f"Parallel execution error: {e}",
                    }

        return [
            (parsed[i][0], parsed[i][1], parsed[i][2], results_by_idx[i])
            for i in range(len(parsed))
        ]

    @property
    def memory_store(self) -> MemoryStore:
        """Return the long-term memory store (for Gradio UI access)."""
        return self._memory_store

    def get_plotly_figure(self):
        """Return the current Plotly figure (or None)."""
        return self._renderer.get_figure()

    def _track_usage(self, response: LLMResponse):
        """Accumulate token usage from an LLMResponse."""
        call_input = response.usage.input_tokens
        call_output = response.usage.output_tokens
        call_thinking = response.usage.thinking_tokens
        self._total_input_tokens += call_input
        self._total_output_tokens += call_output
        self._total_thinking_tokens += call_thinking
        self._api_calls += 1
        log_token_usage(
            agent_name="OrchestratorAgent",
            input_tokens=call_input,
            output_tokens=call_output,
            thinking_tokens=call_thinking,
            cumulative_input=self._total_input_tokens,
            cumulative_output=self._total_output_tokens,
            cumulative_thinking=self._total_thinking_tokens,
            api_calls=self._api_calls,
            tool_context=self._last_tool_context,
            token_log_path=self._token_log_path,
        )
        if self.verbose:
            for thought in response.thoughts:
                # Full text to file/console (untagged)
                self.logger.debug(f"[Thinking] {thought}")
                # Preview for Gradio (tagged) — Gradio handler shows these inline
                preview = thought[:500] + ("..." if len(thought) > 500 else "")
                self.logger.debug(f"[Thinking] {preview}", extra={**tagged("thinking"), "skip_file": True})

    def _send_message(self, message) -> LLMResponse:
        """Send a message on self.chat with timeout/retry and model fallback on 429."""
        try:
            return self._send_with_timeout(self.chat, message)
        except Exception as exc:
            if is_quota_error(exc, adapter=self.adapter) and GEMINI_FALLBACK_MODEL:
                activate_fallback(GEMINI_FALLBACK_MODEL)
                self.chat = self.adapter.create_chat(
                    model=GEMINI_FALLBACK_MODEL,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                )
                self.model_name = GEMINI_FALLBACK_MODEL
                return self._send_with_timeout(self.chat, message)
            raise

    def _send_with_timeout(self, chat, message) -> LLMResponse:
        """Send a message to the LLM with periodic warnings and retry on timeout.

        - Warns every _LLM_WARN_INTERVAL seconds while waiting (10s, 20s, ...).
        - After _LLM_RETRY_TIMEOUT seconds, abandons the call and retries.
        - Retries up to _LLM_MAX_RETRIES times before raising.
        """
        last_exc = None
        for attempt in range(1 + _LLM_MAX_RETRIES):
            future: Future = self._timeout_pool.submit(
                chat.send, message
            )
            t0 = time.monotonic()
            try:
                while True:
                    elapsed = time.monotonic() - t0
                    remaining = _LLM_RETRY_TIMEOUT - elapsed
                    if remaining <= 0:
                        break
                    wait = min(_LLM_WARN_INTERVAL, remaining)
                    try:
                        return future.result(timeout=wait)
                    except TimeoutError:
                        elapsed = time.monotonic() - t0
                        if elapsed >= _LLM_RETRY_TIMEOUT:
                            break
                        self.logger.warning(
                            f"[Orchestrator] LLM API not responding "
                            f"after {elapsed:.0f}s (attempt {attempt + 1})..."
                        )

                elapsed = time.monotonic() - t0
                future.cancel()
                last_exc = TimeoutError(
                    f"LLM API call timed out after {elapsed:.0f}s"
                )
                if attempt < _LLM_MAX_RETRIES:
                    self.logger.warning(
                        f"[Orchestrator] LLM API timed out after "
                        f"{elapsed:.0f}s, retrying ({attempt + 1}/{_LLM_MAX_RETRIES})..."
                    )
                else:
                    self.logger.error(
                        f"[Orchestrator] LLM API timed out after "
                        f"{elapsed:.0f}s, no retries left"
                    )
            except Exception:
                raise

        raise last_exc

    def _extract_grounding_sources(self, response: LLMResponse) -> str:
        """Extract source citations from Google Search grounding metadata."""
        raw = response.raw
        if not raw or not getattr(raw, "candidates", None):
            return ""
        candidate = raw.candidates[0]
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

    def _log_grounding_queries(self, response: LLMResponse) -> None:
        """Log Google Search grounding queries from the raw response (if any)."""
        raw = response.raw
        if raw and getattr(raw, "candidates", None):
            meta = getattr(raw.candidates[0], "grounding_metadata", None)
            if meta and getattr(meta, "web_search_queries", None):
                self.logger.debug(f"[Search] Queries: {meta.web_search_queries}")

    def _google_search(self, query: str) -> dict:
        """Execute a web search query.

        Routes to Gemini grounded search (built-in Google Search) when using a
        Gemini adapter, otherwise falls back to Tavily web search.

        Args:
            query: The search query string.

        Returns:
            Dict with status, answer text, and source URLs.
        """
        if hasattr(self.adapter, "google_search"):
            return self._gemini_grounded_search(query)
        return self._tavily_search(query)

    def _gemini_grounded_search(self, query: str) -> dict:
        """Execute a Google Search query via Gemini's grounding API.

        The generateContent API does not support combining google_search with
        function_declarations in the same request.  This method makes an
        isolated call with only the GoogleSearch tool so Gemini can ground its
        response in real web results.
        """
        try:
            response = self.adapter.google_search(
                query=query,
                model=get_active_model(self.model_name),
            )
            self._last_tool_context = "google_search"
            self._track_usage(response)

            # Extract text
            text = response.text or ""

            # Extract sources from grounding metadata
            sources_text = self._extract_grounding_sources(response)

            raw = response.raw
            if raw and getattr(raw, "candidates", None):
                meta = getattr(raw.candidates[0], "grounding_metadata", None)
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

    def _tavily_search(self, query: str) -> dict:
        """Execute a web search query via Tavily API.

        Used as a fallback when the LLM provider does not have built-in web
        search (i.e. any non-Gemini provider).
        """
        try:
            from tavily import TavilyClient  # lazy import
        except ImportError:
            self.logger.warning("[Search] Web search unavailable — tavily-python not installed")
            return {
                "status": "error",
                "message": "Web search unavailable: tavily-python is not installed. "
                           "Install it with: pip install tavily-python",
            }

        import os
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            self.logger.warning("[Search] Web search unavailable — TAVILY_API_KEY not set")
            return {
                "status": "error",
                "message": "Web search unavailable: TAVILY_API_KEY environment variable not set.",
            }

        try:
            client = TavilyClient(api_key=api_key)
            result = client.search(query, include_answer="basic")

            answer = result.get("answer", "")
            sources = result.get("results", [])

            # Format sources
            sources_text = ""
            if sources:
                source_lines = [f"- [{s.get('title', 'Source')}]({s['url']})" for s in sources if s.get("url")]
                if source_lines:
                    sources_text = "\n\nSources:\n" + "\n".join(source_lines)

            return {
                "status": "success",
                "answer": (answer or "") + sources_text,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tavily search failed: {e}",
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
            Returns None if the metadata lookup fails (fail-open).
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

            # Parse date strings (may or may not have timezone)
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

    @staticmethod
    def _resolve_entry(store, label: str):
        """Resolve a label to a DataEntry, supporting column sub-selection.

        Handles labels like 'PSP_B_DERIVATIVE_FINAL.B_mag' where
        'PSP_B_DERIVATIVE_FINAL' is the store key and 'B_mag' is a column.

        Returns (DataEntry, resolved_label) or (None, None) if not found.
        """
        from data_ops.store import DataEntry

        # Exact match first
        entry = store.get(label)
        if entry is not None:
            return entry, label

        # Try column sub-selection: split from the right and check
        # progressively longer prefixes as parent labels.
        # E.g. "A.B.C" tries "A.B" with col "C", then "A" with col "B.C"
        parts = label.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_label = ".".join(parts[:i])
            col_name = ".".join(parts[i:])
            parent = store.get(parent_label)
            if parent is not None and col_name in parent.data.columns:
                sub_entry = DataEntry(
                    label=label,
                    data=parent.data[[col_name]],
                    units=parent.units,
                    description=f"{parent.description} [{col_name}]" if parent.description else col_name,
                    source=parent.source,
                    metadata=parent.metadata,
                )
                return sub_entry, label
        return None, None

    def _handle_render_plotly_json(self, tool_args: dict) -> dict:
        """Handle render_plotly_json: fill data_label placeholders and render."""
        fig_json = tool_args.get("figure_json", {})
        data_traces = fig_json.get("data", [])
        if not data_traces:
            return {"status": "error", "message": "figure_json.data is required (array of traces)"}

        # Collect all data_label values and resolve entries
        store = get_store()
        entry_map: dict = {}
        for trace in data_traces:
            label = trace.get("data_label")
            if label and label not in entry_map:
                entry, _ = self._resolve_entry(store, label)
                if entry is None:
                    return {"status": "error", "message": f"data_label '{label}' not found in memory"}
                entry_map[label] = entry

        # Validate non-empty data
        for label, entry in entry_map.items():
            if len(entry.data) == 0:
                return {"status": "error",
                        "message": f"Entry '{label}' has no data points"}

        try:
            result = self._renderer.render_plotly_json(fig_json, entry_map)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        if result.get("status") == "success":
            get_operations_log().record(
                tool="render_plotly_json",
                args={"figure_json": fig_json},
                inputs=list(entry_map.keys()),
                outputs=[],
            )

        return result

    def _handle_manage_plot(self, tool_args: dict) -> dict:
        """Handle the manage_plot tool call."""
        action = tool_args.get("action")
        if not action:
            return {"status": "error", "message": "action is required"}

        if action == "reset":
            get_operations_log().record(
                tool="manage_plot",
                args={"action": "reset"},
                inputs=[],
                outputs=[],
            )
            return self._renderer.reset()

        elif action == "get_state":
            return self._renderer.get_current_state()

        elif action == "export":
            filename = tool_args.get("filename", "output.png")
            fmt = tool_args.get("format", "png")
            result = self._renderer.export(filename, format=fmt)

            if result.get("status") == "success":
                get_operations_log().record(
                    tool="manage_plot",
                    args={"action": "export", "filename": filename, "format": fmt},
                    inputs=[],
                    outputs=[],
                )

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
            dataset_id = tool_args["dataset_id"]
            # Return CDF variable names from Master CDF skeleton
            try:
                from data_ops.fetch_cdf import list_cdf_variables
                cdf_vars = list_cdf_variables(dataset_id)
                self.logger.debug(f"[CDF] Listed {len(cdf_vars)} data variables for {dataset_id}")
                return {"status": "success", "parameters": cdf_vars}
            except Exception as e:
                self.logger.debug(f"[CDF] Could not list variables for {dataset_id}: {e}, using metadata cache")
                params = list_parameters(dataset_id)
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
            from knowledge.metadata_client import browse_datasets
            from knowledge.mission_loader import load_mission as _load_mission
            from knowledge.catalog import SPACECRAFT, classify_instrument_type
            mission_id = tool_args["mission_id"]
            # Ensure metadata cache exists (triggers download if needed)
            try:
                _load_mission(mission_id)
            except FileNotFoundError:
                pass
            datasets = browse_datasets(mission_id)
            if datasets is None:
                return {"status": "error", "message": f"No dataset index for '{mission_id}'."}

            # Enrich with instrument/type from mission JSON
            sc = SPACECRAFT.get(mission_id, {})
            ds_to_instrument = {}
            for inst_id, inst in sc.get("instruments", {}).items():
                kws = inst.get("keywords", [])
                for ds_id in inst.get("datasets", []):
                    ds_to_instrument[ds_id] = {"instrument": inst_id, "keywords": kws}

            for ds in datasets:
                info = ds_to_instrument.get(ds["id"], {})
                ds["instrument"] = info.get("instrument", "")
                ds["type"] = classify_instrument_type(info.get("keywords", []))

            return {"status": "success", "mission_id": mission_id,
                    "dataset_count": len(datasets), "datasets": datasets}

        elif tool_name == "list_missions":
            missions = list_missions()
            return {"status": "success", "missions": missions, "count": len(missions)}

        elif tool_name == "get_dataset_docs":
            from knowledge.metadata_client import get_dataset_docs
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

        elif tool_name == "render_plotly_json":
            return self._handle_render_plotly_json(tool_args)

        elif tool_name == "manage_plot":
            return self._handle_manage_plot(tool_args)

        # --- Data Operations Tools ---

        elif tool_name == "fetch_data":
            # Pre-fetch validation: reject dataset IDs not in local cache
            ds_validation = validate_dataset_id(tool_args["dataset_id"])
            if not ds_validation["valid"]:
                return {"status": "error", "message": ds_validation["message"]}

            # Pre-fetch validation: reject parameter IDs not in cached metadata
            param_validation = validate_parameter_id(
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
            if existing is not None and len(existing.time) > 0:
                if existing.is_xarray:
                    existing_start = pd.Timestamp(existing.time[0]).to_pydatetime().replace(tzinfo=None)
                    existing_end = pd.Timestamp(existing.time[-1]).to_pydatetime().replace(tzinfo=None)
                else:
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
                    get_operations_log().record(
                        tool="fetch_data",
                        args={
                            "dataset_id": tool_args["dataset_id"],
                            "parameter_id": tool_args["parameter_id"],
                            "time_range": tool_args.get("time_range", ""),
                            "time_range_resolved": [fetch_start.isoformat(), fetch_end.isoformat()],
                            "already_loaded": True,
                        },
                        outputs=[label],
                    )
                    return response

            try:
                result = fetch_data(
                    dataset_id=tool_args["dataset_id"],
                    parameter_id=tool_args["parameter_id"],
                    time_min=fetch_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    time_max=fetch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    force=tool_args.get("force_large_download", False),
                )
            except Exception as e:
                return {"status": "error", "message": str(e)}

            # Large download needs user permission
            if isinstance(result, dict) and result.get("status") == "confirmation_required":
                return {
                    "status": "clarification_needed",
                    "question": result["message"],
                    "options": [
                        "Yes, proceed with the download",
                        "No, try a shorter time range",
                        "Other (please specify)",
                    ],
                    "context": (
                        f"Dataset {result['dataset_id']}: {result['download_mb']} MB "
                        f"across {result['n_files']} files to download "
                        f"({result['n_cached']} already cached)."
                    ),
                }

            import xarray as xr
            fetched_data = result["data"]
            is_xarray = isinstance(fetched_data, xr.DataArray)

            if is_xarray:
                # xarray DataArray (3D+ variable)
                import numpy as np
                n_time = fetched_data.sizes["time"]

                # Detect all-NaN fetches
                if n_time > 0 and np.all(np.isnan(fetched_data.values)):
                    return {
                        "status": "error",
                        "message": (
                            f"Parameter '{tool_args['parameter_id']}' in dataset "
                            f"'{tool_args['dataset_id']}' returned {n_time} time steps "
                            f"but ALL values are fill/NaN — no real data available "
                            f"for this parameter in the requested time range. "
                            f"Try a different parameter or dataset."
                        ),
                    }

                # NaN percentage
                total_cells = fetched_data.size
                nan_total = int(np.isnan(fetched_data.values).sum())
                nan_pct = round(100 * nan_total / total_cells, 1) if total_cells > 0 else 0.0

                from config import DATA_BACKEND
                entry = DataEntry(
                    label=label,
                    data=fetched_data,
                    units=result["units"],
                    description=result["description"],
                    source=DATA_BACKEND,
                )
                store.put(entry)
                get_operations_log().record(
                    tool="fetch_data",
                    args={
                        "dataset_id": tool_args["dataset_id"],
                        "parameter_id": tool_args["parameter_id"],
                        "time_range": tool_args.get("time_range", ""),
                        "time_range_resolved": [fetch_start.isoformat(), fetch_end.isoformat()],
                        "already_loaded": False,
                    },
                    outputs=[label],
                )
                self.logger.debug(f"[DataOps] Stored '{label}' (xarray {dict(fetched_data.sizes)})", extra=tagged("data_fetched"))
                response = {"status": "success", **entry.summary()}
                response["note"] = (
                    f"This is a {fetched_data.ndim}D variable with dims {dict(fetched_data.sizes)}. "
                    f"Use custom_operation with xarray syntax (da_{tool_args['parameter_id']}) "
                    f"to slice/reduce it to a 2D DataFrame before plotting."
                )

                n_points = n_time
            else:
                # pandas DataFrame (1D/2D variable)
                df = fetched_data
                # Detect all-NaN fetches (parameter has no real data in range)
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
                get_operations_log().record(
                    tool="fetch_data",
                    args={
                        "dataset_id": tool_args["dataset_id"],
                        "parameter_id": tool_args["parameter_id"],
                        "time_range": tool_args.get("time_range", ""),
                        "time_range_resolved": [fetch_start.isoformat(), fetch_end.isoformat()],
                        "already_loaded": False,
                    },
                    outputs=[label],
                )
                self.logger.debug(f"[DataOps] Stored '{label}' ({len(entry.time)} points)", extra=tagged("data_fetched"))
                response = {"status": "success", **entry.summary()}

                n_points = len(df)

            # Warn about very large datasets that may cause slow operations
            if n_points > 500_000:
                response["size_warning"] = (
                    f"Very large dataset ({n_points:,} points). "
                    f"Consider using a shorter time range or a lower-cadence dataset "
                    f"to avoid slow downstream operations."
                )

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
            import numpy as np
            import xarray as xr

            store = get_store()
            labels = tool_args.get("source_labels", [])
            if not labels:
                return {"status": "error", "message": "source_labels is required"}

            sources, err = build_source_map(store, labels)
            if err:
                return {"status": "error", "message": err}

            # Determine if "df" alias exists (only when at least one source is a DataFrame)
            has_df_alias = any(not isinstance(v, xr.DataArray) for v in sources.values())
            df_extra = ["df"] if has_df_alias else []

            # Build source_timeseries map: variable name → is_timeseries flag
            source_ts = {}
            for label in labels:
                entry = store.get(label)
                if entry is not None:
                    suffix = label.rsplit(".", 1)[-1]
                    prefix = "da" if entry.is_xarray else "df"
                    var_name = f"{prefix}_{suffix}"
                    source_ts[var_name] = entry.is_timeseries

            try:
                op_result, warnings = run_multi_source_operation(
                    sources, tool_args["code"], source_timeseries=source_ts,
                )
            except (ValueError, RuntimeError) as e:
                prefix = "Validation" if isinstance(e, ValueError) else "Execution"
                err_msg = f"{prefix} error: {e}"
                get_operations_log().record(
                    tool="custom_operation",
                    args={
                        "source_labels": labels,
                        "code": tool_args["code"],
                        "output_label": tool_args.get("output_label", ""),
                    },
                    inputs=labels,
                    outputs=[],
                    status="error",
                    error=err_msg,
                )
                return {
                    "status": "error",
                    "message": err_msg,
                    "available_variables": list(sources.keys()) + df_extra,
                    "source_info": describe_sources(store, labels),
                }

            first_entry = store.get(labels[0])
            units = tool_args.get("units", first_entry.units if first_entry else "")
            desc = tool_args.get("description", f"Custom operation on {', '.join(labels)}")
            # Result is timeseries if it has a DatetimeIndex (or time dim for xarray)
            if isinstance(op_result, xr.DataArray):
                result_is_ts = "time" in op_result.dims
            else:
                result_is_ts = isinstance(op_result.index, pd.DatetimeIndex)
            entry = DataEntry(
                label=tool_args["output_label"],
                data=op_result,
                units=units,
                description=desc,
                source="computed",
                is_timeseries=result_is_ts,
            )
            store.put(entry)
            get_operations_log().record(
                tool="custom_operation",
                args={
                    "source_labels": labels,
                    "code": tool_args["code"],
                    "output_label": tool_args["output_label"],
                    "description": desc,
                    "units": units,
                },
                inputs=labels,
                outputs=[tool_args["output_label"]],
            )

            for w in warnings:
                self.logger.debug(f"[DataOpsValidation] {w}")

            # Warn on empty or all-NaN results (P0-1 symptom)
            is_xr_result = isinstance(op_result, xr.DataArray)
            n_points = op_result.sizes["time"] if is_xr_result else len(op_result)
            if n_points == 0:
                warnings.append("Result has 0 data points — possible time range mismatch or all-NaN input")
                self.logger.warning(f"[DataOps] custom_operation produced 0 points for '{tool_args['output_label']}'")
            elif is_xr_result:
                if np.all(np.isnan(op_result.values)):
                    warnings.append("Result is entirely NaN — check source data overlap and computation logic")
                    self.logger.warning(f"[DataOps] custom_operation produced all-NaN for '{tool_args['output_label']}'")
            elif op_result.isna().all(axis=None):
                warnings.append("Result is entirely NaN — check source data overlap and computation logic")
                self.logger.warning(f"[DataOps] custom_operation produced all-NaN for '{tool_args['output_label']}'")

            self.logger.debug(f"[DataOps] Custom operation -> '{tool_args['output_label']}' ({n_points} points)")
            result = {
                "status": "success",
                **entry.summary(),
                "source_info": describe_sources(store, labels),
                "available_variables": list(sources.keys()) + df_extra,
            }
            if warnings:
                result["warnings"] = warnings
            return result

        elif tool_name == "store_dataframe":
            try:
                result_df = run_dataframe_creation(tool_args["code"])
            except (ValueError, RuntimeError) as e:
                prefix = "Validation" if isinstance(e, ValueError) else "Execution"
                err_msg = f"{prefix} error: {e}"
                get_operations_log().record(
                    tool="store_dataframe",
                    args={
                        "code": tool_args["code"],
                        "output_label": tool_args.get("output_label", ""),
                    },
                    inputs=[],
                    outputs=[],
                    status="error",
                    error=err_msg,
                )
                return {"status": "error", "message": err_msg}
            entry = DataEntry(
                label=tool_args["output_label"],
                data=result_df,
                units=tool_args.get("units", ""),
                description=tool_args.get("description", "Created from code"),
                source="created",
                is_timeseries=isinstance(result_df.index, pd.DatetimeIndex),
            )
            store = get_store()
            store.put(entry)
            get_operations_log().record(
                tool="store_dataframe",
                args={
                    "code": tool_args["code"],
                    "output_label": tool_args["output_label"],
                    "description": tool_args.get("description", "Created from code"),
                    "units": tool_args.get("units", ""),
                },
                inputs=[],
                outputs=[tool_args["output_label"]],
            )
            self.logger.debug(f"[DataOps] Created DataFrame -> '{tool_args['output_label']}' ({len(result_df)} points)")
            return {"status": "success", **entry.summary()}

        # --- Function Documentation Tools ---

        elif tool_name == "search_function_docs":
            from knowledge.function_catalog import search_functions
            query = tool_args["query"]
            package = tool_args.get("package")
            results = search_functions(query, package=package)
            return {
                "status": "success",
                "query": query,
                "count": len(results),
                "functions": results,
            }

        elif tool_name == "get_function_docs":
            from knowledge.function_catalog import get_function_docstring
            result = get_function_docstring(tool_args["package"], tool_args["function_name"])
            if "error" in result:
                return {"status": "error", "message": result["error"]}
            return {"status": "success", **result}

        # --- Describe & Export Tools ---

        elif tool_name == "describe_data":
            store = get_store()
            entry = store.get(tool_args["label"])
            if entry is None:
                return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

            # --- xarray DataArray path ---
            if entry.is_xarray:
                import numpy as np
                da = entry.data
                dims = dict(da.sizes)
                n_time = dims.get("time", 0)

                # Coordinate info
                coords_info = {}
                for cname, coord in da.coords.items():
                    cvals = coord.values
                    info = {"size": len(cvals), "dtype": str(cvals.dtype)}
                    if np.issubdtype(cvals.dtype, np.number):
                        info["min"] = float(np.nanmin(cvals))
                        info["max"] = float(np.nanmax(cvals))
                    elif np.issubdtype(cvals.dtype, np.datetime64):
                        info["min"] = str(cvals[0])
                        info["max"] = str(cvals[-1])
                    coords_info[cname] = info

                # Global statistics on finite values
                flat = da.values.flatten()
                finite = flat[np.isfinite(flat)]
                nan_count = int(flat.size - finite.size)
                if finite.size > 0:
                    pcts = np.percentile(finite, [25, 50, 75])
                    statistics = {
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "mean": float(np.mean(finite)),
                        "std": float(np.std(finite)),
                        "25%": float(pcts[0]),
                        "50%": float(pcts[1]),
                        "75%": float(pcts[2]),
                    }
                else:
                    statistics = {"min": None, "max": None, "mean": None, "std": None}

                # Time info
                time_start = time_end = time_span = median_cadence = None
                if n_time > 0:
                    times = da.coords["time"].values
                    time_start = str(times[0])
                    time_end = str(times[-1])
                    time_span = str(pd.Timestamp(times[-1]) - pd.Timestamp(times[0]))
                    if n_time > 1:
                        dt = pd.Series(times).diff().dropna()
                        median_cadence = str(dt.median())

                return {
                    "status": "success",
                    "label": entry.label,
                    "units": entry.units,
                    "storage_type": "xarray",
                    "dims": dims,
                    "coordinates": coords_info,
                    "num_points": n_time,
                    "time_start": time_start,
                    "time_end": time_end,
                    "time_span": time_span,
                    "median_cadence": median_cadence,
                    "nan_count": nan_count,
                    "nan_percentage": round(nan_count / flat.size * 100, 1) if flat.size > 0 else 0,
                    "statistics": statistics,
                }

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

            # --- xarray DataArray path ---
            if entry.is_xarray:
                import numpy as np
                da = entry.data
                n_time = da.sizes.get("time", 0)
                n_rows = min(tool_args.get("n_rows", 3), 10)
                position = tool_args.get("position", "both")

                def _xr_time_slice(indices):
                    rows = []
                    for i in indices:
                        sl = da.isel(time=i)
                        vals = sl.values.flatten()
                        finite = vals[np.isfinite(vals)]
                        rows.append({
                            "timestamp": str(da.coords["time"].values[i]),
                            "shape": list(sl.shape),
                            "min": float(np.min(finite)) if finite.size > 0 else None,
                            "max": float(np.max(finite)) if finite.size > 0 else None,
                            "mean": float(np.mean(finite)) if finite.size > 0 else None,
                            "nan_count": int(vals.size - finite.size),
                        })
                    return rows

                result = {
                    "status": "success",
                    "label": entry.label,
                    "units": entry.units,
                    "storage_type": "xarray",
                    "dims": dict(da.sizes),
                    "total_time_steps": n_time,
                }
                if n_time > 0:
                    head_idx = list(range(min(n_rows, n_time)))
                    tail_idx = list(range(max(n_time - n_rows, 0), n_time))
                    if position in ("head", "both"):
                        result["head"] = _xr_time_slice(head_idx)
                    if position in ("tail", "both"):
                        result["tail"] = _xr_time_slice(tail_idx)

                return result

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

            # --- xarray DataArray path: export as NetCDF ---
            if entry.is_xarray:
                da = entry.data
                filename = tool_args.get("filename", "")
                if not filename:
                    safe_label = entry.label.replace(".", "_").replace("/", "_")
                    filename = f"{safe_label}.nc"
                if not filename.endswith(".nc"):
                    filename += ".nc"

                parent = Path(filename).parent
                if parent and str(parent) != "." and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)

                da.to_netcdf(filename, encoding={"time": {"units": "nanoseconds since 1970-01-01"}})
                filepath = str(Path(filename).resolve())
                file_size = Path(filename).stat().st_size

                self.logger.debug(f"[DataOps] Exported xarray '{entry.label}' to {filepath} ({file_size:,} bytes)")

                return {
                    "status": "success",
                    "label": entry.label,
                    "filepath": filepath,
                    "format": "netcdf",
                    "dims": dict(da.sizes),
                    "file_size_bytes": file_size,
                }

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

                # Send to LLM as multimodal content
                if hasattr(self.adapter, "make_bytes_part") and hasattr(self.adapter, "generate_multimodal"):
                    doc_part = self.adapter.make_bytes_part(data=file_bytes, mime_type=mime_type)
                    response = self.adapter.generate_multimodal(
                        model=get_active_model(self.model_name),
                        contents=[doc_part, extraction_prompt],
                    )
                else:
                    response = self.adapter.generate(
                        model=get_active_model(self.model_name),
                        contents=extraction_prompt,
                    )
                self._last_tool_context = "extract_document"
                self._track_usage(response)

                full_text = response.text or ""

                # Save original + extracted text to data_dir/documents/{stem}/
                docs_dir = get_data_dir() / "documents"
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
            self.logger.debug(f"[Router] Delegating to {mission_id} specialist", extra=tagged("delegation"))
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
                snapshot = get_store().list_entries()
                result = self._wrap_delegation_result(sub_result, store_snapshot=snapshot)
                result["mission"] = mission_id
                self.logger.debug(f"[Router] {mission_id} specialist finished", extra=tagged("delegation_done"))
                return result
            except (KeyError, FileNotFoundError):
                return {
                    "status": "error",
                    "message": f"Unknown mission '{mission_id}'. Check the supported missions table.",
                }

        elif tool_name == "delegate_to_visualization":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to Visualization specialist", extra=tagged("delegation"))

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

            # Inject current plot state so the viz agent knows what's displayed
            state = self._renderer.get_current_state()
            if state["has_plot"]:
                if state.get("figure_json"):
                    import json
                    fig_json_str = json.dumps(state["figure_json"], indent=2)
                    full_request += (
                        f"\n\nCurrently displayed: {state['traces']}"
                        f"\n\nCurrent figure_json (modify this, don't rebuild from scratch):\n{fig_json_str}"
                    )
                else:
                    full_request += f"\n\nCurrently displayed: {state['traces']}"
            else:
                full_request += "\n\nNo plot currently displayed."

            sub_result = agent.process_request(full_request)
            self.logger.debug("[Router] Visualization specialist finished", extra=tagged("delegation_done"))
            return self._wrap_delegation_result(sub_result, store_snapshot=get_store().list_entries())

        elif tool_name == "delegate_to_data_ops":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to DataOps specialist", extra=tagged("delegation"))
            agent = self._get_or_create_dataops_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            self.logger.debug("[Router] DataOps specialist finished", extra=tagged("delegation_done"))
            return self._wrap_delegation_result(sub_result, store_snapshot=get_store().list_entries())

        elif tool_name == "delegate_to_data_extraction":
            request = tool_args["request"]
            context = tool_args.get("context", "")
            self.logger.debug("[Router] Delegating to DataExtraction specialist", extra=tagged("delegation"))
            agent = self._get_or_create_data_extraction_agent()
            full_request = f"{request}\n\nContext: {context}" if context else request
            sub_result = agent.process_request(full_request)
            self.logger.debug("[Router] DataExtraction specialist finished", extra=tagged("delegation_done"))
            return self._wrap_delegation_result(sub_result, store_snapshot=get_store().list_entries())

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
            structured_time_range = tool_args.get("time_range", "")
            self.logger.debug(f"[Planner] Planning requested: {reasoning}")
            summary = self._handle_planning_request(
                request, structured_time_range=structured_time_range
            )
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
            task_chat = self.adapter.create_chat(
                model=get_active_model(self.model_name),
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                force_tool_call=True,
                thinking="low",
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
            self._last_tool_context = "task:" + task.description[:50]
            response = self._send_with_timeout(task_chat, task_prompt)
            self._track_usage(response)

            # Process tool calls with loop guard
            guard = LoopGuard(max_total_calls=20, max_iterations=10)
            last_stop_reason = None
            had_successful_tool = False

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

                if not response.tool_calls:
                    break

                function_calls = response.tool_calls

                # Break if LLM is trying to ask for clarification (not supported in task execution)
                if any(fc.name == "ask_clarification" for fc in function_calls):
                    self.logger.debug("[Task] Skipping clarification request")
                    break

                # Check for loops/duplicates/cycling
                call_keys = set()
                for fc in function_calls:
                    call_keys.add(make_call_key(fc.name, fc.args))
                stop_reason = guard.check_calls(call_keys)
                if stop_reason:
                    self.logger.debug(f"[Task] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = fc.args

                    task.tool_calls.append(tool_name)
                    result = self._execute_tool_safe(tool_name, tool_args)

                    if result.get("status") == "success":
                        had_successful_tool = True
                    elif result.get("status") == "error":
                        self.logger.warning(f"[Tool Result: ERROR] {result.get('message', '')}")

                    function_responses.append(
                        self.adapter.make_tool_result_message(
                            tool_name, result, tool_call_id=fc.id
                        )
                    )

                guard.record_calls(call_keys)

                self.logger.debug(f"[LLM] Sending {len(function_responses)} tool result(s) back...")
                tool_names = [fc.name for fc in function_calls]
                self._last_tool_context = "+".join(tool_names)
                response = self._send_with_timeout(task_chat, function_responses)
                self._track_usage(response)

            # Warn if no tools were called (LLM just responded with text)
            if not task.tool_calls:
                log_error(
                    f"Task completed without any tool calls: {task.description}",
                    context={"task_instruction": task.instruction}
                )
                self.logger.warning("[WARNING] No tools were called for this task")

            # Extract text response
            result_text = response.text or "Done."

            if last_stop_reason:
                if last_stop_reason == "cancelled by user":
                    task.status = TaskStatus.FAILED
                    task.error = "Task cancelled by user"
                    result_text += " [CANCELLED]"
                elif had_successful_tool:
                    task.status = TaskStatus.COMPLETED
                    result_text += " [loop guard stopped extra calls]"
                else:
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

        self.logger.debug("[LLM] Generating execution summary...")

        try:
            self._last_tool_context = "plan_summary"
            response = self._send_message(prompt)
            self._track_usage(response)

            text = response.text or plan.progress_summary()
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
                adapter=self.adapter,
                model_name=GEMINI_PLANNER_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                token_log_path=self._token_log_path,
            )
            self.logger.debug(f"[Router] Created PlannerAgent ({GEMINI_PLANNER_MODEL})")
        return self._planner_agent

    def _build_task_result_summary(
        self, task: Task, labels_before: set[str], labels_after: set[str]
    ) -> dict:
        """Build an informative result summary dict for a completed plan task."""
        result_text = (task.result or "")[:500]
        if result_text in ("", "Done.") and task.tool_calls:
            result_text = f"Tools called: {', '.join(task.tool_calls)}"

        new_labels = labels_after - labels_before
        if new_labels:
            new_label_parts = []
            for lbl in sorted(new_labels):
                entry_info = next(
                    (e for e in get_store().list_entries() if e["label"] == lbl),
                    None,
                )
                if entry_info:
                    new_label_parts.append(
                        f"{lbl} ({entry_info.get('shape', '?')}, "
                        f"{entry_info.get('num_points', '?')} pts, "
                        f"units={entry_info.get('units', '?')}, "
                        f"columns={entry_info.get('columns', [])})"
                    )
                else:
                    new_label_parts.append(lbl)
            result_text += f" | New data: {'; '.join(new_label_parts)}"
        elif task.status == TaskStatus.FAILED:
            result_text += " | No new data added."

        warnings = []
        for tr in getattr(task, "tool_results", []):
            if tr.get("quality_warning"):
                warnings.append(f"{tr.get('label', '?')}: {tr['quality_warning']}")
            if tr.get("time_range_note"):
                warnings.append(tr["time_range_note"])
        if warnings:
            result_text += f" | Warnings: {'; '.join(warnings)}"

        return {
            "description": task.description,
            "status": task.status.value,
            "result_summary": result_text,
            "error": task.error,
        }

    def _execute_plan_task(self, task: Task, plan: TaskPlan) -> None:
        """Execute a single plan task, routing to the appropriate agent.

        Updates the task status in place.

        Args:
            task: The task to execute
            plan: The parent plan (for logging context)
        """
        mission_tag = f" [{task.mission}]" if task.mission else ""
        self.logger.debug(f"[Plan]{mission_tag}: {task.description}", extra=tagged("plan_task"))

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
            instr_lower = task.instruction.lower()
            is_export = "export" in instr_lower or ".png" in instr_lower or ".pdf" in instr_lower

            if is_export:
                # Export is a simple dispatch — handle directly, no need for LLM
                self._handle_export_task(task)
            else:
                # Plot tasks: ensure instruction includes actual labels
                has_tool_ref = "render_plotly_json" in instr_lower
                if not has_tool_ref and entries:
                    all_labels = ",".join(e["label"] for e in entries)
                    task.instruction = (
                        f"Use render_plotly_json to plot {all_labels}. "
                        f"Original request: {task.instruction}"
                    )

                # Inject current plot state for the viz agent
                state = self._renderer.get_current_state()
                if state["has_plot"]:
                    if state.get("figure_json"):
                        import json
                        fig_json_str = json.dumps(state["figure_json"], indent=2)
                        task.instruction += (
                            f"\n\nCurrently displayed: {state['traces']}"
                            f"\n\nCurrent figure_json (modify this, don't rebuild from scratch):\n{fig_json_str}"
                        )
                    else:
                        task.instruction += f"\n\nCurrently displayed: {state['traces']}"
                else:
                    task.instruction += "\n\nNo plot currently displayed."

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

    def _handle_planning_request(
        self, user_message: str, *, structured_time_range: str = ""
    ) -> str:
        """Process a complex multi-step request using the plan-execute-replan loop."""
        self.logger.debug("[PlannerAgent] Starting planner for complex request...")

        # Prefer the structured time_range from the tool call (Gemini-resolved).
        # Fall back to regex extraction only when the structured param is empty.
        self._plan_time_range = None
        if structured_time_range:
            try:
                self._plan_time_range = parse_time_range(structured_time_range)
                self.logger.debug(
                    f"[PlannerAgent] Resolved time range (structured): "
                    f"{self._plan_time_range.to_time_range_string()}"
                )
            except (TimeRangeError, ValueError) as e:
                self.logger.debug(
                    f"[PlannerAgent] Structured time_range parse failed: {e}"
                )

        if not self._plan_time_range:
            # Strip memory context before extracting time range — memory contains
            # date references from past sessions that confuse the regex.
            import re as _re
            clean_msg = _re.sub(
                r'\[CONTEXT FROM LONG-TERM MEMORY\].*?\[END MEMORY CONTEXT\]\s*',
                '', user_message, flags=_re.DOTALL
            )
            self._plan_time_range = self._extract_time_range(clean_msg)
            if self._plan_time_range:
                self.logger.debug(
                    f"[PlannerAgent] Resolved time range (regex fallback): "
                    f"{self._plan_time_range.to_time_range_string()}"
                )

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

        # Surface plan summary in Gradio live log
        plan_summary = response.get("summary") or response.get("reasoning", "")
        if plan_summary:
            if len(plan_summary) > 300:
                plan_summary = plan_summary[:300] + "..."
            self.logger.debug(f"[Planning] {plan_summary}", extra=tagged("progress"))

        tasks_preview = response.get("tasks", [])
        if tasks_preview:
            task_lines = []
            for i, t in enumerate(tasks_preview, 1):
                desc = t.get("description", "?")
                ds = t.get("candidate_datasets")
                ds_str = f" ({', '.join(ds)})" if ds else ""
                task_lines.append(f"  {i}. {desc}{ds_str}")
            self.logger.debug(
                "[Planning] Tasks:\n" + "\n".join(task_lines),
                extra=tagged("progress"),
            )

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
            all_candidates_invalid = False
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

                # Validate candidate_datasets against local metadata cache
                if task.candidate_datasets:
                    valid = []
                    invalid = []
                    for ds_id in task.candidate_datasets:
                        v = validate_dataset_id(ds_id)
                        if v["valid"]:
                            valid.append(ds_id)
                        else:
                            invalid.append(ds_id)
                    if invalid:
                        self.logger.debug(
                            f"[Plan] Stripped invalid candidate_datasets "
                            f"from '{task.description}': {invalid}"
                        )
                    if valid:
                        task.candidate_datasets = valid
                    else:
                        # ALL candidates invalid — flag for re-prompt
                        self.logger.warning(
                            f"[Plan] ALL candidate_datasets invalid for "
                            f"'{task.description}': {invalid}"
                        )
                        all_candidates_invalid = True
                new_tasks.append(task)

            # Log validation summary (round 1 is the long one)
            if round_num == 1:
                valid_count = sum(1 for t in new_tasks if t.candidate_datasets)
                total = len(new_tasks)
                self.logger.debug(
                    f"[Planning] Validated datasets for {valid_count}/{total} tasks",
                    extra=tagged("progress"),
                )

            # If any task has all-invalid candidates, re-prompt the planner
            if all_candidates_invalid:
                invalid_ids = []
                for t in new_tasks:
                    if t.candidate_datasets:
                        for ds_id in t.candidate_datasets:
                            v = validate_dataset_id(ds_id)
                            if not v["valid"]:
                                invalid_ids.append(ds_id)
                correction_msg = (
                    "VALIDATION ERROR: The following dataset IDs do not exist "
                    f"in the local metadata cache: {invalid_ids}. "
                    "Re-emit the same tasks using ONLY dataset IDs from the "
                    "Discovery Results. Do NOT invent dataset IDs."
                )
                self.logger.debug(f"[Plan] Sending correction to planner: {correction_msg}")
                response = planner.continue_planning(
                    [{"description": "Dataset ID validation",
                      "status": "failed",
                      "result_summary": correction_msg,
                      "error": correction_msg}],
                    round_num=round_num,
                    max_rounds=MAX_ROUNDS,
                )
                if response is None:
                    self.logger.debug("[PlannerAgent] Planner error after correction, finalizing")
                    break
                # Re-process the corrected response in the next loop iteration
                continue

            plan.add_tasks(new_tasks)
            store.save(plan)

            self.logger.debug(
                f"[PlannerAgent] Round {round_num}: {len(new_tasks)} tasks "
                f"(status={response['status']})",
                extra=tagged("plan_task"),
            )
            self.logger.debug(format_plan_for_display(plan), extra=tagged("plan_task"))
            self.logger.debug(
                f"[Plan] Executing {len(new_tasks)} task(s) (round {round_num})...",
                extra=tagged("progress"),
            )

            # Execute batch — partition into parallelizable fetch tasks and serial tasks
            special_missions = {"__visualization__", "__data_ops__", "__data_extraction__"}
            fetch_tasks = [t for t in new_tasks if t.mission and t.mission not in special_missions]
            other_tasks = [t for t in new_tasks if t not in fetch_tasks]

            round_results = []
            cancelled = False

            # Run fetch tasks in parallel if multiple independent missions
            from config import PARALLEL_FETCH
            if PARALLEL_FETCH and len(fetch_tasks) > 1 and not self._cancel_event.is_set():
                self.logger.debug(
                    f"[Parallel] Executing {len(fetch_tasks)} fetch tasks concurrently: "
                    f"{[t.mission for t in fetch_tasks]}"
                )
                labels_before = set(e['label'] for e in get_store().list_entries())
                max_workers = min(len(fetch_tasks), 3)

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(self._execute_plan_task, t, plan): t
                        for t in fetch_tasks
                    }
                    for future in as_completed(futures):
                        t = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            t.status = TaskStatus.FAILED
                            t.error = str(e)

                labels_after = set(e['label'] for e in get_store().list_entries())
                # Build result summaries for all parallel tasks
                for task in fetch_tasks:
                    round_results.append(
                        self._build_task_result_summary(task, labels_before, labels_after)
                    )
                store.save(plan)
            else:
                # Run fetch tasks serially (0 or 1 task)
                other_tasks = list(new_tasks)  # all tasks go serial
                fetch_tasks = []

            # Run remaining tasks serially (viz, data_ops, data_extraction, or single fetches)
            for i, task in enumerate(other_tasks):
                if self._cancel_event.is_set():
                    self.logger.info("[Cancel] Stopping plan mid-batch")
                    for remaining in other_tasks[i:]:
                        remaining.status = TaskStatus.SKIPPED
                        remaining.error = "Cancelled by user"
                    cancelled = True
                    break
                labels_before = set(e['label'] for e in get_store().list_entries())
                self._execute_plan_task(task, plan)
                labels_after = set(e['label'] for e in get_store().list_entries())
                round_results.append(
                    self._build_task_result_summary(task, labels_before, labels_after)
                )
                store.save(plan)

            if cancelled:
                store.save(plan)
                break

            # Append current store state so planner knows what data exists
            store_entries = get_store().list_entries()
            if store_entries:
                store_details = [
                    {"label": e["label"], "columns": e.get("columns", []),
                     "shape": e.get("shape", ""), "units": e.get("units", ""),
                     "num_points": e.get("num_points", 0)}
                    for e in store_entries
                ]
                for r in round_results:
                    r["data_in_memory"] = store_details

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
        self.logger.debug("[LLM] Sending message to model...")
        self._last_tool_context = "initial_message"
        response = self._send_message(user_message)
        self._track_usage(response)
        self.logger.debug("[LLM] Response received.")
        self._log_grounding_queries(response)

        max_iterations = 10
        iteration = 0
        consecutive_delegation_errors = 0

        while iteration < max_iterations:
            iteration += 1

            if self._cancel_event.is_set():
                self.logger.info("[Cancel] Stopping orchestrator loop")
                break

            if not response.tool_calls:
                break

            function_calls = response.tool_calls

            # Execute tools — parallel when safe, serial otherwise
            tool_results = self._execute_tools_parallel(function_calls)

            function_responses = []
            has_delegation_error = False
            for tc_id, tool_name, tool_args, result in tool_results:
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
                    self.adapter.make_tool_result_message(
                        tool_name, result, tool_call_id=tc_id
                    )
                )

            # Track consecutive delegation failures
            if has_delegation_error:
                consecutive_delegation_errors += 1
            else:
                consecutive_delegation_errors = 0

            tool_names = [fc.name for fc in function_calls]
            self._last_tool_context = "+".join(tool_names)

            if consecutive_delegation_errors >= 2:
                self.logger.debug(f"[Orchestrator] {consecutive_delegation_errors} consecutive delegation failures, stopping retries")
                # Send results back one more time so LLM can produce a final text answer
                response = self._send_message(function_responses)
                self._track_usage(response)
                break

            self.logger.debug(f"[LLM] Sending {len(function_responses)} tool result(s) back to model...")
            response = self._send_message(function_responses)
            self._track_usage(response)
            self.logger.debug("[LLM] Response received.")
            self._log_grounding_queries(response)

        # Extract text response
        text = response.text or "Done."
        text += self._extract_grounding_sources(response)
        return text

    @staticmethod
    def _wrap_delegation_result(sub_result, store_snapshot=None) -> dict:
        """Convert a sub-agent process_request result into a tool result dict.

        If the sub-agent reported failure (stopped due to errors/loops),
        return status='error' so the orchestrator knows not to retry.

        Args:
            sub_result: Dict from sub-agent's process_request ({text, failed, errors}).
            store_snapshot: Optional list of store entry summaries to include,
                so the orchestrator LLM sees concrete data state after delegation.
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
            result = {
                "status": "error",
                "message": f"Sub-agent failed. Errors: {error_summary}",
                "result": text,
            }
        else:
            result = {"status": "success", "result": text}

        if store_snapshot is not None:
            result["data_in_memory"] = [
                {"label": e["label"], "columns": e.get("columns", []),
                 "shape": e.get("shape", ""), "units": e.get("units", ""),
                 "num_points": e.get("num_points", 0)}
                for e in store_snapshot
            ]
        return result

    def _get_or_create_mission_agent(self, mission_id: str) -> MissionAgent:
        """Get a cached mission agent or create a new one. Thread-safe."""
        with self._mission_agents_lock:
            if mission_id not in self._mission_agents:
                pitfalls = self._memory_store.get_scoped_pitfall_texts(f"mission:{mission_id}")
                self._mission_agents[mission_id] = MissionAgent(
                    mission_id=mission_id,
                    adapter=self.adapter,
                    model_name=SUB_AGENT_MODEL,
                    tool_executor=self._execute_tool_safe,
                    verbose=self.verbose,
                    cancel_event=self._cancel_event,
                    pitfalls=pitfalls,
                    token_log_path=self._token_log_path,
                )
                self.logger.debug(f"[Router] Created {mission_id} mission agent ({SUB_AGENT_MODEL})")
            return self._mission_agents[mission_id]

    def _get_or_create_viz_agent(self) -> VisualizationAgent:
        """Get the cached visualization agent or create a new one."""
        if self._viz_agent is None:
            pitfalls = self._memory_store.get_scoped_pitfall_texts("visualization")
            self._viz_agent = VisualizationAgent(
                adapter=self.adapter,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                gui_mode=self.gui_mode,
                cancel_event=self._cancel_event,
                pitfalls=pitfalls,
                token_log_path=self._token_log_path,
            )
            self.logger.debug(f"[Router] Created Visualization agent ({SUB_AGENT_MODEL})")
        return self._viz_agent

    def _get_or_create_dataops_agent(self) -> DataOpsAgent:
        """Get the cached data ops agent or create a new one."""
        if self._dataops_agent is None:
            self._dataops_agent = DataOpsAgent(
                adapter=self.adapter,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                token_log_path=self._token_log_path,
            )
            self.logger.debug("[Router] Created DataOps agent")
        return self._dataops_agent

    def _get_or_create_data_extraction_agent(self) -> DataExtractionAgent:
        """Get the cached data extraction agent or create a new one."""
        if self._data_extraction_agent is None:
            self._data_extraction_agent = DataExtractionAgent(
                adapter=self.adapter,
                model_name=SUB_AGENT_MODEL,
                tool_executor=self._execute_tool_safe,
                verbose=self.verbose,
                cancel_event=self._cancel_event,
                token_log_path=self._token_log_path,
            )
            self.logger.debug("[Router] Created DataExtraction agent")
        return self._data_extraction_agent

    # ---- Long-term memory (background) ----

    def _ensure_memory_agent_started(self) -> None:
        """Start the background memory agent if not already running."""
        if self._memory_agent is not None and self._memory_agent.is_running:
            return
        self._memory_agent = MemoryAgent(
            adapter=self.adapter,
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
                    adapter=self.adapter,
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
            response = self.adapter.generate(
                model=get_active_model(SUB_AGENT_MODEL),
                contents=prompt,
                temperature=0.7,
            )
            self._last_tool_context = "follow_up_suggestions"
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
            history_dicts = self.chat.get_history()
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
            operations=get_operations_log().get_records(),
        )
        self.logger.debug(f"[Session] Saved ({turn_count} turns, {len(store)} data entries)")

    def load_session(self, session_id: str) -> dict:
        """Restore chat history and DataStore from a saved session.

        Args:
            session_id: The session to load.

        Returns:
            The session metadata dict.
        """
        history_dicts, data_dir, metadata, figure_state, operations = self._session_manager.load_session(session_id)

        # Restore chat with saved history — fall back to fresh chat if
        # the adapter can't reconstruct function_call/function_response parts
        if history_dicts:
            try:
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    history=history_dicts,
                    thinking="high",
                )
            except Exception as e:
                self.logger.warning(
                    f"[Session] Could not restore chat history: {e}. "
                    "Starting fresh chat (data still restored)."
                )
                self.chat = self.adapter.create_chat(
                    model=self.model_name,
                    system_prompt=self._system_prompt,
                    tools=self._tool_schemas,
                    thinking="high",
                )
        else:
            self.chat = self.adapter.create_chat(
                model=self.model_name,
                system_prompt=self._system_prompt,
                tools=self._tool_schemas,
                thinking="high",
            )

        # Restore DataStore
        store = get_store()
        store.clear()
        if data_dir.exists():
            count = store.load_from_directory(data_dir)
            self.logger.debug(f"[Session] Restored {count} data entries")

        # Restore operations log
        ops_log = get_operations_log()
        ops_log.clear()
        if operations:
            ops_log.load_from_records(operations)
            self.logger.debug(f"[Session] Restored {len(operations)} operation records")

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
        self.logger.info(f"[User] {user_message}")

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
        self.chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=self._system_prompt,
            tools=self._tool_schemas,
            thinking="high",
        )
        self._current_plan = None
        self._plan_time_range = None
        self._mission_agents.clear()
        self._viz_agent = None
        self._dataops_agent = None
        self._data_extraction_agent = None
        self._planner_agent = None
        self._renderer.reset()
        get_operations_log().clear()

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
        self.logger.debug(format_plan_for_display(plan), extra=tagged("plan_task"))

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

            self.logger.debug(f"[Plan] Resuming step {i+1}/{len(plan.tasks)}: {task.description}", extra=tagged("plan_task"))

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
