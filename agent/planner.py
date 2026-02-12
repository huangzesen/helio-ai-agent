"""
Planning logic for multi-step task handling.

This module provides:
- is_complex_request(): Heuristics to detect when a request needs decomposition
- PlannerAgent: Chat-based planner with plan-execute-replan loop
- format_plan_for_display(): Human-readable plan rendering
"""

import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from google import genai
from google.genai import types

from .logging import get_logger, log_token_usage
from .model_fallback import get_active_model
from .base_agent import _GEMINI_WARN_INTERVAL, _GEMINI_RETRY_TIMEOUT, _GEMINI_MAX_RETRIES
from .tasks import Task, TaskPlan, create_task, create_plan
from .tools import get_tool_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from knowledge.prompt_builder import build_planner_agent_prompt, build_discovery_prompt

logger = get_logger()

# Tool categories the planner can use for dataset discovery
PLANNER_TOOL_CATEGORIES = ["discovery"]
PLANNER_EXTRA_TOOLS = ["list_fetched_data"]

MAX_ROUNDS = 5

# Regex patterns that indicate a complex, multi-step request
COMPLEXITY_INDICATORS = [
    # Multiple conjunctions
    r"\band\b.*\band\b",                        # "fetch X and compute Y and plot Z"
    r"\bthen\b",                                # Sequential language
    r"\bafter\b",                               # Sequential language
    r"\bfirst\b.*\bthen\b",                     # "first X, then Y"
    r"\bfinally\b",                             # Sequential language

    # Comparisons that require multiple data sources
    r"\bcompare\b",                             # "compare X with Y"
    r"\bdifference\s+between\b",               # "difference between X and Y"
    r"\bvs\.?\b",                               # "X vs Y"

    # Multiple data operations
    r"\b(plot|show|visualize)\b.*\b(and|both|with)\b.*\b(plot|show|visualize|data)\b",  # Multiple plots
    r"\b(fetch|get|retrieve)\b.*\band\b.*\b(compute|calculate)\b",  # Fetch + compute
    r"\b(compute|calculate)\b.*\band\b.*\b(plot|show)\b",  # Compute + plot

    # Multiple spacecraft or datasets
    # Note: "wind" alone is ambiguous ("solar wind" vs Wind spacecraft),
    # so we only match "wind" as a spacecraft when NOT preceded by "solar "
    r"\b(psp|parker)\b.*\b(ace|omni|solo|dscovr|mms|stereo)\b",
    r"\b(psp|parker)\b.*(?<!solar )\bwind\b",
    r"\b(ace)\b.*\b(omni|solo|dscovr|mms|stereo)\b",
    r"\b(ace)\b.*(?<!solar )\bwind\b",
    r"(?<!solar )\bwind\b.*\b(ace|omni|solo|psp|parker|dscovr|mms|stereo)\b",
    r"\b(dscovr)\b.*\b(ace|omni|solo|psp|parker|mms|stereo)\b",
    r"\b(mms)\b.*\b(ace|omni|solo|dscovr|stereo)\b",
    r"\b(stereo)\b.*\b(ace|omni|solo|psp|parker|dscovr|mms)\b",

    # Data pipeline phrases
    r"\bsmooth\b.*\band\b.*\bplot\b",           # Smooth and plot
    r"\baverage\b.*\band\b.*\b(compare|plot)\b", # Average and compare/plot
    r"\bmagnitude\b.*\band\b",                  # Magnitude and something else
]


def is_complex_request(text: str) -> bool:
    """Determine if a request is complex enough to warrant task decomposition.

    Uses regex heuristics to detect patterns that suggest multi-step operations.
    Simple requests (single plot, single query) should be handled directly.

    Args:
        text: The user's request

    Returns:
        True if the request appears to need multiple steps
    """
    text_lower = text.lower()

    for pattern in COMPLEXITY_INDICATORS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    return False


# JSON schema for PlannerAgent's structured output
PLANNER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["continue", "done"],
        },
        "reasoning": {
            "type": "string",
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "instruction": {"type": "string"},
                    "mission": {"type": "string"},
                    "candidate_datasets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["description", "instruction"],
            },
        },
        "summary": {
            "type": "string",
        },
    },
    "required": ["status", "reasoning", "tasks"],
}


class PlannerAgent:
    """Chat-based planner that decomposes complex requests into task batches.

    Uses a two-phase approach when tools are available:
    1. **Discovery phase**: A tool-calling session verifies dataset IDs and
       parameter names via discovery tools (search_datasets, list_parameters, etc.).
    2. **Planning phase**: A JSON-schema-enforced session produces the task plan,
       enriched with the discovery context from phase 1.

    Without a tool_executor, skips the discovery phase (legacy mode).
    The planning phase always uses JSON schema enforcement for guaranteed output.
    """

    def __init__(self, client: genai.Client, model_name: str,
                 tool_executor=None, verbose: bool = False,
                 cancel_event=None, token_log_path=None):
        self.client = client
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.verbose = verbose
        self._cancel_event = cancel_event
        self._token_log_path = token_log_path
        self._chat = None
        self._token_usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        self._api_calls = 0
        self._last_tool_context = "send_message"
        self._timeout_pool = ThreadPoolExecutor(max_workers=1)
        self.logger = get_logger()

        # Build function declarations when tools are available
        self._function_declarations = []
        if self.tool_executor is not None:
            for tool_schema in get_tool_schemas(
                categories=PLANNER_TOOL_CATEGORIES,
                extra_names=PLANNER_EXTRA_TOOLS,
            ):
                fd = types.FunctionDeclaration(
                    name=tool_schema["name"],
                    description=tool_schema["description"],
                    parameters=tool_schema["parameters"],
                )
                self._function_declarations.append(fd)

    def _send_with_timeout(self, chat, message):
        """Send a message to Gemini with periodic warnings and retry on timeout."""
        last_exc = None
        for attempt in range(1 + _GEMINI_MAX_RETRIES):
            future: Future = self._timeout_pool.submit(
                chat.send_message, message
            )
            t0 = time.monotonic()
            try:
                while True:
                    elapsed = time.monotonic() - t0
                    remaining = _GEMINI_RETRY_TIMEOUT - elapsed
                    if remaining <= 0:
                        break
                    wait = min(_GEMINI_WARN_INTERVAL, remaining)
                    try:
                        return future.result(timeout=wait)
                    except TimeoutError:
                        elapsed = time.monotonic() - t0
                        if elapsed >= _GEMINI_RETRY_TIMEOUT:
                            break
                        self.logger.warning(
                            f"[PlannerAgent] Gemini API not responding "
                            f"after {elapsed:.0f}s (attempt {attempt + 1})..."
                        )

                elapsed = time.monotonic() - t0
                future.cancel()
                last_exc = TimeoutError(
                    f"Gemini API call timed out after {elapsed:.0f}s"
                )
                if attempt < _GEMINI_MAX_RETRIES:
                    self.logger.warning(
                        f"[PlannerAgent] Gemini API timed out after "
                        f"{elapsed:.0f}s, retrying ({attempt + 1}/{_GEMINI_MAX_RETRIES})..."
                    )
                else:
                    self.logger.error(
                        f"[PlannerAgent] Gemini API timed out after "
                        f"{elapsed:.0f}s, no retries left"
                    )
            except Exception:
                raise
        raise last_exc

    def _track_usage(self, response):
        """Accumulate token usage from a Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        call_input = 0
        call_output = 0
        call_thinking = 0
        if meta:
            call_input = getattr(meta, "prompt_token_count", 0) or 0
            call_output = getattr(meta, "candidates_token_count", 0) or 0
            call_thinking = getattr(meta, "thoughts_token_count", 0) or 0
            self._token_usage["input_tokens"] += call_input
            self._token_usage["output_tokens"] += call_output
            self._token_usage["thinking_tokens"] += call_thinking
        self._api_calls += 1
        log_token_usage(
            agent_name="PlannerAgent",
            input_tokens=call_input,
            output_tokens=call_output,
            thinking_tokens=call_thinking,
            cumulative_input=self._token_usage["input_tokens"],
            cumulative_output=self._token_usage["output_tokens"],
            cumulative_thinking=self._token_usage["thinking_tokens"],
            api_calls=self._api_calls,
            tool_context=self._last_tool_context,
            token_log_path=self._token_log_path,
        )
        if self.verbose:
            from .thinking import extract_thoughts
            from .logging import tagged
            for thought in extract_thoughts(response):
                # Full text to terminal/file (untagged)
                logger.debug(f"[Thinking] {thought}")
                # Preview for Gradio (tagged) — Gradio handler shows these inline
                preview = thought[:500] + ("..." if len(thought) > 500 else "")
                logger.debug(f"[Thinking] {preview}", extra={**tagged("thinking"), "skip_file": True})

    def _parse_response(self, response) -> Optional[dict]:
        """Parse JSON response from Gemini, normalizing mission fields.

        The planning phase always uses JSON schema enforcement, so the
        response text is guaranteed to be valid JSON.
        """
        try:
            text = response.text
            if not text:
                return None
            data = json.loads(text)

            # Normalize mission "null"/"none" strings to None
            for task_data in data.get("tasks", []):
                mission = task_data.get("mission")
                if isinstance(mission, str) and mission.lower() in ("null", "none", ""):
                    task_data["mission"] = None

            return data
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"[PlannerAgent] Failed to parse response: {e}")
            return None

    def _run_discovery(self, user_request: str) -> str:
        """Phase 1: Run discovery tools to gather dataset/parameter info.

        Creates a one-shot tool-calling chat that researches the user's request
        and returns a text summary of what it found.  The raw
        ``list_parameters`` results are captured and appended as a structured
        reference so the planning LLM can select candidate dataset IDs based
        on verified parameter availability.

        Args:
            user_request: The user's original request.

        Returns:
            Text summary of discovery findings with verified parameter reference.
        """
        discovery_prompt = build_discovery_prompt()

        config = types.GenerateContentConfig(
            system_instruction=discovery_prompt,
            tools=[types.Tool(function_declarations=self._function_declarations)],
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level="LOW",
            ),
        )

        chat = self.client.chats.create(
            model=get_active_model(self.model_name),
            config=config,
        )

        if self.verbose:
            logger.debug(f"[PlannerAgent] Discovery phase for: {user_request}")

        self._last_tool_context = "discovery_initial"
        response = self._send_with_timeout(chat, user_request)
        self._track_usage(response)

        # Collect raw tool results so we can extract list_parameters data
        tool_results = {}
        response = run_tool_loop(
            chat=chat,
            response=response,
            tool_executor=self.tool_executor,
            agent_name="PlannerAgent/Discovery",
            max_total_calls=20,
            max_iterations=8,
            track_usage=self._track_usage,
            collect_tool_results=tool_results,
            cancel_event=self._cancel_event,
            send_fn=lambda msg: self._send_with_timeout(chat, msg),
        )

        text = extract_text_from_response(response)
        if self.verbose and text:
            logger.debug(f"[PlannerAgent] Discovery result: {text}")

        # Build a structured parameter reference from raw list_parameters results
        param_ref = self._build_parameter_reference(tool_results)
        if param_ref:
            text = (text or "") + "\n\n" + param_ref

        return text

    @staticmethod
    def _build_parameter_reference(tool_results: dict) -> str:
        """Build a structured dataset reference from collected tool results.

        Combines browse_datasets (broad catalog view) and list_parameters
        (verified parameter details) into a single reference the planning
        LLM uses to select candidate dataset IDs.

        Args:
            tool_results: Dict of {tool_name: [{args, result}, ...]} from
                the discovery tool loop.

        Returns:
            Formatted reference string, or empty string if no data.
        """
        browse_results = tool_results.get("browse_datasets", [])
        lp_results = tool_results.get("list_parameters", [])
        avail_results = tool_results.get("get_data_availability", [])

        if not browse_results and not lp_results:
            return ""

        # Build availability lookup
        availability = {}
        for entry in avail_results:
            ds_id = entry["args"].get("dataset_id", "")
            result = entry["result"]
            if result.get("status") != "error":
                start = result.get("start_date", "?")
                end = result.get("end_date", "?")
                availability[ds_id] = f"{start} to {end}"

        # Build verified parameters lookup
        verified_params = {}
        for entry in lp_results:
            ds_id = entry["args"].get("dataset_id", "unknown")
            result = entry["result"]
            params = result.get("parameters", [])
            if params and result.get("status") != "error":
                verified_params[ds_id] = params

        lines = [
            "## DATASET REFERENCE",
            "",
            "Use ONLY dataset IDs from this reference for candidate_datasets.",
            "",
        ]

        # Section 1: Browse results — grouped by mission, annotated with type
        for entry in browse_results:
            result = entry["result"]
            if result.get("status") == "error":
                continue
            mission_id = result.get("mission_id", "?")
            datasets = result.get("datasets", [])
            if not datasets:
                continue

            lines.append(f"### {mission_id} ({len(datasets)} datasets)")

            # Group by type for readability
            by_type = {}
            for ds in datasets:
                dtype = ds.get("type", "other")
                by_type.setdefault(dtype, []).append(ds)

            for dtype, ds_list in by_type.items():
                lines.append(f"  {dtype}:")
                for ds in ds_list:
                    ds_id = ds["id"]
                    start = ds.get("start_date", "?")
                    stop = ds.get("stop_date", "?")
                    pcnt = ds.get("parameter_count", 0)
                    inst = ds.get("instrument", "")
                    verified = " [VERIFIED]" if ds_id in verified_params else ""
                    inst_tag = f" ({inst})" if inst else ""
                    lines.append(f"    - {ds_id}{inst_tag}: {start} to {stop}, {pcnt} params{verified}")
                lines.append("")

        # Section 2: Verified parameter details (for top picks only)
        if verified_params:
            lines.append("### Verified Parameters")
            lines.append("")
            for ds_id, params in verified_params.items():
                avail = availability.get(ds_id, "unknown")
                lines.append(f"Dataset {ds_id} (available: {avail}):")
                for p in params:
                    name = p.get("name", "?")
                    if name == "Time":
                        continue
                    units = p.get("units") or ""
                    size = p.get("size")
                    desc_parts = []
                    if units:
                        desc_parts.append(units)
                    if size and size != [1]:
                        desc_parts.append(f"size={size}")
                    lines.append(f"  - {name}" + (f" ({', '.join(desc_parts)})" if desc_parts else ""))
                lines.append("")

        # Fallback: if no browse results, still show list_parameters data
        if not browse_results and lp_results:
            lines.append("### Verified Parameters")
            lines.append("")
            for entry in lp_results:
                ds_id = entry["args"].get("dataset_id", "unknown")
                result = entry["result"]
                params = result.get("parameters", [])
                if not params or result.get("status") == "error":
                    lines.append(f"Dataset {ds_id}: NO PARAMETERS AVAILABLE (skip this dataset)")
                    continue
                avail = availability.get(ds_id, "unknown")
                lines.append(f"Dataset {ds_id} (available: {avail}):")
                for p in params:
                    name = p.get("name", "?")
                    if name == "Time":
                        continue
                    units = p.get("units") or ""
                    size = p.get("size")
                    desc_parts = []
                    if units:
                        desc_parts.append(units)
                    if size and size != [1]:
                        desc_parts.append(f"size={size}")
                    lines.append(f"  - {name}" + (f" ({', '.join(desc_parts)})" if desc_parts else ""))
                lines.append("")

        return "\n".join(lines)

    def start_planning(self, user_request: str) -> Optional[dict]:
        """Begin planning by sending the user request to a fresh chat.

        When tools are available, runs a two-phase process:
        1. Discovery phase — calls tools to verify datasets/parameters.
        2. Planning phase — JSON-schema-enforced chat produces the task plan.

        Without tools, goes straight to the planning phase.

        Args:
            user_request: The user's original request.

        Returns:
            Dict with {status, reasoning, tasks, summary} or None on failure.
        """
        try:
            # Phase 1: Discovery (only when tools are available)
            discovery_context = ""
            if self._function_declarations and self.tool_executor:
                discovery_context = self._run_discovery(user_request)

            # Phase 2: Planning (always JSON-schema-enforced)
            system_prompt = build_planner_agent_prompt()

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=PLANNER_RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level="HIGH",
                ),
            )

            self._chat = self.client.chats.create(
                model=get_active_model(self.model_name),
                config=config,
            )

            # Build the planning message with discovery context
            if discovery_context:
                planning_message = (
                    f"{user_request}\n\n"
                    f"## Discovery Results\n\n"
                    f"The following dataset and parameter information was verified:\n\n"
                    f"{discovery_context}"
                )
            else:
                planning_message = user_request

            if self.verbose:
                logger.debug(f"[PlannerAgent] Starting planning for: {user_request}")

            self._last_tool_context = "planning_initial"
            response = self._send_with_timeout(self._chat, planning_message)
            self._track_usage(response)

            result = self._parse_response(response)
            if result and self.verbose:
                logger.debug(
                    f"[PlannerAgent] Round 1: status={result['status']}, "
                    f"{len(result.get('tasks', []))} tasks"
                )
                self._log_plan_details(result, round_num=1)

            return result

        except Exception as e:
            logger.warning(f"[PlannerAgent] Error in start_planning: {e}")
            return None

    def continue_planning(self, round_results: list[dict],
                           round_num: int = 0, max_rounds: int = MAX_ROUNDS) -> Optional[dict]:
        """Send execution results back to the planner for the next round.

        Args:
            round_results: List of dicts with {description, status, result_summary, error}
            round_num: Current round number (1-based).
            max_rounds: Maximum number of rounds allowed.

        Returns:
            Dict with {status, reasoning, tasks, summary} or None on failure.
        """
        if self._chat is None:
            logger.warning("[PlannerAgent] No active chat session for continue_planning")
            return None

        try:
            # Format results as structured text
            lines = ["Execution results:"]
            for r in round_results:
                status = r.get("status", "unknown")
                desc = r.get("description", "")
                line = f"- Task: {desc} | Status: {status}"
                if r.get("result_summary"):
                    line += f" | Result: {r['result_summary']}"
                if r.get("error"):
                    line += f" | Error: {r['error']}"
                lines.append(line)

            # Include current data-store state so planner can make informed decisions
            data_details = None
            for r in round_results:
                if r.get("data_in_memory"):
                    data_details = r["data_in_memory"]
                    break
            if data_details:
                lines.append("\nData currently in memory:")
                for d in data_details:
                    if isinstance(d, dict):
                        cols = d.get("columns", [])
                        cols_str = f", columns={cols}" if cols else ""
                        lines.append(
                            f"  - {d['label']} ({d.get('shape', '?')}, "
                            f"{d.get('num_points', '?')} pts, "
                            f"units={d.get('units', '?')}{cols_str})"
                        )
                    else:
                        # Backward compat: flat label string
                        lines.append(f"  - {d}")
                label_names = [d["label"] if isinstance(d, dict) else d for d in data_details]
                logger.debug(f"[Planner] Data in memory: {', '.join(label_names)}")

            # Collect ALL failed task descriptions (current + previous rounds)
            failed_descs = []
            for r in round_results:
                if r.get("status") == "failed":
                    failed_descs.append(r.get("description", "unknown"))
            if failed_descs:
                lines.append("\n## IMPORTANT: The following tasks FAILED and must NOT be retried:")
                for desc in failed_descs:
                    lines.append(f"  - {desc}")
                lines.append("Do NOT create new tasks that attempt the same searches. "
                             "Proceed with available data or set status='done'.")

            # Round budget awareness
            remaining = max_rounds - round_num
            if remaining <= 2 and remaining > 0:
                lines.append(f"\n## BUDGET WARNING: Only {remaining} round(s) remaining.")
                lines.append("Prioritize essential tasks. Consider setting status='done' with partial results.")
            if remaining <= 0:
                lines.append("\n## FINAL ROUND: This is the last round. Set status='done' unless critical work remains.")

            message = "\n".join(lines)

            if self.verbose:
                logger.debug(f"[PlannerAgent] Sending results:\n{message}")

            self._last_tool_context = f"continue_planning_round{round_num}"
            response = self._send_with_timeout(self._chat, message)
            self._track_usage(response)

            result = self._parse_response(response)
            if result and self.verbose:
                logger.debug(
                    f"[PlannerAgent] Next round: status={result['status']}, "
                    f"{len(result.get('tasks', []))} tasks"
                )
                self._log_plan_details(result, round_num=round_num)

            return result

        except Exception as e:
            logger.warning(f"[PlannerAgent] Error in continue_planning: {e}")
            return None

    def _log_plan_details(self, result: dict, round_num: int) -> None:
        """Log full task details (instructions, candidates, reasoning) for debugging."""
        if not result:
            return
        lines = [f"[PlannerAgent] === Round {round_num} Plan Details ==="]
        if result.get("reasoning"):
            lines.append(f"  Reasoning: {result['reasoning']}")
        for i, task in enumerate(result.get("tasks", []), 1):
            lines.append(f"  Task {i}:")
            lines.append(f"    description: {task.get('description', '?')}")
            lines.append(f"    mission: {task.get('mission', 'null')}")
            lines.append(f"    instruction: {task.get('instruction', '?')}")
            candidates = task.get("candidate_datasets")
            if candidates:
                lines.append(f"    candidate_datasets: {candidates}")
        if result.get("summary"):
            lines.append(f"  Summary: {result['summary']}")
        logger.debug("\n".join(lines))

    def get_token_usage(self) -> dict:
        """Return accumulated token usage."""
        return {
            "input_tokens": self._token_usage["input_tokens"],
            "output_tokens": self._token_usage["output_tokens"],
            "thinking_tokens": self._token_usage["thinking_tokens"],
            "api_calls": 0,  # Planner calls tracked separately from api_calls count
        }

    def reset(self):
        """Reset the chat session."""
        self._chat = None


def format_plan_for_display(plan: TaskPlan) -> str:
    """Format a plan for display to the user.

    Groups tasks by round when round > 0.

    Args:
        plan: The plan to format

    Returns:
        Human-readable string representation
    """
    lines = [f"Plan: {len(plan.tasks)} steps"]
    lines.append("-" * 40)

    # Check if any task has a non-zero round
    has_rounds = any(t.round > 0 for t in plan.tasks)

    if has_rounds:
        # Group by round
        rounds: dict[int, list[tuple[int, Task]]] = {}
        for i, task in enumerate(plan.tasks):
            rounds.setdefault(task.round, []).append((i, task))

        for round_num in sorted(rounds.keys()):
            if round_num > 0:
                lines.append(f"  Round {round_num}:")
            for i, task in rounds[round_num]:
                status_icon = {
                    "pending": "o",
                    "in_progress": "*",
                    "completed": "+",
                    "failed": "x",
                    "skipped": "-",
                }.get(task.status.value, "?")

                mission_tag = f" [{task.mission}]" if task.mission else ""
                lines.append(f"  {i+1}. [{status_icon}]{mission_tag} {task.description}")

                if task.status.value == "failed" and task.error:
                    lines.append(f"       Error: {task.error}")
    else:
        for i, task in enumerate(plan.tasks):
            status_icon = {
                "pending": "o",
                "in_progress": "*",
                "completed": "+",
                "failed": "x",
                "skipped": "-",
            }.get(task.status.value, "?")

            mission_tag = f" [{task.mission}]" if task.mission else ""
            lines.append(f"  {i+1}. [{status_icon}]{mission_tag} {task.description}")

            if task.status.value == "failed" and task.error:
                lines.append(f"       Error: {task.error}")

    lines.append("-" * 40)
    lines.append(f"Progress: {plan.progress_summary()}")

    return "\n".join(lines)
