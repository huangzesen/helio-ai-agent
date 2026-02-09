"""
Planning logic for multi-step task handling.

This module provides:
- is_complex_request(): Heuristics to detect when a request needs decomposition
- PlannerAgent: Chat-based planner with plan-execute-replan loop
- format_plan_for_display(): Human-readable plan rendering
"""

import json
import re
from typing import Optional

from google import genai
from google.genai import types

from .logging import get_logger
from .tasks import Task, TaskPlan, create_task, create_plan
from knowledge.prompt_builder import build_planner_agent_prompt

logger = get_logger()

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

    Uses a stateful Gemini chat session with structured JSON output.
    The planner emits task batches, observes execution results, and adapts.
    """

    def __init__(self, client: genai.Client, model_name: str, verbose: bool = False):
        self.client = client
        self.model_name = model_name
        self.verbose = verbose
        self._chat = None
        self._token_usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

    def _track_usage(self, response):
        """Accumulate token usage from a Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            self._token_usage["input_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
            self._token_usage["output_tokens"] += getattr(meta, "candidates_token_count", 0) or 0
            self._token_usage["thinking_tokens"] += getattr(meta, "thoughts_token_count", 0) or 0
        if self.verbose:
            from .thinking import extract_thoughts
            for thought in extract_thoughts(response):
                preview = thought[:200] + "..." if len(thought) > 200 else thought
                logger.debug(f"[Thinking] {preview}")

    def _parse_response(self, response) -> Optional[dict]:
        """Parse JSON response from Gemini, normalizing mission fields."""
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

    def start_planning(self, user_request: str) -> Optional[dict]:
        """Begin planning by sending the user request to a fresh chat.

        Args:
            user_request: The user's original request.

        Returns:
            Dict with {status, reasoning, tasks, summary} or None on failure.
        """
        try:
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
                model=self.model_name,
                config=config,
            )

            if self.verbose:
                logger.debug(f"[PlannerAgent] Starting planning for: {user_request[:80]}...")

            response = self._chat.send_message(user_request)
            self._track_usage(response)

            result = self._parse_response(response)
            if result and self.verbose:
                logger.debug(
                    f"[PlannerAgent] Round 1: status={result['status']}, "
                    f"{len(result.get('tasks', []))} tasks"
                )

            return result

        except Exception as e:
            logger.warning(f"[PlannerAgent] Error in start_planning: {e}")
            return None

    def continue_planning(self, round_results: list[dict]) -> Optional[dict]:
        """Send execution results back to the planner for the next round.

        Args:
            round_results: List of dicts with {description, status, result_summary, error}

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

            # Include current data-store state so planner avoids redundant tasks
            data_labels = None
            for r in round_results:
                if r.get("data_in_memory"):
                    data_labels = r["data_in_memory"]
                    break
            if data_labels:
                lines.append(f"\nData currently in memory: {', '.join(data_labels)}")

            message = "\n".join(lines)

            if self.verbose:
                logger.debug(f"[PlannerAgent] Sending results:\n{message}")

            response = self._chat.send_message(message)
            self._track_usage(response)

            result = self._parse_response(response)
            if result and self.verbose:
                logger.debug(
                    f"[PlannerAgent] Next round: status={result['status']}, "
                    f"{len(result.get('tasks', []))} tasks"
                )

            return result

        except Exception as e:
            logger.warning(f"[PlannerAgent] Error in continue_planning: {e}")
            return None

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
