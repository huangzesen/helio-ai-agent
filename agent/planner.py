"""
Planning logic for multi-step task handling.

This module provides:
- is_complex_request(): Heuristics to detect when a request needs decomposition
- create_plan(): Uses Gemini to decompose a complex request into tasks

The planning prompt is dynamically generated from the spacecraft catalog
via knowledge/prompt_builder.py — no hardcoded dataset references.
"""

import re
from typing import Optional

from google import genai
from google.genai import types

from .logging import get_logger
from .tasks import Task, TaskPlan, create_task, create_plan
from knowledge.prompt_builder import build_planning_prompt

logger = get_logger()


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


# JSON schema for Gemini's planning output
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "is_complex": {
            "type": "boolean",
            "description": "Whether this request truly requires multiple steps"
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Brief human-readable description of the task"
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Detailed instruction for executing this task, including tool names and parameters"
                    },
                    "mission": {
                        "type": "string",
                        "description": "Spacecraft ID this task belongs to (e.g., 'PSP', 'ACE', 'OMNI'). Null for cross-mission tasks like comparison plots."
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices (0-based) of tasks that must complete before this one. Empty if no dependencies."
                    }
                },
                "required": ["description", "instruction"]
            },
            "description": "Ordered list of tasks to accomplish the user's request"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of why these tasks were chosen"
        }
    },
    "required": ["is_complex", "tasks", "reasoning"]
}


# Generate the planning prompt template once at import time.
# Contains a {user_request} placeholder filled in by create_plan_from_request().
PLANNING_PROMPT = build_planning_prompt()


def create_plan_from_request(
    client: genai.Client,
    model_name: str,
    user_request: str,
    verbose: bool = False,
) -> Optional[TaskPlan]:
    """Use Gemini to decompose a complex request into tasks.

    Args:
        client: Initialized Gemini client
        model_name: Model to use (e.g., "gemini-2.5-flash")
        user_request: The user's original request
        verbose: If True, print debug information

    Returns:
        TaskPlan with decomposed tasks, or None if planning fails
    """
    prompt = PLANNING_PROMPT.replace("{user_request}", user_request)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PLAN_SCHEMA,
            ),
        )

        # Parse the JSON response
        import json
        plan_data = json.loads(response.text)

        if verbose:
            logger.debug("[Planner] Gemini returned: is_complex=%s, %d tasks", plan_data['is_complex'], len(plan_data['tasks']))
            logger.debug("[Planner] Reasoning: %s", plan_data['reasoning'])

        # If Gemini says it's not actually complex, return None to fall back to direct execution
        if not plan_data.get("is_complex", True):
            if verbose:
                logger.debug("[Planner] Request is simple, skipping task decomposition")
            return None

        # Build tasks from the plan
        tasks = []
        for i, task_data in enumerate(plan_data["tasks"]):
            # Normalize mission: "null"/"none" strings → None
            mission = task_data.get("mission")
            if isinstance(mission, str) and mission.lower() in ("null", "none", ""):
                mission = None
            task = create_task(
                description=task_data["description"],
                instruction=task_data["instruction"],
                mission=mission,
            )
            tasks.append(task)

        # Resolve depends_on indices to task IDs
        for i, task_data in enumerate(plan_data["tasks"]):
            dep_indices = task_data.get("depends_on", [])
            for idx in dep_indices:
                if isinstance(idx, int) and 0 <= idx < len(tasks):
                    tasks[i].depends_on.append(tasks[idx].id)

        if not tasks:
            if verbose:
                logger.debug("[Planner] No tasks generated, falling back to direct execution")
            return None

        plan = create_plan(user_request, tasks)

        if verbose:
            logger.debug("[Planner] Created plan with %d tasks:", len(tasks))
            for i, t in enumerate(tasks):
                logger.debug("  %d. %s", i + 1, t.description)

        return plan

    except Exception as e:
        if verbose:
            logger.warning("[Planner] Error creating plan: %s", e)
        return None


def format_plan_for_display(plan: TaskPlan) -> str:
    """Format a plan for display to the user.

    Args:
        plan: The plan to format

    Returns:
        Human-readable string representation
    """
    lines = [f"Plan: {len(plan.tasks)} steps"]
    lines.append("-" * 40)

    for i, task in enumerate(plan.tasks):
        # Use ASCII characters for Windows compatibility
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
