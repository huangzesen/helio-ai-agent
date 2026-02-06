"""
Planning logic for multi-step task handling.

This module provides:
- is_complex_request(): Heuristics to detect when a request needs decomposition
- create_plan(): Uses Gemini to decompose a complex request into tasks
"""

import re
from typing import Optional

from google import genai
from google.genai import types

from .tasks import Task, TaskPlan, create_task, create_plan


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
    r"\b(psp|parker)\b.*\b(ace|omni|solo)\b",   # Two spacecraft mentioned
    r"\b(ace)\b.*\b(omni|solo)\b",              # Two spacecraft mentioned

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


PLANNING_PROMPT = """You are a planning assistant for Autoplot, a scientific data visualization tool.
Your job is to decompose complex user requests into a sequence of discrete tasks.

## Available Tools (with required parameters)

- search_datasets(query): Find spacecraft/instrument datasets by keyword
- list_parameters(dataset_id): Get available parameters for a dataset
- fetch_data(dataset_id, parameter_id, time_range): Pull data into memory. Result stored with label "DATASET.PARAM" format. Time range can be "last week", "last 3 days", or "2024-01-15 to 2024-01-20".
- compute_magnitude(source_label, output_label): Calculate vector magnitude from 3-component data
- compute_arithmetic(label_a, label_b, operation, output_label): Add/subtract/multiply/divide two datasets
- compute_running_average(source_label, window_size, output_label): Smooth data with moving window (window_size is number of points, e.g., 60)
- compute_resample(source_label, cadence_seconds, output_label): Change data cadence
- compute_delta(source_label, output_label, mode): Compute differences (mode="difference") or derivatives (mode="derivative")
- plot_data(dataset_id, parameter_id, time_range): Plot CDAWeb data directly
- plot_computed_data(labels): Plot data from memory. Labels is comma-separated, e.g., "AC_H2_MFI.BGSEc,Bmag_smooth"
- export_plot(filepath): Save plot to PNG

## Important Notes
- When user doesn't specify a time range, use "last week" as default
- Labels for fetched data follow the pattern "DATASET.PARAM" (e.g., "AC_H2_MFI.BGSEc")
- For compute operations, use descriptive output_label names (e.g., "Bmag", "velocity_smooth")
- For running averages, a window_size of 60 points is a reasonable default

## Planning Guidelines
1. Each task should be a single, atomic operation
2. Tasks execute sequentially - later tasks can reference results from earlier tasks
3. For comparisons: fetch both datasets → optional computation → plot together
4. For derived quantities: fetch raw data → compute derived value → plot
5. Keep task count minimal - don't split unnecessarily

## Task Instruction Format
Write each instruction as a direct command to call a specific tool. Do NOT use "Use X with..." format.
Instead, write natural language that clearly states what operation to perform with what values.

Example instructions:
- "Fetch data from dataset AC_H2_MFI, parameter BGSEc, for last week"
- "Compute a running average of AC_H2_MFI.BGSEc with window size 60, save as B_smooth"
- "Plot AC_H2_MFI.BGSEc and B_smooth together"

Analyze the request and return a JSON plan. If the request is actually simple (single step), set is_complex=false and provide a single task.

User request: {user_request}"""


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
    prompt = PLANNING_PROMPT.format(user_request=user_request)

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
            print(f"  [Planner] Gemini returned: is_complex={plan_data['is_complex']}, {len(plan_data['tasks'])} tasks")
            print(f"  [Planner] Reasoning: {plan_data['reasoning']}")

        # If Gemini says it's not actually complex, return None to fall back to direct execution
        if not plan_data.get("is_complex", True):
            if verbose:
                print("  [Planner] Request is simple, skipping task decomposition")
            return None

        # Build tasks from the plan
        tasks = []
        for i, task_data in enumerate(plan_data["tasks"]):
            task = create_task(
                description=task_data["description"],
                instruction=task_data["instruction"],
            )
            tasks.append(task)

        if not tasks:
            if verbose:
                print("  [Planner] No tasks generated, falling back to direct execution")
            return None

        plan = create_plan(user_request, tasks)

        if verbose:
            print(f"  [Planner] Created plan with {len(tasks)} tasks:")
            for i, t in enumerate(tasks):
                print(f"    {i+1}. {t.description}")

        return plan

    except Exception as e:
        if verbose:
            print(f"  [Planner] Error creating plan: {e}")
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

        lines.append(f"  {i+1}. [{status_icon}] {task.description}")

        if task.status.value == "failed" and task.error:
            lines.append(f"       Error: {task.error}")

    lines.append("-" * 40)
    lines.append(f"Progress: {plan.progress_summary()}")

    return "\n".join(lines)
