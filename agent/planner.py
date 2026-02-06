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
- custom_operation(source_label, pandas_code, output_label, description): Apply any pandas/numpy operation. Code operates on `df` (DataFrame) and assigns to `result`. Examples: magnitude, smoothing, resampling, arithmetic, derivatives, normalization, clipping.
- describe_data(label): Get statistical summary (min, max, mean, std, percentiles, NaN count, cadence) of an in-memory timeseries. Use when user says "describe", "summarize", or asks about data characteristics.
- plot_data(dataset_id, parameter_id, time_range): Plot CDAWeb data directly
- plot_computed_data(labels): Plot data from memory. Labels is comma-separated, e.g., "AC_H2_MFI.BGSEc,Bmag_smooth"
- export_plot(filepath): Save current plot to PNG
- save_data(label, filename): Export in-memory timeseries to CSV file. Use when user says "save to file" or "export data".

## Known Dataset IDs and Parameters (use these exact values with fetch_data)
- PSP: dataset=PSP_FLD_L2_MAG_RTN_1MIN, param=psp_fld_l2_mag_RTN_1min (magnetic)
- Solar Orbiter: dataset=SOLO_L2_MAG-RTN-NORMAL-1-MINUTE, param=B_RTN (magnetic)
- ACE: dataset=AC_H2_MFI, param=BGSEc (magnetic vector GSE); dataset=AC_H0_SWE, param=Vp (plasma)
- OMNI: dataset=OMNI_HRO_1MIN (combined, use list_parameters to find params)
- Wind: dataset=WI_H2_MFI, param=BGSE (magnetic vector GSE); dataset=WI_H1_SWE (plasma)
- DSCOVR: dataset=DSCOVR_H0_MAG, param=B1GSE (magnetic vector GSE); dataset=DSCOVR_H1_FC (plasma)
- MMS: dataset=MMS1_FGM_SRVY_L2 (magnetic, use list_parameters); dataset=MMS1_FPI_FAST_L2_DIS-MOMS (plasma)
- STEREO-A: dataset=STA_L2_MAG_RTN (magnetic, use list_parameters); dataset=STA_L2_PLA_1DMAX_1MIN (plasma)

IMPORTANT: Different spacecraft have DIFFERENT parameter names. Do NOT assume one spacecraft's
parameter name works for another (e.g., ACE uses "BGSEc" but Wind uses "BGSE"). When unsure,
include a list_parameters step before fetching.

## Important Notes
- When user doesn't specify a time range, use "last week" as default
- Labels for fetched data follow the pattern "DATASET.PARAM" (e.g., "AC_H2_MFI.BGSEc")
- For compute operations, use descriptive output_label names (e.g., "Bmag", "velocity_smooth")
- For running averages, a window_size of 60 points is a reasonable default
- If you're not sure which parameter to use for a dataset, include a search_datasets step first

## Planning Guidelines
1. Each task should be a single, atomic operation — do ONLY what the instruction says
2. Tasks execute sequentially - later tasks can reference results from earlier tasks
3. For comparisons: fetch both datasets → optional computation → plot together
4. For derived quantities: fetch raw data → compute derived value → plot
5. Keep task count minimal - don't split unnecessarily
6. Do NOT include plotting steps unless the user explicitly asked to plot
7. A "fetch" task should ONLY fetch data, not also plot or describe it

## Task Instruction Format
CRITICAL: Every fetch_data instruction MUST include the exact dataset_id AND parameter_id from the
Known Dataset IDs section above. Never use vague descriptions like "fetch Wind magnetic field" — always
specify "fetch_data with dataset_id=WI_H2_MFI, parameter_id=BGSE".

Every custom_operation instruction MUST include the exact source_label (e.g., "WI_H2_MFI.BGSE").

Example instructions:
- "Fetch data from dataset AC_H2_MFI, parameter BGSEc, for last week"
- "Fetch data from dataset WI_H2_MFI, parameter BGSE, for last week"
- "Compute the magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag"
- "Describe the data labeled ACE_Bmag"
- "Plot ACE_Bmag and Wind_Bmag together"
- "Export the plot to output.png"

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
