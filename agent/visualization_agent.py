"""
Visualization sub-agent.

Owns all visualization operations via three declarative tools:
- plot_data — create plots from in-memory data
- style_plot — apply aesthetics via key-value params
- manage_plot — structural ops (export, reset, zoom, add/remove traces)

The orchestrator delegates visualization requests here, keeping data
operations in mission agents.
"""

import re

from google import genai

from .base_agent import BaseSubAgent
from .tasks import Task
from knowledge.prompt_builder import build_visualization_prompt

# Visualization agent gets its own tool category + list_fetched_data from data_ops
VIZ_TOOL_CATEGORIES = ["visualization"]
VIZ_EXTRA_TOOLS = ["list_fetched_data"]


def _extract_labels_from_instruction(instruction: str) -> list[str]:
    """Extract data labels from a task instruction that has store contents appended.

    The orchestrator appends lines like "  - AC_H0_MFI.Magnitude (37800 pts)"
    to the instruction. This extracts the label portion.
    """
    labels = []
    for match in re.finditer(r"^\s+-\s+(\S+)\s+\(", instruction, re.MULTILINE):
        labels.append(match.group(1))
    return labels


class VisualizationAgent(BaseSubAgent):
    """A Gemini session specialized for visualization.

    Uses three declarative tools (plot_data, style_plot, manage_plot)
    plus list_fetched_data to discover available data.
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        gui_mode: bool = False,
    ):
        self.gui_mode = gui_mode
        super().__init__(
            client=client,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name="Visualization Agent",
            system_prompt=build_visualization_prompt(gui_mode=gui_mode),
            tool_categories=VIZ_TOOL_CATEGORIES,
            extra_tool_names=VIZ_EXTRA_TOOLS,
        )

    def _get_task_prompt(self, task: Task) -> str:
        """Build an explicit task prompt with concrete label values.

        Extracts actual data labels from the instruction (injected by
        _execute_plan_task) and constructs the exact plot_data call
        so Gemini Flash sees the precise command to execute.

        Note: Export tasks are handled directly by the orchestrator and
        never reach this method.
        """
        labels = _extract_labels_from_instruction(task.instruction)
        labels_str = ",".join(labels) if labels else "LABEL1,LABEL2"

        return (
            f"Execute this task: {task.instruction}\n\n"
            f"Your FIRST call must be: "
            f"plot_data(labels=\"{labels_str}\")\n\n"
            "RULES:\n"
            "- Do NOT call manage_plot(action='reset'), manage_plot(action='get_state'), or list_fetched_data.\n"
            "- Call plot_data with the labels shown above.\n"
            "- After plotting, you may call style_plot to adjust titles/labels/axes."
        )
