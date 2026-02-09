"""
Visualization sub-agent.

Owns all visualization operations via a single `execute_visualization`
tool backed by the method registry. The orchestrator delegates visualization
requests here, keeping data operations in mission agents.
"""

from google import genai

from .base_agent import BaseSubAgent
from .tasks import Task
from knowledge.prompt_builder import build_visualization_prompt

# Visualization agent gets its own tool category + list_fetched_data from data_ops
VIZ_TOOL_CATEGORIES = ["visualization"]
VIZ_EXTRA_TOOLS = ["list_fetched_data"]


class VisualizationAgent(BaseSubAgent):
    """A Gemini session specialized for visualization.

    Uses a single `execute_visualization` tool with a method catalog in the
    system prompt, plus `list_fetched_data` to discover available data.
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
        """Add explicit tool-call guidance for visualization tasks."""
        return (
            f"Execute this task: {task.instruction}\n\n"
            "IMPORTANT: Call execute_visualization directly with the appropriate method.\n"
            "- To plot data: execute_visualization(method=\"plot_stored_data\", args={\"labels\": \"LABEL1,LABEL2\"})\n"
            "- To export: execute_visualization(method=\"export\", args={\"filename\": \"output.png\"})\n"
            "- Do NOT call reset or get_plot_state unless the task specifically asks for it.\n"
            "- Do NOT call list_fetched_data — the available data is already listed above."
        )
