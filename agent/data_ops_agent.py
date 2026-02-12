"""
Data operations sub-agent.

Owns data transformation and analysis operations via
custom_operation and describe_data tools. The orchestrator
delegates computation requests here, keeping fetching in mission agents
and visualization in the visualization agent.
"""

from .llm import LLMAdapter
from .base_agent import BaseSubAgent
from .tasks import Task
from knowledge.prompt_builder import build_data_ops_prompt

# DataOps agent gets compute tools + list_fetched_data to discover available data
DATAOPS_TOOL_CATEGORIES = ["data_ops_compute", "conversation"]
DATAOPS_EXTRA_TOOLS = ["list_fetched_data"]


class DataOpsAgent(BaseSubAgent):
    """An LLM session specialized for data transformations and analysis."""

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        cancel_event=None,
        token_log_path=None,
    ):
        super().__init__(
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name="DataOps Agent",
            system_prompt=build_data_ops_prompt(),
            tool_categories=DATAOPS_TOOL_CATEGORIES,
            extra_tool_names=DATAOPS_EXTRA_TOOLS,
            cancel_event=cancel_event,
            token_log_path=token_log_path,
        )

    def _get_task_prompt(self, task: Task) -> str:
        """Strict task prompt to prevent unnecessary post-compute tool calls."""
        return (
            f"Execute this task: {task.instruction}\n\n"
            "RULES:\n"
            "- Do ONLY what the instruction says. Do NOT add extra steps.\n"
            "- After a successful custom_operation or compute_spectrogram, STOP. "
            "Do NOT call list_fetched_data, describe_data, or preview_data afterward.\n"
            "- If the operation fails due to wrong column names, call preview_data ONCE "
            "to check column names, then retry with corrected code.\n"
            "- Return the output label and point count as concise text."
        )
