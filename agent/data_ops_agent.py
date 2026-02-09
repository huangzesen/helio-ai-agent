"""
Data operations sub-agent.

Owns all data transformation, analysis, and export operations via
custom_operation, describe_data, and save_data tools. The orchestrator
delegates computation requests here, keeping fetching in mission agents
and visualization in the visualization agent.
"""

from google import genai

from .base_agent import BaseSubAgent
from knowledge.prompt_builder import build_data_ops_prompt

# DataOps agent gets compute tools + list_fetched_data to discover available data
DATAOPS_TOOL_CATEGORIES = ["data_ops_compute", "conversation"]
DATAOPS_EXTRA_TOOLS = ["list_fetched_data"]


class DataOpsAgent(BaseSubAgent):
    """A Gemini session specialized for data transformations and analysis."""

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        tool_executor,
        verbose: bool = False,
    ):
        super().__init__(
            client=client,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name="DataOps Agent",
            system_prompt=build_data_ops_prompt(),
            tool_categories=DATAOPS_TOOL_CATEGORIES,
            extra_tool_names=DATAOPS_EXTRA_TOOLS,
        )
