"""
Mission-specific sub-agent for executing tasks within a single mission's context.

Each MissionAgent gets a focused system prompt (via build_mission_prompt)
and its own Gemini chat session, so it has deep knowledge of one mission's
data products without context pollution from other missions.
"""

from typing import Optional

from google import genai

from .base_agent import BaseSubAgent
from .tasks import Task
from knowledge.prompt_builder import build_mission_prompt

# Mission sub-agents get discovery + fetch tools — compute is handled by DataOpsAgent
MISSION_TOOL_CATEGORIES = ["discovery", "data_ops_fetch", "conversation"]
MISSION_EXTRA_TOOLS = ["list_fetched_data"]


class MissionAgent(BaseSubAgent):
    """A Gemini session specialized for one spacecraft mission.

    Attributes:
        mission_id: Spacecraft key (e.g., "PSP", "ACE")
    """

    def __init__(
        self,
        mission_id: str,
        client: genai.Client,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        cancel_event=None,
    ):
        self.mission_id = mission_id
        super().__init__(
            client=client,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name=f"{mission_id} Agent",
            system_prompt=build_mission_prompt(mission_id),
            tool_categories=MISSION_TOOL_CATEGORIES,
            extra_tool_names=MISSION_EXTRA_TOOLS,
            cancel_event=cancel_event,
        )

    # ---- Hook overrides ----

    def _on_tool_result(self, tool_name: str, tool_args: dict, result: dict) -> Optional[str]:
        """Intercept clarification_needed results and return formatted question."""
        if result.get("status") == "clarification_needed":
            question = result["question"]
            if result.get("context"):
                question = f"{result['context']}\n\n{question}"
            if result.get("options"):
                question += "\n\nOptions:\n" + "\n".join(
                    f"  {i+1}. {opt}" for i, opt in enumerate(result["options"])
                )
            return question
        return None

    def _should_skip_function_call(self, function_calls: list) -> bool:
        """Skip clarification requests in task execution mode."""
        if any(fc.name == "ask_clarification" for fc in function_calls):
            self.logger.debug(f"[{self.agent_name}] Skipping clarification request")
            return True
        return False

    def _get_task_prompt(self, task: Task) -> str:
        """Strict task prompt to prevent unnecessary post-fetch tool calls."""
        return (
            f"Execute this task: {task.instruction}\n\n"
            "RULES:\n"
            "- Do ONLY what the instruction says. Do NOT add extra steps.\n"
            "- After a successful fetch_data call, STOP. Do NOT call list_fetched_data, "
            "get_data_availability, list_parameters, describe_data, or get_dataset_docs.\n"
            "- Return the stored label and point count as concise text."
        )

    def _get_error_context(self, **kwargs) -> dict:
        """Add mission_id to error context."""
        kwargs["mission"] = self.mission_id
        return kwargs
