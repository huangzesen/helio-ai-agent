"""
Mission-specific sub-agent for executing tasks within a single mission's context.

Phase 2 of the mission-agent architecture. Each MissionAgent gets a focused
system prompt (via build_mission_prompt) and its own Gemini chat session,
so it has deep knowledge of one mission's data products without context
pollution from other missions.

See docs/mission-agent-architecture.md for the full plan.
"""

from typing import Optional

from google import genai
from google.genai import types

from .tools import get_tool_schemas
from .tasks import Task, TaskStatus
from .logging import log_error, log_tool_call, log_tool_result
from knowledge.prompt_builder import build_mission_prompt


class MissionAgent:
    """A Gemini session specialized for one spacecraft mission.

    Attributes:
        mission_id: Spacecraft key (e.g., "PSP", "ACE")
        verbose: Whether to print debug info
    """

    def __init__(
        self,
        mission_id: str,
        client: genai.Client,
        model_name: str,
        tool_executor,
        verbose: bool = False,
    ):
        """Initialize a mission-specific agent.

        Args:
            mission_id: Spacecraft key in the catalog (e.g., "PSP", "ACE")
            client: Initialized Gemini client (shared with main agent)
            model_name: Model to use (e.g., "gemini-2.5-flash")
            tool_executor: Callable(tool_name, tool_args) -> dict that executes tools.
                           Typically AutoplotAgent._execute_tool_safe.
            verbose: If True, print debug info about tool calls.
        """
        self.mission_id = mission_id
        self.client = client
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.verbose = verbose

        # Build mission-focused system prompt from catalog
        self.system_prompt = build_mission_prompt(mission_id)

        # Build function declarations
        function_declarations = []
        for tool_schema in get_tool_schemas():
            fd = types.FunctionDeclaration(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            )
            function_declarations.append(fd)

        # Create Gemini chat with mission-specific context and forced function calling
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[types.Tool(function_declarations=function_declarations)],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        )

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._api_calls = 0

    def _track_usage(self, response):
        """Accumulate token usage from a Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            self._total_input_tokens += getattr(meta, "prompt_token_count", 0) or 0
            self._total_output_tokens += getattr(meta, "candidates_token_count", 0) or 0
        self._api_calls += 1

    def get_token_usage(self) -> dict:
        """Return cumulative token usage for this mission agent."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "api_calls": self._api_calls,
        }

    def execute_task(self, task: Task) -> str:
        """Execute a single task in this mission's context.

        Creates a fresh chat session for each task to avoid context pollution.
        Uses forced function calling (mode="ANY") to ensure tools are invoked.

        Args:
            task: The task to execute

        Returns:
            The text response from Gemini after completing the task
        """
        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []

        if self.verbose:
            print(f"  [{self.mission_id} Agent] Executing: {task.description}")

        try:
            # Fresh chat per task
            chat = self.client.chats.create(
                model=self.model_name,
                config=self.config,
            )
            response = chat.send_message(f"Execute this task: {task.instruction}")
            self._track_usage(response)

            # Process tool calls (limit to 3 iterations)
            max_iterations = 3
            iteration = 0
            previous_calls = set()

            while iteration < max_iterations:
                iteration += 1

                if not response.candidates or not response.candidates[0].content.parts:
                    break

                function_calls = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                        function_calls.append(part.function_call)

                if not function_calls:
                    break

                # Skip clarification requests (not supported in task execution)
                if any(fc.name == "ask_clarification" for fc in function_calls):
                    if self.verbose:
                        print(f"  [{self.mission_id} Agent] Skipping clarification request")
                    break

                # Detect duplicate tool calls
                call_keys = set()
                for fc in function_calls:
                    args_str = str(sorted(dict(fc.args).items())) if fc.args else ""
                    call_keys.add((fc.name, args_str))
                if call_keys and call_keys.issubset(previous_calls):
                    if self.verbose:
                        print(f"  [{self.mission_id} Agent] Duplicate tool call detected, stopping")
                    break

                # Execute tools via the shared executor
                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    task.tool_calls.append(tool_name)
                    result = self.tool_executor(tool_name, tool_args)

                    if self.verbose and result.get("status") == "error":
                        print(f"  [{self.mission_id} Agent] Tool error: {result.get('message', '')}")

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    )

                    args_str = str(sorted(tool_args.items()))
                    previous_calls.add((tool_name, args_str))

                if self.verbose:
                    print(f"  [{self.mission_id} Agent] Sending {len(function_responses)} tool result(s) back...")
                response = chat.send_message(message=function_responses)
                self._track_usage(response)

            # Warn if no tools were called
            if not task.tool_calls:
                log_error(
                    f"Mission task completed without tool calls: {task.description}",
                    context={"mission": self.mission_id, "task_instruction": task.instruction}
                )
                if self.verbose:
                    print(f"  [{self.mission_id} Agent] WARNING: No tools were called")

            # Extract text response
            text_parts = []
            parts = response.candidates[0].content.parts if response.candidates and response.candidates[0].content else None
            if parts:
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

            result_text = "\n".join(text_parts) if text_parts else "Done."
            task.status = TaskStatus.COMPLETED
            task.result = result_text

            if self.verbose:
                print(f"  [{self.mission_id} Agent] Completed: {task.description}")

            return result_text

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            log_error(
                f"Mission agent task failed",
                exc=e,
                context={"mission": self.mission_id, "task": task.description}
            )
            if self.verbose:
                print(f"  [{self.mission_id} Agent] Failed: {task.description} - {e}")
            return f"Error: {e}"
