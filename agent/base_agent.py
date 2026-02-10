"""
Base class for all sub-agents (Mission, DataOps, DataExtraction, Visualization).

Consolidates the shared logic: Gemini chat setup, token tracking, tool-calling
loops (process_request and execute_task), and LoopGuard integration.

Sub-agents override:
- Constructor to provide agent_name, system_prompt, tool_categories, extra_tool_names
- Hook methods for agent-specific behavior (e.g., clarification interception)
"""

import threading
from typing import Optional

from google import genai
from google.genai import types

from .tools import get_tool_schemas
from .tasks import Task, TaskStatus
from .logging import get_logger, log_error
from .loop_guard import LoopGuard, make_call_key
from .model_fallback import get_active_model


class BaseSubAgent:
    """Base class with all shared sub-agent logic.

    Subclasses must call super().__init__() with appropriate parameters.
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        agent_name: str = "SubAgent",
        system_prompt: str = "",
        tool_categories: list[str] | None = None,
        extra_tool_names: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.client = client
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.verbose = verbose
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self._cancel_event = cancel_event
        self.logger = get_logger()

        # Build function declarations from categories
        categories = tool_categories or []
        extra = extra_tool_names or []
        self._function_declarations = []
        for tool_schema in get_tool_schemas(categories=categories, extra_names=extra):
            fd = types.FunctionDeclaration(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            )
            self._function_declarations.append(fd)

        # Config for forced function calling (used by execute_task)
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[types.Tool(function_declarations=self._function_declarations)],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level="LOW",
            ),
        )

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._api_calls = 0

    # ---- Token tracking ----

    def _track_usage(self, response):
        """Accumulate token usage from a Gemini response."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            self._total_input_tokens += getattr(meta, "prompt_token_count", 0) or 0
            self._total_output_tokens += getattr(meta, "candidates_token_count", 0) or 0
            self._total_thinking_tokens += getattr(meta, "thoughts_token_count", 0) or 0
        self._api_calls += 1
        if self.verbose:
            from .thinking import extract_thoughts
            for thought in extract_thoughts(response):
                preview = thought[:200] + "..." if len(thought) > 200 else thought
                self.logger.debug(f"[Thinking] {preview}")

    def get_token_usage(self) -> dict:
        """Return cumulative token usage for this agent."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "thinking_tokens": self._total_thinking_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens + self._total_thinking_tokens,
            "api_calls": self._api_calls,
        }

    # ---- Hook methods for subclass customization ----

    def _on_tool_result(self, tool_name: str, tool_args: dict, result: dict) -> Optional[str]:
        """Hook called after each tool execution in process_request.

        If this returns a non-None string, process_request returns that string
        immediately (used by MissionAgent for clarification interception).
        """
        return None

    def _should_skip_function_call(self, function_calls: list) -> bool:
        """Hook called before executing function calls in execute_task.

        If True, the loop breaks without executing the calls
        (used by MissionAgent to skip ask_clarification in task execution).
        """
        return False

    def _get_task_prompt(self, task: Task) -> str:
        """Hook to customize the task execution prompt.

        Override to add agent-specific instructions (e.g., VisualizationAgent
        adds explicit tool-call guidance).
        """
        return (
            f"Execute this task: {task.instruction}\n\n"
            "CRITICAL: Do ONLY what the instruction says. Do NOT add extra steps.\n"
            "Return results as concise text when done."
        )

    def _get_error_context(self, **kwargs) -> dict:
        """Hook to add agent-specific context to error logs."""
        return kwargs

    # ---- Shared loops ----

    def process_request(self, user_message: str) -> str:
        """Process a user request conversationally (no forced function calling).

        Creates a fresh chat per request to avoid cross-request context pollution.
        """
        self.logger.debug(f"[{self.agent_name}] Processing: {user_message[:80]}...")

        try:
            # Conversational config: no forced function calling
            conv_config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                tools=[types.Tool(function_declarations=self._function_declarations)],
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level="LOW",
                ),
            )
            chat = self.client.chats.create(
                model=get_active_model(self.model_name),
                config=conv_config,
            )
            response = chat.send_message(user_message)
            self._track_usage(response)

            guard = LoopGuard(max_total_calls=20, max_iterations=8)
            consecutive_errors = 0

            while True:
                stop_reason = guard.check_iteration()
                if stop_reason:
                    self.logger.debug(f"[{self.agent_name}] Stopping: {stop_reason}")
                    break

                if self._cancel_event and self._cancel_event.is_set():
                    self.logger.info(f"[{self.agent_name}] Interrupted by user")
                    return "Interrupted by user."

                parts = (
                    response.candidates[0].content.parts
                    if response.candidates and response.candidates[0].content
                    else None
                )
                if not parts:
                    break

                function_calls = []
                for part in parts:
                    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                        function_calls.append(part.function_call)

                if not function_calls:
                    break

                # Check for loops/duplicates/cycling
                call_keys = set()
                for fc in function_calls:
                    call_keys.add(make_call_key(fc.name, dict(fc.args) if fc.args else {}))
                stop_reason = guard.check_calls(call_keys)
                if stop_reason:
                    self.logger.debug(f"[{self.agent_name}] Stopping: {stop_reason}")
                    break

                # Execute tools via the shared executor
                function_responses = []
                all_errors_this_round = True
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    self.logger.debug(f"[{self.agent_name}] Tool: {tool_name}({tool_args})")

                    result = self.tool_executor(tool_name, tool_args)

                    # Hook: let subclass intercept results (e.g., clarification)
                    intercept = self._on_tool_result(tool_name, tool_args, result)
                    if intercept is not None:
                        return intercept

                    if result.get("status") != "error":
                        all_errors_this_round = False

                    if result.get("status") == "error":
                        self.logger.warning(f"[{self.agent_name}] Tool error: {result.get('message', '')}")

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    )

                guard.record_calls(call_keys)

                # Track consecutive error rounds
                if all_errors_this_round:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                if consecutive_errors >= 2:
                    self.logger.warning(f"[{self.agent_name}] {consecutive_errors} consecutive error rounds, stopping")
                    break

                self.logger.debug(f"[{self.agent_name}] Sending {len(function_responses)} tool result(s) back...")
                response = chat.send_message(message=function_responses)
                self._track_usage(response)

            # Extract text response
            text_parts = []
            final_parts = (
                response.candidates[0].content.parts
                if response.candidates and response.candidates[0].content
                else None
            )
            if final_parts:
                for part in final_parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

            return "\n".join(text_parts) if text_parts else "Done."

        except Exception as e:
            ctx = self._get_error_context(request=user_message[:200])
            log_error(
                f"{self.agent_name} request failed",
                exc=e,
                context=ctx,
            )
            self.logger.warning(f"[{self.agent_name}] Failed: {e}")
            return f"Error processing request: {e}"

    def execute_task(self, task: Task) -> str:
        """Execute a single task with forced function calling.

        Creates a fresh chat session for each task to avoid context pollution.
        """
        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []

        self.logger.debug(f"[{self.agent_name}] Executing: {task.description}")

        try:
            chat = self.client.chats.create(
                model=get_active_model(self.model_name),
                config=self.config,
            )
            task_prompt = self._get_task_prompt(task)
            response = chat.send_message(task_prompt)
            self._track_usage(response)

            guard = LoopGuard(max_total_calls=10, max_iterations=3)
            last_stop_reason = None
            had_successful_tool = False

            while True:
                stop_reason = guard.check_iteration()
                if stop_reason:
                    self.logger.debug(f"[{self.agent_name}] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                if self._cancel_event and self._cancel_event.is_set():
                    self.logger.info(f"[{self.agent_name}] Task interrupted by user")
                    last_stop_reason = "cancelled by user"
                    break

                if not response.candidates or not response.candidates[0].content.parts:
                    break

                function_calls = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                        function_calls.append(part.function_call)

                if not function_calls:
                    break

                # Hook: let subclass skip certain calls (e.g., ask_clarification)
                if self._should_skip_function_call(function_calls):
                    self.logger.debug(f"[{self.agent_name}] Skipping function calls per hook")
                    break

                # Check for loops/duplicates/cycling
                call_keys = set()
                for fc in function_calls:
                    call_keys.add(make_call_key(fc.name, dict(fc.args) if fc.args else {}))
                stop_reason = guard.check_calls(call_keys)
                if stop_reason:
                    self.logger.debug(f"[{self.agent_name}] Stopping: {stop_reason}")
                    last_stop_reason = stop_reason
                    break

                # Execute tools via the shared executor
                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    task.tool_calls.append(tool_name)
                    result = self.tool_executor(tool_name, tool_args)

                    if result.get("status") == "success":
                        had_successful_tool = True
                    elif result.get("status") == "error":
                        self.logger.warning(f"[{self.agent_name}] Tool error: {result.get('message', '')}")

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    )

                guard.record_calls(call_keys)

                self.logger.debug(f"[{self.agent_name}] Sending {len(function_responses)} tool result(s) back...")
                response = chat.send_message(message=function_responses)
                self._track_usage(response)

            # Warn if no tools were called
            if not task.tool_calls:
                log_error(
                    f"{self.agent_name} task completed without tool calls: {task.description}",
                    context=self._get_error_context(task_instruction=task.instruction),
                )
                self.logger.warning(f"[{self.agent_name}] No tools were called")

            # Extract text response
            text_parts = []
            parts = response.candidates[0].content.parts if response.candidates and response.candidates[0].content else None
            if parts:
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

            result_text = "\n".join(text_parts) if text_parts else "Done."

            if last_stop_reason:
                if last_stop_reason == "cancelled by user":
                    task.status = TaskStatus.FAILED
                    task.error = f"Task cancelled by user"
                    result_text += f" [CANCELLED]"
                elif had_successful_tool:
                    task.status = TaskStatus.COMPLETED
                    result_text += f" [loop guard stopped extra calls]"
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task stopped by loop guard: {last_stop_reason}"
                    result_text += f" [STOPPED: {last_stop_reason}]"
            else:
                task.status = TaskStatus.COMPLETED

            task.result = result_text

            self.logger.debug(f"[{self.agent_name}] {task.status.value}: {task.description}")

            return result_text

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            ctx = self._get_error_context(task=task.description)
            log_error(
                f"{self.agent_name} task failed",
                exc=e,
                context=ctx,
            )
            self.logger.warning(f"[{self.agent_name}] Failed: {task.description} - {e}")
            return f"Error: {e}"
