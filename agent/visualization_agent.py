"""
Visualization sub-agent.

Owns all visualization operations via a single `execute_visualization`
tool backed by the method registry. The orchestrator delegates visualization
requests here, keeping data operations in mission agents.

Follows the same patterns as MissionAgent: fresh chat per request,
forced function calling for task execution, token tracking.
"""

from google import genai
from google.genai import types

from .tools import get_tool_schemas
from .tasks import Task, TaskStatus
from .logging import get_logger, log_error
from knowledge.prompt_builder import build_visualization_prompt

# Visualization agent gets its own tool category + list_fetched_data from data_ops
VIZ_TOOL_CATEGORIES = ["visualization"]
VIZ_EXTRA_TOOLS = ["list_fetched_data"]


class VisualizationAgent:
    """A Gemini session specialized for visualization.

    Uses a single `execute_visualization` tool with a method catalog in the
    system prompt, plus `list_fetched_data` to discover available data.

    Attributes:
        verbose: Whether to print debug info
        gui_mode: Whether running in GUI mode
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        gui_mode: bool = False,
    ):
        """Initialize the visualization agent.

        Args:
            client: Initialized Gemini client (shared with orchestrator)
            model_name: Model to use (e.g., "gemini-2.5-flash")
            tool_executor: Callable(tool_name, tool_args) -> dict that executes tools.
                           Typically OrchestratorAgent._execute_tool_safe.
            verbose: If True, print debug info about tool calls.
            gui_mode: If True, include GUI-mode instructions in the prompt.
        """
        self.client = client
        self.model_name = model_name
        self.tool_executor = tool_executor
        self.verbose = verbose
        self.gui_mode = gui_mode
        self.logger = get_logger()

        # Build visualization-focused system prompt with method catalog
        self.system_prompt = build_visualization_prompt(gui_mode=gui_mode)

        # Build function declarations (visualization + list_fetched_data)
        function_declarations = []
        for tool_schema in get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        ):
            fd = types.FunctionDeclaration(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            )
            function_declarations.append(fd)

        # Config for forced function calling (task execution)
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[types.Tool(function_declarations=function_declarations)],
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
        """Return cumulative token usage for this visualization agent."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "thinking_tokens": self._total_thinking_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens + self._total_thinking_tokens,
            "api_calls": self._api_calls,
        }

    def process_request(self, user_message: str) -> str:
        """Process a visualization request conversationally.

        Creates a fresh chat per request to avoid cross-request context pollution.
        Allows the agent to respond with text or call tools as needed (no forced
        function calling).

        Args:
            user_message: The visualization request.

        Returns:
            The text response from Gemini after processing.
        """
        self.logger.debug(f"[Visualization Agent] Processing: {user_message[:80]}...")

        try:
            # Conversational config: no forced function calling
            conv_config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                tools=[types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                    ) for t in get_tool_schemas(
                        categories=VIZ_TOOL_CATEGORIES,
                        extra_names=VIZ_EXTRA_TOOLS,
                    )
                ])],
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level="LOW",
                ),
            )
            chat = self.client.chats.create(
                model=self.model_name,
                config=conv_config,
            )
            response = chat.send_message(user_message)
            self._track_usage(response)

            # Process tool calls in a loop (up to 5 iterations)
            max_iterations = 5
            iteration = 0
            previous_calls = set()  # Track (tool_name, args_key) to detect duplicates

            while iteration < max_iterations:
                iteration += 1

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

                # Detect duplicate tool calls (prevents double-plotting)
                call_keys = set()
                for fc in function_calls:
                    args_str = str(sorted(dict(fc.args).items())) if fc.args else ""
                    call_keys.add((fc.name, args_str))
                if call_keys and call_keys.issubset(previous_calls):
                    self.logger.debug("[Visualization Agent] Duplicate tool call detected, stopping")
                    break

                # Execute tools via the shared executor
                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    self.logger.debug(f"[Visualization Agent] Tool: {tool_name}({tool_args})")

                    result = self.tool_executor(tool_name, tool_args)

                    if result.get("status") == "error":
                        self.logger.warning(f"[Visualization Agent] Tool error: {result.get('message', '')}")

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    )

                    # Track this call
                    args_str = str(sorted(tool_args.items()))
                    previous_calls.add((tool_name, args_str))

                self.logger.debug(f"[Visualization Agent] Sending {len(function_responses)} tool result(s) back...")
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
            log_error(
                "Visualization agent request failed",
                exc=e,
                context={"request": user_message[:200]}
            )
            self.logger.warning(f"[Visualization Agent] Failed: {e}")
            return f"Error processing visualization request: {e}"

    def execute_task(self, task: Task) -> str:
        """Execute a single visualization task.

        Creates a fresh chat session with forced function calling (mode="ANY").

        Args:
            task: The task to execute

        Returns:
            The text response from Gemini after completing the task
        """
        task.status = TaskStatus.IN_PROGRESS
        task.tool_calls = []

        self.logger.debug(f"[Visualization Agent] Executing: {task.description}")

        try:
            # Fresh chat per task with forced function calling
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

                # Detect duplicate tool calls
                call_keys = set()
                for fc in function_calls:
                    args_str = str(sorted(dict(fc.args).items())) if fc.args else ""
                    call_keys.add((fc.name, args_str))
                if call_keys and call_keys.issubset(previous_calls):
                    self.logger.debug("[Visualization Agent] Duplicate tool call detected, stopping")
                    break

                # Execute tools via the shared executor
                function_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    task.tool_calls.append(tool_name)
                    result = self.tool_executor(tool_name, tool_args)

                    if result.get("status") == "error":
                        self.logger.warning(f"[Visualization Agent] Tool error: {result.get('message', '')}")

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    )

                    args_str = str(sorted(tool_args.items()))
                    previous_calls.add((tool_name, args_str))

                self.logger.debug(f"[Visualization Agent] Sending {len(function_responses)} tool result(s) back...")
                response = chat.send_message(message=function_responses)
                self._track_usage(response)

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

            self.logger.debug(f"[Visualization Agent] Completed: {task.description}")

            return result_text

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            log_error(
                "Visualization agent task failed",
                exc=e,
                context={"task": task.description}
            )
            self.logger.warning(f"[Visualization Agent] Failed: {task.description} - {e}")
            return f"Error: {e}"
