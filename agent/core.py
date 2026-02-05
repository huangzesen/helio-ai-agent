"""
Core agent logic - orchestrates Gemini calls and tool execution.
"""

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY
from .tools import get_tool_schemas
from .prompts import get_system_prompt, format_tool_result
from .time_utils import parse_time_range, TimeRangeError
from knowledge.catalog import search_by_keywords
from knowledge.hapi_client import list_parameters as hapi_list_parameters
from autoplot_bridge.commands import get_commands


class AutoplotAgent:
    """Main agent class that handles conversation and tool execution."""

    def __init__(self, verbose: bool = False):
        """Initialize the agent.

        Args:
            verbose: If True, print debug info about tool calls.
        """
        self.verbose = verbose

        # Initialize Gemini client
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

        # Build function declarations for Gemini
        function_declarations = []
        for tool_schema in get_tool_schemas():
            fd = types.FunctionDeclaration(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            )
            function_declarations.append(fd)

        # Create tool object with all function declarations
        tool = types.Tool(function_declarations=function_declarations)

        # Store model name and config
        self.model_name = "gemini-2.5-flash"
        self.config = types.GenerateContentConfig(
            system_instruction=get_system_prompt(),
            tools=[tool],
        )

        # Create chat session
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=self.config
        )

        # Autoplot commands (lazy-initialized on first plot)
        self._autoplot = None

    @property
    def autoplot(self):
        """Lazy initialization of Autoplot commands."""
        if self._autoplot is None:
            self._autoplot = get_commands(verbose=self.verbose)
        return self._autoplot

    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Dict with result data (varies by tool)
        """
        if self.verbose:
            print(f"  [Tool: {tool_name}({tool_args})]")

        if tool_name == "search_datasets":
            if self.verbose:
                print(f"  [Catalog] Searching for: {tool_args['query']}")
            result = search_by_keywords(tool_args["query"])
            if result:
                if self.verbose:
                    print(f"  [Catalog] Found matches.")
                return {"status": "success", **result}
            else:
                if self.verbose:
                    print(f"  [Catalog] No matches found.")
                return {"status": "success", "message": "No matching datasets found."}

        elif tool_name == "list_parameters":
            if self.verbose:
                print(f"  [HAPI] Fetching parameters for {tool_args['dataset_id']}...")
            params = hapi_list_parameters(tool_args["dataset_id"])
            if self.verbose:
                print(f"  [HAPI] Got {len(params)} parameters.")
            return {"status": "success", "parameters": params}

        elif tool_name == "plot_data":
            try:
                time_range = parse_time_range(tool_args["time_range"])
            except TimeRangeError as e:
                return {"status": "error", "message": str(e)}
            return self.autoplot.plot_cdaweb(
                dataset_id=tool_args["dataset_id"],
                parameter_id=tool_args["parameter_id"],
                time_range=time_range,
            )

        elif tool_name == "change_time_range":
            try:
                time_range = parse_time_range(tool_args["time_range"])
            except TimeRangeError as e:
                return {"status": "error", "message": str(e)}
            return self.autoplot.set_time_range(time_range)

        elif tool_name == "export_plot":
            filename = tool_args["filename"]
            if not filename.endswith(".png"):
                filename += ".png"
            return self.autoplot.export_png(filename)

        elif tool_name == "get_plot_info":
            return self.autoplot.get_current_state()

        elif tool_name == "ask_clarification":
            # Return the question to show to user
            return {
                "status": "clarification_needed",
                "question": tool_args["question"],
                "options": tool_args.get("options", []),
                "context": tool_args.get("context", ""),
            }

        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    def process_message(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Handles tool calls automatically in a loop until the model
        produces a text response.

        Args:
            user_message: The user's input

        Returns:
            The agent's text response
        """
        # Send message to Gemini
        if self.verbose:
            print(f"  [Gemini] Sending message to model...")
        response = self.chat.send_message(message=user_message)
        if self.verbose:
            print(f"  [Gemini] Response received.")

        # Process tool calls in a loop
        max_iterations = 10  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Check if response has parts
            if not response.candidates or not response.candidates[0].content.parts:
                break

            # Look for function calls
            function_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                    function_calls.append(part.function_call)

            if not function_calls:
                # No function calls - extract text and return
                break

            # Execute each function call
            function_responses = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                result = self._execute_tool(tool_name, tool_args)

                # Handle clarification specially - return immediately
                if result.get("status") == "clarification_needed":
                    question = result["question"]
                    if result.get("context"):
                        question = f"{result['context']}\n\n{question}"
                    if result.get("options"):
                        question += "\n\nOptions:\n" + "\n".join(
                            f"  {i+1}. {opt}" for i, opt in enumerate(result["options"])
                        )
                    return question

                # Create function response using types.Part
                function_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result}
                    )
                )

            # Send function results back to the model
            if self.verbose:
                print(f"  [Gemini] Sending {len(function_responses)} tool result(s) back to model...")
            response = self.chat.send_message(message=function_responses)
            if self.verbose:
                print(f"  [Gemini] Response received.")

        # Extract text response
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        return "\n".join(text_parts) if text_parts else "Done."

    def reset(self):
        """Reset conversation history."""
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=self.config
        )


def create_agent(verbose: bool = False) -> AutoplotAgent:
    """Factory function to create a new agent instance.

    Args:
        verbose: If True, print debug info about tool calls.

    Returns:
        Configured AutoplotAgent instance.
    """
    return AutoplotAgent(verbose=verbose)
