"""
Reusable tool-calling loop for agents that manage their own chat sessions.

Extracted from BaseSubAgent.process_request() so that PlannerAgent (which
keeps a persistent chat across rounds) can also run tool calls without
inheriting from BaseSubAgent.
"""

from google.genai import types

from .logging import get_logger
from .loop_guard import LoopGuard, make_call_key

logger = get_logger()


def extract_text_from_response(response) -> str:
    """Extract concatenated text from a Gemini response, ignoring tool calls.

    Returns empty string if no text parts are found.
    """
    parts = (
        response.candidates[0].content.parts
        if response.candidates and response.candidates[0].content
        else None
    )
    if not parts:
        return ""
    texts = []
    for part in parts:
        if hasattr(part, "text") and part.text:
            texts.append(part.text)
    return "\n".join(texts)


def run_tool_loop(
    chat,
    response,
    tool_executor,
    agent_name: str = "Agent",
    max_total_calls: int = 10,
    max_iterations: int = 5,
    track_usage=None,
    collect_tool_results: dict = None,
):
    """Run a tool-calling loop on an existing chat session.

    Keeps sending tool results back to the model until it stops issuing
    function calls (or a guard limit is hit).

    Args:
        chat: An active ``genai`` chat session.
        response: The initial Gemini response (may already contain tool calls).
        tool_executor: ``(tool_name: str, tool_args: dict) -> dict`` callable.
        agent_name: Label for log messages.
        max_total_calls: Hard cap on total tool invocations.
        max_iterations: Hard cap on loop iterations.
        track_usage: Optional ``(response) -> None`` callback for token accounting.
        collect_tool_results: Optional dict to collect raw tool results.
            If provided, results are stored as ``{tool_name: [result, ...]}``
            so callers can inspect what the tools returned (e.g. parameter lists).

    Returns:
        The final Gemini response after tools stop.
    """
    guard = LoopGuard(max_total_calls=max_total_calls, max_iterations=max_iterations)
    consecutive_errors = 0

    while True:
        stop_reason = guard.check_iteration()
        if stop_reason:
            logger.debug(f"[{agent_name}] Tool loop stopping: {stop_reason}")
            break

        parts = (
            response.candidates[0].content.parts
            if response.candidates and response.candidates[0].content
            else None
        )
        if not parts:
            break

        # Collect function calls from the response
        function_calls = []
        for part in parts:
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                function_calls.append(part.function_call)

        if not function_calls:
            break

        # Check for loops / duplicates / cycling
        call_keys = set()
        for fc in function_calls:
            call_keys.add(make_call_key(fc.name, dict(fc.args) if fc.args else {}))
        stop_reason = guard.check_calls(call_keys)
        if stop_reason:
            logger.debug(f"[{agent_name}] Tool loop stopping: {stop_reason}")
            break

        # Execute each tool
        function_responses = []
        all_errors = True
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}
            logger.debug(f"[{agent_name}] Tool: {tool_name}({tool_args})")

            result = tool_executor(tool_name, tool_args)

            # Collect raw results for callers that need them
            if collect_tool_results is not None:
                collect_tool_results.setdefault(tool_name, []).append({
                    "args": tool_args,
                    "result": result,
                })

            if result.get("status") != "error":
                all_errors = False
            else:
                logger.warning(f"[{agent_name}] Tool error: {result.get('message', '')}")

            function_responses.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        guard.record_calls(call_keys)

        # Bail after 2 consecutive all-error rounds
        if all_errors:
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if consecutive_errors >= 2:
            logger.warning(f"[{agent_name}] {consecutive_errors} consecutive error rounds, stopping")
            break

        # Feed results back to the model
        logger.debug(f"[{agent_name}] Sending {len(function_responses)} tool result(s) back...")
        response = chat.send_message(message=function_responses)
        if track_usage:
            track_usage(response)

    return response
