"""
Reusable tool-calling loop for agents that manage their own chat sessions.

Extracted from BaseSubAgent.process_request() so that PlannerAgent (which
keeps a persistent chat across rounds) can also run tool calls without
inheriting from BaseSubAgent.
"""

import threading

from .logging import get_logger
from .loop_guard import LoopGuard, make_call_key

logger = get_logger()


def extract_text_from_response(response) -> str:
    """Extract concatenated text from a response, ignoring tool calls.

    Accepts both ``LLMResponse`` (has ``.text``) and raw Gemini responses.
    Returns empty string if no text is found.
    """
    # LLMResponse path
    if hasattr(response, "tool_calls") and hasattr(response, "text"):
        return response.text or ""

    # Raw Gemini response fallback
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


def _extract_tool_calls(response):
    """Extract tool calls from either an LLMResponse or raw Gemini response.

    Returns list of objects with .name and .args attributes.
    """
    # LLMResponse path
    if hasattr(response, "tool_calls"):
        return response.tool_calls

    # Raw Gemini response fallback
    parts = (
        response.candidates[0].content.parts
        if getattr(response, "candidates", None) and response.candidates[0].content
        else None
    )
    if not parts:
        return []
    calls = []
    for part in parts:
        if hasattr(part, "function_call") and part.function_call and part.function_call.name:
            calls.append(part.function_call)
    return calls


def run_tool_loop(
    chat,
    response,
    tool_executor,
    agent_name: str = "Agent",
    max_total_calls: int = 10,
    max_iterations: int = 5,
    track_usage=None,
    collect_tool_results: dict = None,
    cancel_event: threading.Event | None = None,
    send_fn=None,
    adapter=None,
):
    """Run a tool-calling loop on an existing chat session.

    Keeps sending tool results back to the model until it stops issuing
    function calls (or a guard limit is hit).

    Args:
        chat: An active chat session (ChatSession or raw genai chat).
        response: The initial response (LLMResponse or raw Gemini response).
        tool_executor: ``(tool_name: str, tool_args: dict) -> dict`` callable.
        agent_name: Label for log messages.
        max_total_calls: Hard cap on total tool invocations.
        max_iterations: Hard cap on loop iterations.
        track_usage: Optional ``(response) -> None`` callback for token accounting.
        collect_tool_results: Optional dict to collect raw tool results.
            If provided, results are stored as ``{tool_name: [result, ...]}``
            so callers can inspect what the tools returned (e.g. parameter lists).
        send_fn: Optional callable ``(message) -> response`` for sending messages.
            If provided, used instead of ``chat.send_message``.  Allows callers
            to inject timeout/retry wrappers.
        adapter: Optional LLMAdapter instance for building tool result messages.
            When provided, uses ``adapter.make_tool_result_message()`` instead
            of raw Gemini ``types.Part.from_function_response()``.

    Returns:
        The final response after tools stop.
    """
    if send_fn is None:
        # Prefer ChatSession.send, fall back to raw genai chat.send_message
        if hasattr(chat, "send"):
            send_fn = chat.send
        else:
            send_fn = lambda msg: chat.send_message(message=msg)

    guard = LoopGuard(max_total_calls=max_total_calls, max_iterations=max_iterations)
    consecutive_errors = 0

    while True:
        stop_reason = guard.check_iteration()
        if stop_reason:
            logger.debug(f"[{agent_name}] Tool loop stopping: {stop_reason}")
            break

        if cancel_event and cancel_event.is_set():
            logger.info(f"[{agent_name}] Tool loop interrupted by user")
            break

        function_calls = _extract_tool_calls(response)
        if not function_calls:
            break

        # Check for loops / duplicates / cycling
        call_keys = set()
        for fc in function_calls:
            args = fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {})
            call_keys.add(make_call_key(fc.name, args))
        stop_reason = guard.check_calls(call_keys)
        if stop_reason:
            logger.debug(f"[{agent_name}] Tool loop stopping: {stop_reason}")
            break

        # Execute each tool
        function_responses = []
        all_errors = True
        for fc in function_calls:
            tool_name = fc.name
            tool_args = fc.args if isinstance(fc.args, dict) else (dict(fc.args) if fc.args else {})
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

            if adapter is not None:
                function_responses.append(adapter.make_tool_result_message(tool_name, result))
            else:
                # Fallback: raw Gemini types (should not happen after full migration)
                from google.genai import types
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
        # Set tool context on the agent (if track_usage is a bound method)
        if track_usage:
            agent_obj = getattr(track_usage, "__self__", None)
            if agent_obj and hasattr(agent_obj, "_last_tool_context"):
                tool_names = [fc.name for fc in function_calls]
                agent_obj._last_tool_context = "+".join(tool_names)
        response = send_fn(function_responses)
        if track_usage:
            track_usage(response)

    return response
