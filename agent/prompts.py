"""
System prompts and response formatting for the agent.

The system prompt is dynamically generated from the spacecraft catalog
via knowledge/prompt_builder.py — no hardcoded spacecraft or dataset tables.
"""

from datetime import datetime

from knowledge.prompt_builder import build_system_prompt

# Generate the system prompt template once at import time.
# Contains a {today} placeholder filled in by get_system_prompt().
_SYSTEM_PROMPT_TEMPLATE = build_system_prompt()


def get_system_prompt(gui_mode: bool = False) -> str:
    """Return the system prompt with current date.

    Args:
        gui_mode: If True, the orchestrator knows GUI mode is active (passed
            through to the autoplot agent, not appended to orchestrator prompt).
    """
    return _SYSTEM_PROMPT_TEMPLATE.replace("{today}", datetime.now().strftime("%Y-%m-%d"))


def format_search_result(result: dict) -> str:
    """Format search_datasets result for display."""
    if not result:
        return "No matching datasets found."

    lines = []
    lines.append(f"Found: {result['spacecraft_name']} ({result['spacecraft']})")

    if result.get("instrument"):
        lines.append(f"Instrument: {result['instrument_name']} ({result['instrument']})")
        lines.append(f"Datasets: {', '.join(result['datasets'])}")
    else:
        lines.append("No specific instrument matched. Available instruments:")
        for inst in result.get("available_instruments", []):
            lines.append(f"  - {inst['name']} ({inst['id']})")

    return "\n".join(lines)


def format_parameters_result(params: list[dict]) -> str:
    """Format list_parameters result for display."""
    if not params:
        return "No plottable parameters found."

    lines = [f"Found {len(params)} plottable parameters:"]
    for p in params[:10]:  # Limit to 10
        units = f" ({p['units']})" if p['units'] else ""
        size_str = f" [vector:{p['size'][0]}]" if p['size'][0] > 1 else ""
        lines.append(f"  - {p['name']}{units}{size_str}")
        if p['description']:
            lines.append(f"      {p['description'][:60]}...")

    if len(params) > 10:
        lines.append(f"  ... and {len(params) - 10} more")

    return "\n".join(lines)


def format_plot_result(result: dict) -> str:
    """Format plot_data result for display."""
    if result.get("status") == "success":
        return f"Plotted {result['dataset_id']}/{result['parameter_id']} for {result['time_range']}"
    else:
        return f"Plot failed: {result.get('message', 'Unknown error')}"


def format_tool_result(tool_name: str, result: dict) -> str:
    """Format any tool result for display."""
    if result.get("status") == "error":
        return f"Error: {result.get('message', 'Unknown error')}"

    if tool_name == "search_datasets":
        return format_search_result(result)
    elif tool_name == "list_parameters":
        return format_parameters_result(result.get("parameters", []))
    elif tool_name == "plot_data":
        return format_plot_result(result)
    elif tool_name == "change_time_range":
        return f"Time range changed to {result['time_range']}"
    elif tool_name == "export_plot":
        return f"Plot exported to {result['filepath']}"
    elif tool_name == "get_plot_info":
        if not result.get("uri"):
            return "No plot is currently displayed."
        return f"Currently showing: {result['uri']}\nTime range: {result['time_range']}"

    return str(result)
