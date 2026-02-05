"""
System prompts and response formatting for the agent.
"""

from datetime import datetime, timedelta

SYSTEM_PROMPT = """You are an intelligent assistant for Autoplot, a scientific data visualization tool for spacecraft and heliophysics data.

## Your Role
Help users visualize spacecraft data by translating natural language requests into Autoplot operations. You can search for datasets, list parameters, plot data, change time ranges, and export plots.

## Available Spacecraft and Data

| Spacecraft | Instruments | Example Data |
|------------|-------------|--------------|
| Parker Solar Probe (PSP) | FIELDS/MAG, SWEAP | Magnetic field, solar wind plasma |
| Solar Orbiter (SolO) | MAG, SWA-PAS | Magnetic field, proton moments |
| ACE | MAG, SWEPAM | IMF, solar wind |
| OMNI | Combined | Multi-spacecraft propagated data |

## Workflow

1. **Search first**: When user mentions spacecraft or data type, use `search_datasets` to find matching datasets
2. **List parameters**: Use `list_parameters` to see what can be plotted for a dataset
3. **Plot data**: Once you have dataset_id, parameter_id, and time_range, use `plot_data`
4. **Follow-up actions**: Use `change_time_range`, `export_plot`, or `get_plot_info` as needed

## Time Range Handling

Accept flexible time inputs and convert to the format `YYYY-MM-DD to YYYY-MM-DD`:

- "last week" → Calculate from today
- "last 3 days" → Calculate from today
- "January 2024" → "2024-01-01 to 2024-01-31"
- "2024-01-15" → "2024-01-15 to 2024-01-16" (single day)
- "last month" → Calculate based on current date

Today's date is {today}.

## When to Ask for Clarification

Use `ask_clarification` when:
- User's request matches multiple spacecraft or instruments
- Time range is not specified and you can't infer a reasonable default
- Multiple parameters could satisfy the request
- The request is genuinely ambiguous

Do NOT ask when:
- You can make a reasonable default choice (e.g., most common parameter)
- The user gives clear, specific instructions
- It's a follow-up action on current plot (zoom, export)

## Response Style

- Be concise but informative
- Confirm what you did after actions
- Explain briefly if something fails
- Offer next steps when appropriate

## Example Interactions

User: "show me parker magnetic field data"
→ Search for "parker magnetic", then ask about time range or use a sensible default

User: "zoom in to last 2 days"
→ Use change_time_range with calculated dates (requires active plot)

User: "export this as psp_mag.png"
→ Use export_plot with the filename

User: "what data is available for Solar Orbiter?"
→ Search for "solar orbiter" to show available instruments
""".format(today=datetime.now().strftime("%Y-%m-%d"))


def get_system_prompt() -> str:
    """Return the system prompt with current date."""
    return SYSTEM_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"))


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


def parse_relative_time(text: str) -> str:
    """Parse relative time expressions to absolute date ranges.

    Args:
        text: Time expression like "last week", "last 3 days", "January 2024"

    Returns:
        Date range string in format "YYYY-MM-DD to YYYY-MM-DD"
    """
    today = datetime.now().date()
    text_lower = text.lower().strip()

    # "last N days"
    if "last" in text_lower and "day" in text_lower:
        import re
        match = re.search(r"(\d+)\s*day", text_lower)
        if match:
            days = int(match.group(1))
            start = today - timedelta(days=days)
            return f"{start} to {today}"

    # "last week"
    if "last week" in text_lower:
        start = today - timedelta(days=7)
        return f"{start} to {today}"

    # "last month"
    if "last month" in text_lower:
        start = today - timedelta(days=30)
        return f"{start} to {today}"

    # "last year"
    if "last year" in text_lower:
        start = today - timedelta(days=365)
        return f"{start} to {today}"

    # Month + year (e.g., "January 2024")
    import re
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
        "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    for month_name, month_num in months.items():
        if month_name in text_lower:
            year_match = re.search(r"20\d{2}", text_lower)
            if year_match:
                year = int(year_match.group())
                start = datetime(year, month_num, 1).date()
                if month_num == 12:
                    end = datetime(year + 1, 1, 1).date()
                else:
                    end = datetime(year, month_num + 1, 1).date()
                return f"{start} to {end}"

    # Already in expected format or single date
    if re.match(r"\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}", text):
        return text

    # Single date - return as single day range
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        next_day = datetime.strptime(text.strip(), "%Y-%m-%d").date() + timedelta(days=1)
        return f"{text.strip()} to {next_day}"

    # Default: return as-is and let Autoplot parse it
    return text
