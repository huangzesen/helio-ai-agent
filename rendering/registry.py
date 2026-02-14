"""
Tool registry for visualization operations.

Describes the visualization tools as structured data.
The VisualizationAgent sub-agent uses this registry to understand what
operations are available and validate arguments before dispatching to
PlotlyRenderer.

Adding a new capability:
    1. Add or update an entry in TOOLS below
    2. Implement the method in plotly_renderer.py
    3. Add dispatch logic in agent/core.py _execute_tool()
"""

TOOLS = [
    {
        "name": "render_plotly_json",
        "description": "Create or update the plot by providing a Plotly figure JSON with data_label placeholders. The system fills in actual data arrays from memory.",
        "parameters": [
            {"name": "figure_json", "type": "object", "required": True,
             "description": "Plotly figure dict with 'data' (array of trace stubs, "
                            "each with 'data_label' and standard Plotly trace properties) "
                            "and 'layout' (standard Plotly layout dict). "
                            "Each trace's 'data_label' fills x from the DataFrame index "
                            "(timestamps for timeseries, numeric values for non-time data) "
                            "and y from column values. "
                            "Multi-panel: define yaxis, yaxis2 with domains."},
        ],
    },
    {
        "name": "manage_plot",
        "description": "Imperative operations on the current figure: export, reset, zoom, get state.",
        "parameters": [
            {"name": "action", "type": "string", "required": True,
             "enum": ["reset", "get_state", "set_time_range", "export"],
             "description": "Action to perform"},
            {"name": "filename", "type": "string", "required": False,
             "description": "Output filename for export action"},
            {"name": "format", "type": "string", "required": False, "default": "png",
             "enum": ["png", "pdf"],
             "description": "Export format (default: png)"},
            {"name": "time_range", "type": "string", "required": False,
             "description": "Time range for set_time_range action (e.g., '2024-01-15 to 2024-01-20')"},
        ],
    },
]

# Build lookup dict for fast access
_TOOL_MAP = {t["name"]: t for t in TOOLS}


def get_method(name: str) -> dict | None:
    """Look up a tool by name.

    Args:
        name: Tool name (e.g., 'render_plotly_json')

    Returns:
        Tool definition dict, or None if not found.
    """
    return _TOOL_MAP.get(name)


def validate_args(name: str, args: dict) -> list[str]:
    """Validate arguments against a tool's parameter spec.

    Args:
        name: Tool name
        args: Arguments dict to validate

    Returns:
        List of error messages. Empty list means valid.
    """
    tool = get_method(name)
    if tool is None:
        return [f"Unknown tool: {name}"]

    errors = []
    for param in tool["parameters"]:
        if param["required"] and param["name"] not in args:
            errors.append(f"Missing required parameter: {param['name']}")
        if param["name"] in args and "enum" in param:
            if args[param["name"]] not in param["enum"]:
                errors.append(
                    f"Invalid value for {param['name']}: '{args[param['name']]}'. "
                    f"Must be one of: {', '.join(str(v) for v in param['enum'])}"
                )
    return errors


def render_method_catalog() -> str:
    """Render the tool registry as a markdown catalog for the LLM prompt.

    Returns:
        Markdown string listing all tools with parameters, descriptions, and examples.
    """
    lines = ["## Visualization Tools", ""]

    for tool in TOOLS:
        # Build parameter signature
        param_parts = []
        for p in tool["parameters"]:
            if p["required"]:
                param_parts.append(p["name"])
            else:
                default = p.get("default", "")
                param_parts.append(f"[{p['name']}={default}]")
        sig = ", ".join(param_parts)
        lines.append(f"### **{tool['name']}**({sig})")
        lines.append(f"{tool['description']}")
        lines.append("")

        # List parameters with types and descriptions
        for p in tool["parameters"]:
            req = "required" if p["required"] else "optional"
            line = f"- `{p['name']}` ({p['type']}, {req}): {p['description']}"
            if "enum" in p:
                vals = ", ".join(f"`{v}`" for v in p["enum"])
                line += f" Values: {vals}"
            lines.append(line)
        lines.append("")

    # Add usage examples
    lines.extend([
        "## Examples",
        "",
        '- New plot: `render_plotly_json(figure_json={"data": [{"type": "scatter", "data_label": "ACE_Bmag"}], "layout": {"title": {"text": "ACE B"}}})`',
        '- Multi-panel: define yaxis/yaxis2 domains in layout, use xaxis/yaxis refs in traces',
        '- Spectrogram: `{"type": "heatmap", "data_label": "ACE_spec", "colorscale": "Viridis"}`',
        "- Zoom: `manage_plot(action=\"set_time_range\", time_range=\"2024-01-15 to 2024-01-20\")`",
        "- Reset: `manage_plot(action=\"reset\")`",
        "- Get state: `manage_plot(action=\"get_state\")`",
        "",
    ])

    return "\n".join(lines)
