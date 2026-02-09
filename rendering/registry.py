"""
Tool registry for visualization operations.

Describes the three declarative visualization tools as structured data.
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
        "name": "plot_data",
        "description": "Create a fresh plot from in-memory timeseries. Supports single-panel overlay or multi-panel layout. Use labels from list_fetched_data.",
        "parameters": [
            {"name": "labels", "type": "string", "required": True,
             "description": "Comma-separated labels of data to plot (e.g., 'Bmag' or 'ACE_Bmag,PSP_Bmag')"},
            {"name": "panels", "type": "array", "required": False,
             "description": "Panel layout as list of label lists, e.g. [['A','B'], ['C']] for 2 panels. Omit for single-panel overlay."},
            {"name": "title", "type": "string", "required": False, "default": "",
             "description": "Optional plot title"},
            {"name": "plot_type", "type": "string", "required": False, "default": "line",
             "enum": ["line", "spectrogram"],
             "description": "Plot type: 'line' (default) or 'spectrogram'"},
            {"name": "colorscale", "type": "string", "required": False, "default": "Viridis",
             "description": "Plotly colorscale for spectrograms (e.g., Viridis, Jet, Plasma)"},
            {"name": "log_y", "type": "boolean", "required": False, "default": False,
             "description": "Log scale on y-axis (spectrogram)"},
            {"name": "log_z", "type": "boolean", "required": False, "default": False,
             "description": "Log scale on color axis (spectrogram intensity)"},
            {"name": "z_min", "type": "number", "required": False,
             "description": "Min value for spectrogram color scale"},
            {"name": "z_max", "type": "number", "required": False,
             "description": "Max value for spectrogram color scale"},
        ],
    },
    {
        "name": "style_plot",
        "description": "Apply aesthetic changes to the current plot. All parameters are optional — pass only what you want to change.",
        "parameters": [
            {"name": "title", "type": "string", "required": False,
             "description": "Plot title"},
            {"name": "x_label", "type": "string", "required": False,
             "description": "X-axis label"},
            {"name": "y_label", "type": "string", "required": False,
             "description": "Y-axis label (string for all panels, or JSON object {panel_num: label})"},
            {"name": "trace_colors", "type": "object", "required": False,
             "description": "Map trace label -> color, e.g. {'ACE Bmag': 'red', 'PSP Bmag': 'blue'}"},
            {"name": "line_styles", "type": "object", "required": False,
             "description": "Map trace label -> {width, dash, mode}, e.g. {'ACE Bmag': {'width': 2, 'dash': 'dot'}}"},
            {"name": "log_scale", "type": "string", "required": False,
             "enum": ["x", "y", "both", "linear"],
             "description": "Set log scale: 'x', 'y', 'both', or 'linear' to reset"},
            {"name": "x_range", "type": "array", "required": False,
             "description": "X-axis range [min, max]"},
            {"name": "y_range", "type": "array", "required": False,
             "description": "Y-axis range [min, max] (or JSON object {panel_num: [min, max]})"},
            {"name": "legend", "type": "boolean", "required": False,
             "description": "Show (true) or hide (false) legend"},
            {"name": "font_size", "type": "integer", "required": False,
             "description": "Global font size in points"},
            {"name": "canvas_size", "type": "object", "required": False,
             "description": "Canvas dimensions: {width: int, height: int}"},
            {"name": "annotations", "type": "array", "required": False,
             "description": "List of annotations: [{text, x, y}, ...]"},
            {"name": "colorscale", "type": "string", "required": False,
             "description": "Plotly colorscale for heatmap traces (e.g., Viridis, Jet)"},
            {"name": "theme", "type": "string", "required": False,
             "description": "Plotly template name (e.g., 'plotly_dark', 'plotly_white')"},
        ],
    },
    {
        "name": "manage_plot",
        "description": "Structural operations on the plot: export, reset, zoom, get state, add/remove traces.",
        "parameters": [
            {"name": "action", "type": "string", "required": True,
             "enum": ["reset", "get_state", "set_time_range", "export", "remove_trace", "add_trace"],
             "description": "Action to perform"},
            {"name": "filename", "type": "string", "required": False,
             "description": "Output filename for export action"},
            {"name": "format", "type": "string", "required": False, "default": "png",
             "enum": ["png", "pdf"],
             "description": "Export format (default: png)"},
            {"name": "time_range", "type": "string", "required": False,
             "description": "Time range for set_time_range action (e.g., '2024-01-15 to 2024-01-20')"},
            {"name": "label", "type": "string", "required": False,
             "description": "Trace label for remove_trace or add_trace actions"},
            {"name": "panel", "type": "integer", "required": False, "default": 1,
             "description": "Target panel (1-based) for add_trace action"},
        ],
    },
]

# Build lookup dict for fast access
_TOOL_MAP = {t["name"]: t for t in TOOLS}


def get_method(name: str) -> dict | None:
    """Look up a tool by name.

    Args:
        name: Tool name (e.g., 'plot_data')

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
        "- Plot stored data: `plot_data(labels=\"ACE_Bmag,PSP_Bmag\", title=\"Comparison\")`",
        "- Multi-panel plot: `plot_data(labels=\"Bmag,Density\", panels=[[\"Bmag\"], [\"Density\"]])`",
        "- Spectrogram: `plot_data(labels=\"ACE_Bmag_spectrogram\", plot_type=\"spectrogram\")`",
        "- Set title: `style_plot(title=\"Solar Wind Speed\")`",
        "- Y-axis label: `style_plot(y_label=\"B (nT)\")`",
        "- Log scale: `style_plot(log_scale=\"y\")`",
        "- Trace color: `style_plot(trace_colors={\"ACE Bmag\": \"red\"})`",
        "- Canvas size: `style_plot(canvas_size={\"width\": 1920, \"height\": 1080})`",
        "- Export PNG: `manage_plot(action=\"export\", filename=\"output.png\")`",
        "- Export PDF: `manage_plot(action=\"export\", filename=\"output.pdf\", format=\"pdf\")`",
        "- Zoom: `manage_plot(action=\"set_time_range\", time_range=\"2024-01-15 to 2024-01-20\")`",
        "- Reset: `manage_plot(action=\"reset\")`",
        "- Get state: `manage_plot(action=\"get_state\")`",
        "- Remove trace: `manage_plot(action=\"remove_trace\", label=\"ACE Bmag\")`",
        "",
    ])

    return "\n".join(lines)
