"""
Method registry for visualization operations.

Describes core visualization capabilities as structured data. The VisualizationAgent
sub-agent uses this registry to understand what operations are available
and validate arguments before dispatching to PlotlyRenderer.

Thin wrappers (title, axis labels, log scale, canvas size, render type, etc.)
have been replaced by the ``custom_visualization`` tool, which lets the LLM
write free-form Plotly code against the current figure.

Adding a new capability:
    1. Add an entry to METHODS below
    2. Implement the method in plotly_renderer.py
    3. Add dispatch logic in agent/core.py _dispatch_viz_method()
"""

METHODS = [
    {
        "name": "plot_stored_data",
        "description": "Plot one or more in-memory timeseries in the plot canvas. Preferred method when data has been fetched or computed. Use labels from list_fetched_data.",
        "parameters": [
            {"name": "labels", "type": "string", "required": True,
             "description": "Comma-separated labels of data to plot (e.g., 'Bmag' or 'ACE_Bmag,PSP_Bmag')"},
            {"name": "title", "type": "string", "required": False, "default": "",
             "description": "Optional plot title"},
            {"name": "filename", "type": "string", "required": False, "default": "",
             "description": "Optional output filename for auto-export to PNG"},
            {"name": "index", "type": "integer", "required": False, "default": -1,
             "description": "Panel index (0-based). Omit or -1 for default auto-layout. Use 0, 1, 2... to target a specific panel."},
        ],
    },
    {
        "name": "set_time_range",
        "description": "Change the time range of the current plot. Use for zooming in/out or shifting to a different period.",
        "parameters": [
            {"name": "time_range", "type": "string", "required": True,
             "description": "New time range (use ' to ' separator, NOT '/'): 'last week', '2024-01-15 to 2024-01-20', etc."},
        ],
    },
    {
        "name": "export",
        "description": "Export the current plot to a file (PNG or PDF).",
        "parameters": [
            {"name": "filename", "type": "string", "required": True,
             "description": "Output filename (extension added if missing)"},
            {"name": "format", "type": "string", "required": False, "default": "png",
             "enum": ["png", "pdf"],
             "description": "Export format: 'png' (default) or 'pdf'"},
        ],
    },
    {
        "name": "get_plot_state",
        "description": "Get information about the current plot: dataset, parameter, and time range.",
        "parameters": [],
    },
    {
        "name": "reset",
        "description": "Reset the plot canvas, clearing all plots and state. Use when starting fresh.",
        "parameters": [],
    },
]

# Build lookup dict for fast access
_METHOD_MAP = {m["name"]: m for m in METHODS}


def get_method(name: str) -> dict | None:
    """Look up a method by name.

    Args:
        name: Method name (e.g., 'export')

    Returns:
        Method definition dict, or None if not found.
    """
    return _METHOD_MAP.get(name)


def validate_args(name: str, args: dict) -> list[str]:
    """Validate arguments against a method's parameter spec.

    Args:
        name: Method name
        args: Arguments dict to validate

    Returns:
        List of error messages. Empty list means valid.
    """
    method = get_method(name)
    if method is None:
        return [f"Unknown method: {name}"]

    errors = []
    for param in method["parameters"]:
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
    """Render the method registry as a markdown catalog for the LLM prompt.

    Returns:
        Markdown string listing all methods with parameters and descriptions.
    """
    lines = ["## Available Methods", ""]
    for method in METHODS:
        # Build parameter signature
        param_parts = []
        for p in method["parameters"]:
            if p["required"]:
                param_parts.append(p["name"])
            else:
                default = p.get("default", "")
                param_parts.append(f"[{p['name']}={default}]")
        sig = ", ".join(param_parts)
        lines.append(f"- **{method['name']}**({sig}) -- {method['description']}")

        # Add enum values if any parameter has them
        for p in method["parameters"]:
            if "enum" in p:
                vals = ", ".join(f"`{v}`" for v in p["enum"])
                lines.append(f"  - {p['name']}: {vals}")

    lines.append("")
    return "\n".join(lines)
