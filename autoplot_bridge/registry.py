"""
Method registry for Autoplot operations.

Describes every Autoplot capability as structured data. The AutoplotAgent
sub-agent uses this registry to understand what operations are available
and validate arguments before dispatching to AutoplotCommands.

Adding a new capability:
    1. Add an entry to METHODS below
    2. Implement the bridge method in commands.py
    3. Add dispatch logic in agent/core.py _dispatch_autoplot_method()
"""

METHODS = [
    {
        "name": "plot_cdaweb",
        "description": "Plot CDAWeb data directly in Autoplot by dataset and parameter ID.",
        "parameters": [
            {"name": "dataset_id", "type": "string", "required": True,
             "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI')"},
            {"name": "parameter_id", "type": "string", "required": True,
             "description": "Parameter to plot (e.g., 'Magnitude')"},
            {"name": "time_range", "type": "string", "required": True,
             "description": "Time range: 'last week', '2024-01-01 to 2024-01-07', etc."},
        ],
    },
    {
        "name": "plot_stored_data",
        "description": "Plot one or more in-memory timeseries in the Autoplot canvas. Use labels from list_fetched_data.",
        "parameters": [
            {"name": "labels", "type": "string", "required": True,
             "description": "Comma-separated labels of data to plot (e.g., 'Bmag' or 'ACE_Bmag,PSP_Bmag')"},
            {"name": "title", "type": "string", "required": False, "default": "",
             "description": "Optional plot title"},
            {"name": "filename", "type": "string", "required": False, "default": "",
             "description": "Optional output filename for auto-export to PNG"},
        ],
    },
    {
        "name": "set_time_range",
        "description": "Change the time range of the current plot. Use for zooming in/out or shifting to a different period.",
        "parameters": [
            {"name": "time_range", "type": "string", "required": True,
             "description": "New time range: 'last week', '2024-01-15 to 2024-01-20', etc."},
        ],
    },
    {
        "name": "export_png",
        "description": "Export the current plot to a PNG image file.",
        "parameters": [
            {"name": "filename", "type": "string", "required": True,
             "description": "Output filename (.png extension added if missing)"},
        ],
    },
    {
        "name": "export_pdf",
        "description": "Export the current plot to a PDF file.",
        "parameters": [
            {"name": "filename", "type": "string", "required": True,
             "description": "Output filename (.pdf extension added if missing)"},
        ],
    },
    {
        "name": "get_plot_state",
        "description": "Get information about the current plot: dataset, parameter, and time range.",
        "parameters": [],
    },
    {
        "name": "reset",
        "description": "Reset the Autoplot canvas, clearing all plots and state. Use when starting fresh.",
        "parameters": [],
    },
    {
        "name": "set_title",
        "description": "Set or change the title displayed above the current plot.",
        "parameters": [
            {"name": "title", "type": "string", "required": True,
             "description": "The title text"},
        ],
    },
    {
        "name": "set_axis_label",
        "description": "Set a label on an axis of the current plot. Only y and z axes are supported.",
        "parameters": [
            {"name": "axis", "type": "string", "required": True,
             "enum": ["y", "z"],
             "description": "Which axis: 'y' or 'z'"},
            {"name": "label", "type": "string", "required": True,
             "description": "The text label"},
        ],
    },
    {
        "name": "toggle_log_scale",
        "description": "Enable or disable logarithmic scale on a plot axis.",
        "parameters": [
            {"name": "axis", "type": "string", "required": True,
             "enum": ["y", "z"],
             "description": "Which axis: 'y' or 'z'"},
            {"name": "enabled", "type": "boolean", "required": True,
             "description": "True for log scale, False for linear"},
        ],
    },
    {
        "name": "set_axis_range",
        "description": "Manually set the value range of a plot axis for zooming into specific values.",
        "parameters": [
            {"name": "axis", "type": "string", "required": True,
             "enum": ["y", "z"],
             "description": "Which axis: 'y' or 'z'"},
            {"name": "min", "type": "number", "required": True,
             "description": "Minimum value"},
            {"name": "max", "type": "number", "required": True,
             "description": "Maximum value"},
        ],
    },
    {
        "name": "save_session",
        "description": "Save the current Autoplot session to a .vap file for later restoration.",
        "parameters": [
            {"name": "filepath", "type": "string", "required": True,
             "description": "Output .vap file path"},
        ],
    },
    {
        "name": "load_session",
        "description": "Load a previously saved Autoplot session from a .vap file.",
        "parameters": [
            {"name": "filepath", "type": "string", "required": True,
             "description": "Path to the .vap file"},
        ],
    },
    {
        "name": "set_render_type",
        "description": "Change how data is rendered in the plot. Use 'spectrogram' for 2D data, 'scatter' for sparse data, 'series' (default) for timeseries.",
        "parameters": [
            {"name": "render_type", "type": "string", "required": True,
             "enum": ["series", "scatter", "spectrogram", "fill_to_zero",
                      "staircase", "color_scatter", "digital", "image",
                      "pitch_angle_distribution", "events_bar", "orbit"],
             "description": "The render type to use"},
            {"name": "index", "type": "integer", "required": False, "default": 0,
             "description": "Plot element index (0 for first/only plot)"},
        ],
    },
    {
        "name": "set_color_table",
        "description": "Set the color table (colormap) for spectrogram or color scatter plots.",
        "parameters": [
            {"name": "name", "type": "string", "required": True,
             "enum": ["apl_rainbow_black0", "black_blue_green_yellow_white",
                      "black_green", "black_red", "blue_white_red",
                      "color_wedge", "grayscale", "matlab_jet",
                      "rainbow", "reverse_rainbow", "wrapped_color_wedge"],
             "description": "Color table name"},
        ],
    },
    {
        "name": "set_canvas_size",
        "description": "Set the canvas (image) size in pixels. Useful before exporting high-resolution images.",
        "parameters": [
            {"name": "width", "type": "integer", "required": True,
             "description": "Canvas width in pixels"},
            {"name": "height", "type": "integer", "required": True,
             "description": "Canvas height in pixels"},
        ],
    },
]

# Build lookup dict for fast access
_METHOD_MAP = {m["name"]: m for m in METHODS}


def get_method(name: str) -> dict | None:
    """Look up a method by name.

    Args:
        name: Method name (e.g., 'set_render_type')

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
