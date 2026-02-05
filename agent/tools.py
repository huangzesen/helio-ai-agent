"""
Tool definitions for Gemini function calling.

Each tool schema defines what the LLM can call and what parameters it needs.
Tools are executed by the agent core based on LLM decisions.
"""

TOOLS = [
    {
        "name": "search_datasets",
        "description": """Search for spacecraft datasets by keyword. Use this when:
- User mentions a spacecraft (Parker, ACE, Solar Orbiter, OMNI)
- User mentions a data type (magnetic field, solar wind, plasma, density)
- User asks what data is available

Returns matching spacecraft, instrument, and dataset information.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'parker magnetic', 'ACE solar wind', 'omni')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_parameters",
        "description": """List plottable parameters for a specific dataset. Use this after search_datasets to find what parameters can be plotted.

Returns list of 1D numeric parameters with names, units, and descriptions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'PSP_FLD_L2_MAG_RTN_1MIN')"
                }
            },
            "required": ["dataset_id"]
        }
    },
    {
        "name": "plot_data",
        "description": """Load and display spacecraft data from CDAWeb. Use this when you have:
- A specific dataset ID
- A parameter name to plot
- A time range

The plot will appear in an Autoplot window.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI', 'PSP_FLD_L2_MAG_RTN_1MIN')"
                },
                "parameter_id": {
                    "type": "string",
                    "description": "Parameter to plot (e.g., 'Magnitude', 'psp_fld_l2_mag_RTN_1min')"
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range. Accepts: relative ('last week', 'last 3 days', 'last month'), month+year ('January 2024'), date range ('YYYY-MM-DD to YYYY-MM-DD'), or datetime range with sub-day precision ('YYYY-MM-DDTHH:MM to YYYY-MM-DDTHH:MM')"
                }
            },
            "required": ["dataset_id", "parameter_id", "time_range"]
        }
    },
    {
        "name": "change_time_range",
        "description": """Change the time range of the current plot. Use this when user wants to:
- Zoom in or out
- Look at a different time period
- Narrow down to specific dates""",
        "parameters": {
            "type": "object",
            "properties": {
                "time_range": {
                    "type": "string",
                    "description": "New time range. Accepts: relative ('last week', 'last 3 days', 'last month'), month+year ('January 2024'), date range ('YYYY-MM-DD to YYYY-MM-DD'), or datetime range with sub-day precision ('YYYY-MM-DDTHH:MM to YYYY-MM-DDTHH:MM')"
                }
            },
            "required": ["time_range"]
        }
    },
    {
        "name": "export_plot",
        "description": "Export the current plot to a PNG image file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Output filename (will be saved as PNG, extension added if missing)"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "get_plot_info",
        "description": "Get information about what is currently plotted, including dataset, parameter, and time range.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "ask_clarification",
        "description": """Ask the user a clarifying question when the request is ambiguous. Use this when:
- Multiple datasets could match the request
- Time range is not specified
- Parameter choice is unclear
- You need more information to proceed

Do NOT guess - ask instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices to present (keep to 3-4 options)"
                },
                "context": {
                    "type": "string",
                    "description": "Brief explanation of why you need this information"
                }
            },
            "required": ["question"]
        }
    }
]


def get_tool_schemas() -> list[dict]:
    """Return tool schemas for Gemini function calling."""
    return TOOLS
