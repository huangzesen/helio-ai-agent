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
        "name": "get_data_availability",
        "description": """Check the available time range for a CDAWeb dataset. Use this to:
- Verify data exists for a requested time range before fetching or plotting
- Tell the user how far back data goes or when it was last updated
- Diagnose "no data" errors by checking if the time range is valid

Returns the earliest and latest available dates for the dataset.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI', 'PSP_FLD_L2_MAG_RTN_1MIN')"
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
    },

    # --- Data Operations Tools ---
    {
        "name": "fetch_data",
        "description": """Fetch timeseries data from CDAWeb HAPI into memory for Python-side operations.
Use this instead of plot_data when the user wants to compute on data (magnitude, averages, differences, etc.).

The data is stored in memory with a label like 'AC_H2_MFI.BGSEc' for later reference by compute and plot tools.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI')"
                },
                "parameter_id": {
                    "type": "string",
                    "description": "Parameter name (e.g., 'BGSEc', 'Magnitude')"
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range (same formats as plot_data)"
                }
            },
            "required": ["dataset_id", "parameter_id", "time_range"]
        }
    },
    {
        "name": "list_fetched_data",
        "description": "Show all timeseries currently held in memory. Returns labels, shapes, units, and time ranges.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "plot_computed_data",
        "description": """Display one or more in-memory timeseries in the Autoplot canvas.
Use this to visualize data fetched with fetch_data or results from compute operations.

Multiple labels are overlaid on the same plot. The result appears in the Autoplot window and can be further manipulated with change_time_range or exported with export_plot.""",
        "parameters": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "string",
                    "description": "Comma-separated labels of data to plot (e.g., 'Bmag' or 'AC_H2_MFI.BGSEc,Bmag')"
                },
                "title": {
                    "type": "string",
                    "description": "Optional plot title"
                },
                "filename": {
                    "type": "string",
                    "description": "Optional output filename (auto-generated if omitted)"
                }
            },
            "required": ["labels"]
        }
    },
    {
        "name": "compute_magnitude",
        "description": """Compute the magnitude (sqrt(x²+y²+z²)) of a vector timeseries.
The source must be a 3-component vector (e.g., magnetic field BGSEc).""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_label": {
                    "type": "string",
                    "description": "Label of the vector timeseries in memory"
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the result (e.g., 'Bmag')"
                }
            },
            "required": ["source_label", "output_label"]
        }
    },
    {
        "name": "compute_arithmetic",
        "description": """Element-wise arithmetic between two timeseries: +, -, *, /.
Both series must have the same shape (use compute_resample to align cadences first).""",
        "parameters": {
            "type": "object",
            "properties": {
                "label_a": {
                    "type": "string",
                    "description": "Label of the first operand"
                },
                "label_b": {
                    "type": "string",
                    "description": "Label of the second operand"
                },
                "operation": {
                    "type": "string",
                    "description": "Arithmetic operation: '+', '-', '*', or '/'"
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the result"
                }
            },
            "required": ["label_a", "label_b", "operation", "output_label"]
        }
    },
    {
        "name": "compute_running_average",
        "description": """Compute a centered moving average to smooth a scalar timeseries.
Uses np.nanmean to skip data gaps. Window size is in number of data points.""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_label": {
                    "type": "string",
                    "description": "Label of the scalar timeseries"
                },
                "window_size": {
                    "type": "integer",
                    "description": "Number of points in the averaging window"
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the smoothed result"
                }
            },
            "required": ["source_label", "window_size", "output_label"]
        }
    },
    {
        "name": "compute_resample",
        "description": """Downsample a timeseries by bin-averaging at a fixed cadence.
Works on both scalar and vector data. Useful for aligning two series to the same time grid.""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_label": {
                    "type": "string",
                    "description": "Label of the timeseries to resample"
                },
                "cadence_seconds": {
                    "type": "number",
                    "description": "New cadence in seconds (e.g., 60 for 1-minute averages)"
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the resampled result"
                }
            },
            "required": ["source_label", "cadence_seconds", "output_label"]
        }
    },
    {
        "name": "compute_delta",
        "description": """Compute differences or time derivatives of a timeseries.
- 'difference' mode: Δv = v[i+1] - v[i]
- 'derivative' mode: dv/dt in units per second

Output has n-1 points with midpoint timestamps.""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_label": {
                    "type": "string",
                    "description": "Label of the source timeseries"
                },
                "mode": {
                    "type": "string",
                    "description": "'difference' or 'derivative'"
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the result"
                }
            },
            "required": ["source_label", "mode", "output_label"]
        }
    },
]


def get_tool_schemas() -> list[dict]:
    """Return tool schemas for Gemini function calling."""
    return TOOLS
