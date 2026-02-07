"""
Tool definitions for Gemini function calling.

Each tool schema defines what the LLM can call and what parameters it needs.
Tools are executed by the agent core based on LLM decisions.
"""

TOOLS = [
    {
        "category": "discovery",
        "name": "search_datasets",
        "description": """Search for spacecraft datasets by keyword. Use this when:
- User mentions a spacecraft (Parker, ACE, Solar Orbiter, OMNI, Wind, DSCOVR, MMS, STEREO)
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
        "category": "discovery",
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
        "category": "discovery",
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
        "category": "plotting",
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
        "category": "plotting",
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
        "category": "plotting",
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
        "category": "plotting",
        "name": "get_plot_info",
        "description": "Get information about what is currently plotted, including dataset, parameter, and time range.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "category": "conversation",
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
        "category": "data_ops",
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
                    "description": "Time range. Use 'YYYY-MM-DD to YYYY-MM-DD' for date ranges (e.g., '2024-01-15 to 2024-01-20'), 'last week'/'last 3 days' for relative, 'January 2024' for a month, or '2024-01-15T06:00 to 2024-01-15T18:00' for sub-day precision. Do NOT use '/' as a separator."
                }
            },
            "required": ["dataset_id", "parameter_id", "time_range"]
        }
    },
    {
        "category": "data_ops",
        "name": "list_fetched_data",
        "description": "Show all timeseries currently held in memory. Returns labels, shapes, units, and time ranges.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "category": "plotting",
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
        "category": "data_ops",
        "name": "custom_operation",
        "description": """Apply a pandas/numpy operation to an in-memory timeseries. This is the universal compute tool — use it for ALL data transformations after fetching data with fetch_data.

The pandas_code must:
- Operate on `df` (a pandas DataFrame with DatetimeIndex)
- Assign the result to `result` (must be a DataFrame or Series with DatetimeIndex)
- Use only `df`, `pd` (pandas), and `np` (numpy) — no imports, no file I/O

Common operations:
- Magnitude: `result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')`
- Arithmetic: `result = df * 2` or `result = df_a + df_b` (use pd.DataFrame constructor for second operand)
- Running average: `result = df.rolling(60, center=True, min_periods=1).mean()`
- Resample: `result = df.resample('60s').mean().dropna(how='all')`
- Difference: `result = df.diff().iloc[1:]`
- Derivative: `dv = df.diff().iloc[1:]; dt_s = df.index.to_series().diff().dt.total_seconds().iloc[1:]; result = dv.div(dt_s, axis=0)`
- Normalize: `result = (df - df.mean()) / df.std()`
- Clip values: `result = df.clip(lower=-50, upper=50)`
- Log transform: `result = np.log10(df.abs().replace(0, np.nan))`
- Interpolate gaps: `result = df.interpolate(method='linear')`
- Select columns: `result = df[['x', 'z']]`
- Detrend: `result = df - df.rolling(100, center=True, min_periods=1).mean()`
- Absolute value: `result = df.abs()`
- Cumulative sum: `result = df.cumsum()`
- Z-score filter: `z = (df - df.mean()) / df.std(); result = df[z.abs() < 3].reindex(df.index)`

Do NOT call this tool when the request cannot be expressed as a pandas/numpy operation (e.g., "email me the data", "upload to server"). Instead, explain to the user what is and isn't possible.""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_label": {
                    "type": "string",
                    "description": "Label of the source timeseries in memory"
                },
                "pandas_code": {
                    "type": "string",
                    "description": "Python code using df (DataFrame), pd (pandas), np (numpy). Must assign to 'result'."
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the result (e.g., 'B_normalized', 'B_clipped')"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the operation"
                }
            },
            "required": ["source_label", "pandas_code", "output_label", "description"]
        }
    },

    # --- Describe & Export Tools ---
    {
        "category": "data_ops",
        "name": "describe_data",
        "description": """Get statistical summary of an in-memory timeseries. Use this when:
- User asks "what does the data look like?" or "summarize the data"
- You want to understand the data before deciding what operations to apply
- User asks about min, max, average, or data quality

Returns statistics (min, max, mean, std, percentiles, NaN count) and the LLM can narrate findings.""",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory (e.g., 'AC_H2_MFI.BGSEc')"
                }
            },
            "required": ["label"]
        }
    },
    {
        "category": "data_ops",
        "name": "save_data",
        "description": """Export an in-memory timeseries to a CSV file. Use this when:
- User asks to save, export, or download data
- User wants data in a file for external use (Excel, MATLAB, etc.)
- User wants to keep a copy of computed results

The CSV file has a datetime column (ISO 8601 UTC) followed by data columns.
If no filename is given, one is auto-generated from the label.""",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory to export"
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g., 'ace_mag.csv'). '.csv' is appended if missing. Default: auto-generated from label."
                }
            },
            "required": ["label"]
        }
    },

    # --- GUI-mode Interactive Tools ---
    {
        "category": "plotting",
        "name": "reset_plot",
        "description": "Reset the Autoplot canvas, clearing all plots and data. Use when the user wants to start fresh or clear the current display.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "category": "plotting",
        "name": "set_plot_title",
        "description": "Set or change the title of the current plot.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title text to display above the plot"}
            },
            "required": ["title"]
        }
    },
    {
        "category": "plotting",
        "name": "set_axis_label",
        "description": "Set a label on an axis of the current plot. Only y and z axes are supported (x-axis labels are auto-managed for time-series data).",
        "parameters": {
            "type": "object",
            "properties": {
                "axis": {"type": "string", "description": "Which axis: 'y' or 'z'"},
                "label": {"type": "string", "description": "The text label to set"}
            },
            "required": ["axis", "label"]
        }
    },
    {
        "category": "plotting",
        "name": "toggle_log_scale",
        "description": "Enable or disable logarithmic scale on a plot axis.",
        "parameters": {
            "type": "object",
            "properties": {
                "axis": {"type": "string", "description": "Which axis: 'y' or 'z'"},
                "enabled": {"type": "boolean", "description": "True for log, False for linear"}
            },
            "required": ["axis", "enabled"]
        }
    },
    {
        "category": "plotting",
        "name": "set_axis_range",
        "description": "Manually set the range of a plot axis. Useful when the user wants to zoom into a specific value range.",
        "parameters": {
            "type": "object",
            "properties": {
                "axis": {"type": "string", "description": "Which axis: 'y' or 'z'"},
                "min": {"type": "number", "description": "Minimum value for the axis"},
                "max": {"type": "number", "description": "Maximum value for the axis"}
            },
            "required": ["axis", "min", "max"]
        }
    },
    {
        "category": "plotting",
        "name": "save_session",
        "description": "Save the current Autoplot session to a .vap file for later restoration.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Output .vap file path"}
            },
            "required": ["filepath"]
        }
    },
    {
        "category": "plotting",
        "name": "load_session",
        "description": "Load a previously saved Autoplot session from a .vap file, restoring all plots and settings.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the .vap file to load"}
            },
            "required": ["filepath"]
        }
    },

    # --- Routing ---
    {
        "category": "routing",
        "name": "delegate_to_mission",
        "description": """Delegate a data request to a mission-specific specialist agent. Use this when:
- The user asks about a specific spacecraft's data (e.g., "show me ACE magnetic field data")
- The user wants to fetch, compute, or describe data from a specific mission
- You need mission-specific knowledge (dataset IDs, parameter names, analysis patterns)

Do NOT delegate:
- Plot follow-ups (zoom, export, time range changes) — handle these directly
- Requests to plot already-loaded data — use plot_computed_data directly
- General questions about capabilities

The specialist will search datasets, fetch data, run computations, and report back what was done. You then decide whether to plot the results.""",
        "parameters": {
            "type": "object",
            "properties": {
                "mission_id": {
                    "type": "string",
                    "description": "Spacecraft mission ID from the supported missions table (e.g., 'PSP', 'ACE', 'SolO', 'OMNI', 'WIND', 'DSCOVR', 'MMS', 'STEREO_A')"
                },
                "request": {
                    "type": "string",
                    "description": "The data request to send to the specialist (e.g., 'fetch magnetic field data for last week')"
                }
            },
            "required": ["mission_id", "request"]
        }
    },
]


def get_tool_schemas(categories: list[str] | None = None) -> list[dict]:
    """Return tool schemas for Gemini function calling.

    Args:
        categories: Optional list of categories to filter by.
            If None, returns all tools. Valid categories:
            "discovery", "plotting", "data_ops", "conversation", "routing".

    Returns:
        List of tool schema dicts.
    """
    if categories is None:
        return TOOLS
    return [t for t in TOOLS if t.get("category") in categories]
