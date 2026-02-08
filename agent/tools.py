"""
Tool definitions for Gemini function calling.

Each tool schema defines what the LLM can call and what parameters it needs.
Tools are executed by the agent core based on LLM decisions.

Categories:
- "discovery": dataset search and parameter listing
- "data_ops": data fetching, computation, statistics, export
- "autoplot": execute_autoplot (registry-driven Autoplot operations)
- "conversation": ask_clarification
- "routing": delegate_to_mission, delegate_to_autoplot
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
        "category": "discovery",
        "name": "browse_datasets",
        "description": """Browse all available science datasets for a mission. Use this when:
- User asks "what datasets are available?" or "what else can I plot?"
- You need to find a dataset not in the recommended list
- User asks about a specific instrument or data type you don't have in your prompt

Returns a filtered list excluding calibration/housekeeping/ephemeris data.
Each entry has: id, description, start_date, stop_date, parameter_count, instrument.""",
        "parameters": {
            "type": "object",
            "properties": {
                "mission_id": {
                    "type": "string",
                    "description": "Mission ID (e.g., 'PSP', 'ACE', 'SolO', 'OMNI', 'WIND', 'DSCOVR', 'MMS', 'STEREO_A')"
                }
            },
            "required": ["mission_id"]
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

    # --- Autoplot Visualization ---
    {
        "category": "autoplot",
        "name": "execute_autoplot",
        "description": "Execute an Autoplot visualization method. See the method catalog in the system prompt for available methods and their parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Method name from the catalog (e.g., 'plot_cdaweb', 'set_render_type', 'export_png')"
                },
                "args": {
                    "type": "object",
                    "description": "Arguments as described in the method catalog"
                }
            },
            "required": ["method"]
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
- Visualization requests (plotting, zoom, export, render changes) — use delegate_to_autoplot
- Requests to plot already-loaded data — use delegate_to_autoplot
- General questions about capabilities

The specialist will search datasets, fetch data, run computations, and report back what was done. You then decide whether to visualize the results.""",
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
    {
        "category": "routing",
        "name": "delegate_to_autoplot",
        "description": """Delegate a visualization request to the Autoplot specialist agent. Use this when:
- The user asks to plot, display, or visualize data
- The user wants to change plot appearance (render type, colors, axis labels, title, log scale)
- The user wants to zoom, export (PNG/PDF), or save/load sessions
- The user wants to resize the canvas

Do NOT delegate:
- Data requests (fetch, compute, describe) — use delegate_to_mission
- Dataset search or parameter listing — handle directly

The specialist has access to all Autoplot visualization methods and can see what data is in memory.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The visualization request (e.g., 'plot ACE_Bmag and PSP_Bmag together', 'switch to scatter plot', 'export as PDF')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about what data is available or what was just done"
                }
            },
            "required": ["request"]
        }
    },
]


def get_tool_schemas(
    categories: list[str] | None = None,
    extra_names: list[str] | None = None,
) -> list[dict]:
    """Return tool schemas for Gemini function calling.

    Args:
        categories: Optional list of categories to filter by.
            If None, returns all tools. Valid categories:
            "discovery", "autoplot", "data_ops", "conversation", "routing".
        extra_names: Optional list of tool names to include regardless of category.
            Useful for giving a sub-agent access to specific tools outside its categories.

    Returns:
        List of tool schema dicts.
    """
    if categories is None and extra_names is None:
        return TOOLS
    if categories is None:
        categories = []
    extra = set(extra_names) if extra_names else set()
    return [
        t for t in TOOLS
        if t.get("category") in categories or t.get("name") in extra
    ]
