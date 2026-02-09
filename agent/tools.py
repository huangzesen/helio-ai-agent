"""
Tool definitions for Gemini function calling.

Each tool schema defines what the LLM can call and what parameters it needs.
Tools are executed by the agent core based on LLM decisions.

Categories:
- "discovery": dataset search and parameter listing
- "data_ops": shared data tools (list_fetched_data)
- "data_ops_fetch": mission-specific data fetching (fetch_data)
- "data_ops_compute": data transformation, statistics, export (custom_operation, describe_data, save_data)
- "data_extraction": unstructured-to-structured data conversion (store_dataframe)
- "visualization": execute_visualization (registry-driven visualization operations)
- "conversation": ask_clarification
- "routing": delegate_to_mission, delegate_to_visualization, delegate_to_data_ops, delegate_to_data_extraction
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
        "category": "discovery",
        "name": "get_dataset_docs",
        "description": """Look up detailed CDAWeb documentation for a dataset. Use this when:
- User asks about coordinate systems, calibration, or data quality
- User asks who the PI or data contact is
- User asks what a parameter measures or how it was derived
- You need domain context to interpret or explain data
Returns instrument descriptions, variable definitions, coordinate info, and PI contact.""",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI')"
                }
            },
            "required": ["dataset_id"]
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
        "category": "data_ops_fetch",
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
        "category": "data_ops_compute",
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

    {
        "category": "data_extraction",
        "name": "store_dataframe",
        "description": """Create a new DataFrame from scratch and store it in memory. Use this when:
- You have text data (event lists, search results, catalogs) that should become a plottable dataset
- The user wants to manually define data points (e.g., from a table in a paper or website)
- You need to create a dataset that doesn't come from CDAWeb HAPI

The pandas_code must:
- Use only `pd` (pandas) and `np` (numpy) — no imports, no file I/O, no `df` variable
- Assign the result to `result` (must be a DataFrame or Series with DatetimeIndex)
- Create a DatetimeIndex from dates using pd.to_datetime() and .set_index()

Examples:
- Event catalog:
  ```
  dates = ['2024-01-01', '2024-02-15', '2024-05-10']
  values = [5.2, 7.8, 6.1]
  result = pd.DataFrame({'x_class_flux': values}, index=pd.to_datetime(dates))
  ```
- Numeric timeseries:
  ```
  result = pd.DataFrame({'value': [1.0, 2.5, 3.0]}, index=pd.date_range('2024-01-01', periods=3, freq='D'))
  ```
- Event catalog with string columns:
  ```
  dates = pd.to_datetime(['2024-01-10', '2024-03-22'])
  result = pd.DataFrame({'class': ['X1.5', 'X2.1'], 'region': ['AR3555', 'AR3590']}, index=dates)
  ```""",
        "parameters": {
            "type": "object",
            "properties": {
                "pandas_code": {
                    "type": "string",
                    "description": "Python code using pd (pandas) and np (numpy) that constructs data and assigns to 'result'. Must produce a DataFrame with DatetimeIndex."
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the stored dataset (e.g., 'xclass_flares_2024', 'cme_catalog')"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the dataset"
                },
                "units": {
                    "type": "string",
                    "description": "Optional units for the data columns (e.g., 'W/m²', 'km/s')"
                }
            },
            "required": ["pandas_code", "output_label", "description"]
        }
    },

    {
        "category": "data_ops_compute",
        "name": "compute_spectrogram",
        "description": """Compute a spectrogram (2D time-frequency or time-energy data) from an in-memory timeseries. Use this when the user wants a spectrogram, power spectral density over time, dynamic spectrum, or frequency-time plot.

The python_code must:
- Operate on `df` (a pandas DataFrame with DatetimeIndex)
- Assign the result to `result` (a DataFrame with DatetimeIndex rows and frequency/energy bin columns)
- Use `df`, `pd` (pandas), `np` (numpy), and `signal` (scipy.signal) — no imports, no file I/O
- Column names MUST be string representations of bin center values (e.g., "0.001", "0.5", "10.0")

Common patterns:
- **Power spectrogram (scipy)**:
  vals = df.iloc[:, 0].dropna().values
  dt = df.index.to_series().diff().dt.total_seconds().median()
  fs = 1.0 / dt
  f, t_seg, Sxx = signal.spectrogram(vals, fs=fs, nperseg=256, noverlap=128)
  times = pd.to_datetime(df.index[0]) + pd.to_timedelta(t_seg, unit='s')
  result = pd.DataFrame(Sxx.T, index=times, columns=[str(freq) for freq in f])

- **Welch PSD (single spectrum, not time-varying)**:
  vals = df.iloc[:, 0].dropna().values
  dt = df.index.to_series().diff().dt.total_seconds().median()
  f, Pxx = signal.welch(vals, fs=1.0/dt, nperseg=256)
  result = pd.DataFrame({'PSD': Pxx}, index=pd.to_datetime(df.index[0]) + pd.to_timedelta(f, unit='s'))

Guidelines:
- Choose nperseg based on data cadence and desired frequency resolution
- Use noverlap=nperseg//2 as a reasonable default
- For large datasets, consider downsampling first with df.resample()""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_label": {
                    "type": "string",
                    "description": "Label of the source timeseries in memory"
                },
                "python_code": {
                    "type": "string",
                    "description": "Python code using df, pd, np, signal (scipy.signal). Must assign to 'result'."
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the output spectrogram (e.g., 'ACE_Bmag_spectrogram')"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the spectrogram"
                },
                "bin_label": {
                    "type": "string",
                    "description": "Y-axis label for the bins (e.g., 'Frequency (Hz)', 'Energy (eV)')"
                },
                "value_label": {
                    "type": "string",
                    "description": "Colorbar label for the values (e.g., 'PSD (nT²/Hz)', 'Flux')"
                }
            },
            "required": ["source_label", "python_code", "output_label", "description"]
        }
    },

    # --- Describe & Export Tools ---
    {
        "category": "data_ops_compute",
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
        "category": "data_ops_compute",
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

    # --- Visualization ---
    {
        "category": "visualization",
        "name": "execute_visualization",
        "description": "Execute a visualization method. See the method catalog in the system prompt for available methods and their parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Method name from the catalog (e.g., 'plot_stored_data', 'set_time_range', 'export')"
                },
                "args": {
                    "type": "object",
                    "description": "Arguments as described in the method catalog"
                }
            },
            "required": ["method"]
        }
    },
    {
        "category": "visualization",
        "name": "custom_visualization",
        "description": """Execute free-form Plotly code to customize the current plot. Use this for ANY plot customization not covered by the core methods (plot_stored_data, set_time_range, export, reset, get_plot_state).

The code runs with these variables available:
- `fig` — the current Plotly Figure (mutate in place)
- `go` — plotly.graph_objects
- `np` — numpy

Common patterns:
- Title: `fig.update_layout(title_text="Solar Wind Speed")`
- Y-axis label: `fig.update_yaxes(title_text="B (nT)", row=1, col=1)`
- Log scale: `fig.update_yaxes(type="log", row=1, col=1)`
- Axis range: `fig.update_yaxes(range=[-10, 10], row=1, col=1)`
- Canvas size: `fig.update_layout(width=1920, height=1080)`
- Scatter mode: `fig.data[0].mode = "markers"`
- Fill to zero: `fig.data[0].fill = "tozeroy"`
- Staircase: `fig.data[0].line = dict(shape="hv", color=fig.data[0].line.color)`
- Horizontal line: `fig.add_hline(y=0, line_dash="dash", line_color="gray")`
- Annotation: `fig.add_annotation(x="2024-01-15", y=5, text="Event")`
- Trace color: `fig.data[0].line.color = "red"`
- Legend off: `fig.update_layout(showlegend=False)`
- Font size: `fig.update_layout(font=dict(size=14))`
- Theme: `fig.update_layout(template="plotly_dark")`

Do NOT use imports — only fig, go, and np are available.""",
        "parameters": {
            "type": "object",
            "properties": {
                "plotly_code": {
                    "type": "string",
                    "description": "Python code that modifies fig in place. Access: fig, go, np."
                }
            },
            "required": ["plotly_code"]
        }
    },

    # --- Full Catalog Search ---
    {
        "category": "discovery",
        "name": "search_full_catalog",
        "description": """Search the full CDAWeb HAPI catalog (2000+ datasets) by keyword. Use this when:
- User asks about a spacecraft or instrument NOT in the supported missions table
- User wants to browse broadly across all available data (e.g., "what magnetospheric data is available?")
- User asks about a mission you don't have a specialist agent for (Cluster, THEMIS, Voyager, GOES, etc.)
- User wants to search by physical quantity across all missions (e.g., "proton density datasets")

Returns matching dataset IDs and titles. Any dataset found can be fetched with fetch_data — you do NOT need a mission agent for uncurated missions.

Do NOT use this for missions already in the routing table (PSP, ACE, etc.) — use delegate_to_mission instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (spacecraft name, instrument, physical quantity, e.g., 'cluster magnetic field', 'voyager 2', 'proton density')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 20)"
                }
            },
            "required": ["query"]
        }
    },

    # --- Search ---
    {
        "category": "discovery",
        "name": "google_search",
        "description": """Search the web using Google Search for real-world context. Use this when:
- User asks about solar events, flares, CMEs, geomagnetic storms, or space weather
- User asks what happened during a specific time period
- User wants scientific context or explanations of heliophysics phenomena
- User asks for an ICME list, event catalog, or recent news

Do NOT use this for finding CDAWeb datasets or fetching spacecraft data — use search_datasets and delegate_to_mission for that.

Returns grounded text with source URLs.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., 'major solar storms January 2024', 'ICME list 2024', 'X-class flare events')"
                }
            },
            "required": ["query"]
        }
    },

    # --- Document Reading ---
    {
        "category": "document",
        "name": "read_document",
        "description": """Read a PDF or image file and extract its text content using Gemini vision.
Supported formats: PDF (.pdf), PNG (.png), JPEG (.jpg, .jpeg), GIF (.gif), WebP (.webp), BMP (.bmp), TIFF (.tiff).
Use this when:
- User uploads or references a PDF or image file
- User wants to read, summarize, or extract content from a document
- User asks questions about a document's contents
The extracted text is saved to ~/.helio-agent/documents/ for persistence across sessions.
Returns the extracted text content and the saved file path.""",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt for targeted extraction (e.g., 'extract the data table', 'list all dates and values'). If not provided, a default extraction prompt is used."
                }
            },
            "required": ["file_path"]
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
- Visualization requests (plotting, zoom, export, render changes) — use delegate_to_visualization
- Requests to plot already-loaded data — use delegate_to_visualization
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
        "name": "delegate_to_visualization",
        "description": """Delegate a visualization request to the visualization specialist agent. Use this when:
- The user asks to plot, display, or visualize data
- The user wants to change plot appearance (render type, colors, axis labels, title, log scale)
- The user wants to zoom, export (PNG/PDF), or save/load sessions
- The user wants to resize the canvas

Do NOT delegate:
- Data requests (fetch, compute, describe) — use delegate_to_mission
- Dataset search or parameter listing — handle directly

The specialist has access to all visualization methods and can see what data is in memory.""",
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
    {
        "category": "routing",
        "name": "delegate_to_data_ops",
        "description": """Delegate data transformation, analysis, or export to the DataOps specialist agent. Use this when:
- The user wants to compute derived quantities (magnitude, smoothing, resampling, derivatives, etc.)
- The user wants statistical summaries (describe data)
- The user wants to export data to CSV

Do NOT delegate:
- Data fetching (use delegate_to_mission — fetching requires mission-specific knowledge)
- Visualization requests (use delegate_to_visualization)
- Creating datasets from text/search results (use delegate_to_data_extraction)
- Dataset search or parameter listing (handle directly or use delegate_to_mission)

The DataOps agent can see all data currently in memory via list_fetched_data.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to compute/analyze/export (e.g., 'compute magnitude of AC_H2_MFI.BGSEc', 'describe ACE_Bmag', 'save ACE_Bmag to CSV')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional: available labels, prior results, or other context"
                }
            },
            "required": ["request"]
        }
    },
    {
        "category": "routing",
        "name": "delegate_to_data_extraction",
        "description": """Delegate text-to-DataFrame conversion to the DataExtraction specialist agent. Use this when:
- The user wants to turn unstructured text into a plottable dataset (event lists, search results, catalogs)
- The user wants to extract data tables from a document (PDF or image)
- You have Google Search results with dates and values that should become a DataFrame
- The user says "create a dataset from..." or "make a timeline of..."

Do NOT delegate:
- Data fetching from CDAWeb (use delegate_to_mission)
- Data transformations on existing in-memory data (use delegate_to_data_ops)
- Visualization requests (use delegate_to_visualization)

The DataExtraction agent can read documents (read_document), create DataFrames (store_dataframe), and see what data is in memory (list_fetched_data).""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to extract and store (e.g., 'Create a DataFrame from these X-class flares: [dates and values]. Label it xclass_flares_2024.')"
                },
                "context": {
                    "type": "string",
                    "description": "Optional: source text, search results, or file path to extract data from"
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
            "discovery", "visualization", "data_ops", "data_ops_fetch",
            "data_ops_compute", "data_extraction", "conversation", "routing".
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
