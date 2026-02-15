"""
Tool definitions for Gemini function calling.

Each tool schema defines what the LLM can call and what parameters it needs.
Tools are executed by the agent core based on LLM decisions.

Categories:
- "discovery": dataset search and parameter listing
- "data_ops": shared data tools (list_fetched_data)
- "data_ops_fetch": mission-specific data fetching (fetch_data)
- "data_ops_compute": data transformation, statistics (custom_operation, describe_data)
- "data_export": save_data (CSV export — orchestrator only, not given to sub-agents)
- "data_extraction": unstructured-to-structured data conversion (store_dataframe)
- "visualization": render_plotly_json, manage_plot (declarative visualization tools)
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
        "name": "list_missions",
        "description": """List all missions with cached metadata. Use this when:
- User asks "what missions/spacecraft are available?"
- You need to see which missions have local data before browsing datasets
- You want a quick overview of the data catalog

Returns mission IDs and dataset counts. No parameters required.""",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
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
        "description": """Fetch timeseries data from CDAWeb into memory for Python-side operations.
Use this to pull data before computing on it (magnitude, averages, differences, etc.) or plotting it.

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
                },
                "force_large_download": {
                    "type": "boolean",
                    "description": "Set to true to override the 1 GB download safety limit. Only use when the user explicitly confirms a large download."
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
        "description": """Apply a pandas/numpy/xarray/scipy/pywt operation to in-memory data. This is the universal compute tool — use it for ALL data transformations after fetching data with fetch_data.

The code must:
- Assign the result to `result` (DataFrame/Series with DatetimeIndex, or xarray DataArray with 'time' dim)
- Use only sandbox variables, `pd` (pandas), `np` (numpy), `xr` (xarray), `scipy` (full scipy), and `pywt` (PyWavelets) — no imports, no file I/O

Each source label becomes a named variable in the sandbox:
- 2D data (DataFrame): `df_<SUFFIX>` where SUFFIX is the part after the last '.' (e.g., 'DATASET.BR' → df_BR)
- 3D+ data (xarray DataArray): `da_<SUFFIX>` (e.g., 'DATASET.EFLUX_VS_PA_E' → da_EFLUX_VS_PA_E)
- If label has no '.', the full label is used as suffix
- The first DataFrame source is also aliased as `df` for backward compatibility

The result can be a DataFrame (with DatetimeIndex) OR an xarray DataArray (with 'time' dim). DataArray results are stored as-is — useful for intermediate xarray→xarray operations or for 2D spectrograms.

xarray operations (DataArray sources — check storage_type in fetch_data response):
- Slice 3D to 2D: `result = da_EFLUX_VS_PA_E.isel(dim1=0)` (keeps as DataArray, can plot as spectrogram)
- Average over a dim: `result = da_EFLUX_VS_PA_E.mean(dim='dim1')` (reduces to 2D DataArray)
- Convert to DataFrame: `result = da_EFLUX_VS_PA_E.isel(dim1=0).to_pandas()` (also valid)
- For spectrogram: fetch a 2D variable (size=[N]) and plot as heatmap with `render_plotly_json`
- For 3D variables (size=[M, N]): use `custom_operation` to slice/average to 2D first, then plot as spectrogram
- IMPORTANT: When producing a DataFrame for heatmap/spectrogram plotting, use meaningful column names
  (e.g., pitch angle values, energy bins, frequency bins) — NOT generic indices ('0', '1', '2').
  The renderer uses column names as y-axis tick values. Use support variables for bin labels.
- For log-scale spectrograms: apply np.log10 in the same or separate custom_operation.
  Example: `result = np.log10(da_EFLUX_VS_PA_E.mean(dim='dim1').clip(min=1e-10))`
  The viz agent CANNOT apply log scaling — do it here.

Single-source operations (one-element array):
- Magnitude: `result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')`
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

Multi-source operations (multiple labels):
- Magnitude from separate components:
  source_labels=['DATASET.BR', 'DATASET.BT', 'DATASET.BN']
  Code: `merged = pd.concat([df_BR, df_BT, df_BN], axis=1); result = merged.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')`
- Cross-cadence merge:
  source_labels=['DATASET_HOURLY.Bmag', 'DATASET_DAILY.density']
  Code: `density_hr = df_density.resample('1h').interpolate(); merged = pd.concat([df_Bmag, density_hr], axis=1); result = merged.dropna()`

Signal processing (scipy + pywt):
- Butterworth bandpass filter:
  `vals = df.iloc[:,0].values; b, a = scipy.signal.butter(4, [0.01, 0.1], btype='band', fs=1.0/60); filtered = scipy.signal.filtfilt(b, a, vals); result = pd.DataFrame({'filtered': filtered}, index=df.index)`
- Power spectrogram:
  `vals = df.iloc[:,0].dropna().values; dt = df.index.to_series().diff().dt.total_seconds().median(); fs = 1.0/dt; f, t_seg, Sxx = scipy.signal.spectrogram(vals, fs=fs, nperseg=256, noverlap=128); times = pd.to_datetime(df.index[0]) + pd.to_timedelta(t_seg, unit='s'); result = pd.DataFrame(Sxx.T, index=times, columns=[str(freq) for freq in f])`
- Wavelet decomposition:
  `coeffs = pywt.wavedec(df.iloc[:,0].values, 'db4', level=5); approx = coeffs[0]; result = pd.DataFrame({'approx': np.interp(np.linspace(0, 1, len(df)), np.linspace(0, 1, len(approx)), approx)}, index=df.index)`
- FFT:
  `vals = df.iloc[:,0].dropna().values; fft_vals = scipy.fft.rfft(vals); freqs = scipy.fft.rfftfreq(len(vals), d=60.0); result = pd.DataFrame({'amplitude': np.abs(fft_vals), 'frequency': freqs}).set_index(pd.date_range(df.index[0], periods=len(freqs), freq='s'))`

Use `search_function_docs` and `get_function_docs` to look up unfamiliar scipy/pywt APIs before writing code.

Do NOT call this tool when the request cannot be expressed as a pandas/numpy operation (e.g., "email me the data", "upload to server"). Instead, explain to the user what is and isn't possible.""",
        "parameters": {
            "type": "object",
            "properties": {
                "source_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels of source timeseries in memory. Each becomes a sandbox variable: df_<SUFFIX> (DataFrame) or da_<SUFFIX> (xarray DataArray) where SUFFIX is the part after the last '.'. First DataFrame also available as 'df'. For single-source ops, pass one-element array."
                },
                "code": {
                    "type": "string",
                    "description": "Python code using df/da_ variables, pd (pandas), np (numpy), xr (xarray), scipy (full scipy), pywt (PyWavelets). Must assign to 'result'."
                },
                "output_label": {
                    "type": "string",
                    "description": "Label for the result (e.g., 'B_normalized', 'B_clipped')"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the operation"
                },
                "units": {
                    "type": "string",
                    "description": "Physical units of the result (e.g., 'nT', 'km/s', 'nT/s', 'cm^-3'). If omitted, inherits from source. Set explicitly when the operation changes dimensions (e.g., derivative adds '/s', multiply changes units, normalize produces dimensionless '')."
                }
            },
            "required": ["source_labels", "code", "output_label", "description"]
        }
    },

    {
        "category": "data_extraction",
        "name": "store_dataframe",
        "description": """Create a new DataFrame from scratch and store it in memory. Use this when:
- You have text data (event lists, search results, catalogs) that should become a plottable dataset
- The user wants to manually define data points (e.g., from a table in a paper or website)
- You need to create a dataset that doesn't come from CDAWeb

The code must:
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
                "code": {
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
            "required": ["code", "output_label", "description"]
        }
    },

    # --- Function Documentation Tools ---
    {
        "category": "function_docs",
        "name": "search_function_docs",
        "description": """Search the scientific computing function catalog by keyword. Use this to find functions for signal processing, spectral analysis, filtering, interpolation, wavelets, statistics, etc.

Returns function names, sandbox call syntax, and one-line summaries.

Cataloged libraries: scipy.signal, scipy.fft, scipy.interpolate, scipy.stats, scipy.integrate, pywt (PyWavelets).""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g., 'bandpass filter', 'spectrogram', 'wavelet', 'interpolate')"
                },
                "package": {
                    "type": "string",
                    "enum": ["scipy.signal", "scipy.fft", "scipy.interpolate", "scipy.stats", "scipy.integrate", "pywt"],
                    "description": "Optional: restrict search to a specific package"
                }
            },
            "required": ["query"]
        }
    },
    {
        "category": "function_docs",
        "name": "get_function_docs",
        "description": """Get the full docstring and signature for a specific function. Use this after search_function_docs to understand function parameters, return values, and usage examples before writing code.""",
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package path (e.g., 'scipy.signal', 'pywt')"
                },
                "function_name": {
                    "type": "string",
                    "description": "Function name (e.g., 'butter', 'cwt', 'spectrogram')"
                }
            },
            "required": ["package", "function_name"]
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
        "name": "preview_data",
        "description": """Preview actual values (first/last N rows) of an in-memory timeseries. Use this when:
- You need to see actual data values to diagnose an issue
- User asks "show me the data" or "what values are in there?"
- A plot looks wrong and you want to check the underlying data
- You want to verify a computation produced correct results

Returns timestamps and values for the requested rows. Use describe_data for statistics instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Label of the data in memory (e.g., 'AC_H2_MFI.BGSEc')"
                },
                "n_rows": {
                    "type": "integer",
                    "description": "Number of rows to show from each end (default: 5, max: 50)"
                },
                "position": {
                    "type": "string",
                    "enum": ["head", "tail", "both"],
                    "description": "Which rows to show: 'head' (first N), 'tail' (last N), or 'both' (default: 'both')"
                }
            },
            "required": ["label"]
        }
    },
    {
        "category": "data_export",
        "name": "save_data",
        "description": """Export an in-memory timeseries to a CSV file.

ONLY use this when the user explicitly asks to save, export, or download data.
Do NOT use this proactively after computations — data stays in memory for plotting.

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
        "name": "render_plotly_json",
        "description": """Create or update the plot by providing a Plotly figure JSON.

You generate a standard Plotly figure dict with `data` (array of traces) and `layout`.
Instead of providing actual data arrays (x, y, z), put a `data_label` field in each
trace dict. The system resolves each label to real data from memory and fills in x/y/z.

## Trace stubs

Each trace in `data` needs:
- `data_label` (string, required): label of the data in memory (from list_fetched_data)
- `type` (string): Plotly trace type — "scatter" (default), "heatmap", "bar", etc.
- All other Plotly trace properties are passed through as-is (mode, line, marker, etc.)
- `xaxis` and `yaxis` (strings): axis references like "x", "x2", "y", "y2" for multi-panel
- Vector data (3-column) is auto-decomposed into (x), (y), (z) component traces

## Layout

Standard Plotly layout dict. For multi-panel plots, define multiple yaxes with domains:
- `yaxis`: {"domain": [0.55, 1], "title": {"text": "nT"}}
- `yaxis2`: {"domain": [0, 0.45], "title": {"text": "Hz"}}
- `xaxis`: {"domain": [0, 1], "anchor": "y"}
- `xaxis2`: {"domain": [0, 1], "anchor": "y2", "matches": "x"}

Shapes, annotations, and all standard Plotly layout properties work directly.

## Automatic processing

The system automatically handles:
- DatetimeIndex → ISO 8601 strings for x-axis
- Vector data (n,3) → 3 separate component traces with color assignment
- Large datasets (>5000 pts) → min-max downsampling
- Very large datasets (>100K pts) → WebGL (scattergl)
- NaN values → None (Plotly requirement)
- Heatmap colorbar positioning from yaxis domain

## Example: single panel

```json
{"data": [{"type": "scatter", "data_label": "ACE_Bmag", "mode": "lines", "line": {"color": "red"}}],
 "layout": {"title": {"text": "ACE Magnetic Field"}, "yaxis": {"title": {"text": "nT"}}}}
```

## Example: two panels

```json
{"data": [
    {"type": "scatter", "data_label": "ACE_Bmag", "xaxis": "x", "yaxis": "y"},
    {"type": "scatter", "data_label": "ACE_density", "xaxis": "x2", "yaxis": "y2"}
  ],
 "layout": {
    "xaxis":  {"domain": [0, 1], "anchor": "y"},
    "xaxis2": {"domain": [0, 1], "anchor": "y2", "matches": "x"},
    "yaxis":  {"domain": [0.55, 1], "anchor": "x", "title": {"text": "B (nT)"}},
    "yaxis2": {"domain": [0, 0.45], "anchor": "x2", "title": {"text": "n (cm⁻³)"}}
  }}
```""",
        "parameters": {
            "type": "object",
            "properties": {
                "figure_json": {
                    "type": "object",
                    "description": "Plotly figure dict with 'data' (array of trace stubs with data_label) and 'layout'."
                }
            },
            "required": ["figure_json"]
        }
    },
    {
        "category": "visualization",
        "name": "manage_plot",
        "description": """Imperative operations on the current figure: export, reset, get state.
Use action parameter to select the operation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["reset", "get_state", "export"],
                    "description": "Action to perform"
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename for export action"
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "pdf"],
                    "description": "Export format: 'png' (default) or 'pdf'"
                }
            },
            "required": ["action"]
        }
    },

    # --- Full Catalog Search ---
    {
        "category": "discovery",
        "name": "search_full_catalog",
        "description": """Search the full CDAWeb catalog (2000+ datasets) by keyword. Use this when:
- User asks about a spacecraft or instrument NOT in the supported missions table
- User wants to browse broadly across all available data (e.g., "what magnetospheric data is available?")
- User asks about a mission you don't have a specialist agent for (Cluster, THEMIS, Voyager, GOES, etc.)
- User wants to search by physical quantity across all missions (e.g., "proton density datasets")

Returns matching dataset IDs and titles. Any dataset found can be fetched with fetch_data — you do NOT need a mission agent for missions not in the routing table.

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

    # --- Web Search (orchestrator + planner only) ---
    {
        "category": "web_search",
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
The extracted text is saved to the agent data directory (default ~/.helio-agent/documents/) for persistence across sessions.
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

    # --- Memory ---
    {
        "category": "memory",
        "name": "recall_memories",
        "description": """Search or browse archived memories from past sessions. Use when:
- The user references something from a previous session ("last time", "before", "we did X")
- You need context about past analyses, preferences, or lessons learned
- The user asks what they've done before or what data they've looked at

Returns a list of archived memory entries with type, content, and date.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g., 'ACE magnetic', 'smoothing'). Leave empty to list recent entries."
                },
                "type": {
                    "type": "string",
                    "enum": ["preference", "summary", "pitfall"],
                    "description": "Optional: filter by memory type"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default 20)"
                }
            },
            "required": []
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
- Visualization requests (plotting, zoom, render changes) — use delegate_to_visualization
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
- The user wants to zoom, set time range, or resize the canvas

Export requests (PNG/PDF) are handled automatically when delegated here — no special handling needed.

Do NOT delegate:
- Data requests (fetch, compute, describe) — use delegate_to_mission
- Dataset search or parameter listing — handle directly

The specialist has access to all visualization methods and can see what data is in memory.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The visualization request (e.g., 'plot ACE_Bmag and PSP_Bmag together', 'switch to scatter plot', 'set log scale on y-axis')"
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
        "description": """Delegate data transformation or analysis to the DataOps specialist agent. Use this when:
- The user wants to compute derived quantities (magnitude, smoothing, resampling, derivatives, etc.)
- The user wants statistical summaries (describe data)

Do NOT delegate:
- Data fetching (use delegate_to_mission — fetching requires mission-specific knowledge)
- Visualization requests (use delegate_to_visualization)
- Creating datasets from text/search results (use delegate_to_data_extraction)
- Dataset search or parameter listing (handle directly or use delegate_to_mission)
- Data export to CSV — only do this when explicitly requested by the user

The DataOps agent can see all data currently in memory via list_fetched_data.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What to compute/analyze (e.g., 'compute magnitude of AC_H2_MFI.BGSEc', 'describe ACE_Bmag')"
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

    # --- request_planning ---
    {
        "category": "routing",
        "name": "request_planning",
        "description": """Activate the multi-step planning system for complex requests that require
coordinated execution across multiple agents. Use this when:
- The user's request requires fetching data from MULTIPLE spacecraft/missions
- The request involves a sequence of 3+ distinct steps (fetch → compute → plot)
- The request asks to compare data from different sources
- The request requires searching the web, extracting data, AND plotting it

Do NOT use this for requests that can be satisfied with 1-2 direct delegations.""",
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The full user request to plan and execute"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this request needs multi-step planning"
                },
                "time_range": {
                    "type": "string",
                    "description": "The resolved time range in 'YYYY-MM-DD to YYYY-MM-DD' format (e.g. '2012-05-01 to 2013-01-31'). Extract from the user's request, converting natural language dates to ISO dates. For sub-day precision use 'YYYY-MM-DDTHH:MM:SS to YYYY-MM-DDTHH:MM:SS'."
                }
            },
            "required": ["request", "reasoning", "time_range"]
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
