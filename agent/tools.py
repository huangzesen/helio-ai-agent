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
- "visualization": plot_data, style_plot, manage_plot (declarative visualization tools)
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

The pandas_code must:
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
- For spectrogram: fetch a 2D variable (size=[N]) and plot directly with `plot_data(plot_type="spectrogram")`
- For 3D variables (size=[M, N]): use `custom_operation` to slice/average to 2D first, then plot as spectrogram

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
                "pandas_code": {
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
            "required": ["source_labels", "pandas_code", "output_label", "description"]
        }
    },

    {
        "category": "data_extraction",
        "name": "store_dataframe",
        "description": """Create a new DataFrame from scratch and store it in memory. Use this when:
- You have text data (event lists, search results, catalogs) that should become a plottable dataset
- The user wants to manually define data points (e.g., from a table in a paper or website)
- You need to create a dataset that doesn't come from CDAWeb

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
        "description": """DEPRECATED: Use `custom_operation` instead — it now has full scipy and pywt in the sandbox.

Compute a spectrogram (2D time-frequency or time-energy data) from an in-memory timeseries. Use this when the user wants a spectrogram, power spectral density over time, dynamic spectrum, or frequency-time plot.

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
        "name": "plot_data",
        "description": """Create a fresh plot from in-memory timeseries data. Use labels from list_fetched_data.
Supports single-panel overlay (default) or multi-panel layout via the panels parameter.
For spectrograms, set plot_type="spectrogram".""",
        "parameters": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "string",
                    "description": "Comma-separated labels of data to plot (e.g., 'Bmag' or 'ACE_Bmag,PSP_Bmag')"
                },
                "panels": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "Panel layout as list of label lists, e.g. [['A','B'], ['C']] for 2 panels. Omit for single-panel overlay."
                },
                "panel_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["line", "spectrogram"]},
                    "description": "Per-panel plot type, parallel to panels array. E.g. ['spectrogram', 'line', 'line']. Omit to use plot_type for all panels."
                },
                "title": {
                    "type": "string",
                    "description": "Optional plot title"
                },
                "plot_type": {
                    "type": "string",
                    "enum": ["line", "spectrogram"],
                    "description": "Default plot type for all panels: 'line' (default) or 'spectrogram'. Override per-panel with panel_types."
                },
                "colorscale": {
                    "type": "string",
                    "description": "Plotly colorscale for spectrograms (e.g., Viridis, Jet, Plasma)"
                },
                "log_y": {
                    "type": "boolean",
                    "description": "Log scale on y-axis (spectrogram)"
                },
                "log_z": {
                    "type": "boolean",
                    "description": "Log scale on color axis (spectrogram intensity)"
                },
                "z_min": {
                    "type": "number",
                    "description": "Min value for spectrogram color scale"
                },
                "z_max": {
                    "type": "number",
                    "description": "Max value for spectrogram color scale"
                },
                "columns": {
                    "type": "integer",
                    "description": "Number of columns for grid layout (default 1). Use 2 for side-by-side epoch comparison."
                },
                "column_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column header labels (e.g. ['Jan 2020', 'Oct 2024']). Length must match columns."
                }
            },
            "required": ["labels"]
        }
    },
    {
        "category": "visualization",
        "name": "style_plot",
        "description": """Apply aesthetic changes to the current plot. All parameters are optional — pass only what you want to change.
Use this for titles, axis labels, log scale, colors, line styles, canvas size, annotations, themes, and legend visibility.""",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Plot title"
                },
                "x_label": {
                    "type": "string",
                    "description": "X-axis label"
                },
                "y_label": {
                    "type": "string",
                    "description": "Y-axis label (applies to all panels)"
                },
                "trace_colors": {
                    "type": "object",
                    "description": "Map trace label -> color, e.g. {'ACE Bmag': 'red'}"
                },
                "line_styles": {
                    "type": "object",
                    "description": "Map trace label -> {width, dash, mode}"
                },
                "log_scale": {
                    "description": "Set y-axis to log scale. String 'y' (all panels log) or 'linear' (all panels linear), or an object mapping panel numbers to 'log'/'linear' for per-panel control, e.g. {'4': 'log', '5': 'log'}"
                },
                "x_range": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "X-axis range [min, max]"
                },
                "y_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Y-axis range [min, max]"
                },
                "legend": {
                    "type": "boolean",
                    "description": "Show (true) or hide (false) legend"
                },
                "font_size": {
                    "type": "integer",
                    "description": "Global font size in points"
                },
                "canvas_size": {
                    "type": "object",
                    "description": "Canvas dimensions: {width: int, height: int}"
                },
                "annotations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of annotations: [{text, x, y}, ...]"
                },
                "colorscale": {
                    "type": "string",
                    "description": "Plotly colorscale for heatmap traces"
                },
                "theme": {
                    "type": "string",
                    "description": "Plotly template name (e.g., 'plotly_dark', 'plotly_white')"
                },
                "vlines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "string",
                                "description": "Timestamp string for the vertical line position (required)"
                            },
                            "label": {
                                "type": "string",
                                "description": "Text label displayed at the top of the line"
                            },
                            "color": {
                                "type": "string",
                                "description": "Line color (default: 'red')"
                            },
                            "dash": {
                                "type": "string",
                                "description": "Line dash style: 'solid', 'dash', 'dot', 'dashdot'"
                            },
                            "width": {
                                "type": "number",
                                "description": "Line width in pixels (default: 1.5)"
                            }
                        },
                        "required": ["x"]
                    },
                    "description": "Vertical lines: [{x, label, color, dash, width}, ...]. x is a timestamp string."
                },
                "vrects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x0": {
                                "type": "string",
                                "description": "Start timestamp (required)"
                            },
                            "x1": {
                                "type": "string",
                                "description": "End timestamp (required)"
                            },
                            "label": {
                                "type": "string",
                                "description": "Text label centered above the highlighted region"
                            },
                            "color": {
                                "type": "string",
                                "description": "Fill color (default: semi-transparent light blue)"
                            },
                            "opacity": {
                                "type": "number",
                                "description": "Fill opacity 0-1 (default: 0.3)"
                            }
                        },
                        "required": ["x0", "x1"]
                    },
                    "description": "Highlighted time ranges: [{x0, x1, label, color, opacity}, ...]. x0/x1 are timestamp strings."
                }
            },
            "required": []
        }
    },
    {
        "category": "visualization",
        "name": "manage_plot",
        "description": """Structural operations on the plot: export, reset, zoom, get state, add/remove traces.
Use action parameter to select the operation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["reset", "get_state", "set_time_range", "export", "remove_trace", "add_trace"],
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
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range for set_time_range action (e.g., '2024-01-15 to 2024-01-20')"
                },
                "label": {
                    "type": "string",
                    "description": "Trace label for remove_trace or add_trace actions"
                },
                "panel": {
                    "type": "integer",
                    "description": "Target panel (1-based) for add_trace action"
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

    # ---- Pipeline tools (orchestrator only) ----

    {
        "category": "pipeline",
        "name": "save_pipeline",
        "description": """Save the current session's data workflow as a reusable pipeline. Use this when the user wants to save their current analysis as a template they can re-run later with different time ranges.

You MUST provide the steps array — cherry-pick from the tool calls made in this session, keeping only the successful data-producing and visualization steps. For each step, write a clear 'intent' describing what it does in plain English.

Time ranges in fetch_data steps should use the variable "$TIME_RANGE" so the pipeline can be replayed for different dates.""",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the pipeline (e.g., 'ACE B-field Overview')"
                },
                "description": {
                    "type": "string",
                    "description": "What this pipeline does (e.g., 'Fetch ACE mag data, compute magnitude, two-panel plot')"
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered list of pipeline steps. Each step: {tool_name, tool_args, intent, produces (optional), depends_on (optional), critical (optional, default true)}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "tool_args": {"type": "object"},
                            "intent": {"type": "string", "description": "Plain-English description of what this step does"},
                            "produces": {"type": "array", "items": {"type": "string"}, "description": "DataStore labels created by this step"},
                            "depends_on": {"type": "array", "items": {"type": "integer"}, "description": "Step IDs (1-based) that must succeed first"},
                            "critical": {"type": "boolean", "description": "If true (default), failure aborts dependent steps"}
                        },
                        "required": ["tool_name", "tool_args", "intent"]
                    }
                },
                "variables": {
                    "type": "object",
                    "description": "Template variables. Keys start with '$' (e.g., '$TIME_RANGE'). Values: {type, description, default}. If omitted, $TIME_RANGE is auto-detected from fetch_data steps."
                }
            },
            "required": ["name", "description", "steps"]
        }
    },
    {
        "category": "pipeline",
        "name": "run_pipeline",
        "description": """Execute a saved pipeline. By default runs deterministically (no LLM, zero tokens, consistent output).

If the user requests modifications (e.g., 'run my ACE pipeline but with red lines'), pass them in the 'modifications' field — this triggers LLM-mediated mode where the pipeline runs first, then the LLM applies only the requested changes.""",
        "parameters": {
            "type": "object",
            "properties": {
                "pipeline_id": {
                    "type": "string",
                    "description": "The pipeline ID (slug) to execute"
                },
                "variable_overrides": {
                    "type": "object",
                    "description": "Override template variables. Example: {\"$TIME_RANGE\": \"2026-01-01 to 2026-01-31\"}"
                },
                "modifications": {
                    "type": "string",
                    "description": "Natural language description of changes to apply after pipeline execution (triggers LLM-mediated mode)"
                }
            },
            "required": ["pipeline_id"]
        }
    },
    {
        "category": "pipeline",
        "name": "list_pipelines",
        "description": "List all saved pipelines with their names, descriptions, step counts, and variables.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "category": "pipeline",
        "name": "render_spec",
        "description": """Render a plot from a unified plot specification. Used by the pipeline system to combine plot_data + style_plot into a single step.

The spec is a single JSON object containing all layout fields (from plot_data) and aesthetic fields (from style_plot). Same spec = same plot, always.

Layout fields: labels, panels, panel_types, title, plot_type, colorscale, log_y, log_z, z_min, z_max, columns, column_titles.
Style fields: x_label, y_label, trace_colors, line_styles, log_scale, x_range, y_range, legend, font_size, canvas_size, annotations, theme, vlines, vrects.

All fields are optional except 'labels'.""",
        "parameters": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "description": "Unified plot specification containing all plot layout and style fields"
                }
            },
            "required": ["spec"]
        }
    },
    {
        "category": "pipeline",
        "name": "delete_pipeline",
        "description": "Delete a saved pipeline by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "pipeline_id": {
                    "type": "string",
                    "description": "The pipeline ID (slug) to delete"
                }
            },
            "required": ["pipeline_id"]
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
