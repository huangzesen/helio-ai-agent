# Capability Summary

Current state of the helio-ai-agent project as of February 2026.

## What It Does

An AI agent that lets users explore and visualize spacecraft/heliophysics data through natural language. Users type requests like "Show me ACE magnetic field data for last week" and the agent searches datasets, fetches data, computes derived quantities, and renders plots — all through conversation.

## Architecture

```
User input
  |
  v
main.py  (readline CLI, --verbose/--gui/--model flags, token usage on exit)
  |  - Commands: quit, reset, status, retry, cancel, errors, capabilities, help
  |  - Single-command mode: python main.py "request"
  |  - Checks for incomplete plans on startup
  |
  v
agent/core.py  OrchestratorAgent  (LLM-driven orchestrator)
  |  - Routes: data requests -> mission agents, visualization -> autoplot agent
  |  - Complex multi-mission requests -> planner -> sub-agents
  |  - Token usage tracking (input/output/api_calls, includes all sub-agents)
  |
  +---> agent/autoplot_agent.py   Autoplot sub-agent (visualization-only tools)
  |       AutoplotAgent           Focused Gemini session for all visualization
  |       execute_autoplot()      Registry method dispatch (16 methods)
  |       autoplot_script()       Direct ScriptContext/DOM code (AST-validated sandbox)
  |       process_request()       Full conversational mode (max 10 iter)
  |       execute_task()          Forced function calling for plan tasks (max 3 iter)
  |                               System prompt includes method catalog + DOM reference
  |
  +---> agent/mission_agent.py    Mission sub-agents (data-only tools)
  |       MissionAgent            Focused Gemini session per spacecraft mission
  |       execute_task()          Forced function calling for plan tasks (max 3 iter)
  |       process_request()       Full conversational mode (max 10 iter)
  |                               Rich system prompt with data ops docs + recommended datasets
  |                               No plotting tools — reports results to orchestrator
  |
  +---> agent/prompts.py           Prompt formatting + tool result formatters
  |       get_system_prompt()      Dynamic system prompt with {today} date
  |       format_tool_result()     Format tool outputs for Gemini conversation
  |
  +---> agent/planner.py          Task planning
  |       is_complex_request()    Regex heuristics for complexity detection
  |       create_plan_from_request()  Gemini JSON output for task decomposition
  |                                   Tags tasks with mission IDs, __autoplot__, dependencies
  |
  +---> agent/tasks.py            Task management
  |       Task, TaskPlan          Data structures (mission, depends_on fields)
  |       TaskStore               JSON persistence to ~/.helio-agent/tasks/
  |
  +---> knowledge/                Dataset discovery + prompt generation
  |       missions/*.json          Per-mission JSON files (8 files, HAPI-derived + hand-curated)
  |       mission_loader.py        Lazy-loading cache, routing table, dataset access
  |       catalog.py               Thin routing layer (loads from JSON, backward-compat SPACECRAFT dict)
  |       prompt_builder.py        Slim system prompt (routing table only) + rich mission/autoplot prompts
  |       hapi_client.py           CDAWeb HAPI /info endpoint (parameter metadata, cached, browse_datasets)
  |
  +---> data_ops/                 Python-side data pipeline (pandas-backed)
  |       fetch.py                  HAPI /data endpoint -> pandas DataFrames (pd.read_csv)
  |       store.py                  In-memory DataStore singleton (label -> DataEntry w/ DataFrame)
  |       custom_ops.py             AST-validated sandboxed executor for LLM-generated pandas/numpy code
  |       plotting.py               Matplotlib fallback (DEPRECATED — plotting goes through Autoplot)
  |
  +---> autoplot_bridge/          Java visualization via JPype
  |       connection.py             JVM startup, ScriptContext singleton, conditional headless flag
  |       registry.py               Method registry (16 methods) — single source of truth for capabilities
  |       script_runner.py          AST-validated sandbox for direct ScriptContext/DOM code
  |       commands.py               plot_cdaweb, set_time_range, export_png/pdf, plot_dataset
  |                                 set_render_type, set_color_table, set_canvas_size
  |                                 execute_script (delegates to script_runner)
  |                                 numpy->QDataSet conversion, overplot with color management
  |                                 GUI mode: reset, title, axis labels, log scale, axis range,
  |                                 save/load session (.vap)
  |
  +---> scripts/                  Tooling
          generate_mission_data.py  Auto-populate JSON from CDAWeb HAPI catalog
          fetch_hapi_cache.py       Download HAPI /info metadata to local cache
          agent_server.py           TCP socket server for multi-turn agent testing
          run_agent_tests.py        Integration test suite (6 scenarios)
          stress_test.py            Stress testing
```

## Tools (14 tool schemas)

### Dataset Discovery
| Tool | Purpose |
|------|---------|
| `search_datasets` | Keyword search across spacecraft/instruments (local catalog) |
| `browse_datasets` | Browse all science datasets for a mission (filtered by calibration exclusion lists) |
| `list_parameters` | List plottable parameters for a dataset (HAPI /info) |
| `get_data_availability` | Check available time range for a dataset (HAPI /info) |

### Autoplot Visualization
| Tool | Purpose |
|------|---------|
| `execute_autoplot` | Execute any of 16 Autoplot methods via the method registry |
| `autoplot_script` | Execute custom ScriptContext/DOM code for advanced visualization (multi-panel, annotations, styling) |

The `execute_autoplot` tool dispatches to the method registry (`autoplot_bridge/registry.py`), which describes 16 operations: `plot_cdaweb`, `plot_stored_data`, `set_time_range`, `export_png`, `export_pdf`, `get_plot_state`, `reset`, `set_title`, `set_axis_label`, `toggle_log_scale`, `set_axis_range`, `save_session`, `load_session`, `set_render_type`, `set_color_table`, `set_canvas_size`.

The `autoplot_script` tool provides direct access to Autoplot's ScriptContext and DOM API via an AST-validated sandbox (`autoplot_bridge/script_runner.py`). Pre-imported namespace includes `sc` (ScriptContext), `dom` (Application), `Color`, `RenderType`, `DatumRangeUtil`, `DatumRange`, `Units`, `DDataSet`, `QDataSet`, `store` (DataStore), and `to_qdataset` (converts stored data labels to QDataSet). Blocks imports, exec/eval, JClass/jpype, and dunder access.

### Data Operations (fetch -> custom_operation -> plot)
| Tool | Purpose |
|------|---------|
| `fetch_data` | Pull HAPI data into memory (label: `DATASET.PARAM`) |
| `list_fetched_data` | Show all in-memory timeseries |
| `custom_operation` | LLM-generated pandas/numpy code (AST-validated, sandboxed) — handles magnitude, arithmetic, smoothing, resampling, derivatives, and any other transformation |
| `describe_data` | Statistical summary of in-memory data (min/max/mean/std/percentiles/NaN) |
| `save_data` | Export in-memory timeseries to CSV file |

### Conversation
| Tool | Purpose |
|------|---------|
| `ask_clarification` | Ask user when request is ambiguous |

### Routing
| Tool | Purpose |
|------|---------|
| `delegate_to_mission` | LLM-driven delegation to a mission specialist sub-agent |
| `delegate_to_autoplot` | LLM-driven delegation to the Autoplot visualization sub-agent |

## Sub-Agent Architecture

### OrchestratorAgent (agent/core.py)
- Sees tools: discovery, data_ops, conversation, routing (NOT autoplot)
- Routes data requests to MissionAgent, visualization to AutoplotAgent
- Handles multi-step plans with mission-tagged task dispatch

### MissionAgent (agent/mission_agent.py)
- Sees tools: discovery, data_ops, conversation (NOT autoplot or routing)
- One agent per spacecraft, cached per session
- Rich system prompt with recommended datasets, data ops docs, analysis patterns

### AutoplotAgent (agent/autoplot_agent.py)
- Sees tools: `execute_autoplot` + `autoplot_script` + `list_fetched_data` (3 tools total)
- System prompt includes the method catalog, DOM hierarchy reference, and script examples
- `execute_autoplot`: Registry-driven dispatch for standard operations (16 methods)
- `autoplot_script`: Direct ScriptContext/DOM code for advanced visualization (multi-panel, styling, annotations)
- Handles all visualization: plotting, customization, export, render type changes

## Supported Spacecraft

| Spacecraft | Instruments | Example Datasets |
|-----------|-------------|-----------------|
| Parker Solar Probe (PSP) | FIELDS/MAG, SWEAP | `PSP_FLD_L2_MAG_RTN_1MIN` |
| Solar Orbiter (SolO) | MAG, SWA-PAS | Magnetic field, proton moments |
| ACE | MAG, SWEPAM | `AC_H2_MFI`, `AC_H0_SWE` |
| OMNI | Combined | `OMNI_HRO_1MIN` |
| Wind | MFI, SWE | `WI_H2_MFI`, `WI_H1_SWE` |
| DSCOVR | MAG, FC | `DSCOVR_H0_MAG`, `DSCOVR_H1_FC` |
| MMS | FGM, FPI-DIS | `MMS1_FGM_SRVY_L2` |
| STEREO-A | MAG, PLASTIC | `STA_L2_MAG_RTN` |

## Time Range Parsing

Handled by `agent/time_utils.py`. Accepts:
- Relative: `"last week"`, `"last 3 days"`, `"last month"`
- Month+year: `"January 2024"`
- Single date: `"2024-01-15"` (full day)
- Date range: `"2024-01-15 to 2024-01-20"`
- Datetime range: `"2024-01-15T06:00 to 2024-01-15T18:00"`
- Space-separated datetime: `"2024-01-15 12:00:00 to 2024-01-16"`

All times are UTC. Outputs `TimeRange` objects with `start`/`end` datetimes. Converts to Autoplot format via `to_autoplot_string()`.

## Key Implementation Details

### Method Registry (`autoplot_bridge/registry.py`)
- Single source of truth for all Autoplot capabilities (16 methods)
- Each method has: name, description, typed parameters (with enums for constrained values)
- `render_method_catalog()` renders the registry into markdown for the LLM prompt
- `get_method(name)` and `validate_args(name, args)` for dispatch and validation
- Adding a new capability = add registry entry + implement bridge method. No tool schema changes needed.

### Script Runner (`autoplot_bridge/script_runner.py`)
- AST-validated sandbox for executing direct ScriptContext/DOM code (analogous to `data_ops/custom_ops.py` for pandas)
- Pre-imported namespace: `sc` (ScriptContext), `dom` (Application), `Color`, `RenderType`, `DatumRangeUtil`, `DatumRange`, `Units`, `DDataSet`, `QDataSet`, `store` (DataStore), `to_qdataset` (converts stored data labels to QDataSet)
- Blocks: imports, exec/eval/compile/open, JClass/jpype (prevents arbitrary Java class construction), os/sys/subprocess/socket/shutil/pathlib/importlib, dunder access, global/nonlocal, async
- `result` assignment is optional (most Autoplot ops are void side-effects); if assigned, string value is captured
- `print()` output is captured and returned in the result dict
- `_format_java_exception()` unwraps JPype Java exceptions for readable error messages

### Autoplot Bridge (`autoplot_bridge/commands.py`)
- **numpy -> QDataSet conversion**: `_numpy_to_qdataset()` converts datetime64[ns] to Units.t2000 (seconds since 2000-01-01) and float64 values to DDataSet objects via JPype.
- **Vector decomposition**: (n,3) arrays are split into 3 scalar series (`.x`, `.y`, `.z`) because rank-2 QDataSets render as spectrograms.
- **Overplot**: `setLayoutOverplot(n)` + `plot(idx, ds)` for multiple series on one panel.
- **Color management**: Golden ratio HSB color generation on first plot, cached per label. New additions to existing plots default to black. Colors persist across `plot_dataset` calls via `_label_colors` dict.
- **Render types**: `set_render_type()` switches between series, scatter, spectrogram, fill_to_zero, staircase_plus, digital.
- **Color tables**: `set_color_table()` for viridis, plasma, jet, etc. (spectrograms and 2D plots).
- **Canvas sizing**: `set_canvas_size()` for custom width/height.
- **PDF export**: `export_pdf()` mirrors the export_png pattern.

### Data Pipeline (`data_ops/`)
- `DataEntry` wraps a `pd.DataFrame` (DatetimeIndex + float64 columns) with backward-compat `.time` and `.values` properties for the Autoplot bridge.
- `DataStore` is a singleton dict keyed by label. The LLM chains tools automatically: fetch -> custom_operation -> plot.
- `custom_ops.py`: AST-validated, sandboxed executor for LLM-generated pandas/numpy code. Replaces all hardcoded compute functions — the LLM writes the pandas code directly.
- HAPI CSV parsing uses `pd.read_csv()` with `pd.to_numeric(errors="coerce")` for robust handling. Detects HAPI JSON error responses (e.g., code 1201 "no data for time range") before attempting CSV parsing.

### Agent Loop (`agent/core.py`)
- Gemini decides which tools to call via function calling.
- Tool results are fed back to Gemini as function responses.
- Loop continues until Gemini produces a text response (or 10 iterations).
- Token usage accumulated from `response.usage_metadata` (prompt_token_count, candidates_token_count).

### LLM-Driven Routing (`agent/core.py`, `agent/mission_agent.py`, `agent/autoplot_agent.py`)
- **Routing**: The OrchestratorAgent (LLM) decides whether to handle a request directly or delegate via `delegate_to_mission` (data) or `delegate_to_autoplot` (visualization) tools. No regex-based routing — the LLM uses conversation context and the routing table to decide.
- **Mission sub-agents**: Each spacecraft has a data specialist with rich system prompt (recommended datasets, data ops docs, analysis patterns). Agents are cached per session. Sub-agents have **data-only tools** (discovery, data_ops, conversation) — no plotting or routing tools.
- **Autoplot sub-agent**: Visualization specialist with `execute_autoplot` + `autoplot_script` + `list_fetched_data` tools. System prompt includes the method catalog, DOM hierarchy reference, and script examples. Handles all plotting, customization, and export.
- **Tool separation**: Tools have a `category` field (`discovery`, `autoplot`, `data_ops`, `conversation`, `routing`). `get_tool_schemas(categories=..., extra_names=...)` filters tools by category. Orchestrator sees `["discovery", "data_ops", "conversation", "routing"]`. AutoplotAgent sees `["autoplot"]` + `list_fetched_data` extra.
- **Post-delegation plotting**: After `delegate_to_mission` returns data, the orchestrator uses `delegate_to_autoplot` to visualize results.
- **Slim orchestrator**: System prompt contains a routing table (mission names + capabilities) plus delegation instructions. No dataset IDs or analysis tips — those live in mission sub-agents.

### Multi-Step Requests
- The orchestrator naturally chains tool calls in its conversation loop (up to 10 iterations)
- "Compare PSP and ACE" → `delegate_to_mission("PSP", ...)` → `delegate_to_mission("ACE", ...)` → `delegate_to_autoplot(plot both)` — all in one `process_message` call
- Complex plans tag tasks with `mission="__autoplot__"` for visualization dispatch
- Legacy planner infrastructure (`agent/planner.py`, `agent/tasks.py`) is retained for programmatic use

### Per-Mission JSON Knowledge (`knowledge/missions/*.json`)
- **8 JSON files**: One per mission (psp.json, ace.json, etc.) with HAPI-derived metadata + hand-curated profiles.
- **Recommended datasets**: All datasets in the instrument section are shown as recommended. Additional datasets are discoverable via `browse_datasets`.
- **Calibration exclusion lists**: Per-mission `_calibration_exclude.json` files filter out calibration, housekeeping, and ephemeris datasets from browse results. Uses glob patterns and exact IDs.
- **Auto-generation**: `scripts/generate_mission_data.py` queries CDAWeb HAPI to populate parameters, dates, descriptions.
- **Loader**: `knowledge/mission_loader.py` provides lazy-loading cache, routing table, and dataset access.

## Configuration

`.env` file at project root:
```
GOOGLE_API_KEY=<gemini-api-key>
AUTOPLOT_JAR=<path-to-autoplot.jar>
JAVA_HOME=<optional, auto-detected>
```

## Running

```bash
python main.py              # Normal mode (headless)
python main.py --verbose    # Show tool calls, timing, errors
python main.py --gui        # Interactive GUI mode (Autoplot window visible)
python main.py --gui -v     # GUI mode with verbose logging
python main.py -m MODEL     # Specify Gemini model (default: gemini-2.5-flash)
python main.py "request"    # Single-command mode (non-interactive, exits after response)
```

### GUI Mode

When launched with `--gui`, Autoplot starts with its native Swing window visible instead of headless mode. Plots appear instantly in the GUI — no need to export to PNG. The AutoplotAgent's system prompt is adjusted to avoid suggesting PNG exports. GUI-specific operations (reset, title, labels, log scale, axis range, save/load session) are available through the same `execute_autoplot` tool.

### CLI Commands

| Command | Description |
|---------|-------------|
| `quit` / `exit` | Exit the program |
| `reset` | Clear conversation history |
| `status` | Show current multi-step plan progress |
| `retry` | Retry the first failed task in current plan |
| `cancel` | Cancel current plan, skip remaining tasks |
| `errors` | Show recent errors from log files |
| `capabilities` / `caps` | Show detailed capability summary |
| `help` | Show welcome message and help |

### Logging (`agent/logging.py`)
- Log files stored in `~/.helio-agent/logs/agent_YYYYMMDD.log`
- Daily rotation, detailed error logging with stack traces
- `log_error()`: Captures context and full stack traces for debugging
- `log_tool_call()` / `log_tool_result()`: Tracks all tool invocations
- `log_plan_event()`: Records plan lifecycle events
- `print_recent_errors()`: CLI command to review recent errors

## Tests

```bash
python -m pytest tests/test_store.py tests/test_custom_ops.py   # 56 tests, data ops
python -m pytest tests/                                          # All tests (~461 tests)
```

Note: `test_agent.py` requires `google-genai` which may not be in the conda env.

## Dependencies

```
google-genai>=1.60.0    # Gemini API (was google-generativeai, migrated)
jpype1==1.5.0           # Java-Python bridge
python-dotenv>=1.0.0    # .env loading
requests>=2.28.0        # HAPI HTTP calls
numpy>=1.24.0           # Array operations
pandas>=2.0.0           # DataFrame-based data pipeline
matplotlib>=3.7.0       # Fallback plotting (deprecated path)
pytest>=7.0.0           # Test framework
```
