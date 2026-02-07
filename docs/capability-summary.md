# Capability Summary

Current state of the helio-ai-agent project as of February 2026.

## What It Does

An AI agent that lets users explore and visualize spacecraft/heliophysics data through natural language. Users type requests like "Show me ACE magnetic field data for last week" and the agent searches datasets, fetches data, computes derived quantities, and renders plots — all through conversation.

## Architecture

```
User input
  |
  v
main.py  (readline CLI, --verbose/--gui flags, token usage on exit)
  |  - Commands: quit, reset, status, retry, cancel, help
  |  - Checks for incomplete plans on startup
  |
  v
agent/core.py  AutoplotAgent  (LLM-driven orchestrator)
  |  - LLM decides: handle directly OR delegate via delegate_to_mission tool
  |  - Complex multi-mission requests -> planner -> sub-agents
  |  - Mission agent cache (reused across session)
  |  - Token usage tracking (input/output/api_calls, includes sub-agents)
  |
  +---> agent/planner.py    Task planning
  |       is_complex_request()    Regex heuristics for complexity detection
  |       create_plan_from_request()  Gemini JSON output for task decomposition
  |                                   Tags tasks with mission IDs and dependencies
  |
  +---> agent/tasks.py      Task management
  |       Task, TaskPlan         Data structures (mission, depends_on fields)
  |       TaskStore              JSON persistence to ~/.helio-agent/tasks/
  |
  +---> agent/mission_agent.py  Mission sub-agents (data-only tools)
  |       MissionAgent           Focused Gemini session per spacecraft mission
  |       execute_task()         Forced function calling for plan tasks (max 3 iter)
  |       process_request()      Full conversational mode for delegated requests (max 10 iter)
  |                              Rich system prompt with data ops docs + tiered datasets
  |                              No plotting tools — reports results to main agent for plotting
  |
  +---> knowledge/         Dataset discovery + prompt generation
  |       missions/*.json    Per-mission JSON files (8 files, HAPI-derived + hand-curated)
  |       mission_loader.py  Lazy-loading cache, routing table, tier-filtered dataset access
  |       catalog.py         Thin routing layer (loads from JSON, backward-compat SPACECRAFT dict)
  |       prompt_builder.py  Slim system prompt (routing table only) + rich mission prompts
  |       hapi_client.py     CDAWeb HAPI /info endpoint (parameter metadata, cached)
  |
  +---> data_ops/           Python-side data pipeline (pandas-backed)
  |       fetch.py            HAPI /data endpoint -> pandas DataFrames (pd.read_csv)
  |       store.py            In-memory DataStore singleton (label -> DataEntry w/ DataFrame)
  |       custom_ops.py       AST-validated sandboxed executor for LLM-generated pandas/numpy code
  |       plotting.py         Matplotlib fallback (DEPRECATED — plotting goes through Autoplot)
  |
  +---> autoplot_bridge/    Java visualization via JPype
  |       connection.py       JVM startup, ScriptContext singleton, conditional headless flag
  |       commands.py          plot_cdaweb, set_time_range, export_png, plot_dataset
  |                            numpy->QDataSet conversion, overplot with color management
  |                            GUI mode: reset, title, axis labels, log scale, axis range,
  |                            save/load session (.vap)
  |
  +---> scripts/            Tooling
          generate_mission_data.py  Auto-populate JSON from CDAWeb HAPI catalog
```

## Tools (22 total)

### Dataset Discovery
| Tool | Purpose |
|------|---------|
| `search_datasets` | Keyword search across spacecraft/instruments (local catalog) |
| `list_parameters` | List plottable parameters for a dataset (HAPI /info) |
| `get_data_availability` | Check available time range for a dataset (HAPI /info) |

### Autoplot Visualization
| Tool | Purpose |
|------|---------|
| `plot_data` | Plot CDAWeb data directly via Autoplot URI |
| `change_time_range` | Zoom/pan the current plot |
| `export_plot` | Export current plot to PNG (auto-opens in default viewer) |
| `get_plot_info` | Get current URI and time range |

### Interactive GUI (available with `--gui` flag)
| Tool | Purpose |
|------|---------|
| `reset_plot` | Clear the Autoplot canvas (all plots and state) |
| `set_plot_title` | Set or change the plot title |
| `set_axis_label` | Set label on y or z axis |
| `toggle_log_scale` | Switch axis between linear and log scale |
| `set_axis_range` | Manually set min/max range on an axis |
| `save_session` | Save current Autoplot session to .vap file |
| `load_session` | Restore a previously saved .vap session |

### Data Operations (fetch -> custom_operation -> plot)
| Tool | Purpose |
|------|---------|
| `fetch_data` | Pull HAPI data into memory (label: `DATASET.PARAM`) |
| `list_fetched_data` | Show all in-memory timeseries |
| `custom_operation` | LLM-generated pandas/numpy code (AST-validated, sandboxed) — handles magnitude, arithmetic, smoothing, resampling, derivatives, and any other transformation |
| `plot_computed_data` | Display in-memory data in Autoplot canvas (overplot support) |
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

### Autoplot Bridge (`autoplot_bridge/commands.py`)
- **numpy -> QDataSet conversion**: `_numpy_to_qdataset()` converts datetime64[ns] to Units.t2000 (seconds since 2000-01-01) and float64 values to DDataSet objects via JPype.
- **Vector decomposition**: (n,3) arrays are split into 3 scalar series (`.x`, `.y`, `.z`) because rank-2 QDataSets render as spectrograms.
- **Overplot**: `setLayoutOverplot(n)` + `plot(idx, ds)` for multiple series on one panel.
- **Color management**: Golden ratio HSB color generation on first plot, cached per label. New additions to existing plots default to black. Colors persist across `plot_dataset` calls via `_label_colors` dict.

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

### LLM-Driven Routing (`agent/core.py`, `agent/mission_agent.py`)
- **Routing**: The main agent (LLM) decides whether to handle a request directly or delegate via the `delegate_to_mission` tool. No regex-based routing — the LLM uses conversation context and the routing table to decide. Complex multi-mission requests go through the planner.
- **Mission sub-agents**: Each spacecraft has a data specialist with rich system prompt (tiered datasets, data ops docs, analysis patterns). Agents are cached per session. Sub-agents have **data-only tools** (discovery, data_ops, conversation) — no plotting or routing tools.
- **Tool separation**: Tools have a `category` field (`discovery`, `plotting`, `data_ops`, `conversation`, `routing`). `get_tool_schemas(categories=...)` filters tools by category. Sub-agents get `["discovery", "data_ops", "conversation"]`.
- **Post-delegation plotting**: After `delegate_to_mission` returns, the main agent decides whether to plot using `plot_computed_data`. This keeps plotting centralized in the orchestrator.
- **Slim main agent**: System prompt contains a routing table (mission names + capabilities) plus after-delegation instructions. No dataset IDs or analysis tips — those live in mission sub-agents.

### Multi-Step Requests
- The main agent naturally chains tool calls in its conversation loop (up to 10 iterations)
- "Compare PSP and ACE" → `delegate_to_mission("PSP", ...)` → `delegate_to_mission("ACE", ...)` → `plot_computed_data(both labels)` — all in one `process_message` call
- No separate planner needed; the LLM decides the sequence based on context
- Legacy planner infrastructure (`agent/planner.py`, `agent/tasks.py`) is retained for programmatic use but not auto-invoked

### Per-Mission JSON Knowledge (`knowledge/missions/*.json`)
- **8 JSON files**: One per mission (psp.json, ace.json, etc.) with HAPI-derived metadata + hand-curated profiles.
- **Tiered datasets**: `primary` (default, shown prominently) and `advanced` (higher-res, specialized). Tier is hand-curated, preserved on auto-generation.
- **Auto-generation**: `scripts/generate_mission_data.py` queries CDAWeb HAPI to populate parameters, dates, descriptions. Preserves profile and tier values.
- **Loader**: `knowledge/mission_loader.py` provides lazy-loading cache, routing table, and tier-filtered dataset access.

## Configuration

`.env` file at project root:
```
GOOGLE_API_KEY=<gemini-api-key>
AUTOPLOT_JAR=<path-to-autoplot.jar>
JAVA_HOME=<optional, auto-detected>
```

## Running

```bash
python main.py            # Normal mode (headless)
python main.py --verbose  # Show tool calls, timing, errors
python main.py --gui      # Interactive GUI mode (Autoplot window visible)
python main.py --gui -v   # GUI mode with verbose logging
```

### GUI Mode

When launched with `--gui`, Autoplot starts with its native Swing window visible instead of headless mode. Plots appear instantly in the GUI — no need to export to PNG. The 7 GUI tools (reset, title, labels, log scale, axis range, save/load session) become available. The LLM's system prompt is adjusted to avoid suggesting PNG exports.

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
python -m pytest tests/                                          # All tests
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
```
