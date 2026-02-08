# Capability Summary

Current state of the helio-ai-agent project as of February 2026.

## What It Does

An AI agent that lets users explore and visualize spacecraft/heliophysics data through natural language. Users type requests like "Show me ACE magnetic field data for last week" and the agent searches datasets, fetches data, computes derived quantities, and renders interactive Plotly plots — all through conversation.

## Architecture

```
User input
  |
  v
main.py  (readline CLI, --verbose/--model flags, token usage on exit)
  |  - Commands: quit, reset, status, retry, cancel, errors, sessions, capabilities, help
  |  - Flags: --continue/-c (resume latest), --session/-s ID (resume specific)
  |  - Single-command mode: python main.py "request"
  |  - Auto-saves session every turn; checks for incomplete plans on startup
  |
  v
gradio_app.py  (browser-based chat UI, inline Plotly plots, data table sidebar)
  |  - Wraps same OrchestratorAgent as main.py
  |  - Flags: --share, --port, --verbose, --model
  |
  v
agent/core.py  OrchestratorAgent  (LLM-driven orchestrator)
  |  - Routes: fetch -> mission agents, compute -> DataOps agent, viz -> visualization agent
  |  - Complex multi-mission requests -> planner -> sub-agents
  |  - Token usage tracking (input/output/api_calls, includes all sub-agents)
  |  - Models: Gemini 3 Pro Preview (orchestrator), Gemini 3 Flash Preview (sub-agents)
  |  - Configurable via GEMINI_MODEL / GEMINI_SUB_AGENT_MODEL env vars
  |
  +---> agent/visualization_agent.py  Visualization sub-agent (visualization-only tools)
  |       VisualizationAgent         Focused Gemini session for all visualization
  |       execute_visualization()    Registry method dispatch (5 core methods)
  |       custom_visualization()     Free-form Plotly code sandbox
  |       process_request()          Full conversational mode (max 5 iter, duplicate detection)
  |       execute_task()             Forced function calling for plan tasks (max 3 iter)
  |                                  System prompt includes method catalog + Plotly cookbook
  |
  +---> agent/data_ops_agent.py   DataOps sub-agent (compute/describe/export tools)
  |       DataOpsAgent            Focused Gemini session for data transformations
  |       execute_task()          Forced function calling for plan tasks (max 3 iter)
  |       process_request()       Full conversational mode (max 5 iter, duplicate detection)
  |                               System prompt with computation patterns + code guidelines
  |                               No fetch or plot tools — operates on in-memory data
  |
  +---> agent/mission_agent.py    Mission sub-agents (fetch-only tools)
  |       MissionAgent            Focused Gemini session per spacecraft mission
  |       execute_task()          Forced function calling for plan tasks (max 3 iter)
  |       process_request()       Full conversational mode (max 5 iter, duplicate + error detection)
  |                               Rich system prompt with recommended datasets + analysis patterns
  |                               No compute or plot tools — reports fetched labels to orchestrator
  |
  +---> agent/prompts.py           Prompt formatting + tool result formatters
  |       get_system_prompt()      Dynamic system prompt with {today} date
  |
  +---> agent/planner.py          Task planning
  |       is_complex_request()    Regex heuristics for complexity detection
  |       create_plan_from_request()  Gemini JSON output for task decomposition
  |                                   Tags tasks with mission IDs, __visualization__, dependencies
  |
  +---> agent/session.py           Session persistence
  |       SessionManager          Save/load chat history + DataStore to ~/.helio-agent/sessions/
  |                                Auto-save every turn, --continue/--session CLI flags
  |
  +---> agent/tasks.py            Task management
  |       Task, TaskPlan          Data structures (mission, depends_on fields)
  |       TaskStore               JSON persistence to ~/.helio-agent/tasks/
  |
  +---> knowledge/                Dataset discovery + prompt generation
  |       missions/*.json          Per-mission JSON files (8 curated + 44 auto-generated, 52 total)
  |       mission_loader.py        Lazy-loading cache, routing table, dataset access
  |       mission_prefixes.py      Shared CDAWeb dataset ID prefix map (40+ missions)
  |       cdaweb_catalog.py        Full CDAWeb HAPI catalog fetch/cache/search (2000+ datasets)
  |       catalog.py               Thin routing layer (loads from JSON, backward-compat SPACECRAFT dict)
  |       prompt_builder.py        Slim system prompt (routing table + catalog search) + rich mission/visualization prompts
  |       hapi_client.py           CDAWeb HAPI /info endpoint (parameter metadata, cached, browse_datasets)
  |
  +---> data_ops/                 Python-side data pipeline (pandas-backed)
  |       fetch.py                  HAPI /data endpoint -> pandas DataFrames (pd.read_csv)
  |       store.py                  In-memory DataStore singleton (label -> DataEntry w/ DataFrame)
  |       custom_ops.py             AST-validated sandboxed executor for LLM-generated pandas/numpy code
  |
  +---> rendering/                Plotly-based visualization engine
  |       plotly_renderer.py        Interactive Plotly figures, multi-panel, WebGL, PNG/PDF export via kaleido
  |       registry.py               Method registry (5 core methods) — single source of truth for viz capabilities
  |       custom_viz_ops.py         AST-validated sandbox for LLM-generated Plotly code (titles, labels, etc.)
  |
  +---> scripts/                  Tooling
          generate_mission_data.py  Auto-populate JSON from CDAWeb HAPI catalog
          fetch_hapi_cache.py       Download HAPI /info metadata to local cache
          agent_server.py           TCP socket server for multi-turn agent testing
          run_agent_tests.py        Integration test suite (6 scenarios)
          stress_test.py            Stress testing
```

## Tools (20 tool schemas)

### Dataset Discovery
| Tool | Purpose |
|------|---------|
| `search_datasets` | Keyword search across spacecraft/instruments (local catalog) |
| `browse_datasets` | Browse all science datasets for a mission (filtered by calibration exclusion lists) |
| `list_parameters` | List plottable parameters for a dataset (HAPI /info) |
| `get_data_availability` | Check available time range for a dataset (HAPI /info) |
| `get_dataset_docs` | Fetch CDAWeb documentation for a dataset (instrument info, coordinates, PI contact) |
| `search_full_catalog` | Search full CDAWeb HAPI catalog (2000+ datasets) by keyword |
| `google_search` | Web search via Google Search grounding (isolated Gemini API call) |

### Visualization
| Tool | Purpose |
|------|---------|
| `execute_visualization` | Execute core visualization methods via the method registry (5 methods) |
| `custom_visualization` | Execute free-form Plotly code to customize the current plot |

The `execute_visualization` tool dispatches to the method registry (`rendering/registry.py`), which describes 5 core operations: `plot_stored_data`, `set_time_range`, `export`, `get_plot_state`, `reset`. The `custom_visualization` tool handles all other customization (titles, axis labels, log scale, canvas size, render type, annotations, trace styling, etc.) via LLM-generated Plotly code in an AST-validated sandbox (`rendering/custom_viz_ops.py`).

### Data Operations (fetch -> custom_operation -> plot)
| Tool | Purpose |
|------|---------|
| `fetch_data` | Pull HAPI data into memory (label: `DATASET.PARAM`) |
| `list_fetched_data` | Show all in-memory timeseries |
| `custom_operation` | LLM-generated pandas/numpy code (AST-validated, sandboxed) — handles magnitude, arithmetic, smoothing, resampling, derivatives, and any other transformation |
| `store_dataframe` | Create a new DataFrame from scratch and store it in memory (event lists, catalogs, manual data) |
| `describe_data` | Statistical summary of in-memory data (min/max/mean/std/percentiles/NaN) |
| `save_data` | Export in-memory timeseries to CSV file |

### Document Conversion
| Tool | Purpose |
|------|---------|
| `convert_to_markdown` | Convert files (PDF, DOCX, PPTX, XLSX, HTML, images, etc.) to Markdown using markitdown |

### Conversation
| Tool | Purpose |
|------|---------|
| `ask_clarification` | Ask user when request is ambiguous |

### Routing
| Tool | Purpose |
|------|---------|
| `delegate_to_mission` | LLM-driven delegation to a mission specialist sub-agent |
| `delegate_to_data_ops` | LLM-driven delegation to the data ops specialist sub-agent |
| `delegate_to_visualization` | LLM-driven delegation to the visualization sub-agent |

## Sub-Agent Architecture (4 agents)

### OrchestratorAgent (agent/core.py)
- Sees tools: discovery, conversation, routing, document + `list_fetched_data` extra
- Routes: data fetching -> MissionAgent, computation -> DataOpsAgent, visualization -> VisualizationAgent
- Handles multi-step plans with mission-tagged task dispatch (`__data_ops__`, `__visualization__`)

### MissionAgent (agent/mission_agent.py)
- Sees tools: discovery, data_ops_fetch, conversation + `list_fetched_data` extra
- One agent per spacecraft, cached per session
- Rich system prompt with recommended datasets and analysis patterns
- No compute tools — reports fetched data labels to orchestrator

### DataOpsAgent (agent/data_ops_agent.py)
- Sees tools: data_ops_compute (`custom_operation`, `store_dataframe`, `describe_data`, `save_data`), conversation + `list_fetched_data` extra
- Singleton, cached per session
- System prompt with computation patterns and code guidelines
- No fetch tools — operates on already-fetched data in memory

### VisualizationAgent (agent/visualization_agent.py)
- Sees tools: `execute_visualization` + `custom_visualization` + `list_fetched_data` (3 tools total)
- System prompt includes the method catalog and Plotly cookbook
- `execute_visualization`: Registry-driven dispatch for core operations (5 methods)
- `custom_visualization`: Free-form Plotly code for any customization (titles, labels, scales, render types, annotations, etc.)
- Handles all visualization: plotting, customization, export

## Supported Spacecraft

### Curated Missions (8) — Rich prompts with analysis patterns

| Spacecraft | Instruments | Example Datasets |
|-----------|-------------|-----------------|
| Parker Solar Probe (PSP) | FIELDS/MAG, SWEAP | `PSP_FLD_L2_MAG_RTN_1MIN` |
| Solar Orbiter (SolO) | MAG, SWA-PAS | Magnetic field, proton moments |
| ACE | MAG, SWEPAM | `AC_H2_MFI`, `AC_H0_SWE` |
| OMNI | Combined | `OMNI_HRO_1MIN` |
| Wind | MFI, SWE | `WI_H2_MFI`, `WI_H1_SWE` |
| DSCOVR | MAG, FC | `DSCOVR_H0_MAG`, `DSCOVR_H1_FC` |
| MMS | FGM, FPI-DIS | `MMS1_FGM_SRVY_L2` |
| STEREO-A | MAG, PLASTIC | `STA_L1_MAG_RTN` |

### Full CDAWeb Catalog Access (2000+ datasets)

All CDAWeb datasets are searchable via the `search_full_catalog` tool, including missions like STEREO-B, THEMIS, Cluster, Van Allen Probes, GOES, Voyager 1/2, Ulysses, Geotail, Polar, IMAGE, FAST, SOHO, Juno, MAVEN, MESSENGER, Cassini, New Horizons, IMP-8, ISEE, Arase/ERG, TIMED, TWINS, IBEX, and more.

New missions can be added as curated missions (with mission agent + rich prompts) by creating a JSON file in `knowledge/missions/` via `scripts/generate_mission_data.py --create-new`. The shared prefix map in `knowledge/mission_prefixes.py` maps dataset ID prefixes to mission identifiers.

## Time Range Parsing

Handled by `agent/time_utils.py`. Accepts:
- Relative: `"last week"`, `"last 3 days"`, `"last month"`, `"last year"`
- Month+year: `"January 2024"`
- Single date: `"2024-01-15"` (full day)
- Date range: `"2024-01-15 to 2024-01-20"`
- Datetime range: `"2024-01-15T06:00 to 2024-01-15T18:00"`
- Space-separated datetime: `"2024-01-15 12:00:00 to 2024-01-16"`
- Single datetime: `"2024-01-15T06:00"` (1-hour window)

All times are UTC. Outputs `TimeRange` objects with `start`/`end` datetimes.

## Key Implementation Details

### Method Registry (`rendering/registry.py`)
- Describes 5 core visualization methods: `plot_stored_data`, `set_time_range`, `export`, `get_plot_state`, `reset`
- Each method has: name, description, typed parameters (with enums for constrained values)
- `render_method_catalog()` renders the registry into markdown for the LLM prompt
- `get_method(name)` and `validate_args(name, args)` for dispatch and validation
- Thin wrappers (titles, labels, scales, render types) replaced by `custom_visualization` tool
- Adding a new Plotly customization = LLM already knows it (no code changes needed)

### Custom Visualization Sandbox (`rendering/custom_viz_ops.py`)
- AST-validated sandbox for LLM-generated Plotly code
- Operates on the current `plotly.graph_objects.Figure` object
- Handles titles, axis labels, log scale, axis ranges, canvas sizing, render types, annotations, trace styling, and any other Plotly customization
- Same security model as `data_ops/custom_ops.py` — blocks imports, exec/eval, os/sys access, dunder access

### Data Pipeline (`data_ops/`)
- `DataEntry` wraps a `pd.DataFrame` (DatetimeIndex + float64 columns).
- `DataStore` is a singleton dict keyed by label. The LLM chains tools automatically: fetch -> custom_operation -> plot.
- `custom_ops.py`: AST-validated, sandboxed executor for LLM-generated pandas/numpy code. Replaces all hardcoded compute functions — the LLM writes the pandas code directly.
- HAPI CSV parsing uses `pd.read_csv()` with `pd.to_numeric(errors="coerce")` for robust handling. Detects HAPI JSON error responses (e.g., code 1201 "no data for time range") before attempting CSV parsing.

### Agent Loop (`agent/core.py`)
- Gemini decides which tools to call via function calling.
- Tool results are fed back to Gemini as function responses.
- Orchestrator loop continues until Gemini produces a text response (or 10 iterations), with consecutive delegation error tracking (breaks after 2 failures).
- Sub-agent loops limited to 5 iterations with duplicate call detection and consecutive error tracking.
- Token usage accumulated from `response.usage_metadata` (prompt_token_count, candidates_token_count).

### LLM-Driven Routing (`agent/core.py`, `agent/mission_agent.py`, `agent/data_ops_agent.py`, `agent/visualization_agent.py`)
- **Routing**: The OrchestratorAgent (LLM) decides whether to handle a request directly or delegate via `delegate_to_mission` (fetching), `delegate_to_data_ops` (computation), or `delegate_to_visualization` (visualization) tools. No regex-based routing — the LLM uses conversation context and the routing table to decide.
- **Mission sub-agents**: Each spacecraft has a data fetching specialist with rich system prompt (recommended datasets, analysis patterns). Agents are cached per session. Sub-agents have **fetch-only tools** (discovery, data_ops_fetch, conversation) — no compute, plot, or routing tools.
- **DataOps sub-agent**: Data transformation specialist with `custom_operation`, `store_dataframe`, `describe_data`, `save_data` + `list_fetched_data`. System prompt includes computation patterns and code guidelines. Singleton, cached per session.
- **Visualization sub-agent**: Visualization specialist with `execute_visualization` + `custom_visualization` + `list_fetched_data` tools. System prompt includes the method catalog and Plotly cookbook. Handles all plotting, customization, and export.
- **Tool separation**: Tools have a `category` field (`discovery`, `visualization`, `data_ops`, `data_ops_fetch`, `data_ops_compute`, `conversation`, `routing`, `document`). `get_tool_schemas(categories=..., extra_names=...)` filters tools by category. Orchestrator sees `["discovery", "conversation", "routing", "document"]` + `list_fetched_data` extra. MissionAgent sees `["discovery", "data_ops_fetch", "conversation"]` + `list_fetched_data` extra. DataOpsAgent sees `["data_ops_compute", "conversation"]` + `list_fetched_data` extra. VisualizationAgent sees `["visualization"]` (`execute_visualization` + `custom_visualization`) + `list_fetched_data` extra.
- **Post-delegation flow**: After `delegate_to_mission` returns data labels, the orchestrator uses `delegate_to_data_ops` for computation and then `delegate_to_visualization` to visualize results.
- **Slim orchestrator**: System prompt contains a routing table (mission names + capabilities) plus delegation instructions. No dataset IDs or analysis tips — those live in mission sub-agents.

### Multi-Step Requests
- The orchestrator naturally chains tool calls in its conversation loop (up to 10 iterations, with consecutive delegation error guard)
- "Compare PSP and ACE" -> `delegate_to_mission("PSP", ...)` -> `delegate_to_mission("ACE", ...)` -> `delegate_to_visualization(plot both)` — all in one `process_message` call
- "Fetch ACE mag, compute magnitude, plot" -> `delegate_to_mission("ACE", fetch)` -> `delegate_to_data_ops(compute magnitude)` -> `delegate_to_visualization(plot)`
- Complex plans tag tasks with `mission="__visualization__"` for visualization dispatch, `mission="__data_ops__"` for compute dispatch
- Planner infrastructure (`agent/planner.py`, `agent/tasks.py`) supports programmatic multi-step plans

### Per-Mission JSON Knowledge (`knowledge/missions/*.json`)
- **8 curated JSON files** + 44 auto-generated skeletons (52 total). Curated missions have hand-written profiles (analysis patterns, coordinate systems, data caveats). Auto-generated missions have minimal profiles populated from HAPI metadata.
- **Shared prefix map**: `knowledge/mission_prefixes.py` maps CDAWeb dataset ID prefixes to mission identifiers (40+ mission groups).
- **Full catalog search**: `knowledge/cdaweb_catalog.py` provides `search_full_catalog` tool — searches all 2000+ CDAWeb datasets by keyword, with 24-hour local cache.
- **Recommended datasets**: All datasets in the instrument section are shown as recommended. Additional datasets are discoverable via `browse_datasets`.
- **Calibration exclusion lists**: Per-mission `_calibration_exclude.json` files filter out calibration, housekeeping, and ephemeris datasets from browse results. Uses glob patterns and exact IDs.
- **Auto-generation**: `scripts/generate_mission_data.py` queries CDAWeb HAPI to populate parameters, dates, descriptions. Use `--create-new` to create skeleton JSON files for new missions.
- **Loader**: `knowledge/mission_loader.py` provides lazy-loading cache, routing table, and dataset access.

### Session Persistence (`agent/session.py`)
- `SessionManager` saves and restores chat history + DataStore across process restarts
- Storage layout: `~/.helio-agent/sessions/{session_id}/` with `metadata.json`, `history.json`, and `data/*.pkl`
- Auto-save after every turn in `process_message()` — survives crashes
- DataStore persistence uses `save_to_directory()` / `load_from_directory()` with pickle + `_index.json`
- Chat history round-trips via `Content.model_dump(exclude_none=True)` → JSON → `chats.create(history=...)`
- CLI flags: `--continue` / `-c` (resume latest), `--session` / `-s ID` (resume specific)
- CLI command: `sessions` — list saved sessions
- Gradio: Sessions accordion in sidebar with Load / New / Delete buttons
- Sub-agent state not persisted (fresh chats per request); PlotlyRenderer resets on load (user can re-plot)

### Auto-Clamping Time Ranges
- `_validate_time_range()` in `agent/core.py` auto-adjusts requested time ranges to fit dataset availability windows
- Handles partial overlaps (clamps to available range) and full mismatches (informs user of available range)
- Fail-open: if HAPI metadata call fails, proceeds without validation

### Default Plot Styling
- `_DEFAULT_LAYOUT` in `rendering/plotly_renderer.py` sets explicit white backgrounds (`paper_bgcolor`, `plot_bgcolor`) and dark font color
- Prevents Gradio dark theme CSS from making plots appear black
- Applied in `_ensure_figure()` and `_grow_panels()`

### Gradio Streaming
- `gradio_app.py` supports real-time streaming of agent verbose output
- Agent output unified through Python logging (commit 413eada)

### Google Search Grounding
- `google_search` tool provides web search via Google Search grounding API
- Implemented as a custom function tool that makes an isolated Gemini API call with only GoogleSearch configured (Gemini API does not support google_search + function_declarations in the same call)
- Returns grounded text with source URLs
- Search results can be turned into plottable datasets via the `store_dataframe` tool (google_search → delegate_to_data_ops → store_dataframe → plot)

## Configuration

`.env` file at project root:
```
GOOGLE_API_KEY=<gemini-api-key>
GEMINI_MODEL=<optional, default: gemini-3-pro-preview>
GEMINI_SUB_AGENT_MODEL=<optional, default: gemini-3-flash-preview>
```

## Running

```bash
python main.py              # Normal mode (auto-saves session)
python main.py --verbose    # Show tool calls, timing, errors
python main.py --continue   # Resume most recent session
python main.py --session ID # Resume specific session by ID
python main.py -m MODEL     # Specify Gemini model (overrides .env)
python main.py "request"    # Single-command mode (non-interactive, exits after response)
```

### Gradio Web UI

```bash
python gradio_app.py                # Launch on localhost:7860
python gradio_app.py --share        # Generate a public Gradio URL
python gradio_app.py --port 8080    # Custom port
python gradio_app.py --verbose      # Show tool call details
python gradio_app.py --model MODEL  # Override model
```

Displays interactive Plotly figures inline, data table sidebar, and token usage tracking.

### CLI Commands

| Command | Description |
|---------|-------------|
| `quit` / `exit` | Exit the program |
| `reset` | Clear conversation history (starts new session) |
| `status` | Show current multi-step plan progress |
| `retry` | Retry the first failed task in current plan |
| `cancel` | Cancel current plan, skip remaining tasks |
| `errors` | Show recent error from log files |
| `sessions` | List saved sessions (most recent 10) |
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
python -m pytest tests/test_store.py tests/test_custom_ops.py   # Data ops tests
python -m pytest tests/test_session.py                           # Session persistence tests
python -m pytest tests/                                          # All tests (675 tests)
```

## Dependencies

```
google-genai>=1.60.0    # Gemini API
python-dotenv>=1.0.0    # .env loading
requests>=2.28.0        # HAPI HTTP calls
numpy>=1.24.0           # Array operations
pandas>=2.0.0           # DataFrame-based data pipeline
plotly>=5.18.0          # Interactive scientific data visualization
kaleido>=0.2.1          # Static image export for Plotly (PNG, PDF)
gradio>=4.44.0          # Browser-based chat UI
matplotlib>=3.7.0       # Legacy plotting (unused in main pipeline)
markitdown[all]>=0.1.0  # Document-to-Markdown conversion (PDF, DOCX, PPTX, etc.)
pytest>=7.0.0           # Test framework
```
