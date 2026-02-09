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
  |  - Flags: --refresh (update time ranges), --refresh-full (rebuild primary),
  |           --refresh-all (rebuild all missions from CDAWeb)
  |  - Single-command mode: python main.py "request"
  |  - Auto-saves session every turn; checks for incomplete plans on startup
  |  - Mission data menu on startup (interactive refresh prompt)
  |
  v
gradio_app.py  (browser-based chat UI, inline Plotly plots, data table sidebar)
  |  - Wraps same OrchestratorAgent as main.py
  |  - Flags: --share, --port, --quiet, --model
  |  - Flags: --refresh, --refresh-full, --refresh-all (same as main.py)
  |  - Multimodal file upload (PDF, images)
  |  - Browse & Fetch sidebar (mission → dataset → parameter dropdowns, direct HAPI fetch)
  |
  v
agent/core.py  OrchestratorAgent  (LLM-driven orchestrator)
  |  - Routes: fetch -> mission agents, compute -> DataOps agent, viz -> visualization agent
  |  - Complex multi-mission requests -> planner -> sub-agents
  |  - Token usage tracking (input/output/thinking/api_calls, includes all sub-agents)
  |  - Models: Gemini 3 Pro Preview (orchestrator), Gemini 3 Flash Preview (sub-agents)
  |  - Configurable via GEMINI_MODEL / GEMINI_SUB_AGENT_MODEL env vars
  |  - Thinking levels: HIGH (orchestrator + planner), LOW (all sub-agents)
  |
  +---> agent/visualization_agent.py  Visualization sub-agent (visualization-only tools)
  |       VisualizationAgent         Focused Gemini session for all visualization
  |       plot_data()                Create plots from in-memory data (overlay or multi-panel)
  |       style_plot()               Apply aesthetics via key-value params (no code gen)
  |       manage_plot()              Structural ops: export, reset, zoom, add/remove traces
  |       process_request()          Full conversational mode (max 5 iter, duplicate detection)
  |       execute_task()             Forced function calling for plan tasks (max 3 iter)
  |                                  System prompt includes tool catalog (declarative, no code gen)
  |
  +---> agent/data_ops_agent.py   DataOps sub-agent (compute/describe/export tools)
  |       DataOpsAgent            Focused Gemini session for data transformations
  |       execute_task()          Forced function calling for plan tasks (max 3 iter)
  |       process_request()       Full conversational mode (max 5 iter, duplicate detection)
  |                               System prompt with computation patterns + code guidelines
  |                               No fetch or plot tools — operates on in-memory data
  |
  +---> agent/data_extraction_agent.py  DataExtraction sub-agent (text-to-DataFrame)
  |       DataExtractionAgent     Focused Gemini session for unstructured-to-structured conversion
  |       execute_task()          Forced function calling for plan tasks (max 3 iter)
  |       process_request()       Full conversational mode (max 5 iter, duplicate detection)
  |                               Tools: store_dataframe, read_document, ask_clarification
  |                               Turns search results, documents, event lists into DataFrames
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
  |       PlannerAgent            Chat-based planner with plan-execute-replan loop
  |                               Emits task batches, observes results, adapts plan
  |                               Uses structured JSON output (no tool calling)
  |                               Model: GEMINI_PLANNER_MODEL (defaults to GEMINI_MODEL)
  |
  +---> agent/session.py           Session persistence
  |       SessionManager          Save/load chat history + DataStore to ~/.helio-agent/sessions/
  |                                Auto-save every turn, --continue/--session CLI flags
  |
  +---> agent/memory.py            Long-term memory (cross-session)
  |       MemoryStore              Persist preferences + summaries to ~/.helio-agent/memory.json
  |                                Inject into prompts, extract at session boundaries
  |
  +---> agent/tasks.py            Task management
  |       Task, TaskPlan          Data structures (mission, depends_on fields)
  |       TaskStore               JSON persistence to ~/.helio-agent/tasks/
  |
  +---> knowledge/                Dataset discovery + prompt generation
  |       missions/*.json          Per-mission JSON files (52 total, all auto-generated from CDAWeb)
  |       mission_loader.py        Lazy-loading cache, routing table, dataset access
  |       mission_prefixes.py      Shared CDAWeb dataset ID prefix map (40+ missions)
  |       cdaweb_metadata.py       CDAWeb REST API client — InstrumentType-based grouping
  |       cdaweb_catalog.py        Full CDAWeb HAPI catalog fetch/cache/search (2000+ datasets)
  |       catalog.py               Thin routing layer (loads from JSON, backward-compat SPACECRAFT dict)
  |       prompt_builder.py        Slim system prompt (routing table + catalog search) + rich mission/visualization prompts
  |       hapi_client.py           CDAWeb HAPI /info endpoint (parameter metadata, cached, browse_datasets)
  |       startup.py               Mission data startup: status check, interactive refresh menu, CLI flag resolution
  |       bootstrap.py             Mission JSON auto-generation from CDAWeb HAPI catalog
  |
  +---> data_ops/                 Python-side data pipeline (pandas-backed)
  |       fetch.py                  HAPI /data endpoint -> pandas DataFrames (pd.read_csv)
  |       store.py                  In-memory DataStore singleton (label -> DataEntry w/ DataFrame)
  |       custom_ops.py             AST-validated sandboxed executor for LLM-generated pandas/numpy code
  |
  +---> rendering/                Plotly-based visualization engine
  |       plotly_renderer.py        Interactive Plotly figures, multi-panel, WebGL, PNG/PDF export via kaleido
  |       registry.py               Tool registry (3 declarative tools) — single source of truth for viz capabilities
  |
  +---> scripts/                  Tooling
          generate_mission_data.py  Auto-populate JSON from CDAWeb HAPI catalog
          fetch_hapi_cache.py       Download HAPI /info metadata to local cache
          agent_server.py           TCP socket server for multi-turn agent testing
          run_agent_tests.py        Integration test suite (6 scenarios)
          test_dataset_loading.py   End-to-end dataset loading test across all HAPI datasets
          regression_test_20260207.py  Regression tests from 2026-02-07 session
          stress_test.py            Stress testing
```

## Tools (24 tool schemas)

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
| `plot_data` | Create plots from in-memory data (single panel overlay or multi-panel layout) |
| `style_plot` | Apply aesthetics via key-value params: titles, labels, colors, scales, fonts, annotations |
| `manage_plot` | Structural operations: export (PNG/PDF), reset, zoom/time range, add/remove traces |

Three declarative tools replace the old `execute_visualization` + `custom_visualization` approach. All customization is done via bounded parameter sets — no free-form code generation. The tool registry (`rendering/registry.py`) describes all 3 tools with their parameters and examples.

### Data Operations (fetch -> custom_operation -> plot)
| Tool | Purpose |
|------|---------|
| `fetch_data` | Pull HAPI data into memory (label: `DATASET.PARAM`) |
| `list_fetched_data` | Show all in-memory timeseries |
| `custom_operation` | LLM-generated pandas/numpy code (AST-validated, sandboxed) — handles magnitude, arithmetic, smoothing, resampling, derivatives, and any other transformation |
| `compute_spectrogram` | LLM-generated scipy.signal code to compute spectrograms from timeseries (AST-validated, sandboxed) |
| `describe_data` | Statistical summary of in-memory data (min/max/mean/std/percentiles/NaN) |
| `save_data` | Export in-memory timeseries to CSV file |

### Data Extraction
| Tool | Purpose |
|------|---------|
| `store_dataframe` | Create a new DataFrame from scratch and store it in memory (event lists, catalogs, search results, manual data) |

### Document Reading
| Tool | Purpose |
|------|---------|
| `read_document` | Read PDF and image files using Gemini vision (extracts text, tables, charts) |

### Conversation
| Tool | Purpose |
|------|---------|
| `ask_clarification` | Ask user when request is ambiguous |

### Routing
| Tool | Purpose |
|------|---------|
| `delegate_to_mission` | LLM-driven delegation to a mission specialist sub-agent |
| `delegate_to_data_ops` | LLM-driven delegation to the data ops specialist sub-agent |
| `delegate_to_data_extraction` | LLM-driven delegation to the data extraction specialist sub-agent |
| `delegate_to_visualization` | LLM-driven delegation to the visualization sub-agent |
| `request_planning` | Activate multi-step planning system for complex requests (orchestrator can trigger dynamically) |

## Sub-Agent Architecture (5 agents)

### OrchestratorAgent (agent/core.py)
- Sees tools: discovery, conversation, routing, document + `list_fetched_data` extra
- Routes: data fetching -> MissionAgent, computation -> DataOpsAgent, text-to-data -> DataExtractionAgent, visualization -> VisualizationAgent
- Handles multi-step plans with mission-tagged task dispatch (`__data_ops__`, `__data_extraction__`, `__visualization__`)

### MissionAgent (agent/mission_agent.py)
- Sees tools: discovery, data_ops_fetch, conversation + `list_fetched_data` extra
- One agent per spacecraft, cached per session
- Rich system prompt with recommended datasets and analysis patterns
- No compute tools — reports fetched data labels to orchestrator

### DataOpsAgent (agent/data_ops_agent.py)
- Sees tools: data_ops_compute (`custom_operation`, `describe_data`, `save_data`), conversation + `list_fetched_data` extra
- Singleton, cached per session
- System prompt with computation patterns and code guidelines
- No fetch tools — operates on already-fetched data in memory

### DataExtractionAgent (agent/data_extraction_agent.py)
- Sees tools: data_extraction (`store_dataframe`), document (`read_document`), conversation (`ask_clarification`) + `list_fetched_data` extra
- Singleton, cached per session
- System prompt with extraction patterns, DataFrame creation guidelines, and document reading workflow
- Turns unstructured text (search results, document tables, event catalogs) into structured DataFrames
- No fetch, compute, or plot tools — creates data from text only

### VisualizationAgent (agent/visualization_agent.py)
- Sees tools: `plot_data` + `style_plot` + `manage_plot` + `list_fetched_data` (4 tools total)
- System prompt includes the tool catalog with parameter descriptions and examples
- `plot_data`: Create plots from in-memory data (overlay or multi-panel)
- `style_plot`: Declarative aesthetics via key-value params (no code generation)
- `manage_plot`: Structural ops (export, reset, zoom, add/remove traces)
- Handles all visualization: plotting, customization, export

## Supported Spacecraft

### Primary Missions (52, all auto-generated from CDAWeb)

All 52 mission JSON files are auto-generated from CDAWeb HAPI metadata via `scripts/generate_mission_data.py`. Key missions include PSP, Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, STEREO-A, Cluster, THEMIS, Van Allen Probes, GOES, Voyager 1/2, Ulysses, and more.

### Full CDAWeb Catalog Access (2000+ datasets)

All CDAWeb datasets are searchable via the `search_full_catalog` tool. New missions can be added by creating a JSON file in `knowledge/missions/` via `scripts/generate_mission_data.py --create-new`. The shared prefix map in `knowledge/mission_prefixes.py` maps dataset ID prefixes to mission identifiers.

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

### Tool Registry (`rendering/registry.py`)
- Describes 3 declarative visualization tools: `plot_data`, `style_plot`, `manage_plot`
- Each tool has: name, description, typed parameters (with enums for constrained values)
- `render_method_catalog()` renders the registry into markdown for the LLM prompt
- `get_method(name)` and `validate_args(name, args)` for dispatch and validation
- All customization (titles, labels, scales, colors, fonts, annotations) handled via `style_plot` key-value params — no free-form code generation

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
- Token usage accumulated from `response.usage_metadata` (prompt_token_count, candidates_token_count, thoughts_token_count).

### LLM-Driven Routing (`agent/core.py`, `agent/mission_agent.py`, `agent/data_ops_agent.py`, `agent/visualization_agent.py`)
- **Routing**: The OrchestratorAgent (LLM) decides whether to handle a request directly or delegate via `delegate_to_mission` (fetching), `delegate_to_data_ops` (computation), `delegate_to_data_extraction` (text-to-DataFrame), or `delegate_to_visualization` (visualization) tools. No regex-based routing — the LLM uses conversation context and the routing table to decide.
- **Mission sub-agents**: Each spacecraft has a data fetching specialist with rich system prompt (recommended datasets, analysis patterns). Agents are cached per session. Sub-agents have **fetch-only tools** (discovery, data_ops_fetch, conversation) — no compute, plot, or routing tools.
- **DataOps sub-agent**: Data transformation specialist with `custom_operation`, `describe_data`, `save_data` + `list_fetched_data`. System prompt includes computation patterns and code guidelines. Singleton, cached per session.
- **DataExtraction sub-agent**: Text-to-DataFrame specialist with `store_dataframe`, `read_document`, `ask_clarification` + `list_fetched_data`. System prompt includes extraction patterns and DataFrame creation guidelines. Singleton, cached per session.
- **Visualization sub-agent**: Visualization specialist with `plot_data` + `style_plot` + `manage_plot` + `list_fetched_data` tools. System prompt includes the tool catalog with parameter descriptions and examples. Handles all plotting, customization, and export via declarative tools (no code generation).
- **Tool separation**: Tools have a `category` field (`discovery`, `visualization`, `data_ops`, `data_ops_fetch`, `data_ops_compute`, `data_extraction`, `conversation`, `routing`, `document`). `get_tool_schemas(categories=..., extra_names=...)` filters tools by category. Orchestrator sees `["discovery", "conversation", "routing", "document"]` + `list_fetched_data` extra. MissionAgent sees `["discovery", "data_ops_fetch", "conversation"]` + `list_fetched_data` extra. DataOpsAgent sees `["data_ops_compute", "conversation"]` + `list_fetched_data` extra. DataExtractionAgent sees `["data_extraction", "document", "conversation"]` + `list_fetched_data` extra. VisualizationAgent sees `["visualization"]` (`plot_data` + `style_plot` + `manage_plot`) + `list_fetched_data` extra.
- **Post-delegation flow**: After `delegate_to_mission` returns data labels, the orchestrator uses `delegate_to_data_ops` for computation, `delegate_to_data_extraction` for text-to-DataFrame conversion, and then `delegate_to_visualization` to visualize results.
- **Slim orchestrator**: System prompt contains a routing table (mission names + capabilities) plus delegation instructions. No dataset IDs or analysis tips — those live in mission sub-agents.

### Multi-Step Requests (Hybrid Planning)
- Simple requests are handled by the orchestrator's conversation loop (up to 10 iterations, with consecutive delegation error guard)
- "Compare PSP and ACE" -> `delegate_to_mission("PSP", ...)` -> `delegate_to_mission("ACE", ...)` -> `delegate_to_visualization(plot both)` — all in one `process_message` call
- Complex requests use **hybrid routing** to the **PlannerAgent** for plan-execute-replan:
  1. **Regex pre-filter**: `is_complex_request()` regex heuristics catch obvious complex cases (free, no API cost) and route directly to planner
  2. **Orchestrator override**: The orchestrator (with HIGH thinking) can also call `request_planning` tool for complex cases the regex missed
  3. PlannerAgent decomposes the request into task batches using structured JSON output
  4. Each batch is executed by routing tasks to the appropriate sub-agent
  5. Results are fed back to the PlannerAgent, which decides to continue or finish
  6. Maximum 5 rounds of replanning (configurable via `MAX_ROUNDS`)
  7. If the planner fails, falls back to direct orchestrator execution
- Tasks are tagged with `mission="__visualization__"` for visualization dispatch, `mission="__data_ops__"` for compute dispatch, `mission="__data_extraction__"` for text-to-DataFrame dispatch

### Thinking Levels (Gemini ThinkingConfig)
- **HIGH**: Orchestrator (`agent/core.py`) and PlannerAgent (`agent/planner.py`) — deep reasoning for routing decisions and plan decomposition
- **LOW**: MissionAgent, VisualizationAgent, DataOpsAgent, DataExtractionAgent — fast execution with minimal thinking overhead
- Thinking tokens tracked separately in `get_token_usage()` across all agents
- Verbose mode logs thought previews (first 200 chars) via `agent/thinking.py` utilities
- Task plans persist to `~/.helio-agent/tasks/` with round tracking for multi-round plans

### Per-Mission JSON Knowledge (`knowledge/missions/*.json`)
- **52 mission JSON files**, all auto-generated from CDAWeb HAPI metadata. Profiles include instrument groupings, dataset parameters, and time ranges populated by `scripts/generate_mission_data.py`.
- **Shared prefix map**: `knowledge/mission_prefixes.py` maps CDAWeb dataset ID prefixes to mission identifiers (40+ mission groups).
- **CDAWeb InstrumentType grouping**: `knowledge/cdaweb_metadata.py` fetches the CDAWeb REST API to get authoritative InstrumentType per dataset (18+ categories like "Magnetic Fields (space)", "Plasma and Solar Wind"). Bootstrap uses this to group datasets into meaningful instrument categories with keywords, instead of dumping everything into "General".
- **Full catalog search**: `knowledge/cdaweb_catalog.py` provides `search_full_catalog` tool — searches all 2000+ CDAWeb datasets by keyword, with 24-hour local cache.
- **Recommended datasets**: All datasets in the instrument section are shown as recommended. Additional datasets are discoverable via `browse_datasets`.
- **Calibration exclusion lists**: Per-mission `_calibration_exclude.json` files filter out calibration, housekeeping, and ephemeris datasets from browse results. Uses glob patterns and exact IDs.
- **Auto-generation**: `scripts/generate_mission_data.py` queries CDAWeb HAPI to populate parameters, dates, descriptions. Use `--create-new` to create skeleton JSON files for new missions.
- **Loader**: `knowledge/mission_loader.py` provides lazy-loading cache, routing table, and dataset access. Routing table derives capabilities from instrument keywords (magnetic field, plasma, energetic particles, electric field, radio/plasma waves, geomagnetic indices, ephemeris, composition, coronagraph, imaging).

### Long-term Memory (`agent/memory.py`)
- Cross-session memory that persists user preferences and session summaries
- Storage: `~/.helio-agent/memory.json` — global, not per-session
- Two memory types: `"preference"` (plot styles, spacecraft of interest, workflow habits) and `"summary"` (what was analyzed in each session)
- Automatic extraction at session boundaries via lightweight Gemini Flash call (no tools, no thinking)
- Injection: prepends memory context to user messages in `process_message()` (avoids chat recreation)
- Deduplication: skips new memories that are substrings of existing ones
- Capped at 15 preferences + 15 summaries per injection to keep prompt size reasonable
- Global enable/disable toggle + per-memory enable/disable
- Gradio UI: "Long-term Memory" accordion in right sidebar with toggle, delete, and clear controls
- CLI: memories extracted automatically on session exit (`main.py`)


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
- `gradio_app.py` streams live progress (tool calls, downloads, thinking) by default; use `--quiet` to disable
- Agent output unified through Python logging (commit 413eada)

### Mission Data Startup (`knowledge/startup.py`)
- Shared startup logic used by both `main.py` and `gradio_app.py`
- `get_mission_status()` scans mission JSONs and reports count, datasets, last refresh date
- `show_mission_menu()` presents interactive refresh options on startup
- `resolve_refresh_flags()` maps CLI flags (`--refresh`, `--refresh-full`, `--refresh-all`) to actions
- `run_mission_refresh()` invokes bootstrap to refresh time ranges, rebuild primary missions, or rebuild all missions
- After refresh, clears mission_loader and hapi_client caches

### Google Search Grounding
- `google_search` tool provides web search via Google Search grounding API
- Implemented as a custom function tool that makes an isolated Gemini API call with only GoogleSearch configured (Gemini API does not support google_search + function_declarations in the same call)
- Returns grounded text with source URLs
- Search results can be turned into plottable datasets via the `store_dataframe` tool (google_search → delegate_to_data_extraction → store_dataframe → plot)

## Configuration

`.env` file at project root:
```
GOOGLE_API_KEY=<gemini-api-key>
GEMINI_MODEL=<optional, default: gemini-3-pro-preview>
GEMINI_SUB_AGENT_MODEL=<optional, default: gemini-3-flash-preview>
GEMINI_PLANNER_MODEL=<optional, default: GEMINI_MODEL>
```

## Running

```bash
python main.py               # Normal mode (auto-saves session)
python main.py --verbose     # Show tool calls, timing, errors
python main.py --continue    # Resume most recent session
python main.py --session ID  # Resume specific session by ID
python main.py -m MODEL      # Specify Gemini model (overrides .env)
python main.py "request"     # Single-command mode (non-interactive, exits after response)
python main.py --refresh     # Refresh dataset time ranges (fast — start/stop dates only)
python main.py --refresh-full  # Full rebuild of primary mission data
python main.py --refresh-all   # Download ALL missions from CDAWeb (full rebuild)
```

### Gradio Web UI

```bash
python gradio_app.py                # Launch on localhost:7860 (live progress on by default)
python gradio_app.py --share        # Generate a public Gradio URL
python gradio_app.py --port 8080    # Custom port
python gradio_app.py --quiet        # Hide live progress log
python gradio_app.py --model MODEL  # Override model
python gradio_app.py --refresh      # Refresh dataset time ranges before launch
```

Features:
- Interactive Plotly figures displayed above the chat
- Multimodal file upload (PDF, images) via drag-and-drop
- Browse & Fetch sidebar: mission → dataset → parameter cascade dropdowns with direct HAPI fetch
- Data table sidebar showing all in-memory timeseries
- Data preview with head/tail rows for any label
- Token usage and memory tracking
- Session management (load/new/delete) in sidebar accordion
- Example prompts for quick start
- Verbose mode streams live debug logs in collapsible `<details>` blocks

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
python -m pytest tests/test_memory.py                            # Long-term memory tests
python -m pytest tests/                                          # All tests
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
tqdm>=4.60.0            # Progress bars for bootstrap/data downloads
pytest>=7.0.0           # Test framework
```
