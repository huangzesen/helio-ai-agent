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
  |  - Flags: --refresh (update time ranges), --refresh-full (rebuild all),
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
  |  - Browse & Fetch sidebar (mission → dataset → parameter dropdowns, direct CDF fetch)
  |
  v
agent/core.py  OrchestratorAgent  (LLM-driven orchestrator)
  |  - Routes: fetch -> mission agents, compute -> DataOps agent, viz -> visualization agent
  |  - Complex multi-mission requests -> planner -> sub-agents
  |  - Token usage tracking (input/output/thinking/api_calls, includes all sub-agents)
  |  - Models: Gemini 3 Pro Preview (orchestrator), Gemini 3 Flash Preview (sub-agents)
  |  - Configurable via ~/.helio-agent/config.json (model / sub_agent_model keys)
  |  - Thinking levels: HIGH (orchestrator + planner), LOW (all sub-agents)
  |
  +---> agent/visualization_agent.py  Visualization sub-agent (spec-based workflow)
  |       VisualizationAgent         Focused Gemini session for all visualization
  |       update_plot_spec()         Create/update plots via single unified JSON spec
  |       list_fetched_data()        Discover available data in memory
  |       process_request()          Full conversational mode (max 5 iter, duplicate detection)
  |       execute_task()             Forced function calling for plan tasks (max 3 iter)
  |                                  System prompt with spec field reference and examples
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
  |       execute_task()          Forced function calling for plan tasks (max 5 iter)
  |       process_request()       Full conversational mode (max 5 iter, duplicate + error detection)
  |                               Two-mode task prompt: candidate inspection vs direct fetch
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
  |       MemoryStore              Persist preferences + summaries + pitfalls to ~/.helio-agent/memory.json
  |                                Inject into prompts, extract at session boundaries
  |
  +---> agent/memory_agent.py     Passive MemoryAgent (self-evolving session analysis)
  |       MemoryAgent              Monitors log growth + conversation turns, auto-triggers analysis
  |                                Extracts pitfalls (operational knowledge) + error patterns
  |                                Saves reports to ~/.helio-agent/reports/
  |
  +---> agent/tasks.py            Task management
  |       Task, TaskPlan          Data structures (mission, depends_on fields)
  |       TaskStore               JSON persistence to ~/.helio-agent/tasks/
  |
  +---> knowledge/                Dataset discovery + prompt generation
  |       function_catalog.py      Auto-generated searchable catalog of scipy/pywt functions (search + docstring retrieval)
  |       missions/*.json          Per-mission JSON files (52 total, all auto-generated from CDAWeb)
  |       mission_loader.py        Lazy-loading cache, routing table, dataset access
  |       mission_prefixes.py      Shared CDAWeb dataset ID prefix map (40+ missions)
  |       cdaweb_metadata.py       CDAWeb REST API client — InstrumentType-based grouping
  |       cdaweb_catalog.py        Full CDAWeb catalog fetch/cache/search (CDAS REST API)
  |       catalog.py               Thin routing layer (loads from JSON, backward-compat SPACECRAFT dict)
  |       prompt_builder.py        Slim system prompt (routing table + catalog search) + rich mission/visualization prompts
  |       metadata_client.py       Dataset metadata (3-layer cache: memory → file → Master CDF)
  |       master_cdf.py            Master CDF skeleton download + parameter metadata extraction
  |       startup.py               Mission data startup: status check, interactive refresh menu, CLI flag resolution
  |       bootstrap.py             Mission JSON auto-generation from CDAS REST + Master CDF
  |
  +---> data_ops/                 Python-side data pipeline (pandas-backed)
  |       fetch.py                  Data fetching via CDF backend
  |       fetch_cdf.py              CDF data fetching + Master CDF-based variable listing
  |       store.py                  In-memory DataStore singleton (label -> DataEntry w/ DataFrame)
  |       custom_ops.py             AST-validated sandboxed executor for LLM-generated pandas/numpy code
  |
  +---> rendering/                Plotly-based visualization engine
  |       plotly_renderer.py        Interactive Plotly figures, multi-panel, WebGL, PNG/PDF export via kaleido
  |       registry.py               Tool registry (3 declarative tools) — single source of truth for viz capabilities
  |
  +---> scripts/                  Tooling
          generate_mission_data.py  Auto-populate JSON from CDAS REST + Master CDF
          fetch_metadata_cache.py   Download metadata cache (Master CDF)
          agent_server.py           TCP socket server for multi-turn agent testing
          run_agent_tests.py        Integration test suite (6 scenarios)
          test_dataset_loading.py   End-to-end dataset loading test across all datasets
          regression_test_20260207.py  Regression tests from 2026-02-07 session
          stress_test.py            Stress testing
```

## Tools (35 tool schemas)

### Dataset Discovery
| Tool | Purpose |
|------|---------|
| `search_datasets` | Keyword search across spacecraft/instruments (local catalog) |
| `browse_datasets` | Browse all science datasets for a mission (filtered by calibration exclusion lists) |
| `list_parameters` | List plottable parameters for a dataset (Master CDF / local cache) |
| `get_data_availability` | Check available time range for a dataset (local cache / CDAS REST) |
| `get_dataset_docs` | Fetch CDAWeb documentation for a dataset (instrument info, coordinates, PI contact) |
| `search_full_catalog` | Search full CDAWeb catalog (2000+ datasets, CDAS REST primary) by keyword |
| `google_search` | Web search — Gemini uses built-in Google Search grounding; non-Gemini providers use Tavily fallback |

### Visualization
| Tool | Purpose |
|------|---------|
| `update_plot_spec` | Create or update plots via a single unified JSON spec. Layout changes trigger re-render; style-only changes are applied in-place. |
| `manage_plot` | Structural operations: export (PNG/PDF), reset, zoom/time range, add/remove traces |

The viz agent uses `update_plot_spec` and `manage_plot` for all visualization operations. The system diffs the new spec against the current one and decides whether to re-render (layout changed) or restyle in-place (only aesthetics changed). The tool registry (`rendering/registry.py`) describes both tools with their parameters and examples. `plot_data` and `style_plot` have been removed from the LLM-facing tool set (the Python methods remain as internal implementation).

### Plot Self-Review
Every `update_plot_spec` re-render returns a `review` field with structured metadata for LLM self-assessment:
- **`trace_summary`**: per-trace name, panel, point count, y-range, gap status
- **`warnings`**: heuristic checks — cluttered panels (>6 traces), resolution mismatches (>10x point count difference), suspicious y-ranges (possible fill values), invisible traces (all NaN/missing data), empty panels
- **`hint`**: one-line summary of panel layout and trace assignments

The LLM inspects this metadata within the existing tool loop and can self-correct (resample, split panels, filter fill values) before responding to the user — no extra LLM call or image export needed.

### Data Operations (fetch -> custom_operation -> plot)
| Tool | Purpose |
|------|---------|
| `fetch_data` | Pull data into memory via CDF download (label: `DATASET.PARAM`) |
| `list_fetched_data` | Show all in-memory timeseries |
| `custom_operation` | LLM-generated pandas/numpy/scipy/pywt code (AST-validated, sandboxed) — handles magnitude, arithmetic, smoothing, resampling, derivatives, filtering, spectrograms, wavelets, and any other transformation |
| `compute_spectrogram` | DEPRECATED — use `custom_operation` instead (which now has full scipy in the sandbox) |
| `describe_data` | Statistical summary of in-memory data (min/max/mean/std/percentiles/NaN) |
| `preview_data` | Preview actual values (first/last N rows) of in-memory timeseries for debugging or inspection |
| `save_data` | Export in-memory timeseries to CSV file |

### Data Extraction
| Tool | Purpose |
|------|---------|
| `store_dataframe` | Create a new DataFrame from scratch and store it in memory (event lists, catalogs, search results, manual data) |

### Function Documentation
| Tool | Purpose |
|------|---------|
| `search_function_docs` | Search scientific computing function catalog by keyword (scipy.signal, scipy.fft, scipy.interpolate, scipy.stats, scipy.integrate, pywt) |
| `get_function_docs` | Get full docstring and signature for a specific function |

### Document Reading
| Tool | Purpose |
|------|---------|
| `read_document` | Read PDF and image files using Gemini vision (extracts text, tables, charts) |

### Memory
| Tool | Purpose |
|------|---------|
| `recall_memories` | Search or browse archived memories from past sessions (preferences, summaries, pitfalls) |

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

### Pipeline
| Tool | Purpose |
|------|---------|
| `save_pipeline` | Save session's data workflow as a reusable pipeline template |
| `run_pipeline` | Execute a saved pipeline — deterministic by default, or with LLM-mediated modifications |
| `list_pipelines` | List all saved pipelines with names, descriptions, step counts, and variables |
| `delete_pipeline` | Delete a saved pipeline by ID |
| `render_spec` | Render a plot from a unified spec (used by pipeline replay) |

## Sub-Agent Architecture (5 agents)

### OrchestratorAgent (agent/core.py)
- Sees tools: discovery, conversation, routing, document, pipeline + `list_fetched_data` extra
- Routes: data fetching -> MissionAgent, computation -> DataOpsAgent, text-to-data -> DataExtractionAgent, visualization -> VisualizationAgent
- Handles multi-step plans with mission-tagged task dispatch (`__data_ops__`, `__data_extraction__`, `__visualization__`)

### MissionAgent (agent/mission_agent.py)
- Sees tools: discovery, data_ops_fetch, conversation + `list_fetched_data` extra
- One agent per spacecraft, cached per session
- Rich system prompt with recommended datasets and analysis patterns
- **Two-mode operation**: when planner provides `candidate_datasets`, inspects candidates via `list_parameters` and selects best dataset/parameters autonomously; otherwise executes exact instructions directly
- Handles all-NaN fallback: skips empty parameters, tries next candidate dataset
- No compute tools — reports fetched data labels to orchestrator
- See `docs/planning-workflow.md` for detailed flow

### DataOpsAgent (agent/data_ops_agent.py)
- Sees tools: data_ops_compute (`custom_operation`, `describe_data`, `save_data`), function_docs (`search_function_docs`, `get_function_docs`), conversation + `list_fetched_data` extra
- **Two-phase compute**: Think phase (explore data + research function APIs) then Execute phase (write code with enriched context)
- Think phase uses ephemeral chat with function_docs + data inspection tools (same pattern as PlannerAgent discovery)
- Sandbox includes full `scipy` and `pywt` (PyWavelets) for signal processing, wavelets, filtering, interpolation, etc.
- Function documentation catalog auto-generated from scipy submodules and pywt docstrings
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
- Sees tools: `update_plot_spec` + `list_fetched_data` (2 tools total)
- `update_plot_spec`: Single tool for creating and modifying plots via a unified JSON spec
  - Layout changes (labels, panels, panel_types, etc.) trigger full re-render via `render_from_spec()`
  - Style-only changes (title, colors, font, etc.) are applied in-place via `style()`
  - Orchestrator injects current spec into viz agent context for diffing
- The viz agent owns all visualization: `update_plot_spec` + `manage_plot` + `list_fetched_data`
- `plot_data` and `style_plot` have been removed from LLM-facing tools (Python methods remain as internal implementation)

## Supported Spacecraft

### Primary Missions (52, all auto-generated from CDAWeb)

All 52 mission JSON files are auto-generated from CDAS REST API + Master CDF metadata via `scripts/generate_mission_data.py`. Key missions include PSP, Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, STEREO-A, Cluster, THEMIS, Van Allen Probes, GOES, Voyager 1/2, Ulysses, and more.

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
- Describes 2 visualization tools: `update_plot_spec`, `manage_plot`
- Each tool has: name, description, typed parameters (with enums for constrained values)
- `render_method_catalog()` renders the registry into markdown for the LLM prompt
- `get_method(name)` and `validate_args(name, args)` for dispatch and validation

### Data Pipeline (`data_ops/`)
- `DataEntry` wraps a `pd.DataFrame` (DatetimeIndex + float64 columns).
- `DataStore` is a singleton dict keyed by label. The LLM chains tools automatically: fetch -> custom_operation -> plot.
- `custom_ops.py`: AST-validated, sandboxed executor for LLM-generated pandas/numpy/scipy/pywt code. Replaces all hardcoded compute functions — the LLM writes the code directly. Sandbox includes `pd`, `np`, `xr`, `scipy` (full scipy), and `pywt` (PyWavelets).
- Data fetching uses the CDF backend exclusively — downloads CDF files from CDAWeb REST API, caches locally, reads with cdflib. Errors propagate directly for the agent to learn from.

### LLM Abstraction Layer (`agent/llm/`)
- **Phases 1-3 complete (February 2026)**: All LLM SDK calls go through `agent/llm/` adapter layer. Three adapters implemented.
- `agent/llm/base.py` — Abstract types: `ToolCall`, `UsageMetadata`, `LLMResponse`, `FunctionSchema`, `ChatSession` ABC, `LLMAdapter` ABC
- `agent/llm/gemini_adapter.py` — `GeminiAdapter` + `GeminiChatSession` wrapping `google-genai` SDK
- `agent/llm/openai_adapter.py` — `OpenAIAdapter` for OpenAI-compatible providers (OpenAI, DeepSeek, Qwen, Ollama, etc.)
- `agent/llm/anthropic_adapter.py` — `AnthropicAdapter` for Anthropic Claude models
- Escape hatches for provider-specific features via `LLMResponse.raw` field and adapter-specific methods
- Phase 4 pending: session persistence normalization, CLI `--provider` flag, Gradio UI selector

### Agent Loop (`agent/core.py`)
- LLM decides which tools to call via function calling (through `LLMAdapter` interface).
- Tool results are fed back via `adapter.make_tool_result_message()`.
- Orchestrator loop continues until LLM produces a text response (or 10 iterations), with consecutive delegation error tracking (breaks after 2 failures).
- Sub-agent loops limited to 5 iterations with duplicate call detection and consecutive error tracking.
- Token usage accumulated from `LLMResponse.usage` (input_tokens, output_tokens, thinking_tokens).

### LLM-Driven Routing (`agent/core.py`, `agent/mission_agent.py`, `agent/data_ops_agent.py`, `agent/visualization_agent.py`)
- **Routing**: The OrchestratorAgent (LLM) decides whether to handle a request directly or delegate via `delegate_to_mission` (fetching), `delegate_to_data_ops` (computation), `delegate_to_data_extraction` (text-to-DataFrame), or `delegate_to_visualization` (visualization) tools. No regex-based routing — the LLM uses conversation context and the routing table to decide.
- **Mission sub-agents**: Each spacecraft has a data fetching specialist with rich system prompt (recommended datasets, analysis patterns). Agents are cached per session. Sub-agents have **fetch-only tools** (discovery, data_ops_fetch, conversation) — no compute, plot, or routing tools.
- **DataOps sub-agent**: Data transformation specialist with `custom_operation`, `describe_data`, `save_data` + `list_fetched_data`. System prompt includes computation patterns and code guidelines. Singleton, cached per session.
- **DataExtraction sub-agent**: Text-to-DataFrame specialist with `store_dataframe`, `read_document`, `ask_clarification` + `list_fetched_data`. System prompt includes extraction patterns and DataFrame creation guidelines. Singleton, cached per session.
- **Visualization sub-agent**: Visualization specialist with `update_plot_spec` + `manage_plot` + `list_fetched_data` tools. Owns all visualization operations (plotting, styling, export, reset, zoom, traces). Uses spec-based workflow: the orchestrator injects the current plot spec into the context, and the viz agent emits a complete desired spec via `update_plot_spec`. The handler diffs layout vs style fields to decide re-render or restyle.
- **Tool separation**: Tools have a `category` field (`discovery`, `visualization`, `data_ops`, `data_ops_fetch`, `data_ops_compute`, `data_extraction`, `function_docs`, `conversation`, `routing`, `document`). `get_tool_schemas(categories=..., extra_names=...)` filters tools by category. Orchestrator sees `["discovery", "conversation", "routing", "document"]` + `list_fetched_data` extra. MissionAgent sees `["discovery", "data_ops_fetch", "conversation"]` + `list_fetched_data` extra. DataOpsAgent sees `["data_ops_compute", "conversation"]` + `list_fetched_data`, `search_function_docs`, `get_function_docs` extras. DataExtractionAgent sees `["data_extraction", "document", "conversation"]` + `list_fetched_data` extra. VisualizationAgent sees `["visualization"]` + `list_fetched_data`, `manage_plot` extras → `update_plot_spec` + `manage_plot` + `list_fetched_data`.
- **Post-delegation flow**: After `delegate_to_mission` returns data labels, the orchestrator uses `delegate_to_data_ops` for computation, `delegate_to_data_extraction` for text-to-DataFrame conversion, and then `delegate_to_visualization` to visualize results.
- **Slim orchestrator**: System prompt contains a routing table (mission names + capabilities) plus delegation instructions. No dataset IDs or analysis tips — those live in mission sub-agents.

### Multi-Step Requests (Hybrid Planning)
- Simple requests are handled by the orchestrator's conversation loop (up to 10 iterations, with consecutive delegation error guard)
- "Compare PSP and ACE" -> `delegate_to_mission("PSP", ...)` -> `delegate_to_mission("ACE", ...)` -> `delegate_to_visualization(plot both)` — all in one `process_message` call
- Complex requests use **hybrid routing** to the **PlannerAgent** for plan-execute-replan:
  1. **Regex pre-filter**: `is_complex_request()` regex heuristics catch obvious complex cases (free, no API cost) and route directly to planner
  2. **Orchestrator override**: The orchestrator (with HIGH thinking) can also call `request_planning` tool for complex cases the regex missed
  3. PlannerAgent runs **discovery phase** (tool-calling) then **planning phase** (JSON-schema-enforced)
  4. Fetch tasks use **physics-intent instructions** + `candidate_datasets` list — planner does NOT specify parameter names
  5. Mission agents inspect candidates, select best dataset/parameters, handle all-NaN fallback
  6. Results (with actual stored labels) are fed back to the PlannerAgent, which decides to continue or finish
  7. Maximum 5 rounds of replanning (configurable via `MAX_ROUNDS`)
  8. If the planner fails, falls back to direct orchestrator execution
  9. See `docs/planning-workflow.md` for the full detailed flow
- Tasks are tagged with `mission="__visualization__"` for visualization dispatch, `mission="__data_ops__"` for compute dispatch, `mission="__data_extraction__"` for text-to-DataFrame dispatch

### Thinking Levels
- Controlled via `create_chat(thinking="high"|"low"|"default")` in the adapter layer
- **HIGH**: Orchestrator (`agent/core.py`) and PlannerAgent (`agent/planner.py`) — deep reasoning for routing decisions and plan decomposition
- **LOW**: MissionAgent, VisualizationAgent, DataOpsAgent, DataExtractionAgent — fast execution with minimal thinking overhead
- Thinking tokens tracked separately in `get_token_usage()` across all agents
- Verbose mode logs full thoughts to terminal/file, plus 500-char tagged previews for Gradio via `agent/thinking.py` utilities
- Task plans persist to `~/.helio-agent/tasks/` with round tracking for multi-round plans

### Per-Mission JSON Knowledge (`knowledge/missions/*.json`)
- **52 mission JSON files**, all auto-generated from CDAS REST API + Master CDF metadata. Profiles include instrument groupings, dataset parameters, and time ranges populated by `scripts/generate_mission_data.py`.
- **Shared prefix map**: `knowledge/mission_prefixes.py` maps CDAWeb dataset ID prefixes to mission identifiers (40+ mission groups).
- **CDAWeb InstrumentType grouping**: `knowledge/cdaweb_metadata.py` fetches the CDAWeb REST API to get authoritative InstrumentType per dataset (18+ categories like "Magnetic Fields (space)", "Plasma and Solar Wind"). Bootstrap uses this to group datasets into meaningful instrument categories with keywords, instead of dumping everything into "General".
- **Full catalog search**: `knowledge/cdaweb_catalog.py` provides `search_full_catalog` tool — searches all 2000+ CDAWeb datasets by keyword (CDAS REST API), with 24-hour local cache.
- **Master CDF metadata**: `knowledge/master_cdf.py` downloads CDF skeleton files from CDAWeb and extracts parameter metadata (names, types, units, fill values, sizes). Cached to `~/.helio-agent/master_cdfs/`. Used as the network source for parameter metadata.
- **3-layer metadata resolution**: `knowledge/metadata_client.py` resolves dataset metadata through: in-memory cache → local file cache → Master CDF download. Master CDF results are persisted to the local file cache for subsequent use.
- **Recommended datasets**: All datasets in the instrument section are shown as recommended. Additional datasets are discoverable via `browse_datasets`.
- **Calibration exclusion lists**: Per-mission `_calibration_exclude.json` files filter out calibration, housekeeping, and ephemeris datasets from browse results. Uses glob patterns and exact IDs.
- **Auto-generation**: `scripts/generate_mission_data.py` queries CDAS REST API for catalog + Master CDF for parameters. Use `--create-new` to create skeleton JSON files for new missions.
- **Loader**: `knowledge/mission_loader.py` provides in-memory cache, routing table, and dataset access. Routing table derives capabilities from instrument keywords (magnetic field, plasma, energetic particles, electric field, radio/plasma waves, geomagnetic indices, ephemeris, composition, coronagraph, imaging).

### Long-term Memory (`agent/memory.py`)
- Cross-session memory that persists user preferences, session summaries, and operational pitfalls
- Storage: `~/.helio-agent/memory.json` — global, not per-session
- Three memory types: `"preference"` (plot styles, spacecraft of interest, workflow habits), `"summary"` (what was analyzed in each session), and `"pitfall"` (operational lessons learned)
- Pitfalls have a `scope` field: `"generic"` (default), `"mission:<ID>"` (e.g., `"mission:PSP"`), or `"visualization"`
  - Generic pitfalls injected into orchestrator prompt as "Operational Knowledge"
  - Mission-scoped pitfalls injected into the corresponding MissionAgent's task prompt
  - Visualization-scoped pitfalls injected into the VisualizationAgent's task prompt
  - MemoryAgent auto-detects scope during extraction; migration script available at `scripts/migrate_pitfall_scopes.py`
- Automatic extraction at session boundaries via lightweight Gemini Flash call (no tools, no thinking)
- Injection: prepends memory context to user messages in `process_message()` (avoids chat recreation)
- Deduplication: skips new memories that are substrings of existing ones
- Capped at 15 preferences + 15 summaries + 20 pitfalls per injection to keep prompt size reasonable
- Global enable/disable toggle + per-memory enable/disable
- Gradio UI: "Long-term Memory" accordion in right sidebar with toggle, delete, and clear controls
- CLI: memories extracted automatically on session exit (`main.py`)

### Passive MemoryAgent (`agent/memory_agent.py`)
- Monitors log growth, error count, and conversation turns — triggers analysis automatically
- Triggered at end of every `process_message()` via lightweight file-stat check (no LLM call until threshold met)
- Thresholds: +10KB log growth, 5+ ERROR/WARNING lines, or 10+ user turns since last analysis
- Single-shot Gemini Flash call analyzes conversation + log content to extract:
  - Preferences and session summaries (stored in MemoryStore)
  - Pitfalls — generalizable operational lessons (stored in MemoryStore, injected into future prompts)
  - Error patterns — structured bug reports (saved as markdown to `~/.helio-agent/reports/`)
- State tracked in `~/.helio-agent/memory_agent_state.json` (last log offset, turn count, timestamp)
- Handles log rotation (new day), caps log content at 50KB, reads with `errors='replace'`
- All exceptions caught — never breaks the main agent flow


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
- Fail-open: if metadata call fails (Master CDF), proceeds without validation

### Default Plot Styling
- `_DEFAULT_LAYOUT` in `rendering/plotly_renderer.py` sets explicit white backgrounds (`paper_bgcolor`, `plot_bgcolor`) and dark font color
- Prevents Gradio dark theme CSS from making plots appear black
- Applied in `_ensure_figure()` and `_grow_panels()`

### Figure Sizing
- Renderer sets explicit defaults: `autosize=False`, 300px per panel height, 1100px width
- Prevents Plotly.js from recalculating dimensions on toolbar interactions (zoom, pan, reset)
- `update_plot_spec` re-render review metadata includes `figure_size` (current) and `sizing_recommendation` (suggested)
- Viz agent includes `canvas_size` in the spec when sizing should differ from defaults (4+ panels use compact 250px/panel, spectrograms use ≥400px height and 1200px width)

### Gradio Streaming
- `gradio_app.py` streams live progress by default; use `--quiet` to disable
- Agent output unified through Python logging
- `_ListHandler` in `gradio_app.py` uses tag-based filtering via `GRADIO_VISIBLE_TAGS` — only agent lifecycle, plan events, data fetched, thinking previews, and errors are shown. See the Logging section for the full tag table and how to add new categories.

### Mission Data Startup (`knowledge/startup.py`)
- Shared startup logic used by both `main.py` and `gradio_app.py`
- `get_mission_status()` scans mission JSONs and reports count, datasets, last refresh date
- `show_mission_menu()` presents interactive refresh options on startup
- `resolve_refresh_flags()` maps CLI flags (`--refresh`, `--refresh-full`, `--refresh-all`) to actions
- `run_mission_refresh()` invokes bootstrap to refresh time ranges or rebuild all missions
- After refresh, clears mission_loader and metadata_client caches

### Automatic Model Fallback (`agent/model_fallback.py`)
- When any Gemini API call hits a 429 RESOURCE_EXHAUSTED (quota/rate limit), all agents automatically switch to `GEMINI_FALLBACK_MODEL` for the remainder of the session
- Session-level global flag — once activated, every subsequent `client.chats.create()` and `models.generate_content()` call uses the fallback model
- The OrchestratorAgent's persistent chat is recreated with the fallback model on first 429 error
- Sub-agents (BaseSubAgent, PlannerAgent, MemoryAgent) use `get_active_model()` at chat/call creation time, so they pick up the fallback automatically
- Configurable via `fallback_model` in `~/.helio-agent/config.json` (default: `gemini-2.5-flash`)
- If the fallback model also fails, the error propagates normally (no retry chain)

### Empty Session Auto-Cleanup
- On startup, `SessionManager` auto-removes sessions with no chat history and no stored data
- Prevents clutter from abandoned or crashed sessions
- Session save is skipped when there's nothing to persist

### First-Run Full Download
- On first run (no mission JSONs exist), `ensure_missions_populated()` calls `populate_missions()` which downloads the full CDAWeb catalog + Master CDF parameter metadata (~5-10 minutes, one-time)
- Subsequent startups are instant (JSON files already exist)
- Two refresh paths: `--refresh` (lightweight time-range update) and `--refresh-full` (destructive rebuild)
- Shows progress via tqdm in terminal, logger-based progress in Gradio live log

### Web Search (`google_search` tool)
- **Gemini provider**: Uses built-in Google Search grounding API via an isolated Gemini API call (Gemini API does not support google_search + function_declarations in the same call)
- **Non-Gemini providers** (OpenAI, Anthropic, OpenRouter, etc.): Falls back to Tavily web search (`TAVILY_API_KEY` env var required; `tavily-python` package)
- **No search backend available**: Warning logged, error returned to LLM — agent continues without search
- Returns grounded text with source URLs
- Search results can be turned into plottable datasets via the `store_dataframe` tool (google_search → delegate_to_data_extraction → store_dataframe → plot)

## Configuration

**`.env`** at project root (secret only):
```
GOOGLE_API_KEY=<gemini-api-key>
TAVILY_API_KEY=<tavily-api-key>  # Optional — enables web search for non-Gemini providers
```

**`~/.helio-agent/config.json`** (user-editable, all optional — defaults shown):
```json
{
  "model": "gemini-3-pro-preview",
  "sub_agent_model": "gemini-3-flash-preview",
  "planner_model": null,
  "fallback_model": "gemini-2.5-flash",
  "llm_provider": "gemini",
  "llm_api_key": null,
  "llm_base_url": null,
  "data_backend": "cdf",
  "catalog_search_method": "semantic",
  "max_preferences": 15,
  "max_summaries": 10,
  "max_pitfalls": 20,
  "memory_poll_interval_seconds": 30,
  "memory_log_growth_threshold_kb": 10,
  "memory_error_count_threshold": 5,
  "memory_max_log_bytes_kb": 50
}
```

See `config.template.json` for a copyable template. If the file doesn't exist, built-in defaults are used.

## Running

```bash
python main.py               # Normal mode (auto-saves session)
python main.py --verbose     # Show tool calls, timing, errors
python main.py --continue    # Resume most recent session
python main.py --session ID  # Resume specific session by ID
python main.py -m MODEL      # Specify Gemini model (overrides .env)
python main.py "request"     # Single-command mode (non-interactive, exits after response)
python main.py --refresh     # Refresh dataset time ranges (fast — start/stop dates only)
python main.py --refresh-full  # Full rebuild of all mission data
python main.py --refresh-all   # Full rebuild of all missions (same as --refresh-full)
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
- Browse & Fetch sidebar: mission → dataset → parameter cascade dropdowns with direct CDF fetch
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
- Log files stored in `~/.helio-agent/logs/agent_YYYYMMDD_HHMMSS.log` (one per session)
- Detailed error logging with stack traces
- `log_error()`: Captures context and full stack traces for debugging
- `log_tool_call()` / `log_tool_result()`: Tracks all tool invocations
- `log_plan_event()`: Records plan lifecycle events (tagged for Gradio)
- `print_recent_errors()`: CLI command to review recent errors

#### Tag-Based Log Filtering (Gradio vs Terminal)
All log calls can be tagged with `extra=tagged("category")`. The Gradio live log handler only shows records whose `log_tag` is in `GRADIO_VISIBLE_TAGS` (plus all WARNING/ERROR). Terminal and file handlers show everything.

**`GRADIO_VISIBLE_TAGS`** (defined in `agent/logging.py`):
| Tag | Example message | Source |
|-----|-----------------|--------|
| `"delegation"` | `[Router] Delegating to PSP specialist` | `agent/core.py` |
| `"delegation_done"` | `[Router] PSP specialist finished` | `agent/core.py` |
| `"plan_event"` | `Plan created: a1b2c3d4...` | `agent/logging.py:log_plan_event()` |
| `"plan_task"` | `[Plan] [PSP]: Fetch magnetic field data` | `agent/core.py` |
| `"data_fetched"` | `[DataOps] Stored 'AC_H2_MFI.BGSEc' (10080 points)` | `agent/core.py` |
| `"thinking"` | `[Thinking] The user wants...` (first 500 chars) | `core.py`, `base_agent.py`, `planner.py` |
| `"error"` | Real errors with context/stack traces | `agent/logging.py:log_error()` |

**What is NOT shown in Gradio** (terminal/file only): `[CDF]`, `[Gemini]`, `[Tool:]` calls, full thinking text, internal tool-result warnings/errors, DataOps plumbing. Only `log_error()` errors appear in Gradio (tagged `"error"`); per-tool `logger.warning("Tool error: ...")` lines are untagged and filtered out.

**To add a new category to Gradio:**
1. Tag the log call: `logger.debug("...", extra=tagged("my_tag"))`
2. Add `"my_tag"` to `GRADIO_VISIBLE_TAGS` in `agent/logging.py`

No filter logic changes needed. The `tagged()` helper returns `{"log_tag": tag}` for use as the `extra` kwarg.

**Thinking log records**: Each thought emits two records — the full untagged text (goes to terminal/file only) and a 500-char truncated preview tagged `"thinking"` (shown in Gradio). This happens in `_track_usage()` in `core.py`, `base_agent.py`, and `planner.py`.

## Tests

```bash
python -m pytest tests/test_store.py tests/test_custom_ops.py   # Data ops tests
python -m pytest tests/test_session.py                           # Session persistence tests
python -m pytest tests/test_memory.py tests/test_memory_agent.py # Memory + MemoryAgent tests
python -m pytest tests/                                          # All tests
```

## Dependencies

```
google-genai>=1.60.0    # Gemini API (via agent/llm/gemini_adapter.py)
python-dotenv>=1.0.0    # .env loading
requests>=2.28.0        # HTTP calls (CDAS REST, Master CDF, CDF downloads)
cdflib>=1.3.0           # CDF file reading (Master CDF metadata, data files)
numpy>=1.24.0           # Array operations
scipy>=1.10.0           # Signal processing, FFT, interpolation, statistics
PyWavelets>=1.8.0       # Wavelet transforms (CWT, DWT, packets)
pandas>=2.0.0           # DataFrame-based data pipeline
plotly>=5.18.0          # Interactive scientific data visualization
kaleido>=0.2.1          # Static image export for Plotly (PNG, PDF)
gradio>=4.44.0          # Browser-based chat UI
matplotlib>=3.7.0       # Legacy plotting (unused in main pipeline)
tqdm>=4.60.0            # Progress bars for bootstrap/data downloads
pytest>=7.0.0           # Test framework
tavily-python>=0.5.0    # Tavily web search (fallback for non-Gemini providers)
```
