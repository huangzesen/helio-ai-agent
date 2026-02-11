# Architecture Flowchart

## High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACES                       │
│                                                         │
│   main.py (CLI)              gradio_app.py (Web UI)     │
│   ├─ readline REPL           ├─ Chatbot + Plot viewer   │
│   ├─ --verbose/--continue    ├─ Browse & Fetch sidebar  │
│   └─ session resume          └─ Data table + preview    │
└────────────────────┬────────────────────────────────────┘
                     │ process_message(user_input)
                     ▼
┌─────────────────────────────────────────────────────────┐
│            ORCHESTRATOR AGENT  (Gemini 3 Pro)           │
│                  agent/core.py                          │
│                                                         │
│  • Routes requests via LLM function calling             │
│  • Injects long-term memory into system prompt          │
│  • Aggregates token usage across all sub-agents         │
│  • Auto-saves session after every turn                  │
│  • Model fallback on 429 quota errors                   │
│                                                         │
│  Tools: discovery, routing, conversation,               │
│         document, web_search, memory                    │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
```

---

## Agent Routing

```
              ┌──────────────────────────┐
              │  is_complex_request()?   │
              └─────┬──────────┬─────────┘
                YES │          │ NO
                    ▼          ▼
         ┌──────────────┐  LLM picks delegation tool:
         │PlannerAgent  │    delegate_to_mission
         │(Gemini 3 Pro)│    delegate_to_data_ops
         │              │    delegate_to_data_extraction
         │ 1. Discovery │    delegate_to_visualization
         │    phase     │
         │ 2. Planning  │
         │    phase     │
         │ 3. Execute   │
         │    tasks     │
         │ 4. Replan    │
         │    (max 5x)  │
         └──────────────┘
```

---

## Sub-Agent Hierarchy

```
OrchestratorAgent (HIGH thinking, Gemini 3 Pro)
│
├── MissionAgent (LOW thinking, Gemini 3 Flash)
│   ├─ One instance per spacecraft (cached)
│   ├─ Tools: search_datasets, list_parameters,
│   │         fetch_data, browse_datasets,
│   │         get_data_availability, get_dataset_docs,
│   │         search_full_catalog
│   └─ Rich per-mission prompt with recommended datasets
│
├── DataOpsAgent (LOW thinking, Gemini 3 Flash)
│   ├─ Singleton (cached)
│   ├─ Tools: custom_operation, describe_data,
│   │         preview_data, save_data, compute_spectrogram
│   └─ AST-validated sandbox for LLM-generated code
│
├── DataExtractionAgent (LOW thinking, Gemini 3 Flash)
│   ├─ Singleton (cached)
│   ├─ Tools: store_dataframe, read_document,
│   │         ask_clarification
│   └─ Converts unstructured text → DataFrames
│
├── VisualizationAgent (LOW thinking, Gemini 3 Flash)
│   ├─ Singleton (cached)
│   ├─ Tools: plot_data, style_plot, manage_plot
│   └─ Declarative key-value params (no code generation)
│
├── PlannerAgent (HIGH thinking, Gemini 3 Pro)
│   ├─ Two-phase: discovery (tools) → planning (JSON schema)
│   ├─ Emits task batches, observes results, replans
│   └─ Max 5 rounds of replanning
│
└── MemoryAgent (background daemon thread, Gemini Flash)
    ├─ Runs every 30s, monitors log growth
    ├─ Extracts: preferences, summaries, pitfalls
    └─ Auto-consolidates when memories > 30
```

---

## Data Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    FETCH     │     │    STORE     │     │   COMPUTE    │     │     PLOT     │
│              │     │              │     │              │     │              │
│ HAPI /data   │────▶│  DataStore   │────▶│  AST sandbox │────▶│ PlotlyRender │
│ endpoint     │     │  (singleton) │     │              │     │              │
│              │     │              │     │  Allowed:    │     │  plot_data() │
│ fetch.py     │     │  store.py    │     │  pd, np,     │     │  style()     │
│              │     │              │     │  signal      │     │  manage()    │
│ CSV → pandas │     │  DataEntry:  │     │              │     │              │
│ DataFrame    │     │  ├─ label    │     │  custom_ops  │     │  Downsample  │
│ fill → NaN   │     │  ├─ data(DF) │     │  .py         │     │  >5k pts     │
│              │     │  ├─ units    │     │              │     │              │
│              │     │  ├─ source   │     │  Blocked:    │     │  WebGL       │
│              │     │  └─ metadata │     │  imports,    │     │  >100k pts   │
│              │     │              │     │  exec/eval,  │     │              │
│              │     │  put() / get │     │  dunder,     │     │  Export:     │
│              │     │  list/remove │     │  global      │     │  PNG/PDF via │
│              │     │              │     │              │     │  kaleido     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            ▲                     │
                            │    result stored    │
                            └─────────────────────┘
```

---

## Tool Execution Flow

```
User message
    │
    ▼
OrchestratorAgent
    │ LLM function call
    ▼
┌─────────────────────────────────────────────────────────┐
│          _execute_tool(tool_name, args)                  │
│                agent/core.py                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  DISCOVERY (6 tools)         DATA OPS (8 tools)         │
│  ├─ search_datasets          ├─ fetch_data              │
│  │   → catalog.search()      │   → fetch.fetch_hapi()   │
│  ├─ browse_datasets          │   → DataStore.put()       │
│  │   → hapi_client           ├─ custom_operation         │
│  ├─ list_parameters          │   → AST validate          │
│  │   → hapi_client           │   → sandbox execute       │
│  ├─ get_data_availability    │   → DataStore.put()       │
│  │   → hapi_client           ├─ compute_spectrogram      │
│  ├─ get_dataset_docs         ├─ describe_data            │
│  │   → hapi_client           ├─ preview_data             │
│  └─ search_full_catalog      ├─ save_data                │
│      → cdaweb_catalog        ├─ list_fetched_data        │
│                              └─ store_dataframe           │
│                                                          │
│  VISUALIZATION (3 tools)     ROUTING (5 tools)           │
│  ├─ plot_data                ├─ delegate_to_mission      │
│  │   → PlotlyRenderer       │   → MissionAgent           │
│  ├─ style_plot               ├─ delegate_to_data_ops     │
│  │   → PlotlyRenderer       │   → DataOpsAgent           │
│  └─ manage_plot              ├─ delegate_to_data_extract  │
│      → PlotlyRenderer       │   → DataExtractionAgent    │
│                              ├─ delegate_to_visualization │
│  OTHER (4 tools)             │   → VisualizationAgent    │
│  ├─ ask_clarification        └─ request_planning          │
│  ├─ google_search                → PlannerAgent           │
│  ├─ read_document                                        │
│  └─ recall_memories                                      │
│                                                          │
│  All tools return: {status: "success"/"error", ...}     │
└─────────────────────────────────────────────────────────┘
    │ function response
    ▼
LLM produces text reply → User
```

---

## Knowledge Base

```
┌─────────────────────────────────────────────────────────┐
│               KNOWLEDGE BASE  (knowledge/)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  missions/*.json (52 files)                             │
│  ├─ PSP, ACE, OMNI, Wind, DSCOVR, MMS, STEREO-A, ...  │
│  ├─ id, name, keywords, profile                        │
│  └─ instruments → datasets → parameters (HAPI metadata)│
│           │                                             │
│           ▼                                             │
│  mission_loader.py ──────────────▶ prompt_builder.py    │
│  ├─ load_mission(id)             ├─ build_mission_prompt│
│  ├─ load_all_missions()          ├─ build_viz_prompt    │
│  └─ get_routing_table()          ├─ build_data_ops_prmt │
│           │                      └─ build_planner_prmt  │
│           ▼                              │              │
│  catalog.py                              │              │
│  └─ search_by_keywords()                 ▼              │
│                                  System prompts for     │
│  hapi_client.py                  all agents (dynamic,   │
│  ├─ list_parameters()            generated from JSON)   │
│  ├─ get_dataset_time_range()                            │
│  ├─ get_dataset_docs()                                  │
│  └─ browse_datasets()                                   │
│           │                                             │
│  cdaweb_catalog.py                                      │
│  └─ search_catalog() (2000+ datasets, 24h cache)       │
│                                                         │
│  Cache: ~/.helio-agent/hapi_cache/{mission}/            │
└─────────────────────────────────────────────────────────┘
```

---

## PlannerAgent Flow

```
User: "Compare PSP and ACE magnetic field data for January 2024"
    │
    ▼
is_complex_request() → YES (multi-spacecraft)
    │
    ▼
┌─────────────── PHASE 1: DISCOVERY ───────────────┐
│  PlannerAgent gets discovery tools                │
│  ├─ search_datasets("PSP magnetic field")         │
│  ├─ search_datasets("ACE magnetic field")         │
│  ├─ list_parameters(PSP_FLD_L2_MAG_RTN_1MIN)     │
│  └─ list_parameters(AC_H2_MFI)                    │
│  Verifies datasets and parameters exist           │
└──────────────────────┬────────────────────────────┘
                       ▼
┌─────────────── PHASE 2: PLANNING ────────────────┐
│  JSON-schema-enforced chat (no tools)             │
│  Outputs structured TaskPlan:                     │
│  ├─ Task 1: fetch PSP mag data (mission: PSP)     │
│  ├─ Task 2: fetch ACE mag data (mission: ACE)     │
│  ├─ Task 3: compute magnitudes (type: data_ops)   │
│  └─ Task 4: plot comparison (type: visualization)  │
└──────────────────────┬────────────────────────────┘
                       ▼
┌─────────────── PHASE 3: EXECUTION ───────────────┐
│  Orchestrator executes tasks in batches:          │
│  ├─ Batch 1: Task 1 + Task 2 (parallel fetch)    │
│  ├─ Batch 2: Task 3 (depends on 1+2)             │
│  └─ Batch 3: Task 4 (depends on 3)               │
│                                                   │
│  Each task → appropriate sub-agent:               │
│  ├─ fetch tasks → MissionAgent.execute_task()     │
│  ├─ compute   → DataOpsAgent.execute_task()       │
│  └─ plot      → VisualizationAgent.execute_task() │
└──────────────────────┬────────────────────────────┘
                       ▼
┌─────────────── PHASE 4: REPLAN (if needed) ──────┐
│  Feed task results back to PlannerAgent            │
│  ├─ If all succeeded → done                        │
│  ├─ If failures → replan (up to 5 rounds)          │
│  └─ If planner fails → fallback to orchestrator    │
└───────────────────────────────────────────────────┘
```

---

## Memory System

```
┌────────────────────── RUNTIME ──────────────────────┐
│                                                      │
│  OrchestratorAgent                                   │
│  └─ Injects MemoryStore.build_prompt_section()       │
│     into system prompt at session start              │
│                                                      │
│  recall_memories tool                                │
│  └─ Searches MemoryStore + cold storage              │
│                                                      │
└──────────────────────────────────────────────────────┘

┌────────────────── BACKGROUND DAEMON ────────────────┐
│                                                      │
│  MemoryAgent (daemon thread, every 30s)              │
│  │                                                   │
│  ├─ 1. Check thresholds (no LLM):                    │
│  │      log growth > 10KB? errors > 5?               │
│  │                                                   │
│  ├─ 2. If threshold met → single-shot Gemini Flash:  │
│  │      Extract from logs:                           │
│  │      ├─ preferences (plot styles, workflows)      │
│  │      ├─ summaries (session analysis)              │
│  │      ├─ pitfalls (operational lessons)            │
│  │      └─ error_patterns → reports/                 │
│  │                                                   │
│  ├─ 3. Save to MemoryStore (memory.json)             │
│  │                                                   │
│  └─ 4. Consolidate if count > 30:                    │
│         ├─ Merge/prune via LLM                       │
│         └─ Archive evicted → memory_cold.json        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Session Persistence

```
~/.helio-agent/
├── sessions/{session_id}/
│   ├── metadata.json          ← model, turn_count, timestamps, token_usage
│   ├── history.json           ← Gemini Content dicts (base64-encoded bytes)
│   ├── data/
│   │   ├── {label}.pkl        ← pickled DataFrames
│   │   └── _index.json        ← label → {filename, units, description, source}
│   └── figure.json            ← PlotlyRenderer state (traces, layout)
│
├── memory.json                ← long-term memories (preferences, summaries, pitfalls)
├── memory_cold.json           ← archived/evicted memories
├── memory_agent_state.json    ← log offset, last analysis timestamp
│
├── tasks/{plan_id}.json       ← PlannerAgent task plans
├── reports/report_*.md        ← MemoryAgent error reports
│
├── logs/
│   └── agent_YYYYMMDD.log    ← daily rotation, DEBUG level always captured
│
├── hapi_cache/{mission}/
│   ├── {dataset_id}.json      ← HAPI /info metadata
│   └── _index.json            ← dataset list
│
└── documents/                 ← saved PDF/image text extractions
```

---

## Rendering Pipeline

```
VisualizationAgent
    │ plot_data / style_plot / manage_plot
    ▼
┌──────────────────────────────────────────────┐
│         PlotlyRenderer  (stateful)           │
│         rendering/plotly_renderer.py          │
├──────────────────────────────────────────────┤
│                                              │
│  State:                                      │
│  ├─ _figure (go.Figure)                      │
│  ├─ _panel_count (subplot rows)              │
│  ├─ _current_time_range                      │
│  ├─ _label_colors (stable color assignment)  │
│  ├─ _trace_labels / _trace_panels            │
│  └─ _spectrogram_traces                      │
│                                              │
│  plot_data():                                │
│  ├─ Reads DataEntry from DataStore           │
│  ├─ Single-panel overlay or multi-panel      │
│  ├─ Line plots or spectrograms (heatmap)     │
│  ├─ Downsample: >5k pts → min-max decimation │
│  ├─ WebGL: >100k pts → Scattergl            │
│  └─ Returns plot review (trace summary,      │
│     warnings, hints)                         │
│                                              │
│  style():                                    │
│  ├─ Titles, axis labels, log scale           │
│  ├─ Trace colors, line styles, legend        │
│  ├─ Font size, canvas size, annotations      │
│  └─ Colorscale, theme                        │
│                                              │
│  manage():                                   │
│  ├─ reset, get_state                         │
│  ├─ set_time_range (zoom)                    │
│  ├─ export (PNG/PDF via kaleido)             │
│  ├─ remove_trace, add_trace                  │
│  └─ auto-opens unless web_mode=True          │
│                                              │
│  Validation via registry.py:                 │
│  └─ 3 tool schemas with typed parameters     │
│                                              │
└──────────────────────────────────────────────┘
    │
    ▼
  Gradio: gr.Plot (interactive)
  CLI: export → PNG/PDF → OS viewer
```

---

## Logging & Observability

```
                    agent/logging.py
                    ├─ GRADIO_VISIBLE_TAGS (frozenset)
                    ├─ tagged(tag) → {"log_tag": tag}
                    └─ _SessionFilter: injects session_id + log_tag default

  Log call with tag                    Log call without tag
  logger.debug("...",                  logger.debug("...")
    extra=tagged("delegation"))
         │                                    │
         ▼                                    ▼
  record.log_tag = "delegation"        record.log_tag = ""
         │                                    │
         ├──────────────┬─────────────────────┤
         ▼              ▼                     ▼
   ┌──────────┐  ┌────────────┐        ┌──────────┐
   │  Gradio  │  │  Terminal   │        │   File   │
   │_ListHdlr │  │  console    │        │ handler  │
   │          │  │  handler    │        │          │
   │ tag in   │  │ DEBUG if    │        │ always   │
   │ VISIBLE? │  │ --verbose   │        │ DEBUG    │
   │ YES→show │  │ else WARN+  │        │          │
   │ NO →skip │  │             │        │          │
   │          │  │ shows ALL   │        │ shows    │
   │ (all lvl │  │ log records │        │ ALL log  │
   │  by tag) │  │             │        │ records  │
   └──────────┘  └────────────┘        └──────────┘

  Visible tags:
    delegation       → [Router] Delegating to X specialist
    delegation_done  → [Router] X specialist finished
    plan_event       → Plan created / completed / failed
    plan_task        → [Plan] task description + plan summary
    data_fetched     → [DataOps] Stored 'label' (N points)
    thinking         → [Thinking] ... (first 500 chars)
    error            → log_error() real errors with stack traces
```

---

## Gradio Web UI Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     gradio_app.py                             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  _agent = create_agent(verbose, gui_mode, web_mode, model)   │
│  Same OrchestratorAgent as main.py                            │
│                                                               │
│  ┌─────────────────────┐  ┌──────────────────────────────┐   │
│  │   Main Panel        │  │   Sidebar                    │   │
│  │                     │  │                              │   │
│  │  gr.Chatbot         │  │  Plot Viewer (gr.Plot)      │   │
│  │  ├─ Text messages   │  │  ├─ Interactive Plotly fig   │   │
│  │  ├─ Inline plots    │  │  └─ Updates after each turn  │   │
│  │  └─ File uploads    │  │                              │   │
│  │                     │  │  Data Table (gr.Dataframe)   │   │
│  │  Text Input         │  │  ├─ Label, Points, Units     │   │
│  │  └─ Submit button   │  │  ├─ Time Range, Source       │   │
│  │                     │  │  └─ Preview (head/tail)      │   │
│  │  Example Prompts    │  │                              │   │
│  │                     │  │  Browse & Fetch              │   │
│  └─────────────────────┘  │  ├─ Mission dropdown         │   │
│                           │  ├─ Dataset dropdown          │   │
│                           │  ├─ Parameter dropdown        │   │
│                           │  └─ Time pickers + Fetch btn  │   │
│                           │                              │   │
│                           │  Sessions Accordion          │   │
│                           │  ├─ Load / New / Delete       │   │
│                           │  └─ Session list              │   │
│                           │                              │   │
│                           │  Memory Accordion            │   │
│                           │  ├─ Toggle / Delete / Clear   │   │
│                           │  └─ Memory list               │   │
│                           │                              │   │
│                           │  Token Usage                 │   │
│                           │  └─ Input/Output/Thinking     │   │
│                           └──────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## End-to-End Example Flow

```
User: "Show me ACE magnetic field data for last week"
 │
 ▼
main.py → OrchestratorAgent.process_message()
 │
 ├─ Memory injection: MemoryStore.build_prompt_section()
 │
 ▼
Gemini 3 Pro (HIGH thinking):
 "This is a single-spacecraft fetch+plot request"
 → calls delegate_to_mission(mission="ACE", request="...")
 │
 ▼
MissionAgent("ACE") created/cached
 ├─ System prompt from prompt_builder.build_mission_prompt("ACE")
 │   includes recommended datasets, analysis patterns
 │
 ├─ Gemini Flash (LOW thinking):
 │   → calls search_datasets(keywords="ACE magnetic field")
 │   → result: AC_H2_MFI recommended
 │
 ├─ → calls list_parameters(dataset_id="AC_H2_MFI")
 │   → result: BGSEc (3-component GSE mag field, nT)
 │
 ├─ → calls fetch_data(dataset="AC_H2_MFI", parameter="BGSEc",
 │                      time_min="2026-02-02", time_max="2026-02-09")
 │   → data_ops/fetch.py fetches from HAPI
 │   → DataStore.put("AC_H2_MFI.BGSEc", DataEntry(...))
 │
 └─ Returns: "Fetched AC_H2_MFI.BGSEc (10080 points, nT)"
 │
 ▼
Back in OrchestratorAgent:
 Gemini 3 Pro sees fetch result
 → calls delegate_to_visualization(request="plot AC_H2_MFI.BGSEc")
 │
 ▼
VisualizationAgent created/cached
 ├─ → calls plot_data(labels=["AC_H2_MFI.BGSEc"],
 │                     title="ACE Magnetic Field (GSE)")
 │   → PlotlyRenderer.plot_data()
 │   → Reads DataEntry from DataStore
 │   → Creates go.Figure with 3 traces (Bx, By, Bz)
 │   → Downsamples if > 5000 points
 │   → Returns plot review with trace summary
 │
 └─ Returns: "Plot created with 3 components"
 │
 ▼
OrchestratorAgent produces final text:
 "Here's the ACE magnetic field data (BGSEc) for the past week..."
 │
 ├─ SessionManager.save_session() (auto-save)
 ├─ MemoryAgent monitors log (background)
 │
 ▼
User sees: text response + interactive Plotly plot
```
