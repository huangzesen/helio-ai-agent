# Planning & Data Fetch Workflow

How requests flow through the agent system, from user input to plotted data.

**Last updated**: 2026-02-10

---

## Overview

The system has two main processing paths:

1. **Simple requests** — handled directly by the orchestrator's LLM conversation loop with delegation tools
2. **Complex requests** — routed to a plan-execute-replan loop with specialized sub-agents

The key design principle is **separation of concerns**:

- **OrchestratorAgent** routes requests and executes tool calls
- **PlannerAgent** decides *what* to do (physics intent + candidate datasets)
- **MissionAgent** decides *how* to fetch (inspects datasets, selects parameters)
- **DataOpsAgent** handles transformations (magnitude, smoothing, etc.)
- **DataExtractionAgent** converts unstructured text to DataFrames
- **VisualizationAgent** handles plotting (Plotly renderer)
- **MemoryAgent** runs in the background extracting operational knowledge

---

## Part 1: Message Processing (`process_message`)

Every user message enters through `OrchestratorAgent.process_message()`. The flow is:

```
User message
  │
  ├─ 1. Clear cancellation event
  ├─ 2. Inject long-term memory context (if enabled)
  ├─ 3. Complexity routing:
  │     ├─ is_complex_request() == True  → _handle_planning_request()
  │     └─ is_complex_request() == False → _process_single_message()
  ├─ 4. Auto-save session (if enabled)
  └─ 5. Ensure MemoryAgent daemon thread is running
```

### Memory Injection

Before routing, the orchestrator calls `self._memory_store.build_prompt_section()`. If non-empty, it prepends a memory context block to the user message:

```
[CONTEXT FROM LONG-TERM MEMORY]
## Preferences
- User prefers log scale for particle data
## Operational Knowledge
- PSP merged datasets often have NaN-only parameters
[END MEMORY CONTEXT]

<original user message>
```

This is injected into the user message text (not the system prompt) to avoid recreating the chat session.

---

## Part 2: Simple Request Path (`_process_single_message`)

For non-complex requests, the orchestrator handles everything in a single LLM conversation loop.

### Orchestrator Tools

The orchestrator sees these tool categories:
- **Discovery**: `search_datasets`, `list_parameters`, `get_data_availability`, `browse_datasets`, `list_missions`, `get_dataset_docs`, `search_full_catalog`
- **Web search**: `google_search`
- **Conversation**: `ask_clarification`
- **Document**: `read_document`
- **Memory**: `recall_memories`
- **Routing**: `delegate_to_mission`, `delegate_to_visualization`, `delegate_to_data_ops`, `delegate_to_data_extraction`, `request_planning`
- **Extras**: `list_fetched_data`, `preview_data`

The orchestrator does **not** directly see `fetch_data`, `custom_operation`, `plot_data`, `style_plot`, or `manage_plot` — those are handled by sub-agents via delegation.

### Conversation Loop

```
Send user message to orchestrator chat
  │
  Loop (max 10 iterations):
  │  ├─ Check cancellation
  │  ├─ Extract function_calls from LLM response
  │  ├─ If no function calls → break (LLM produced a text response)
  │  ├─ Execute each tool call via _execute_tool_safe()
  │  ├─ If ask_clarification was called → return question to user immediately
  │  ├─ Track consecutive delegation errors (sub-agent failures)
  │  │     └─ If >= 2 consecutive failures → send results once more, then break
  │  └─ Send function_responses back to chat
  │
  Return text from final LLM response
```

### Delegation Tools

When the LLM calls a delegation tool, the orchestrator creates a sub-agent and passes the request:

| Delegation Tool | Sub-Agent | What Happens |
|---|---|---|
| `delegate_to_mission(mission_id, request)` | `MissionAgent(mission_id)` | Fetches data for a spacecraft. Current store contents injected into request. |
| `delegate_to_data_ops(request)` | `DataOpsAgent` | Runs transformations on in-memory data. |
| `delegate_to_data_extraction(request)` | `DataExtractionAgent` | Converts text/documents to DataFrames. |
| `delegate_to_visualization(request)` | `VisualizationAgent` | Creates/styles/exports plots. Export requests intercepted before LLM (regex for "export", ".png", ".pdf"). |
| `request_planning(request)` | `PlannerAgent` | Escalates to the planning path (catches complex cases the regex missed). |

Each sub-agent call returns a dict `{text, failed, errors}`. The orchestrator wraps this via `_wrap_delegation_result()` — if the sub-agent failed with errors, the last 3 error messages are summarized and returned as `status: "error"`.

### Typical Simple Flow

A request like "Show me ACE magnetic field data for last week" follows this path:

```
Orchestrator LLM → delegate_to_mission("ACE", "Fetch mag field for last week")
  → MissionAgent("ACE").process_request(...)
    → list_parameters("AC_H2_MFI")
    → fetch_data("AC_H2_MFI", "BGSEc", "2026-02-03 to 2026-02-10")
    → returns {text: "Fetched AC_H2_MFI.BGSEc, 10080 points"}
  ← result fed back to orchestrator LLM
Orchestrator LLM → delegate_to_visualization("Plot AC_H2_MFI.BGSEc")
  → VisualizationAgent.process_request(...)
    → plot_data(labels="AC_H2_MFI.BGSEc")
    → returns {text: "Created plot with 1 panel"}
  ← result fed back to orchestrator LLM
Orchestrator LLM → text response to user
```

---

## Part 3: Complex Request Path — Planning

### Complexity Detection

`is_complex_request()` in `planner.py` uses 18+ regex patterns (`COMPLEXITY_INDICATORS`) to detect:

- **Multiple conjunctions**: `\band\b.*\band\b`, `\bthen\b`, `\bafter\b`, `\bfirst\b.*\bthen\b`, `\bfinally\b`
- **Comparisons**: `\bcompare\b`, `\bdifference\s+between\b`, `\bvs\.?\b`
- **Multi-step operations**: plot+and+plot, fetch+and+compute, compute+and+plot
- **Multiple spacecraft**: cross-product of PSP, ACE, OMNI, Wind, DSCOVR, MMS, STEREO, SolO (with special handling for "wind" vs "solar wind")
- **Pipeline phrases**: smooth+and+plot, average+and+compare, magnitude+and

Returns True if **any** pattern matches (case-insensitive). The orchestrator (with HIGH thinking) can also call `request_planning` tool during the simple path to catch complex cases the regex misses.

### Time Range Resolution

Before planning begins, the orchestrator calls `_extract_time_range()`:

1. Tries `parse_time_range()` on the full user message text
2. Falls back to regex extraction of time clauses:
   - `for <clause>`, `from <clause>`, `during <clause>`
   - Explicit date ranges, relative phrases (`last week`, `last 3 days`)
   - Month names

If resolved, appends to the planning message:
```
Resolved time range: 2024-01-10 to 2024-01-17. Use this exact range for ALL fetch tasks.
```

This prevents each sub-agent from re-interpreting relative expressions like "last week" at different wall-clock times.

---

## Part 4: The Plan-Execute-Replan Loop (`_handle_planning_request`)

### Step 1: Start Planning

Calls `PlannerAgent.start_planning(planning_msg)`, which runs two phases:

**Phase 1 — Discovery (tool-calling session):**

A temporary Gemini chat with discovery tools. Uses thinking level LOW.

- Tool categories: `["discovery"]` + extras `["list_fetched_data"]`
- Loop limit: max 20 tool calls, 8 iterations (via `run_tool_loop`)
- Identifies relevant spacecraft/instruments
- Calls `list_parameters()` for candidate datasets
- Calls `get_data_availability()` to check time coverage
- Builds a **VERIFIED PARAMETER REFERENCE** section from raw tool results:

```
## VERIFIED PARAMETER REFERENCE

REFERENCE: Verified parameters per dataset. Recommend these dataset IDs as candidates;
the mission agent selects specific parameters.

Dataset AC_H2_MFI (available: 1998-02-05 to 2025-12-31):
  - BGSEc (nT, double, size=[3])
  - Magnitude (nT, double)

Dataset AC_H0_MFI (available: 1997-09-02 to 2025-12-31):
  - BGSEc (nT, double, size=[3])
```

**Phase 2 — Planning (JSON-schema-enforced session):**

A separate Gemini chat produces the task plan as structured JSON. Uses thinking level HIGH.

- Response format: `response_mime_type="application/json"` with schema enforcement
- Receives: user request + discovery context (if available)
- Mission "null"/"none" strings normalized to `None`

The JSON schema (`PLANNER_RESPONSE_SCHEMA`):

```json
{
  "status": "continue" | "done",
  "reasoning": "why these tasks are needed",
  "tasks": [
    {
      "description": "human-readable summary",
      "instruction": "physics-intent instruction for the executing agent",
      "mission": "ACE" | "__data_ops__" | "__visualization__" | "__data_extraction__" | null,
      "candidate_datasets": ["AC_H2_MFI", "AC_H0_MFI"]
    }
  ],
  "summary": "optional final summary"
}
```

### Step 2: Task Structure — Physics Intent, Not Parameter Names

The critical design decision: **fetch task instructions describe physical quantities, not parameter names.**

| Old approach (before refactor) | Current approach |
|-------------------------------|-------------|
| `"Fetch data from dataset AC_H2_MFI, parameter BGSEc, for last week"` | `"Fetch magnetic field vector components for last week"` |
| Planner specifies exact `dataset_id` + `parameter_id` | Planner specifies `candidate_datasets: ["AC_H2_MFI", "AC_H0_MFI"]` |
| Mission agent blindly calls `fetch_data` | Mission agent inspects candidates, selects best dataset + parameters |
| All-NaN parameter → task "succeeds" with useless data | All-NaN parameter → agent skips it, tries next candidate |

### Step 3: Execute-Replan Loop

The orchestrator runs up to `MAX_ROUNDS` (5) iterations:

```
┌──────────────────────────────────────────────────┐
│ Planner produces task batch (or previous round's │
│   continue_planning response)                    │
│   ↓                                              │
│ Orchestrator creates Task objects for this batch  │
│   - Sets task.round = round_num                  │
│   - Sets task.candidate_datasets from planner    │
│   - Adds to TaskPlan, saves to TaskStore         │
│   ↓                                              │
│ For each task in batch:                           │
│   - Check cancellation (skip remaining as SKIPPED│
│     if cancelled)                                │
│   - Snapshot store labels (before)               │
│   - Inject context into instruction:             │
│     - Canonical time range                       │
│     - Current store contents (labels + columns)  │
│     - Candidate datasets (for mission tasks)     │
│   - Route to appropriate sub-agent               │
│   - Snapshot store labels (after)                │
│   - Record new_labels = after - before           │
│   - Build result summary (truncated to 500 chars)│
│   ↓                                              │
│ If planner said status=="done" → break           │
│ Otherwise → continue_planning(round_results)     │
│   - Planner sees: task results + new labels +    │
│     data_in_memory + failed task list + budget   │
│   - Planner decides: "continue" or "done"        │
│     - If "continue": produce next task batch     │
│     - If "done": plan complete                   │
└──────────────────────────────────────────────────┘
```

### Step 4: Task Routing (`_execute_plan_task`)

Each task is routed based on its `mission` field:

| `task.mission` | Agent | Method | Tools Available |
|---|---|---|---|
| `"ACE"`, `"PSP"`, etc. | `MissionAgent(mission_id)` | `execute_task(task)` | discovery, fetch_data, ask_clarification, list_fetched_data |
| `"__data_ops__"` | `DataOpsAgent` | `execute_task(task)` | custom_operation, describe_data, preview_data, save_data, list_fetched_data |
| `"__data_extraction__"` | `DataExtractionAgent` | `execute_task(task)` | store_dataframe, read_document, ask_clarification, list_fetched_data |
| `"__visualization__"` | `VisualizationAgent` | `execute_task(task)` | plot_data, style_plot, manage_plot, list_fetched_data |
| `null` or unknown | Orchestrator itself | `_execute_task(task)` | All orchestrator tools |

**Special handling before routing:**

- **Visualization tasks**: Checks if task is an export (handled directly without LLM via `_handle_export_task`). Ensures data labels are present in the instruction. Sets renderer time range from the plan's canonical time range.
- **Mission tasks**: Injects `Candidate datasets to inspect: ...` marker into the instruction when `task.candidate_datasets` is present.

### Step 5: Candidate Dataset Injection

When a task has `candidate_datasets` and routes to a mission agent, the orchestrator appends:

```
Candidate datasets to inspect: AC_H2_MFI, AC_H0_MFI
```

The mission agent's `_get_task_prompt()` detects this marker and switches to **candidate inspection mode**.

### Step 6: Mission Agent — Two Operating Modes

**Mode A: Candidate inspection** (when "Candidate datasets to inspect:" is present)

The mission agent prompt instructs it to:
1. Call `list_parameters` for each candidate dataset
2. Evaluate which dataset has the best parameter coverage for the physical quantity requested
3. Call `fetch_data` for each relevant parameter from the selected dataset
4. If a parameter returns all-NaN data, skip it and try the next candidate
5. After ALL fetch_data calls succeed, STOP IMMEDIATELY
6. Report stored label(s) and point count

LoopGuard allows up to 12 tool calls and 5 iterations (e.g., 2 `list_parameters` + 3 `fetch_data` + 1 retry).

**Mode B: Direct execution** (no candidates — exact instruction)

The mission agent prompt instructs it to:
1. Do ONLY what the instruction says (typically a single `fetch_data` call)
2. Do NOT call extraneous tools
3. After successful `fetch_data`, STOP
4. Report stored label and point count

This mode is used for tasks from the replan loop where labels are already known from previous round results.

Both modes append **scoped pitfalls** (operational knowledge from past sessions, scoped to this mission) to the prompt.

### Step 7: Result Reporting Back to Planner

After each task completes, the orchestrator builds a result summary:

```
- Task: Fetch ACE mag data | Status: completed | Result: Tools called: list_parameters, fetch_data, fetch_data | New data labels: AC_H2_MFI.BGSEc, AC_H2_MFI.Magnitude
```

The `continue_planning()` call includes:
- **Per-task results**: description, status, result_summary (truncated to 500 chars), error
- **Data in memory**: full list of all stored labels (so planner avoids redundant fetches)
- **Failed task list**: `"## IMPORTANT: The following tasks FAILED and must NOT be retried: ..."` (accumulated across all rounds)
- **Budget warnings**: When remaining rounds <= 2

The planner uses **actual stored labels** in subsequent round instructions (e.g., "Compute magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag").

### Step 8: Plan Finalization

After the loop ends, the orchestrator:
1. Sets plan status (CANCELLED / FAILED / COMPLETED) based on task outcomes
2. Saves to TaskStore
3. Resets the planner agent
4. Generates a summary of what was done via `_summarize_plan_execution(plan)`
5. Clears `_current_plan` and `_plan_time_range`

---

## Part 5: Sub-Agent Architecture

All sub-agents extend `BaseSubAgent` and share a common tool execution loop with two modes:

### Conversational Mode (`process_request`)

Used when the orchestrator delegates via `delegate_to_*` tools (simple path).

- Creates a **fresh chat per request** (no forced function calling)
- LoopGuard: max 20 tool calls, 8 iterations
- **Consecutive error tracking**: stops after 2 consecutive rounds where all tools error
- Returns `{text, failed, errors}` dict

### Task Execution Mode (`execute_task`)

Used when the planner dispatches tasks (complex path).

- Creates a **fresh chat per task** with **forced function calling** (mode="ANY")
- LoopGuard: max 12 tool calls, 5 iterations
- Each sub-agent overrides `_get_task_prompt(task)` to provide focused instructions
- Status determination:
  - Stopped by loop guard + had a successful tool → COMPLETED (with note about loop limit)
  - Stopped by loop guard + no successful tool → FAILED
  - Cancelled → FAILED
  - Otherwise → COMPLETED

### LoopGuard (`loop_guard.py`)

Three-layer protection against runaway loops:

1. **Hard total-call limit**: If `total_calls + len(new_calls) > max_total_calls`, stop.
2. **Subset duplicate detection**: If all proposed calls are a subset of previously made calls, stop.
3. **Cycle detection**: If the exact batch (as frozenset) appeared in the last 3 batches, stop.

### Sub-Agent Specifics

| Agent | Tool Categories | Extras | Task Prompt Key Rules |
|---|---|---|---|
| MissionAgent | discovery, data_ops_fetch, conversation | list_fetched_data | Two-mode (candidate inspection / direct). Injects mission-scoped pitfalls. Skips `ask_clarification` in task mode. |
| DataOpsAgent | data_ops_compute, conversation | list_fetched_data | Do ONLY what instruction says. After successful `custom_operation`, STOP. If wrong column names, call `preview_data` ONCE to check. |
| DataExtractionAgent | data_extraction, document, conversation | list_fetched_data | Default BaseSubAgent task prompt. |
| VisualizationAgent | visualization | list_fetched_data | First call must be `plot_data(labels=...)`. No `manage_plot(action='reset/export')`. May call `style_plot` after. Injects visualization-scoped pitfalls. |

---

## Part 6: Tool Execution (`_execute_tool`)

All tool calls funnel through `OrchestratorAgent._execute_tool(tool_name, tool_args)`, a large dispatch function. Key behaviors:

### Discovery Tools
- `search_datasets`: keyword search across the local mission JSON catalog
- `list_parameters`: 3-layer metadata resolution (memory → file cache → Master CDF)
- `get_data_availability`: dataset time range from metadata cache
- `browse_datasets`: all science datasets for a mission (filtered by calibration exclusion lists)
- `search_full_catalog`: searches all 2000+ CDAWeb datasets

### Data Fetch
- `fetch_data`: validates dataset/parameter IDs, parses time range, auto-clamps to dataset availability, detects duplicate fetches, reports NaN-only columns

### Compute
- `custom_operation`: AST-validated sandboxed execution of LLM-generated pandas/numpy/scipy code

### Visualization
- `plot_data` → `_handle_plot_data`: auto-splits by units, creates multi-panel layouts
- `style_plot` → `_handle_style_plot`: key-value styling params
- `manage_plot` → `_handle_manage_plot`: export, reset, zoom, add/remove traces

### Special
- `google_search`: isolated Gemini API call with GoogleSearch tool only (not combinable with function_declarations)
- `ask_clarification`: returns `clarification_needed` status, halts the loop immediately
- `read_document`: Gemini multimodal (vision) for PDFs and images

`_execute_tool_safe` wraps all calls with try/except, sanitizes NaN/Inf values via `_sanitize_for_json`, and logs results.

---

## Part 7: Memory System

### Long-term Memory (`memory.py`)

Cross-session memory persisted at `~/.helio-agent/memory.json`:

- **Three types**: `preference` (plot styles, spacecraft of interest), `summary` (session summaries), `pitfall` (operational knowledge)
- **Scoped pitfalls**: `"generic"` (injected into orchestrator prompt), `"mission:<ID>"` (injected into MissionAgent task prompts), `"visualization"` (injected into VisualizationAgent task prompts)
- **Injection**: prepended to user messages in `process_message()` (avoids chat recreation)
- **Deduplication**: skips new memories that are substrings of existing ones
- **Caps**: 15 preferences + 10 summaries + 20 pitfalls per injection
- **Cold storage**: archived memories go to `memory_cold.json`, searchable via `recall_memories` tool

### MemoryAgent (Background Daemon, `memory_agent.py`)

Runs as a daemon thread, polling every 30 seconds:

- **Triggers** (`_should_analyze`):
  - Log growth >= 10 KB since last analysis
  - 5+ ERROR/WARNING lines in new log content
- **Analysis**: Single-shot Gemini Flash call (temperature=0.2) on new log content (capped at 50 KB)
- **Extracts**: preferences, summaries, pitfalls (with auto-detected scope), error patterns (saved as markdown reports to `~/.helio-agent/reports/`)
- **Consolidation**: If memory count > 30, runs LLM-based consolidation, archives evicted entries to cold storage
- **Safety**: All exceptions caught — never breaks the main agent flow

---

## Part 8: Error Handling & Resilience

### All-NaN Fallback
Mission agents skip parameters that return all-NaN data and try the next candidate dataset. Common with merged datasets like `PSP_COHO1HR_MERGED_MAG_PLASMA`.

### Failed Task Tracking
The planner is told not to retry failed tasks across rounds:
```
## IMPORTANT: The following tasks FAILED and must NOT be retried:
  - Fetch PSP plasma density
Do NOT create new tasks that attempt the same searches.
```

### Round Budget Warnings
```
## BUDGET WARNING: Only 2 round(s) remaining.
## FINAL ROUND: This is the last round. Set status='done' unless critical work remains.
```

### Consecutive Error Guard
- **Orchestrator**: breaks after 2 consecutive delegation failures
- **Sub-agents**: `process_request` stops after 2 consecutive rounds where ALL tools error

### LoopGuard Limits
| Context | Max Tool Calls | Max Iterations |
|---|---|---|
| `BaseSubAgent.process_request` | 20 | 8 |
| `BaseSubAgent.execute_task` | 12 | 5 |
| Planner discovery `run_tool_loop` | 20 | 8 |
| Orchestrator `_execute_task` | 10 | 5 |
| Orchestrator `_process_single_message` | — (no LoopGuard) | 10 (hard limit) |

### Auto-Clamping Time Ranges
`_validate_time_range()` in `core.py` auto-adjusts requested time ranges to fit dataset availability windows. Handles partial overlaps (clamps) and full mismatches (informs user). Fail-open: proceeds without validation if metadata call fails.

### Model Fallback
When any Gemini API call hits a 429 RESOURCE_EXHAUSTED error, all agents switch to `GEMINI_FALLBACK_MODEL` (default: `gemini-2.5-flash`) for the session. The orchestrator's persistent chat is recreated with the fallback model.

### Cancellation
Users can cancel at any point. The `_cancel_event` (threading.Event) is checked:
- Between tasks in the plan batch
- Between rounds in the plan-execute loop
- Inside sub-agent tool loops (both `process_request` and `execute_task`)
- Inside planner discovery tool loop

Remaining tasks are marked SKIPPED.

---

## Part 9: Models and Configuration

| Role | Default Model | Config |
|---|---|---|
| Orchestrator | `gemini-3-pro-preview` | `GEMINI_MODEL` env var |
| Sub-agents (Mission, DataOps, DataExtraction, Visualization) | `gemini-3-flash-preview` | `GEMINI_SUB_AGENT_MODEL` env var |
| Planner | Same as orchestrator | `GEMINI_PLANNER_MODEL` env var |
| Fallback (all agents) | `gemini-2.5-flash` | `GEMINI_FALLBACK_MODEL` env var |
| MemoryAgent | Flash model | Uses sub-agent model |

**Thinking levels:**
- **HIGH**: Orchestrator and PlannerAgent (deep reasoning for routing and plan decomposition)
- **LOW**: All sub-agents (fast execution with minimal thinking overhead)

---

## Concrete Example

**User**: "Compare ACE and Wind magnetic field, compute magnitude of each, plot them"

### Complexity Detection
`is_complex_request()` matches "compare" + multiple spacecraft → routes to `_handle_planning_request()`.

### Time Range Resolution
`_extract_time_range()` doesn't find an explicit time range → planner decides (typically defaults to recent data).

### Round 0: Discovery + Planning

**Discovery phase** calls:
- `list_parameters("AC_H2_MFI")` → finds BGSEc, Magnitude
- `list_parameters("WI_H2_MFI")` → finds BGSE, BGSM
- `get_data_availability("AC_H2_MFI")` → 1998 to present

**Planning phase** produces:
```json
{
  "status": "continue",
  "reasoning": "Need to fetch magnetic field data from both missions",
  "tasks": [
    {
      "description": "Fetch ACE mag data",
      "instruction": "Fetch magnetic field vector components for last week",
      "mission": "ACE",
      "candidate_datasets": ["AC_H2_MFI", "AC_H0_MFI"]
    },
    {
      "description": "Fetch Wind mag data",
      "instruction": "Fetch magnetic field vector components for last week",
      "mission": "WIND",
      "candidate_datasets": ["WI_H2_MFI", "WI_H0_MFI"]
    }
  ]
}
```

### Round 0: Execution

**ACE MissionAgent** (candidate inspection mode):
1. `list_parameters("AC_H2_MFI")` → sees BGSEc (3-component vector, nT)
2. `list_parameters("AC_H0_MFI")` → sees BGSEc (similar)
3. Picks AC_H2_MFI (higher resolution), calls `fetch_data("AC_H2_MFI", "BGSEc", "last week")`
4. Reports: "Stored label: AC_H2_MFI.BGSEc, 10080 points"

**Wind MissionAgent** (candidate inspection mode):
1. `list_parameters("WI_H2_MFI")` → sees BGSE (3-component vector, nT)
2. Calls `fetch_data("WI_H2_MFI", "BGSE", "last week")`
3. Reports: "Stored label: WI_H2_MFI.BGSE, 10080 points"

**Results sent to planner** via `continue_planning()`:
```
- Task: Fetch ACE mag data | Status: completed | New data labels: AC_H2_MFI.BGSEc
- Task: Fetch Wind mag data | Status: completed | New data labels: WI_H2_MFI.BGSE
Data currently in memory: AC_H2_MFI.BGSEc, WI_H2_MFI.BGSE
```

### Round 1: Compute

Planner produces (using labels from round 0 results):
```json
{
  "status": "continue",
  "tasks": [
    {
      "description": "Compute ACE Bmag",
      "instruction": "Compute magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag",
      "mission": "__data_ops__"
    },
    {
      "description": "Compute Wind Bmag",
      "instruction": "Compute magnitude of WI_H2_MFI.BGSE, save as Wind_Bmag",
      "mission": "__data_ops__"
    }
  ]
}
```

**DataOpsAgent** executes each task:
1. `custom_operation(code="...", input_labels=["AC_H2_MFI.BGSEc"], output_label="ACE_Bmag")`
2. `custom_operation(code="...", input_labels=["WI_H2_MFI.BGSE"], output_label="Wind_Bmag")`

### Round 2: Visualize

```json
{
  "status": "done",
  "tasks": [
    {
      "description": "Plot comparison",
      "instruction": "Use plot_data to plot ACE_Bmag and Wind_Bmag together with title 'ACE vs Wind Magnetic Field Magnitude'",
      "mission": "__visualization__"
    }
  ],
  "summary": "Fetched ACE and Wind magnetic field data, computed magnitudes, and plotted them together."
}
```

**VisualizationAgent** executes:
1. `plot_data(labels="ACE_Bmag, Wind_Bmag")` — creates plot with auto-panel layout
2. `style_plot(title="ACE vs Wind Magnetic Field Magnitude")` — applies title

Plan status → COMPLETED.

---

## Key Files

| File | Role |
|------|------|
| `agent/core.py` | OrchestratorAgent: message routing, tool execution, plan loop, delegation |
| `agent/planner.py` | PlannerAgent: discovery + JSON-schema planning, `is_complex_request()`, replan with results |
| `agent/base_agent.py` | BaseSubAgent: LoopGuard integration, `process_request` / `execute_task` loops |
| `agent/mission_agent.py` | MissionAgent: two-mode task prompt (candidate inspection vs direct) |
| `agent/data_ops_agent.py` | DataOpsAgent: data transformation task prompt |
| `agent/visualization_agent.py` | VisualizationAgent: plot task prompt with label extraction |
| `agent/data_extraction_agent.py` | DataExtractionAgent: text-to-DataFrame conversion |
| `agent/tasks.py` | Task / TaskPlan / TaskStore dataclasses and persistence |
| `agent/tools.py` | Tool schemas (26+ tools across 11 categories) |
| `agent/loop_guard.py` | LoopGuard: call limits, duplicate/cycle detection |
| `agent/tool_loop.py` | `run_tool_loop()`: standalone tool loop for planner discovery |
| `agent/memory.py` | MemoryStore: cross-session memory persistence |
| `agent/memory_agent.py` | MemoryAgent: background daemon for knowledge extraction |
| `agent/model_fallback.py` | Automatic model fallback on quota errors |
| `agent/time_utils.py` | TimeRange parsing |
| `knowledge/prompt_builder.py` | All agent prompts: planner, mission, discovery, orchestrator, data ops, visualization |
| `data_ops/store.py` | In-memory DataStore singleton (label → DataEntry) |
| `data_ops/fetch.py` | CDF data fetching (delegates to fetch_cdf.py) |
| `data_ops/custom_ops.py` | AST-validated sandboxed executor |
| `rendering/plotly_renderer.py` | Plotly figure management |
| `rendering/registry.py` | Visualization tool registry (3 declarative tools) |
| `config.py` | Model and backend configuration |
