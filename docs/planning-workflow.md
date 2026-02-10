# Planning & Data Fetch Workflow

How complex multi-step requests flow through the agent system, from user input to plotted data.

**Last updated**: 2026-02-10

---

## Overview

When a user asks something complex (e.g., "Compare ACE and PSP magnetic field for last week, compute magnitudes, and plot them"), the system uses a **plan-execute-replan loop** with specialized sub-agents. The key design principle is **separation of concerns**:

- **PlannerAgent** decides *what* to do (physics intent + candidate datasets)
- **MissionAgent** decides *how* to fetch (inspects datasets, selects parameters)
- **DataOpsAgent** handles transformations (magnitude, smoothing, etc.)
- **VisualizationAgent** handles plotting (Plotly renderer)

This separation prevents the planner from guessing parameter names (which vary across datasets and often have all-NaN values), while letting mission-specialist agents make informed decisions about data quality.

---

## The Full Flow

### 1. Complexity Detection

`OrchestratorAgent.process_message()` checks whether the request is complex via `is_complex_request()` — a set of regex heuristics that detect multi-spacecraft references, sequential language ("then", "after"), comparison keywords ("compare", "vs"), and multi-step operations.

- **Simple request** → handled directly by the orchestrator's Gemini session
- **Complex request** → routed to `_handle_planning_request()`

### 2. Time Range Resolution

Before planning begins, the orchestrator attempts to parse a canonical time range from the user message (`_extract_time_range()`). If found, it's injected into the planning message:

```
"Resolved time range: 2024-01-10 to 2024-01-17. Use this exact range for ALL fetch tasks."
```

This prevents each sub-agent from re-interpreting relative expressions like "last week" at different times.

### 3. Discovery Phase

The `PlannerAgent.start_planning()` method runs a two-phase process:

**Phase 1 — Discovery (tool-calling session):**
A temporary Gemini chat with discovery tools (`search_datasets`, `browse_datasets`, `list_parameters`, `get_data_availability`, `list_fetched_data`) researches the user's request. It:

1. Identifies which spacecraft/instruments are relevant
2. Calls `list_parameters()` for each candidate dataset to get exact parameter names
3. Optionally calls `get_data_availability()` to check time coverage
4. Returns a text summary plus a **VERIFIED PARAMETER REFERENCE** block

The parameter reference is structured like:

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

This gives the planner visibility into what datasets contain, so it can choose appropriate candidates — but the planner does NOT put specific parameter names in task instructions.

**Phase 2 — Planning (JSON-schema-enforced session):**
A separate Gemini chat produces the task plan as structured JSON. The schema enforces:

```json
{
  "status": "continue" | "done",
  "reasoning": "...",
  "tasks": [
    {
      "description": "human-readable summary",
      "instruction": "physics-intent instruction for the executing agent",
      "mission": "ACE" | "__data_ops__" | "__visualization__" | null,
      "candidate_datasets": ["AC_H2_MFI", "AC_H0_MFI"]
    }
  ],
  "summary": "..."
}
```

### 4. Task Structure — Physics Intent, Not Parameter Names

The critical design decision: **fetch task instructions describe physical quantities, not parameter names.**

| Old approach (before refactor) | New approach |
|-------------------------------|-------------|
| `"Fetch data from dataset AC_H2_MFI, parameter BGSEc, for last week"` | `"Fetch magnetic field vector components for last week"` |
| Planner specifies exact `dataset_id` + `parameter_id` | Planner specifies `candidate_datasets: ["AC_H2_MFI", "AC_H0_MFI"]` |
| Mission agent blindly calls `fetch_data` | Mission agent inspects candidates, selects best dataset + parameters |
| All-NaN parameter → task "succeeds" with useless data | All-NaN parameter → agent skips it, tries next candidate |
| Label mismatch → false "Expected label not found" warning | No expected-label validation — actual labels reported to planner |

### 5. Plan Execution Loop

The orchestrator runs up to `MAX_ROUNDS` (5) iterations of:

```
┌─────────────────────────────────────────────┐
│ Planner produces task batch                  │
│   ↓                                          │
│ Orchestrator creates Task objects             │
│   ↓                                          │
│ For each task in batch:                       │
│   - Snapshot store labels (before)            │
│   - Route to appropriate sub-agent            │
│   - Snapshot store labels (after)             │
│   - Record new_labels = after - before        │
│   - Build result summary for planner          │
│   ↓                                          │
│ Send results back to planner                  │
│   ↓                                          │
│ Planner decides: "continue" or "done"         │
│   - If "continue": produce next task batch    │
│   - If "done": plan complete                  │
└─────────────────────────────────────────────┘
```

### 6. Task Routing

The orchestrator routes each task based on its `mission` field:

| `task.mission` | Agent | What it does |
|---------------|-------|-------------|
| `"ACE"`, `"PSP"`, etc. | `MissionAgent(mission_id)` | Data fetching for that spacecraft |
| `"__visualization__"` | `VisualizationAgent` | Plotting via `plot_data`, `style_plot`, `manage_plot` |
| `"__data_ops__"` | `DataOpsAgent` | Transformations via `custom_operation`, `describe_data` |
| `"__data_extraction__"` | `DataExtractionAgent` | DataFrame creation via `store_dataframe` |
| `null` or unknown | Orchestrator itself | General tasks |

### 7. Candidate Dataset Injection

When a task has `candidate_datasets` and routes to a mission agent, the orchestrator appends to the task instruction:

```
Candidate datasets to inspect: AC_H2_MFI, AC_H0_MFI
```

The mission agent's `_get_task_prompt()` detects this marker and switches to **candidate inspection mode**.

### 8. Mission Agent — Two Operating Modes

**Mode A: Candidate inspection** (when "Candidate datasets to inspect:" is present)

The mission agent:
1. Calls `list_parameters` for each candidate dataset
2. Evaluates which dataset has the best parameter coverage for the physical quantity requested
3. Calls `fetch_data` for each relevant parameter from the selected dataset
4. If a parameter returns all-NaN data, skips it and tries the next candidate
5. Reports stored label(s) and point count

LoopGuard allows up to 12 tool calls and 5 iterations for this workflow (e.g., 2 `list_parameters` + 3 `fetch_data` + 1 retry).

**Mode B: Direct execution** (no candidates — exact instruction)

The mission agent:
1. Does exactly what the instruction says (typically a single `fetch_data` call)
2. Does NOT call `list_parameters`, `describe_data`, or other exploratory tools
3. Reports stored label and point count

This mode is used for tasks from the replan loop where labels are already known from previous round results.

### 9. Result Reporting Back to Planner

After each task completes, the orchestrator builds a result summary:

```
- Task: Fetch ACE mag data | Status: completed | Result: Tools called: list_parameters, fetch_data, fetch_data | New data labels: AC_H2_MFI.BGSEc, AC_H2_MFI.Magnitude
```

Key signals included:
- **New data labels**: what was actually stored (label = `DATASET.PARAM`)
- **Tool calls made**: what the agent did
- **Data currently in memory**: full list of all stored labels (so planner avoids redundant fetches)
- **Failed task tracking**: planner is told not to retry failed tasks

The planner uses the **actual stored labels** in subsequent round instructions (e.g., "Compute magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag").

### 10. Downstream Tasks Use Actual Labels

After fetching, the planner creates compute and visualization tasks using the labels reported in execution results — not guessed names:

```json
{
  "description": "Compute ACE Bmag",
  "instruction": "Compute magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag",
  "mission": "__data_ops__"
}
```

```json
{
  "description": "Plot comparison",
  "instruction": "Use plot_data to plot ACE_Bmag and Wind_Bmag together",
  "mission": "__visualization__"
}
```

---

## Concrete Example

**User**: "Compare ACE and Wind magnetic field, compute magnitude of each, plot them"

### Round 1: Discovery + Planning

Discovery phase calls:
- `list_parameters("AC_H2_MFI")` → finds BGSEc, Magnitude
- `list_parameters("WI_H2_MFI")` → finds BGSE, BGSM
- `get_data_availability("AC_H2_MFI")` → 1998 to present

Planner produces:
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

### Round 1: Execution

ACE MissionAgent:
1. `list_parameters("AC_H2_MFI")` → sees BGSEc (3-component vector, nT)
2. `list_parameters("AC_H0_MFI")` → sees BGSEc (similar)
3. Picks AC_H2_MFI (higher resolution), calls `fetch_data("AC_H2_MFI", "BGSEc", "last week")`
4. Reports: "Stored label: AC_H2_MFI.BGSEc, 10080 points"

Wind MissionAgent (runs in parallel):
1. `list_parameters("WI_H2_MFI")` → sees BGSE (3-component vector, nT)
2. Calls `fetch_data("WI_H2_MFI", "BGSE", "last week")`
3. Reports: "Stored label: WI_H2_MFI.BGSE, 10080 points"

Results sent to planner:
```
- Task: Fetch ACE mag data | Status: completed | New data labels: AC_H2_MFI.BGSEc
- Task: Fetch Wind mag data | Status: completed | New data labels: WI_H2_MFI.BGSE
Data currently in memory: AC_H2_MFI.BGSEc, WI_H2_MFI.BGSE
```

### Round 2: Compute

Planner produces (using labels from round 1 results):
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

### Round 3: Visualize

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

---

## Error Handling & Resilience

### All-NaN Fallback
If a mission agent fetches a parameter and gets all-NaN values, it skips that parameter and tries the next candidate dataset. This is a common issue with merged datasets like `PSP_COHO1HR_MERGED_MAG_PLASMA` where some parameters have gaps.

### Failed Task Tracking
The planner is explicitly told not to retry failed tasks:
```
## IMPORTANT: The following tasks FAILED and must NOT be retried:
  - Fetch PSP plasma density
Do NOT create new tasks that attempt the same searches.
```

### Round Budget
The planner receives budget warnings as rounds deplete:
```
## BUDGET WARNING: Only 2 round(s) remaining.
## FINAL ROUND: This is the last round. Set status='done' unless critical work remains.
```

### LoopGuard (per sub-agent)
Each sub-agent call is protected by a `LoopGuard(max_total_calls=12, max_iterations=5)` that:
- Caps total tool calls to prevent runaway loops
- Detects duplicate tool calls (same function + args) and stops
- Allows enough headroom for candidate inspection (2 `list_parameters` + 3-4 `fetch_data`)

### Cancellation
Users can cancel at any point. The orchestrator checks `_cancel_event` between tasks and between rounds, marking remaining tasks as SKIPPED.

---

## Key Files

| File | Role |
|------|------|
| `agent/core.py` | Orchestrator: complexity detection, plan loop, task routing, candidate injection |
| `agent/planner.py` | PlannerAgent: discovery + JSON-schema planning, replan with results |
| `agent/mission_agent.py` | MissionAgent: two-mode task prompt (candidate inspection vs direct) |
| `agent/base_agent.py` | BaseSubAgent: LoopGuard, tool execution loop, usage tracking |
| `agent/tasks.py` | Task/TaskPlan dataclasses with `candidate_datasets` field |
| `knowledge/prompt_builder.py` | All agent prompts: planner, mission, discovery, orchestrator |
| `agent/tools.py` | Tool schemas (26 tools across 7 categories) |
| `data_ops/fetch.py` | HAPI data fetching with dataset/parameter validation |
| `data_ops/store.py` | In-memory DataStore singleton (label → DataEntry) |
