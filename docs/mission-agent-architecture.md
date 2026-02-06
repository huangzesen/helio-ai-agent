# Mission-Specific Agent Architecture (Phased Plan)

## Context

The current system has a monolithic prompt with spacecraft knowledge hardcoded in 3 places. The vision is a multi-agent architecture where:

1. **Main agent** (orchestrator) understands user intent and decomposes into tasks
2. **Mission sub-agents** (PSP, ACE, etc.) each have deep knowledge of one mission's 100+ data products
3. **Parallel execution** for independent tasks across different missions
4. **Report-back protocol** where sub-agents return results to the main agent

The existing planner/task system (`planner.py`, `tasks.py`, `core.py:_execute_task()`) already handles decomposition and sequential execution. The evolution is to: tag tasks with missions, dispatch to mission-specific agent sessions, and run independent tasks in parallel.

**This is implemented in 3 phases — Phase 1 is the foundation.**

---

## Phase 1: Mission Knowledge Modules + Dynamic Prompts (FOUNDATION)

**Status: COMPLETE**

This is the foundation. Without it, mission sub-agents have nothing to load.

### 1a. Enhance `knowledge/catalog.py` with spacecraft-level mission profiles

Added `"profile"` dicts at the spacecraft level with domain knowledge that HAPI doesn't provide.
Instrument-level metadata (parameter names, units, descriptions) is fetched from HAPI at runtime
via `list_parameters` — not hardcoded in the catalog.

```python
"PSP": {
    "name": "Parker Solar Probe",
    "keywords": [...],  # unchanged
    "profile": {
        "description": "Inner heliosphere probe studying the solar corona and young solar wind",
        "coordinate_systems": ["RTN"],
        "typical_cadence": "1-minute",
        "data_caveats": ["RTN frame rotates with spacecraft orbital position"],
        "analysis_patterns": [
            "Switchback detection: compute radial component sign changes in Br",
            "Parker spiral angle: atan2(Bt, Br) compared to expected spiral",
        ]
    },
    "instruments": {
        "FIELDS/MAG": {
            "name": "FIELDS Magnetometer",
            "keywords": ["magnetic", "field", "mag", "b-field", "bfield"],
            "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"],
            # No parameter/unit metadata — comes from HAPI at runtime
        },
    }
}
```

**Files:** `knowledge/catalog.py` (+90 lines for spacecraft profiles on all 8 missions)

### 1b. Create `knowledge/prompt_builder.py`

Pure functions that generate prompt sections from the catalog. Dataset reference tables
list dataset IDs and types only — the agent uses `list_parameters` (HAPI) to discover
parameter names and units at runtime.

- `generate_spacecraft_overview()` — spacecraft/instruments table for system prompt
- `generate_dataset_quick_reference()` — dataset ID + type table (no hardcoded parameter names)
- `generate_planner_dataset_reference()` — dataset reference for planner prompt
- `generate_mission_profiles()` — domain knowledge (caveats, analysis tips, coordinate systems)
- **`build_mission_prompt(mission_id)`** — focused prompt for ONE mission (foundation for Phase 2)
- `build_system_prompt()` — assembles full system prompt (all missions)
- `build_planning_prompt()` — assembles planning prompt

**Files:** `knowledge/prompt_builder.py` (NEW, ~230 lines)

### 1c. Refactor `agent/prompts.py` and `agent/planner.py`

Replace hardcoded strings with generated content:

```python
# agent/prompts.py
from knowledge.prompt_builder import build_system_prompt
_SYSTEM_PROMPT_TEMPLATE = build_system_prompt()  # cached at import

def get_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(today=...)
```

```python
# agent/planner.py
from knowledge.prompt_builder import build_planning_prompt
_PLANNING_PROMPT_TEMPLATE = build_planning_prompt()  # cached at import
```

**Files:** `agent/prompts.py` (-160 lines, +10), `agent/planner.py` (-65 lines, +10)

### 1d. Tests

- `tests/test_prompt_builder.py` (NEW, ~80 lines): verify all spacecraft/datasets appear in generated sections
- Existing `tests/test_catalog.py`: must pass unchanged (backward compat)

### 1e. Update docs

- `docs/capability-summary.md`: note dynamic prompt generation
- `CLAUDE.md`: update "adding new spacecraft" instructions

### Phase 1 File Summary

| File | Action | Change |
|------|--------|--------|
| `knowledge/catalog.py` | Modify | +90 lines (spacecraft-level mission profiles) |
| `knowledge/prompt_builder.py` | **New** | ~230 lines (section generators + `build_mission_prompt()`) |
| `agent/prompts.py` | Modify | -160 lines, +10 (import generated prompt) |
| `agent/planner.py` | Modify | -65 lines, +10 (import generated reference) |
| `tests/test_prompt_builder.py` | **New** | 23 unit tests |
| `docs/capability-summary.md` | Modify | +5 lines |
| `CLAUDE.md` | Modify | +10 lines |

**Unchanged:** `agent/core.py`, `agent/tasks.py`, `agent/tools.py`, `knowledge/hapi_client.py`, all `data_ops/`, all `autoplot_bridge/`, all existing tests (288 total pass).

### Design Decision: HAPI for Parameter Metadata

The original plan included instrument-level `profile` dicts with `primary_parameters`, `units`,
and `tips`. These were removed because CDAWeb HAPI already provides this metadata via
`list_parameters`. The catalog now stores only what HAPI doesn't know: domain knowledge
(analysis patterns, caveats, coordinate system context) and keyword mappings for NLP routing.

### Why Phase 1 First

Phase 1 is the **critical foundation**:
- `build_mission_prompt(mission_id)` is exactly what `MissionAgent.__init__()` will call in Phase 2
- Catalog profiles give mission agents domain knowledge; HAPI gives parameter details
- Single source of truth prevents the previous 3-file duplication problem
- Zero risk — no behavioral changes, just prompt generation refactor

Without Phase 1, Phase 2 would have to hardcode mission-specific prompts (repeating the old problem at a larger scale).

---

## Phase 2: Mission Sub-Agents + Dispatch Protocol

**Status: COMPLETE**

### 2a. Add `mission` and `depends_on` fields to Task

Added `mission: Optional[str]` and `depends_on: list[str]` to the Task dataclass.
The planner now outputs mission tags and dependency indices, which are resolved to
task IDs during plan creation. Serialization is backward-compatible — old plans
without these fields load with defaults (None, []).

### 2b. Create `agent/mission_agent.py`

`MissionAgent` is a Gemini session specialized for one mission. It receives a
focused system prompt via `build_mission_prompt(mission_id)` (from Phase 1) and
uses a shared `tool_executor` callable (from the main agent) so all tools work
identically. Each task gets a fresh chat session to avoid context pollution.

Key design: the `tool_executor` parameter is `AutoplotAgent._execute_tool_safe`,
so mission agents share the same Autoplot bridge, data store, and error handling
as the main agent — no tool duplication.

### 2c. Planner tags tasks with missions

The planning prompt now instructs Gemini to:
- Tag each task with its spacecraft ID (e.g., `"mission": "PSP"`)
- Declare dependencies via `"depends_on": [0, 1]` (0-based task indices)
- Use `null` mission for cross-mission tasks (comparison plots, exports)

The `PLAN_SCHEMA` includes optional `mission` (string) and `depends_on` (array of int)
fields. Dependency indices are resolved to task UUIDs in `create_plan_from_request()`.

### 2d. Dispatch in `core.py`

`_process_complex_request()` now:
1. Creates one `MissionAgent` per unique mission in the plan
2. Dispatches mission-tagged tasks to the corresponding agent
3. Falls back to main agent for cross-mission tasks (mission=None)
4. Checks dependencies before each task — skips tasks whose dependencies failed
5. Aggregates token usage from all mission agents into the main agent's totals

Execution is still sequential (Phase 3 adds parallelism).

### Phase 2 File Summary

| File | Action | Change |
|------|--------|--------|
| `agent/tasks.py` | Modify | +2 fields (mission, depends_on), updated serialization |
| `agent/mission_agent.py` | **New** | ~180 lines (MissionAgent class) |
| `agent/planner.py` | Modify | Schema + dependency resolution + mission display |
| `agent/core.py` | Modify | Mission agent creation + dispatch logic |
| `knowledge/prompt_builder.py` | Modify | Mission tagging + dependency instructions in planner prompt |
| `tests/test_mission_agent.py` | **New** | 9 tests (prompt, import, interface) |
| `tests/test_tasks.py` | Modify | +7 tests for new fields |
| `tests/test_planner.py` | Modify | +1 test for mission-tagged display |
| `tests/test_prompt_builder.py` | Modify | +2 tests for mission tagging in planner prompt |

---

## Phase 3: Parallel Execution (LATER)

**Status: NOT STARTED — depends on Phase 2**

### 3a. Dependency-aware parallel dispatch

```python
import concurrent.futures

def _process_complex_request(self, user_message):
    plan = create_plan_from_request(...)
    completed_tasks = {}

    while not plan.is_complete():
        # Find tasks whose dependencies are all completed
        ready = [t for t in plan.tasks
                 if t.status == TaskStatus.PENDING
                 and all(dep in completed_tasks for dep in t.depends_on)]

        if not ready:
            break  # deadlock or done

        # Execute ready tasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for task in ready:
                agent = mission_agents.get(task.mission, self)
                futures[executor.submit(agent.execute_task, task)] = task

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                completed_tasks[task.id] = future.result()
```

### 3b. Progress reporting

Main agent reports: "PSP agent working... ACE agent completed... Waiting for both to finish..."

**Files:** `agent/core.py` (parallel execution logic)
