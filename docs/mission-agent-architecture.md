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

### 1a. Enhance `knowledge/catalog.py` with rich mission profiles

Add optional `"profile"` dicts with detailed data product info per mission. Existing API unchanged.

```python
"PSP": {
    "name": "Parker Solar Probe",
    "keywords": [...],  # unchanged
    "profile": {
        "description": "Inner heliosphere probe studying solar corona",
        "coordinate_systems": ["RTN"],
        "typical_cadence": "1-minute",
        "data_caveats": ["RTN rotates with spacecraft position"],
        "analysis_patterns": [
            "Switchback detection: compute radial component sign changes",
            "Parker spiral angle: atan2(Bt, Br) compared to expected"
        ]
    },
    "instruments": {
        "FIELDS/MAG": {
            ...,  # existing fields unchanged
            "profile": {
                "primary_parameters": {
                    "psp_fld_l2_mag_RTN_1min": "3-component B-field vector (RTN)"
                },
                "units": "nT",
                "tips": "Compute magnitude for overview"
            }
        }
    }
}
```

**Files:** `knowledge/catalog.py` (+120 lines for profiles on all 8 missions)

### 1b. Create `knowledge/prompt_builder.py`

Pure functions that generate prompt sections from the catalog:

- `generate_spacecraft_overview()` — replaces hardcoded table in prompts.py
- `generate_dataset_quick_reference()` — replaces hardcoded dataset table in prompts.py
- `generate_planner_dataset_reference()` — replaces hardcoded table in planner.py
- `generate_mission_profiles()` — detailed per-mission context (new)
- **`build_mission_prompt(mission_id)`** — generates a focused prompt for ONE mission (foundation for Phase 2)
- `build_system_prompt()` — assembles full system prompt (all missions)
- `build_planning_prompt()` — assembles planning prompt

**Files:** `knowledge/prompt_builder.py` (NEW, ~250 lines)

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
| `knowledge/catalog.py` | Modify | +120 lines (mission profiles) |
| `knowledge/prompt_builder.py` | **New** | ~250 lines (section generators + `build_mission_prompt()`) |
| `agent/prompts.py` | Modify | -160 lines, +10 (import generated prompt) |
| `agent/planner.py` | Modify | -65 lines, +10 (import generated reference) |
| `tests/test_prompt_builder.py` | **New** | ~80 lines (unit tests) |
| `docs/capability-summary.md` | Modify | +5 lines |
| `CLAUDE.md` | Modify | +10 lines |

**Unchanged:** `agent/core.py`, `agent/tasks.py`, `agent/tools.py`, `knowledge/hapi_client.py`, all `data_ops/`, all `autoplot_bridge/`, all existing tests.

### Why Phase 1 First

Phase 1 is the **critical foundation**:
- `build_mission_prompt(mission_id)` is exactly what `MissionAgent.__init__()` will call in Phase 2
- Rich catalog profiles give mission agents their specialized knowledge
- Single source of truth prevents the current 3-file duplication problem
- Zero risk — no behavioral changes, just prompt generation refactor

Without Phase 1, Phase 2 would have to hardcode mission-specific prompts (repeating the current problem at a larger scale).

### Verification (Phase 1)

1. `python -m pytest tests/` — all existing + new tests pass
2. `python main.py "show me ACE magnetic field last week"` — identical behavior
3. `python main.py --verbose` — check token count within 10% of original
4. Add test mission to catalog → verify it appears in generated prompt
5. Call `build_mission_prompt("PSP")` → verify it produces a focused PSP-only prompt

---

## Phase 2: Mission Sub-Agents + Dispatch Protocol (NEXT)

**Status: NOT STARTED — depends on Phase 1**

### 2a. Add `mission` field to Task

```python
# agent/tasks.py — Task dataclass
@dataclass
class Task:
    id: str
    description: str
    instruction: str
    mission: Optional[str] = None  # NEW: "PSP", "ACE", None for cross-mission
    depends_on: list[str] = field(default_factory=list)  # NEW: task IDs this depends on
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    ...
```

### 2b. Create `agent/mission_agent.py`

```python
class MissionAgent:
    """A Gemini session specialized for one mission."""

    def __init__(self, mission_id: str, client, model_name, verbose=False):
        self.mission_id = mission_id
        # Build mission-specific prompt from catalog
        self.prompt = build_mission_prompt(mission_id)  # from prompt_builder
        self.tools = get_tool_schemas()  # shared tools (+ mission-specific later)

        # Create Gemini chat with mission-focused context
        self.chat = client.chats.create(
            model=model_name,
            config=GenerateContentConfig(
                system_instruction=self.prompt,
                tools=[Tool(function_declarations=...)],
                tool_config=ToolConfig(
                    function_calling_config=FunctionCallingConfig(mode="ANY")
                ),
            )
        )

    def execute_task(self, task: Task) -> str:
        """Execute a task in this mission's context."""
        # Same logic as current _execute_task(), but using mission-specific chat
        ...
```

### 2c. Enhance planner to tag tasks with missions

Update `PLANNING_PROMPT` to output:
```json
{
    "tasks": [
        {"description": "...", "instruction": "...", "mission": "PSP"},
        {"description": "...", "instruction": "...", "mission": "ACE"},
        {"description": "...", "instruction": "...", "mission": null, "depends_on": [0, 1]}
    ]
}
```

### 2d. Update task dispatcher in `core.py`

```python
def _process_complex_request(self, user_message):
    plan = create_plan_from_request(...)

    # Create mission agents for each unique mission in the plan
    mission_agents = {}
    for task in plan.tasks:
        if task.mission and task.mission not in mission_agents:
            mission_agents[task.mission] = MissionAgent(
                task.mission, self.client, self.model_name
            )

    # Execute tasks (sequential for now, parallel in Phase 3)
    for task in plan.tasks:
        if task.mission and task.mission in mission_agents:
            mission_agents[task.mission].execute_task(task)
        else:
            self._execute_task(task)  # cross-mission tasks use main agent

    return self._summarize_plan_execution(plan)
```

**Files:** `agent/tasks.py` (+2 fields), `agent/mission_agent.py` (NEW, ~150 lines), `agent/planner.py` (schema update), `agent/core.py` (dispatch logic)

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
