# Over-Engineering & Dead Code Report

*Generated: 2026-02-08*

## CRITICAL — Massive Duplication

### 1. Four sub-agents are 85% copy-paste (~700 lines)

`mission_agent.py`, `visualization_agent.py`, `data_ops_agent.py`, `data_extraction_agent.py` each duplicate:
- `_track_usage()` — identical in all 5 agents (including orchestrator)
- `get_token_usage()` — identical in 4 sub-agents
- `process_request()` — ~80 lines duplicated 4x (only log prefix differs)
- `execute_task()` — ~60 lines duplicated 4x

**Fix:** Extract a `BaseSubAgent` class. Each sub-agent becomes ~20 lines setting categories, prompt builder, and name.

### 2. Three near-identical sandbox executors in `data_ops/custom_ops.py` (~150 lines)

`execute_custom_operation()`, `execute_dataframe_creation()`, `execute_spectrogram_computation()` share ~80% identical code (builtins setup, exec, result extraction, validation). Plus three identical `run_*` validate-then-execute wrappers.

**Fix:** One `_execute_sandboxed(namespace, code)` function. The three public functions become 5-line namespace-setup wrappers.

---

## HIGH — Dead Code

| What | Where | Notes |
|------|-------|-------|
| Entire deprecated matplotlib module | `data_ops/plotting.py` (103 lines) | Only imported by its own test. Docstring says "DEPRECATED" |
| `format_tool_result()` + 3 helpers | `agent/prompts.py` | References obsolete tool names (`plot_data`, `change_time_range`). Imported in `core.py` but never called |
| 3 dead generator functions | `knowledge/prompt_builder.py` | `generate_spacecraft_overview()`, `generate_dataset_quick_reference()`, `generate_mission_profiles()` — only called by tests |
| `get_thinking_tokens()` | `agent/thinking.py` | Never imported. Every agent has inline version in `_track_usage()` |
| `_safe_filename()` | `agent/session.py` | Defined, never called |
| `has()` and `remove()` | `data_ops/store.py` | Only called by tests, never in production |

---

## HIGH — Tests That Test Nothing Real (~50-60 tests)

### Handler re-implementation (worst offender)

`test_describe_save.py` and `test_features_01_04.py` **copy-paste handler logic from `core.py` into local helper functions**, then test those local copies. If `core.py` changes, these tests stay green while real code is broken. Affects ~25 tests.

```python
# test_describe_save.py — this is NOT calling core.py, it's a LOCAL copy
def _describe(label):
    entry = store.get(label)
    if entry is None:
        return {"status": "error", ...}
    # ... 40 lines of re-implemented logic ...
```

### Tautological Python tests

Tests that re-implement trivial logic inline and verify Python builtins work:

```python
# test_describe_save.py — tests that str.replace works
def test_auto_filename_generation(self):
    label = "AC_H2_MFI.BGSEc"
    safe_label = label.replace(".", "_").replace("/", "_")
    filename = f"{safe_label}.csv"
    assert filename == "AC_H2_MFI_BGSEc.csv"  # no application code called!
```

```python
# test_features_01_04.py — tests that str.endswith works
def test_csv_extension_handling(self):
    result = input_name
    if not result.endswith(".csv"):
        result += ".csv"
    assert result == expected  # no application code called!
```

### hasattr existence tests (~18 tests across 3 files)

```python
# Repeated in test_visualization_agent.py, test_mission_agent.py, test_routing.py
def test_has_process_request(self):
    assert hasattr(VisualizationAgent, "process_request")
    assert callable(getattr(VisualizationAgent, "process_request"))
```

These can never catch behavioral regressions. If someone deletes the method, the `ImportError` in production catches it before any test would.

---

## MEDIUM — Over-Engineered But Has Some Value

### Tests

| Pattern | Files | Count | Issue |
|---------|-------|-------|-------|
| Prompt substring tests | `test_prompt_builder.py` | ~40 | Each test calls builder + checks 1 string. Tightly coupled to prompt wording, not behavior |
| Repetitive tool-filtering | `test_routing.py`, `test_visualization_agent.py` | ~15 | Same `get_tool_schemas()` call repeated per-tool; one combined assertion would suffice |
| Mock-dominated tests | `test_cdaweb_catalog.py`, `test_features_01_04.py` | ~8 | Mock setup is 10 lines, assertion is 1 line. Tests the mock wiring, not the code |
| Schema structure tests | `test_planner.py` | 7 | Tests that a dict has certain keys — the Gemini API would reject bad schemas anyway |
| Constant-pinning tests | `test_planner.py` | 1 | `assert MAX_ROUNDS == 5` — breaks on any valid config change |

### Code

| What | Where | Issue |
|------|-------|-------|
| `create_agent()` factory | `agent/core.py:1848` | Pass-through wrapper — just call `OrchestratorAgent(...)` directly |
| `validate_plotly_code()` | `rendering/custom_viz_ops.py:21` | One-liner that calls `validate_pandas_code(code, require_result=False)` |
| Token aggregation | `agent/core.py:249-301` | 5 copy-paste blocks; should be a loop over a list of agents |
| `Task.depends_on` field | `agent/tasks.py:48` | Defined, serialized, tested — but never read by any execution logic |
| `TaskStore._plan_path()` | `agent/tasks.py` | O(n) file scan per operation; should use ID-based filenames |
| Duplicate logging | `agent/core.py:528-531` | `log_tool_call()` AND `self.logger.debug()` write the same thing |
| `format_plan_for_display()` | `agent/planner.py:294-332` | Character-for-character identical rendering in both branches |

### Bug Found

`agent/logging.py:202` — `get_recent_errors()` uses `today.replace(day=today.day - i)` which crashes across month boundaries. Should be `today - timedelta(days=i)`.

---

## Clean Files (no issues found)

These are well-designed with meaningful tests:

- **`test_store.py`** — Tests real `DataStore`/`DataEntry` behavior
- **`test_custom_ops.py`** — Excellent AST sandbox tests
- **`test_capability_boundary.py`** — Well-designed sandbox limit probing
- **`test_custom_viz_ops.py`** — Clean validation + execution tests
- **`test_session.py`** — Good round-trip persistence with actual file I/O
- **`test_spectrogram.py`** — Strong end-to-end compute/store/render coverage
- **`test_plotly_renderer.py`** — Tests actual Plotly figure construction
- **`rendering/plotly_renderer.py`** — Clean, no unnecessary abstractions
- **`data_ops/store.py`** — Simple and correct (aside from unused `has`/`remove`)
- **`knowledge/mission_loader.py`** — Clean lazy-loading cache

---

## By the Numbers

| Category | Estimated Count |
|----------|----------------|
| Meaningless tests (test nothing real) | ~50-60 of 678 |
| Over-engineered tests (marginal value) | ~40-50 |
| Dead production code lines | ~260+ |
| Duplicated production code lines | ~850+ |

The two biggest wins would be: (1) extracting a `BaseSubAgent` to eliminate ~700 lines of agent duplication, and (2) deleting dead code (`plotting.py`, dead prompt formatters, dead prompt_builder functions).
