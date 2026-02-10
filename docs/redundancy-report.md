# Code Redundancy Report

**Date**: 2026-02-08 (updated 2026-02-09 — marked resolved items after BaseSubAgent extraction)
**Scope**: Full codebase (active source files, excluding tests and deleted `autoplot_bridge/`)
**Estimated total refactoring effort**: 12-18 hours

---

## Summary

9 findings across the codebase. Most redundancy stems from rapid prototyping that created similar patterns in parallel modules (`core.py` vs `mission_agent.py` vs `data_ops_agent.py` vs `data_extraction_agent.py` vs `visualization_agent.py`) and lack of shared abstractions for common patterns (singletons, Gemini interactions, token tracking).

All identified redundancy is safe to consolidate without breaking functionality.

### Findings by Severity

| Severity | Count | Estimated Hours |
|----------|-------|-----------------|
| High     | 2     | 7-10            |
| Medium   | 3     | 3-5             |
| Low      | 4     | 2-3             |

---

## High-Impact Findings

### 1. Duplicate Task Execution Logic — **RESOLVED**

> **RESOLVED (2026-02-09):** `BaseSubAgent` in `agent/base_agent.py` provides shared `execute_task()`. All four sub-agents inherit it.

**Files**:
- `agent/core.py::_execute_task()`
- `agent/mission_agent.py::execute_task()`

**Description**: Both agents have nearly identical `execute_task()` methods with the same structure: task status management, fresh chat creation, tool call loop with `mode="ANY"`, duplicate detection, iteration limits, and error handling. ~120 lines duplicated.

**Fix**: Extract shared logic into a base class or mixin.

**Effort**: 4-6 hours

---

### 2. Tool Call Loop Duplication — **RESOLVED**

> **RESOLVED (2026-02-09):** `BaseSubAgent` in `agent/base_agent.py` provides shared `process_request()` with the tool-calling loop, duplicate detection, and consecutive error tracking. All four sub-agents inherit it.

**Files**:
- `agent/core.py::_process_single_message()`
- `agent/mission_agent.py::process_request()`
- `agent/data_ops_agent.py::process_request()`
- `agent/data_extraction_agent.py::process_request()`
- `agent/visualization_agent.py::process_request()`

**Description**: All five agents have nearly identical tool-calling loops: extract function_calls from response parts, execute each tool, build function_responses, send back to model. ~80 lines duplicated across 5 files.

**Fix**: Extract into a shared base class.

**Effort**: 3-4 hours

---

## Medium-Impact Findings

### 3. Token Usage Tracking Duplication — **RESOLVED**

> **RESOLVED (2026-02-09):** `BaseSubAgent` provides shared `_track_usage()` and `get_token_usage()`. All four sub-agents inherit them.

**Files**:
- `agent/core.py::_track_usage()` / `get_token_usage()`
- `agent/mission_agent.py::_track_usage()` / `get_token_usage()`
- `agent/data_ops_agent.py::_track_usage()` / `get_token_usage()`
- `agent/data_extraction_agent.py::_track_usage()` / `get_token_usage()`
- `agent/visualization_agent.py::_track_usage()` / `get_token_usage()`

**Description**: All five agents have identical token tracking methods.

**Fix**: Create a `TokenTrackingMixin` base class.

**Effort**: 1-2 hours

---

### 4. Singleton Pattern Duplication

**Files**:
- `data_ops/store.py` (DataStore singleton)
- `agent/tasks.py` (TaskStore singleton)

**Description**: The singleton pattern is implemented identically in 2 modules: module-level `_variable: Optional[Type] = None`, getter function, and reset function.

**Fix**: Create a shared singleton decorator or factory.

**Effort**: 1-2 hours

---

### 5. Dead Code: Matplotlib Plotting Module

**File**: `data_ops/plotting.py`

**Description**: Entire module is unused — never imported anywhere. All plotting goes through the Plotly renderer (`rendering/plotly_renderer.py`).

**Fix**: Delete the file.

**Effort**: 15 minutes

---

## Low-Impact Findings

### 6. Repeated Error Result Dicts

**File**: `agent/core.py::_execute_tool()` (12+ occurrences)

`{"status": "error", "message": ...}` and `{"status": "success", ...}` constructed manually everywhere.

**Fix**: Add `_error_result(msg)` and `_success_result(**kwargs)` helper methods.

**Effort**: 30 minutes

---

### 7. Gemini Text Extraction (multiple occurrences) — **RESOLVED**

> **RESOLVED (2026-02-09):** Text extraction handled in `BaseSubAgent.process_request()`.

**Files**: `agent/core.py`, `agent/mission_agent.py`, `agent/data_ops_agent.py`, `agent/data_extraction_agent.py`, `agent/visualization_agent.py`

Same pattern for extracting text parts from a Gemini response repeated across all agent files.

**Fix**: Create `extract_text_from_response(response, default="Done.")` utility.

**Effort**: 30 minutes

---

### 8. Keyword Mapping Duplication

**File**: `knowledge/mission_loader.py::get_routing_table()`

Hardcoded keyword-to-capability mappings that could be centralized.

**Fix**: Extract keyword-to-category mappings into a shared config.

**Effort**: 1 hour

---

### 9. Verbose Flag Propagation — **RESOLVED**

> **RESOLVED (2026-02-09):** `BaseSubAgent.__init__()` handles `self.verbose` for all sub-agents.

**Files**: `agent/core.py`, `agent/mission_agent.py`, `agent/data_ops_agent.py`, `agent/data_extraction_agent.py`, `agent/visualization_agent.py`

All five agent classes store `self.verbose = verbose` and use it for conditional printing.

**Fix**: Create a `VerboseLoggingMixin` with a shared `log()` method, or move into a base class.

**Effort**: 30 minutes

---

## Recommended Refactoring Plan

### Phase 1: Quick Wins (1-2 hours)

1. Delete `data_ops/plotting.py` (Finding 5)
2. Add `_error_result()` / `_success_result()` helpers (Finding 6)
3. Add `extract_text_from_response()` utility (Finding 7)

### Phase 2: Agent Base Class (6-10 hours) — **DONE**

> **Completed (2026-02-09):** `BaseSubAgent` extracted in `agent/base_agent.py`. All four sub-agents inherit shared logic. Findings 1, 2, 3, 7, and 9 resolved.

```
BaseSubAgent (agent/base_agent.py)
  - _track_usage() / get_token_usage()     (Finding 3)
  - process_request() with tool call loop  (Finding 2)
  - execute_task()                         (Finding 1)
  - text extraction                        (Finding 7)
  - verbose logging                        (Finding 9)
  |
  +-- MissionAgent (mission specialist)
  +-- DataOpsAgent (data ops specialist)
  +-- DataExtractionAgent (data extraction specialist)
  +-- VisualizationAgent (visualization specialist)
```

### Phase 3: Shared Utilities (2-3 hours)

1. Singleton decorator in a shared utility (Finding 4)
2. Keyword-to-capability mappings (Finding 8)

### Phase 4: Validation

- Run full test suite after each phase (632 tests across 24 files)
- Verify no behavior changes (pure refactoring)
- Update `docs/capability-summary.md` with new architecture
