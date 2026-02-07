# Code Redundancy Report

**Date**: 2026-02-06
**Scope**: Full codebase (20 source files, excluding tests)
**Estimated total refactoring effort**: 25-36 hours

---

## Summary

14 findings across the codebase. Most redundancy stems from rapid prototyping that created similar patterns in parallel modules (`core.py` vs `mission_agent.py`) and lack of shared abstractions for common patterns (singletons, Gemini interactions, token tracking).

All identified redundancy is safe to consolidate without breaking functionality.

### Findings by Severity

| Severity | Count | Estimated Hours |
|----------|-------|-----------------|
| High     | 3     | 10-15           |
| Medium   | 4     | 10-14           |
| Low      | 7     | 5-7             |

---

## High-Impact Findings

### 1. Singleton Pattern Duplication

**Files**:
- `data_ops/store.py` (lines 100-115)
- `agent/tasks.py` (lines 326-342)
- `autoplot_bridge/commands.py` (lines 365-374)

**Description**: The singleton pattern is implemented identically in 3 modules: module-level `_variable: Optional[Type] = None`, getter function, and reset function.

```python
# data_ops/store.py
_store: Optional[DataStore] = None

def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
    return _store

def reset_store() -> None:
    global _store
    _store = None

# agent/tasks.py - NEARLY IDENTICAL
_store: Optional[TaskStore] = None

def get_task_store() -> TaskStore:
    global _store
    if _store is None:
        _store = TaskStore()
    return _store

def reset_task_store():
    global _store
    _store = None

# autoplot_bridge/commands.py - SAME PATTERN
_commands = None

def get_commands(verbose: bool = False) -> AutoplotCommands:
    global _commands
    if _commands is None:
        _commands = AutoplotCommands(verbose=verbose)
    return _commands
```

**Fix**: Create a shared singleton decorator or factory in a `utils/patterns.py` module:
```python
def singleton(cls):
    _instances = {}
    def get_instance(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    def reset():
        _instances.pop(cls, None)
    get_instance.reset = reset
    return get_instance
```

**Effort**: 2-3 hours

---

### 2. Duplicate Task Execution Logic

**Files**:
- `agent/core.py::_execute_task()` (lines 547-683)
- `agent/mission_agent.py::execute_task()` (lines 222-348)

**Description**: Both agents have nearly identical `execute_task()` methods with the same structure: task status management, fresh chat creation, tool call loop with `mode="ANY"`, duplicate detection, iteration limits, and error handling. ~120 lines duplicated.

```python
# Both have this identical structure:
def _execute_task(self, task: Task) -> str:
    task.status = TaskStatus.IN_PROGRESS
    task.tool_calls = []

    # Fresh chat with mode="ANY"
    task_config = types.GenerateContentConfig(
        system_instruction=get_system_prompt(),
        tools=[...],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )
    task_chat = self.client.chats.create(...)
    response = task_chat.send_message(f"Execute this task: {task.instruction}")

    # Tool call loop with duplicate detection
    max_iterations = 3
    iteration = 0
    previous_calls = set()
    # ... (same loop structure in both files)
```

**Fix**: Extract shared logic into a base class or mixin:
```python
class TaskExecutorMixin:
    def execute_task_with_tools(self, task, config, tool_executor):
        # Shared implementation
        pass
```

**Effort**: 4-6 hours

---

### 3. Tool Call Loop Duplication

**Files**:
- `agent/core.py::_process_single_message()` (lines 876-934)
- `agent/mission_agent.py::process_request()` (lines 138-197)

**Description**: Both methods have nearly identical tool-calling loops: extract function_calls from response parts, execute each tool, build function_responses, send back to model. ~80 lines duplicated.

```python
# Identical in both files:
while iteration < max_iterations:
    iteration += 1

    parts = (
        response.candidates[0].content.parts
        if response.candidates and response.candidates[0].content
        else None
    )
    if not parts:
        break

    function_calls = []
    for part in parts:
        if hasattr(part, "function_call") and part.function_call and part.function_call.name:
            function_calls.append(part.function_call)

    if not function_calls:
        break

    function_responses = []
    for fc in function_calls:
        tool_name = fc.name
        tool_args = dict(fc.args) if fc.args else {}
        result = self.tool_executor(tool_name, tool_args)
        function_responses.append(types.Part.from_function_response(...))

    response = chat.send_message(message=function_responses)
    self._track_usage(response)
```

**Fix**: Extract into a shared utility:
```python
# agent/tool_loop.py
def execute_tool_call_loop(chat, initial_response, tool_executor,
                           track_usage_fn, max_iterations=10, verbose=False):
    """Generic tool-calling loop for Gemini chat sessions."""
    pass
```

**Effort**: 3-4 hours

---

## Medium-Impact Findings

### 4. Token Usage Tracking Duplication

**Files**:
- `agent/core.py::_track_usage()` (lines 97-103)
- `agent/core.py::get_token_usage()` (lines 105-123)
- `agent/mission_agent.py::_track_usage()` (lines 84-90)
- `agent/mission_agent.py::get_token_usage()` (lines 92-99)

**Description**: Both agents have identical token tracking methods.

```python
# Identical in both files:
def _track_usage(self, response):
    meta = getattr(response, "usage_metadata", None)
    if meta:
        self._total_input_tokens += getattr(meta, "prompt_token_count", 0) or 0
        self._total_output_tokens += getattr(meta, "candidates_token_count", 0) or 0
    self._api_calls += 1
```

**Fix**: Create a `TokenTrackingMixin` base class.

**Effort**: 1-2 hours

---

### 5. Time Range Validation Pattern

**Files**:
- `agent/core.py::_execute_tool()` plot_data handler (lines 234-250)
- `agent/core.py::_execute_tool()` fetch_data handler (lines 301-335)

**Description**: Identical parse-validate-error flow repeated in two tool handlers.

```python
# Repeated in both plot_data and fetch_data handlers:
try:
    time_range = parse_time_range(tool_args["time_range"])
except TimeRangeError as e:
    return {"status": "error", "message": str(e)}
validation = self._validate_time_range(
    tool_args["dataset_id"], time_range.start, time_range.end
)
if validation and validation.startswith("No data available"):
    return {"status": "error", "message": validation}
# ... do work ...
if validation:
    result["warning"] = validation
```

**Fix**: Create a helper method:
```python
def _parse_and_validate_time_range(self, dataset_id, time_range_str):
    """Returns (TimeRange, warning_or_None). Raises ValueError on failure."""
    time_range = parse_time_range(time_range_str)
    validation = self._validate_time_range(dataset_id, time_range.start, time_range.end)
    if validation and validation.startswith("No data available"):
        raise ValueError(validation)
    return time_range, validation
```

**Effort**: 1-2 hours

---

### 6. Dead Code: Matplotlib Plotting Module

**File**: `data_ops/plotting.py` (103 lines)

**Description**: Entire module marked DEPRECATED, never imported anywhere. All plotting goes through Autoplot via `autoplot_bridge.commands`.

**Fix**: Delete the file.

**Effort**: 15 minutes

---

### 7. Keyword Mapping Duplication

**File**: `knowledge/mission_loader.py::get_routing_table()` (lines 98-106)

**Description**: Hardcoded keyword-to-capability mappings that are conceptually similar to logic in `catalog.py`.

```python
for kw in inst.get("keywords", []):
    if kw in ("magnetic", "field", "mag", "b-field", "bfield", "imf", "mfi", "fgm", "impact"):
        capabilities.add("magnetic field")
    elif kw in ("plasma", "solar wind", "proton", "density", "velocity", ...):
        capabilities.add("plasma")
```

**Fix**: Extract keyword-to-category mappings into a shared config:
```python
# knowledge/keyword_mappings.py
KEYWORD_TO_CAPABILITY = {
    "magnetic field": ["magnetic", "field", "mag", "b-field", ...],
    "plasma": ["plasma", "solar wind", "proton", "density", ...],
}
```

**Effort**: 1-2 hours

---

## Low-Impact Findings

### 8. Repeated Error Result Dicts

**File**: `agent/core.py::_execute_tool()` (12+ occurrences)

`{"status": "error", "message": ...}` and `{"status": "success", ...}` constructed manually everywhere.

**Fix**: Add `_error_result(msg)` and `_success_result(**kwargs)` helper methods.

**Effort**: 30 minutes

---

### 9. Gemini Text Extraction (5 occurrences)

**Files**: `agent/core.py` (3 places), `agent/mission_agent.py` (2 places)

Same pattern for extracting text parts from a Gemini response repeated 5 times.

```python
text_parts = []
parts = (
    response.candidates[0].content.parts
    if response.candidates and response.candidates[0].content
    else None
)
if parts:
    for part in parts:
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
return "\n".join(text_parts) if text_parts else "Done."
```

**Fix**: Create `extract_text_from_response(response, default="Done.")` utility.

**Effort**: 30 minutes

---

### 10. Filepath Normalization Duplication

**Files**: `agent/core.py` (lines 260-263) and `autoplot_bridge/commands.py` (lines 148-149)

Both add `.png` extension if missing. The bridge layer should own all normalization.

**Fix**: Remove `.png` check from `core.py`; let `commands.py` handle it.

**Effort**: 15 minutes

---

### 11. Cache Pattern Variations

**Files**: `knowledge/hapi_client.py`, `knowledge/mission_loader.py`, `agent/core.py`

Three different caching patterns for similar purposes (memoization of lookups).

**Fix**: Standardize on one pattern or use `functools.lru_cache` where applicable.

**Effort**: 1-2 hours

---

### 12. Verbose Flag Propagation

**Files**: `agent/core.py`, `agent/mission_agent.py`, `autoplot_bridge/commands.py`

All three classes store `self.verbose = verbose` and use it for conditional printing.

**Fix**: Create a `VerboseLoggingMixin` with a shared `log()` method.

**Effort**: 1 hour

---

### 13. String-to-Enum Conversion Without Error Handling

**File**: `agent/tasks.py` (lines 77, 131)

`TaskStatus(data["status"])` and `PlanStatus(data["status"])` can raise `ValueError` on invalid strings.

**Fix**: Add try-except with fallback to default status.

**Effort**: 30 minutes

---

## Recommended Refactoring Plan

### Phase 1: Quick Wins (2-3 hours)

1. Delete `data_ops/plotting.py` (Finding 6)
2. Add `_error_result()` / `_success_result()` helpers (Finding 8)
3. Add `extract_text_from_response()` utility (Finding 9)
4. Remove duplicate `.png` normalization from `core.py` (Finding 10)

### Phase 2: Agent Base Class (8-12 hours)

Create a `BaseAgent` class that both `AutoplotAgent` and `MissionAgent` inherit from:

```
BaseAgent
  - _track_usage() / get_token_usage()     (Finding 4)
  - execute_tool_call_loop()               (Finding 3)
  - execute_task()                         (Finding 2)
  - extract_text_from_response()           (Finding 9)
  - verbose logging                        (Finding 12)
  |
  +-- AutoplotAgent (main orchestrator)
  +-- MissionAgent (mission specialist)
```

This single refactor addresses findings 2, 3, 4, 9, and 12 together.

### Phase 3: Shared Utilities (5-7 hours)

1. Singleton decorator in `utils/patterns.py` (Finding 1)
2. Time range parse+validate helper (Finding 5)
3. Keyword-to-capability mappings (Finding 7)
4. Standardize caching pattern (Finding 11)
5. Enum conversion safety (Finding 13)

### Phase 4: Validation

- Run full test suite after each phase
- Verify no behavior changes (pure refactoring)
- Update `docs/capability-summary.md` with new architecture
