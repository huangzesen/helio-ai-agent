# Plan: Savable & Replayable Pipelines

## Context

The helio-agent currently drives all data operations through LLM tool-calling: fetch → compute → plot. The "recipe" for any analysis exists only in the LLM's ephemeral conversation. Users who want the same plot for a different date range must re-describe the entire workflow from scratch. This leads to inconsistency (LLM may choose different parameters/styling each time) and wasted tokens.

**Goal**: Let users save a proven sequence of tool calls as a **pipeline template**, then replay it with a new date range — either deterministically (fast, consistent, no LLM) or with LLM-mediated modifications (flexible).

**Key pain point this solves**: Currently when the user asks to modify something (e.g. "change the color to red"), the LLM often replots everything from scratch with new code, frequently messing things up. With a pipeline, the LLM can see exactly which step handles styling and modify *only* that step, leaving fetch/compute steps untouched.

## Design

### Pipeline = ordered list of recorded tool calls + template variables

```json
{
  "id": "ace-bfield-overview",
  "name": "ACE B-field Overview",
  "description": "Fetch ACE mag data, compute magnitude, two-panel plot",
  "variables": {
    "$TIME_RANGE": {"type": "time_range", "default": "last 7 days"}
  },
  "steps": [
    {
      "step_id": 1,
      "tool_name": "fetch_data",
      "tool_args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc", "time_range": "$TIME_RANGE"},
      "intent": "Fetch ACE magnetic field vector in GSE",
      "produces": ["AC_H2_MFI.BGSEc"],
      "depends_on": [],
      "critical": true
    },
    {
      "step_id": 2,
      "tool_name": "custom_operation",
      "tool_args": {"source_labels": ["AC_H2_MFI.BGSEc"], "pandas_code": "...", "output_label": "ACE_Bmag"},
      "intent": "Compute scalar magnetic field magnitude",
      "produces": ["ACE_Bmag"],
      "depends_on": [1],
      "critical": true
    },
    {
      "step_id": 3,
      "tool_name": "plot_data",
      "tool_args": {"labels": "AC_H2_MFI.BGSEc,ACE_Bmag", "panels": [["AC_H2_MFI.BGSEc"], ["ACE_Bmag"]]},
      "intent": "Two-panel plot: vector components on top, magnitude on bottom",
      "depends_on": [1, 2],
      "critical": false
    },
    {
      "step_id": 4,
      "tool_name": "style_plot",
      "tool_args": {"y_label": {"1": "B (nT)", "2": "|B| (nT)"}, "trace_colors": {"ACE_Bmag": "black"}},
      "intent": "Label axes, color magnitude black",
      "depends_on": [3],
      "critical": false
    }
  ]
}
```

Key fields:
- **`intent`**: Natural-language description per step — makes pipeline LLM-readable for modifications
- **`produces`**: DataStore labels created — for dependency tracking
- **`depends_on`**: Step IDs that must succeed first
- **`critical`**: If true, failure aborts dependent steps; if false (styling), failure is logged but continues
- **`variables`**: `$TIME_RANGE` (and optionally `$MISSION`, user-defined) substituted at execution time

### Two execution modes

1. **Deterministic** (no `modifications` arg): Walk steps in order, substitute variables, call `_execute_tool()` directly. No LLM, zero tokens, consistent output.

2. **LLM-mediated** (`modifications` provided, e.g. "use red lines and add running average"): The executor runs all steps deterministically first, then hands control to the LLM with the pipeline context + user's modification request. The LLM sees the completed pipeline and applies **only the relevant changes** — e.g. calling `style_plot` with new colors, or inserting a `custom_operation` for a running average. It does NOT re-execute fetch/compute steps that don't need changing.

   The prompt explicitly instructs: *"The pipeline has been executed. The data and plot are ready. Apply ONLY the user's requested modifications — do not re-fetch data or re-plot unless the modification requires it."*

   This directly fixes the current problem where the LLM replots everything from scratch when asked for a small style change.

### Recording (always-on + smart filter)

A `PipelineRecorder` passively captures ALL tool calls in `_execute_tool_safe()` every session (zero overhead — just appending to a list). When the user says "save this as a pipeline", the LLM calls `save_pipeline` which:
1. Reads the full recording buffer
2. **LLM selects which steps to include** — filtering out exploratory/failed calls (e.g. search_datasets, failed fetches). The LLM provides the `steps` array in its `save_pipeline` tool call, cherry-picking from the recording.
3. LLM provides `intent` descriptions and `name`/`description`
4. Time ranges in `fetch_data` steps are auto-parameterized to `$TIME_RANGE`
5. Dependencies are inferred from label references

### Mission swapping

When the user says "run my ACE pipeline for Wind instead", this triggers **LLM-mediated mode**. The LLM reads the pipeline step intents (e.g. "Fetch magnetic field vector in GSE"), understands the physics intent, and substitutes equivalent Wind datasets/parameters. No pre-configured mission mapping needed — the LLM already knows spacecraft equivalences from the catalog.

## Files to Create

### `agent/pipeline.py` (new, ~250 lines)

Follows the same pattern as `agent/tasks.py` (dataclasses + JSON persistence).

```python
# Dataclasses
PipelineStep       # step_id, tool_name, tool_args, intent, produces, depends_on, critical
PipelineVariable   # type, description, default, current
Pipeline           # id, name, description, variables, steps, created_at, source_session
                   # Methods: to_dict(), from_dict(), to_llm_context()

# Storage
PipelineStore      # save/load/list/delete from ~/.helio-agent/pipelines/*.json
                   # Modeled after TaskStore in agent/tasks.py (lines 181-199)

# Recording
PipelineRecorder   # record(tool_name, tool_args, result) — passive buffer
                   # Only records: fetch_data, custom_operation, compute_spectrogram,
                   #   store_dataframe, plot_data, style_plot, manage_plot
                   # get_recording() → list[dict], clear()

# Execution
PipelineExecutor   # execute(pipeline, variable_overrides, tool_executor_fn) → result dict
                   # Walks steps, substitutes variables, calls tool_executor_fn
                   # Handles depends_on + critical failure propagation
```

## Files to Modify

### `agent/tools.py` — Add 4 tool schemas

New category: `"pipeline"`

| Tool | Required args | Optional args | Purpose |
|------|--------------|---------------|---------|
| `save_pipeline` | `name`, `description`, `steps` | `variables` | Save session's tool calls as pipeline. LLM must provide `steps` (cherry-picked from recording buffer, with intents). |
| `run_pipeline` | `pipeline_id` | `variable_overrides`, `modifications` | Execute a saved pipeline |
| `list_pipelines` | — | — | List all saved pipelines |
| `delete_pipeline` | `pipeline_id` | — | Delete a pipeline |

### `agent/core.py` — 4 changes

1. **Import** `pipeline` module (top of file, ~line 18)

2. **Add `PipelineRecorder` to `__init__()`** (~1 line):
   ```python
   self._pipeline_recorder = PipelineRecorder()
   ```

3. **Hook recording into `_execute_tool_safe()`** (line ~1875, after `result = self._execute_tool(...)`, before sanitize):
   ```python
   self._pipeline_recorder.record(tool_name, tool_args, result)
   ```

4. **Add tool handlers in `_execute_tool()`** (~line 1843, before the `else: unknown tool` block):
   ```python
   elif tool_name == "save_pipeline":
       return self._handle_save_pipeline(tool_args)
   elif tool_name == "run_pipeline":
       return self._handle_run_pipeline(tool_args)
   elif tool_name == "list_pipelines":
       return self._handle_list_pipelines(tool_args)
   elif tool_name == "delete_pipeline":
       return self._handle_delete_pipeline(tool_args)
   ```

5. **Add `"pipeline"` to `ORCHESTRATOR_EXTRA_TOOLS`** (or create a new category in `ORCHESTRATOR_CATEGORIES`) so the orchestrator LLM sees these tools.

6. **Implement 4 handler methods** on `OrchestratorAgent`:
   - `_handle_save_pipeline`: Read recording buffer or use explicit `steps`, construct Pipeline, save via PipelineStore
   - `_handle_run_pipeline`: If `modifications` → LLM-mediated (inject pipeline context, call `_process_single_message`); else → deterministic (use `PipelineExecutor`)
   - `_handle_list_pipelines`: Return PipelineStore.list_pipelines()
   - `_handle_delete_pipeline`: PipelineStore.delete()

### No changes needed to:
- Sub-agents (mission, visualization, data_ops) — they are unaware of pipelines
- Rendering layer — pipelines use the existing `_execute_tool()` dispatch
- Data ops layer — same tool interface
- Planner — orthogonal (though a completed plan can be saved as a pipeline)
- Session management — pipelines are separate persistent artifacts

## Failure Handling

| Scenario | Behavior |
|----------|----------|
| Critical step fails (e.g. dataset unavailable) | All dependent steps skipped; result reports partial completion |
| Non-critical step fails (e.g. styling) | Logged, execution continues |
| Column name changed upstream | custom_operation sandbox error caught; dependent steps skipped |
| User wants to fix a failed step | "Run pipeline with modifications: fix step 2" → LLM-mediated mode |

## Implementation Progress

**Phase 1 — MVP (deterministic replay)** — COMPLETE (commit 7286125, feature/pipeline branch)
1. ~~Create `agent/pipeline.py`~~ — Pipeline, PipelineStep, PipelineVariable, PipelineStore, PipelineRecorder, PipelineExecutor (~350 lines)
2. ~~Add 4 tool schemas to `agent/tools.py`~~ — save_pipeline, run_pipeline, list_pipelines, delete_pipeline
3. ~~Hook PipelineRecorder into `core.py:_execute_tool_safe()`~~ — single line, records after every tool call
4. ~~Add tool handlers + `_handle_save_pipeline` / `_handle_run_pipeline`~~ — 4 handlers on OrchestratorAgent
5. ~~Add pipeline tools to orchestrator tool set~~ — added "pipeline" to ORCHESTRATOR_CATEGORIES

**Phase 2 — LLM-mediated replay** — COMPLETE (same commit)
1. ~~Implement `Pipeline.to_llm_context()`~~ — human-readable formatting for LLM context injection
2. ~~Implement modifications path in `_handle_run_pipeline()`~~ — runs deterministic first, then LLM applies only requested changes
3. "Save modified pipeline" flow — available via existing save_pipeline (user can save after modifications)

**Phase 3 — Unified Plot Spec** — COMPLETE
1. ~~`render_from_spec()` on PlotlyRenderer~~ — takes a single spec dict + DataEntry list, calls plot_data + style in one shot
2. ~~`render_spec` tool schema + handler in core.py~~ — resolves labels from DataStore, dispatches to renderer
3. ~~`merge_plot_steps()` in pipeline.py~~ — auto-merges consecutive plot_data + style_plot into render_spec during save
4. ~~Updated `_handle_run_pipeline` modifications prompt~~ — guides LLM to use style_plot for style-only changes (no re-fetch)
5. ~~`render_spec` added to RECORDABLE_TOOLS~~

**Phase 4 — Polish** — NOT STARTED
1. Add pipeline listing to Gradio sidebar
2. Pipeline export/import for sharing

**Tests** — 54 pipeline tests + 7 render_from_spec tests, all passing. Covers:
- PipelineStore: save/load/list/delete round-trip, overwrite, nonexistent
- PipelineExecutor: variable substitution, dependency ordering, critical/non-critical failure propagation, transitive skip, exceptions
- PipelineRecorder: recording filter (only recordable tools), error filtering, deep copy, clear
- Dataclass round-trips: PipelineStep, Pipeline, PipelineVariable
- Variable substitution: strings, dicts, lists, nested, multiple vars, scalars
- Slugify: basic, special chars, empty, whitespace
- merge_plot_steps: plot_data alone → render_spec, plot_data + style_plot merged, multiple style_plot merged, style overrides plot fields, mixed steps preserve order, depends_on remapped, internal deps removed, intents combined, two separate groups, empty input
- render_from_spec: basic spec, panels + style, equivalence with plot_data + style, trace colors, empty entries error, title override, vlines

## Verification

1. **Unit tests** (no API key needed): `venv/bin/python -m pytest tests/test_pipeline.py -v`

2. **Integration test** (requires API key):
   - Start agent → "Show me ACE magnetic field for last week" → "Save this as a pipeline called ACE overview" → reset session → "Run my ACE overview pipeline for January 2026" → verify plot produced
   - "Run my ACE overview but with red magnitude line" → verify LLM modifies style_plot step

3. **Manual check**:
   - `ls ~/.helio-agent/pipelines/` → verify JSON files created
   - Read the JSON → verify structure matches schema above
   - Run pipeline twice with same time range → verify identical output (consistency goal)
