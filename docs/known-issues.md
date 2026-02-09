# Known Issues

Tracking bugs and issues in helio-ai-agent.

**Last updated**: February 2026

---

## Open Issues

### 1. OMNI data parsing fails for recent time ranges

**Status**: Open
**Severity**: Medium
**Component**: `data_ops/fetch.py`

**Description**: Fetching OMNI_HRO_1MIN data for "last 3 days" fails with:
```
could not convert string to float: ''
```

The HAPI CSV response contains empty strings that can't be parsed as floats. "Last week" works, suggesting recent data has gaps.

**Workaround**: Use "last week" or older time ranges.

**Fix needed**: Handle empty strings in CSV parsing (treat as NaN).

---

### 2. Gemini doesn't follow task instructions precisely

**Status**: Mitigated
**Severity**: Low
**Component**: `agent/core.py`, `agent/planner.py`, `agent/base_agent.py`

**Description**: During multi-step task execution, Gemini sometimes calls extra tools not requested in the task instruction. For example:
- Task: "Search for flares" -> Gemini also creates DataFrames and plots bar charts
- Task: "Find date" -> Gemini also delegates to visualization

This happens despite using `tool_config mode="ANY"` which forces function calling.

**Mitigation (2026-02-09)**: Added "CRITICAL: Do ONLY what the instruction says" prompt constraints to both `_execute_task()` (core.py) and `_get_task_prompt()` (base_agent.py). Also added anti-duplication rules 8-10 to the planner prompt. This significantly reduces over-execution but Gemini may still occasionally add extra steps.

---

### 3. Multi-step tasks lose context between steps

**Status**: Open
**Severity**: Medium
**Component**: `agent/core.py`

**Description**: Each task in a multi-step plan uses a fresh chat session (to force function calling). This means:
- Task 2 doesn't know what Task 1 fetched
- Gemini may re-fetch data or ask for parameters already specified

The task instruction includes the expected labels (e.g., "compute running average of OMNI_HRO_1MIN.flow_speed"), but Gemini doesn't have context that this data was just fetched.

**Workaround**: Be explicit in task instructions about data labels.

**Fix needed**: Pass task results/context to subsequent tasks, or use a shared memory description.

---

## Resolved Issues

### Fixed in 2026-02-09 LoopGuard batch

| Issue | Description | Fix |
|-------|-------------|-----|
| VisualizationAgent infinite loop | Export task called `reset()` and `get_plot_state()` hundreds of times in alternation, never exporting | Created shared `LoopGuard` class (`agent/loop_guard.py`) with 3-layer protection: hard total-call limit, subset-based duplicate detection, cycle detection. Applied to all 5 agents (10 loops). |
| Inconsistent call fingerprinting | Old `str(sorted(dict(fc.args).items()))` produced non-deterministic keys from Protobuf Struct objects | `make_call_key()` uses `json.dumps(sort_keys=True)` for deterministic serialization |
| Viz tasks call wrong tools | Planner generated vague instructions like "Plot X together", VisualizationAgent called reset/get_plot_state instead of plot_stored_data | (1) Explicit tool-call guidance in `execute_task` prompt, (2) planner examples now start with tool name ("Use plot_stored_data to..."), (3) viz prompt differentiates conversational vs task-execution workflow |
| `describe_data` crashes on string columns | `KeyError: 'min'` when describing DataFrames with non-numeric columns (e.g., flare class strings) | Added dtype check: numeric columns get full stats, string/object columns get count/unique/top |
| Massive Plotly error messages waste context | `custom_visualization` errors returned full Plotly property list (1000+ lines) | Truncate RuntimeError messages to 500 chars |
| Planner generates duplicate tasks | Round 3 re-created tasks from Rounds 1-2 because result summaries said "Done." with no details | (1) Enriched result summaries to include tool calls list when text is empty, (2) added planner rules 8-10 prohibiting task repetition |
| Tasks over-execute (do more than asked) | "Search for flares" task also created DataFrames and plotted bar charts, wasting 3-4 of 5 iterations | Added "CRITICAL: Do ONLY what the instruction says" constraint to all task execution prompts |
| Viz agent calls reset/get_plot_state instead of plotting | VisualizationAgent ignored "Do NOT call reset" prompt; called plot_stored_data with empty labels | (1) `_get_task_prompt` now injects actual labels from data store, (2) `_execute_plan_task` prepends "Use plot_stored_data to plot LABEL1,LABEL2" when planner omits labels |
| Export tasks waste viz agent iterations | Export routed to VisualizationAgent which called reset/get_plot_state instead of exporting | Export tasks now intercepted by orchestrator's `_handle_export_task()` — direct dispatch, no LLM needed |

### Fixed in 2026-02-08 stability batch

| Issue | Description | Fix |
|-------|-------------|-----|
| Agent loops forever | Requesting data outside available time range caused ~100 API calls (10 orchestrator × 10 mission agent iterations) | Reduced sub-agent iterations 10→5, added duplicate call detection, consecutive error tracking (break after 2), "STOP: Do not retry" in error messages |
| Duplicate Plotly labels | `process_request()` in VisualizationAgent and DataOpsAgent lacked duplicate call detection, causing double-plotting | Added `previous_calls` tracking to all sub-agent `process_request()` methods |
| Black Plotly plots | Gradio dark theme CSS inherited into Plotly figures, making plots appear black | Added explicit `_DEFAULT_LAYOUT` (white backgrounds, dark font) in `plotly_renderer.py` |
| Time range out of bounds | Fetching data for dates outside dataset availability caused errors or empty results | Auto-clamp time ranges to dataset availability window in `_validate_time_range()` |

### Fixed in 2026-02-07 refactor batch

| Issue | Description | Fix |
|-------|-------------|-----|
| Autoplot bridge removed | Entire `autoplot_bridge/` package with JPype/Java dependency | Replaced with pure-Python Plotly renderer (`rendering/`) |
| 10 thin viz wrappers | `set_title`, `set_axis_label`, `toggle_log_scale`, etc. | Replaced by single `custom_visualization` tool (Plotly sandbox) |
| JVM crashes | 4+ panels crashed JVM, `waitUntilIdle` race condition | No longer applicable — Plotly has no JVM |
| Relative paths rejected | `export_png/pdf` with relative paths failed | Resolve to absolute with `Path.resolve()` |
| DatetimeIndex for rolling | Time-based rolling windows (`'2H'`) failed | Ensure DatetimeIndex before executing code |
| MMS @0/@1 naming | MMS datasets not found | Updated JSON with @0 suffixed IDs |
| Unnecessary clarification | Agent asked when dataset+param provided | Strengthened "Do NOT ask" prompt rules |
| Stored labels unclear | Downstream agents couldn't find labels | Mission agent now reports exact stored labels |
| STEREO-A dataset | STA_L2_MAG_RTN not in HAPI | Updated to STA_L1_MAG_RTN + STA_L2_MAGPLASMA_1M |
| Wind high-cadence | WI_H2_MFI fetches too slow for multi-day | Added WI_H0_MFI (1-min) with guidance |

### Fixed in commit e58be47 (2026-02-05)

| Issue | Description | Fix |
|-------|-------------|-----|
| Infinite recursion | `_execute_tool_safe` called itself | Changed to call `_execute_tool` |
| Tasks not calling tools | Chat context pollution prevented function calling | Fresh chat per task with `mode="ANY"` |
| Unicode encoding error | Windows can't display special characters | Use ASCII characters instead |
| NoneType iteration | `response.candidates[0].content.parts` was None | Added None check before iteration |
| Clarification loops | `mode="ANY"` forced endless `ask_clarification` calls | Break on clarification, limit to 3 iterations |

---

## Reporting New Issues

When adding issues, include:
- **Status**: Open / In Progress / Resolved
- **Severity**: Critical / High / Medium / Low
- **Component**: Which file(s) are affected
- **Description**: What happens, error messages
- **Workaround**: Any temporary fix
- **Fix needed**: What should be done
