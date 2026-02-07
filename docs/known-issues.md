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

### 2. JVM not running for multi-step plot_computed_data

**Status**: Open
**Severity**: Medium
**Component**: `autoplot_bridge/`, `agent/core.py`

**Description**: When running multi-step tasks, `plot_computed_data` fails with:
```
Java Virtual Machine is not running
```

This happens because each task uses a fresh chat session, and the JVM is only started when `plot_data` is called (which initializes Autoplot). The `plot_computed_data` tool expects the JVM to already be running.

**Workaround**: Run a simple `plot_data` command first to start the JVM, then run the multi-step request.

**Fix needed**: Either:
- Lazy-initialize JVM in `plot_computed_data` if not running
- Or start JVM at agent initialization (slower startup)

---

### 3. Gemini doesn't follow task instructions precisely

**Status**: Open
**Severity**: Low
**Component**: `agent/core.py`, `agent/planner.py`

**Description**: During multi-step task execution, Gemini sometimes calls extra tools not requested in the task instruction. For example:
- Task: "Fetch data" → Gemini calls `fetch_data` then also tries `plot_computed_data`
- Task: "Compute running average" → Gemini asks for clarification instead

This happens despite using `tool_config mode="ANY"` which forces function calling.

**Workaround**: None currently. Tasks still complete but may have unexpected side effects.

**Fix needed**:
- More explicit task instructions from planner
- Or parse task instruction to validate which tools should be called

---

### 4. Multi-step tasks lose context between steps

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

### 5. Autoplot `waitUntilIdle` race condition in headless mode

**Status**: Open (upstream)
**Severity**: Low
**Component**: Autoplot (upstream) — `DasCanvas.waitUntilIdle()`

**Description**: Autoplot intermittently logs:
```
INFO: strange bug where update event didn't clear dirty flags, reposting.
```
This occurs during multi-panel or overlay plots in headless mode. The race condition is in Autoplot's internal rendering thread synchronization (`DasCanvas.waitUntilIdle()`), not in our bridge code.

**Impact**: Usually self-recovers, but may contribute to JVM crashes when creating 4+ panels rapidly (see ISSUE-01 in test log).

**Workaround**: None (upstream issue). Our 3-panel maximum guard mitigates the worst case.

**Fix needed**: Upstream Autoplot fix. Monitor for Autoplot updates.

---

## Resolved Issues

### Fixed in 2026-02-07 bug fix batch

| Issue | Description | Fix |
|-------|-------------|-----|
| Relative paths rejected | `export_png/pdf`, `save_session` with relative paths failed | Resolve to absolute with `Path.resolve()` |
| Render type mapping | `fill_to_zero`, `staircase`, etc. rejected by Java | Map snake_case to camelCase enum values |
| DOM title API | Agent used non-existent `dom.setTitle()` | Updated prompt: title is on `dom.getPlots(i)` |
| Color table on line plots | Setting color table on non-spectrogram silently failed | Guard checks render type first |
| DatetimeIndex for rolling | Time-based rolling windows (`'2H'`) failed | Ensure DatetimeIndex before executing code |
| CDAWeb param crash | Invalid parameter crashed JVM | Validate via HAPI before calling Autoplot |
| to_qdataset vectors | Vector data failed with unhelpful error | Improved error with component examples |
| 4-panel JVM crash | 4+ panels crashed JVM | 3-panel max guard in commands + script_runner |
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
| Unicode encoding error | Windows can't display `○✓✗` characters | Use ASCII `o+x-` instead |
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
