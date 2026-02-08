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

**Status**: Open
**Severity**: Low
**Component**: `agent/core.py`, `agent/planner.py`

**Description**: During multi-step task execution, Gemini sometimes calls extra tools not requested in the task instruction. For example:
- Task: "Fetch data" -> Gemini calls `fetch_data` then also tries to plot
- Task: "Compute running average" -> Gemini asks for clarification instead

This happens despite using `tool_config mode="ANY"` which forces function calling.

**Workaround**: None currently. Tasks still complete but may have unexpected side effects.

**Fix needed**:
- More explicit task instructions from planner
- Or parse task instruction to validate which tools should be called

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
