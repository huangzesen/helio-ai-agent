# Bug Summary Report
**Date range**: 2026-02-14 12:04:21 to 12:39:29 (35 minutes)
**Log sources**: Local (`~/.helio-agent/logs/agent_20260214_120421.log`)
**Generated**: 2026-02-14

## Executive Summary
Analysis of a 413KB log (3,609 lines) from a 35-minute session reveals **4 recurring critical bugs** affecting visualization agent reliability and data fetch operations. The session involved Voyager 1 heliopause crossing analysis with multiple planning rounds. Key findings: visualization agent repeatedly returns empty Plotly JSON (4 failures), LLM API response times are extremely slow (73 timeout warnings, 1 actual 60s timeout), data quality issues with NaN-filled parameters, and a data shape mismatch bug in CDF fetch.

## Critical & High Priority Issues

### Issue 1: Visualization Agent Returns Empty Plotly JSON
- **Severity**: Critical
- **Source**: Lines 625-634, 1235-1248, 2032-2045, 2888-2900
- **First seen**: 2026-02-14 12:11:05
- **Frequency**: 4 occurrences (100% failure rate in first attempts before think→execute retry)
- **Description**: VisualizationAgent calls `render_plotly_json` with an empty `figure_json` object (`{}`), causing validation error: "figure_json.data is required (array of traces)". This happens on the *first* render attempt in every visualization task. The agent then hits the duplicate call guard and the task fails.
- **Stack trace**: None (validation error, not exception)
- **Pattern**:
  ```
  Tool call: render_plotly_json({'figure_json': {}})
  ERROR: figure_json.data is required (array of traces)
  [Visualization Agent] Stopping: duplicate tool calls detected
  [Visualization Agent] failed: <task_name>
  ```
- **Likely root cause**: Gemini 3 Flash Preview model is generating malformed tool calls when first invoked as VisualizationAgent. The empty dict `{}` suggests the LLM is hallucinating the schema or experiencing context confusion. This may be related to the long LLM response times (20-30s) seen immediately before these failures.
- **Suggested fix**:
  1. **Short-term**: Add validation in `visualization_agent.py` to detect empty `figure_json` before calling the tool, and retry with explicit schema reminder in the prompt.
  2. **Medium-term**: Implement the think→execute retry loop already added for other agents (see commit 7ae2e7f). The log shows successful renders later in the session (lines 3597-3600), suggesting retries work.
  3. **Long-term**: Investigate if switching to Gemini 2.5 Pro or adding function calling examples reduces this error.
- **Known issue?**: No — not listed in `docs/known-issues.md`. This is a **new bug** that should be added.

### Issue 2: Excessive LLM API Response Times
- **Severity**: High
- **Source**: Throughout log
- **First seen**: 2026-02-14 12:04:44 (Orchestrator, first user message)
- **Frequency**: 73 "not responding after Xs" warnings, 1 actual 60s timeout (line 842)
- **Description**: LLM API calls consistently take 10-60 seconds to respond. Pattern: most calls trigger "not responding after 10s" warnings, many extend to 20s-30s. One Visualization Agent call timed out after 60s and required retry. This is **much slower than normal** (typical Gemini Flash responses: 1-5s).
- **Timeline of worst case**:
  ```
  12:12:00 | WARNING | [Visualization Agent] not responding after 10s
  12:12:10 | WARNING | not responding after 10s (attempt 1)
  12:12:20 | WARNING | not responding after 20s (attempt 1)
  12:12:30 | WARNING | not responding after 30s (attempt 1)
  12:12:40 | WARNING | not responding after 40s (attempt 1)
  12:12:50 | WARNING | not responding after 50s (attempt 1)
  12:13:00 | WARNING | LLM API timed out after 60s, retrying (1/2)
  ```
- **Likely root cause**:
  1. **Gemini API backend issues** on 2026-02-14 midday — possible rate limiting, regional latency, or service degradation.
  2. **Large context windows** — system prompt is ~12K tokens and grows with conversation history. By end of session: 924K input tokens cumulative (line 156 of token log).
  3. **Thinking budget** — Flash Preview uses extended thinking on complex requests. 7,475 thinking tokens accumulated (line 156), adding latency.
- **Suggested fix**:
  1. Add retry backoff strategy with exponential delays (currently retries immediately).
  2. Implement Gemini context caching (90% discount on repeated prompt tokens, reduces latency).
  3. Monitor API status before long sessions: check `https://status.cloud.google.com/`.
  4. Consider degrading to Flash (non-preview) if timeouts exceed threshold.
- **Known issue?**: No — timeout *handling* exists, but consistent 10-30s response times are abnormal and should be investigated.

### Issue 3: VOYAGER1 `protonDensity` Parameter Returns All NaN
- **Severity**: Medium (data quality, not code bug)
- **Source**: Lines 243-249, 444-451
- **First seen**: 2026-02-14 12:07:03
- **Frequency**: 2 fetch attempts (both failed)
- **Description**: Fetching `protonDensity` from `VOYAGER1_COHO1HR_MERGED_MAG_PLASMA` for time range `2012-08-01 to 2012-09-15` returns 1,081 rows, but **all values are fill/NaN**. Error message correctly identifies the issue: "ALL values are fill/NaN — no real data available for this parameter in the requested time range."
- **Context**:
  ```
  Tool: fetch_data
  Args: {
    'parameter_id': 'protonDensity',
    'dataset_id': 'VOYAGER1_COHO1HR_MERGED_MAG_PLASMA',
    'time_range': '2012-08-01 to 2012-09-15'
  }
  Result: error (all NaN)
  ```
- **Likely root cause**: This is **upstream data quality**, not an agent bug. Voyager 1 was in the heliopause region in Aug 2012 — plasma density measurements may be unavailable because:
  1. The instrument (PLS) stopped returning data in 1980.
  2. The COHO merged dataset interpolates/extrapolates, but has no real measurements for this period.
  3. PWS (Plasma Wave Science) *can* derive density from plasma oscillations, but it's in a different dataset.
- **Agent behavior**: Correctly detected the all-NaN condition, logged the error, and (eventually) worked around it by finding alternative datasets. Good!
- **Suggested fix**:
  1. **Documentation**: Add to `knowledge/missions/voyager1.json` metadata that `protonDensity` in COHO is unreliable after 1980; suggest PWS-derived density instead.
  2. **Prompt guidance**: Update Voyager1 agent prompt to prefer PWS density over COHO `protonDensity`.
  3. **Auto-fallback**: In `fetch_data`, if all-NaN detected and parameter is "density", suggest alternative datasets in error message.
- **Known issue?**: No — add to known-issues.md under "Data Quality" section.

### Issue 4: CDF Fetch Dimension Mismatch for `Radius` Parameter
- **Severity**: High
- **Source**: Lines 1897-1902
- **First seen**: 2026-02-14 12:22:47
- **Frequency**: 1 occurrence (but blocking data fetch)
- **Description**: Fetching the `Radius` parameter from `VOYAGER1_48S_MAG-VIM` dataset fails with pandas ValueError: "Length of values (366) does not match length of index (156909)". This is a **data shape mismatch** — the `Radius` variable has 366 values (likely daily cadence, ~1 year of data), but the time index has 156,909 points (48-second cadence).
- **Context**:
  ```
  Tool: fetch_data
  Args: {
    'parameter_id': 'Radius',
    'dataset_id': 'VOYAGER1_48S_MAG-VIM',
    'time_range': '2012-02-25 to 2013-08-25'
  }
  Expected: 156,909 rows (48-sec cadence)
  Actual: 366 values (daily cadence)
  Error: Length of values (366) does not match length of index (156909)
  ```
- **Likely root cause**: The `VOYAGER1_48S_MAG-VIM` dataset contains **mixed-cadence variables**:
  - Magnetic field components (BN, BR, BT, F1): 48-second cadence → 156,909 points
  - Ephemeris variables (Radius, latitude, longitude): Daily cadence → 366 points

  The CDF fetch code in `data_ops/fetch.py` (or `master_cdf.py`) assumes all variables in a dataset share the same time axis. When it reads the 48-second `Epoch` as the index, then tries to assign 366 daily `Radius` values to it, pandas raises ValueError.
- **Stack trace**: None in log (error caught and returned as tool error)
- **Suggested fix**:
  1. **Immediate**: In `fetch.py` or `master_cdf.py`, check if variable has its own time dimension (e.g., `Epoch_ephem` for ephemeris). If dimensions don't match primary `Epoch`, either:
     - Skip the variable with a warning, OR
     - Interpolate/resample to match the primary cadence, OR
     - Store it as a separate DataEntry with its own time axis.
  2. **Better**: Detect mixed-cadence datasets during metadata bootstrap and annotate variables with their cadence. Update `list_parameters` to show cadence, so LLM doesn't try to fetch incompatible parameters together.
  3. **Best**: Allow `DataEntry` to hold multiple time axes (like xarray Datasets). This is a larger refactor.
- **Known issue?**: No — this is a **new critical bug**. Mixed-cadence CDF files are common in heliophysics (especially for spacecraft ephemeris). This will affect PSP, SolO, MMS, and other missions.

## Medium Priority Issues

### Issue 5: CDF Metadata Sync Warnings (Epoch Variables)
- **Severity**: Low
- **Source**: Lines 230, 232, 235, 254, 348
- **First seen**: 2026-02-14 12:07:03
- **Frequency**: 5 warnings across 4 datasets
- **Description**: Metadata synchronization detects variables present in data CDF but not in master CDF skeleton. Examples:
  - `VOYAGER1_CRS_DAILY_FLUX`: `Epoch` (data-only)
  - `VOYAGER1_COHO1HR_MERGED_MAG_PLASMA`: `Epoch` (data-only)
  - `VOYAGER1_48S_MAG-VIM`: `Epoch, Epoch_ephem, spacecraftID` (data-only)
  - `VOYAGER-1_LECP_ELEC-BGND-COR-1D`: `Epoch, decimalYear, doy, year` (data-only)
- **Pattern**: All warnings involve time-related variables (`Epoch*`) or auxiliary metadata (`spacecraftID`, `year`, `doy`).
- **Likely root cause**: CDAWeb master CDF skeletons exclude certain "support data" variables that are present in actual data files. This is **expected behavior** for time variables and metadata, not a bug.
- **Impact**: Cosmetic only — metadata sync still succeeds ("updated cache — X master-only, Y data-only, Z confirmed").
- **Suggested fix**:
  1. Filter out known support variables from the warning (e.g., suppress warnings for variables with `VAR_TYPE == "support_data"` or names matching `Epoch*`, `*Year`, `*ID`).
  2. Downgrade to DEBUG level instead of WARNING for data-only variables.
- **Known issue?**: No, but very low priority.

### Issue 6: MemoryAgent Consolidation Timeout
- **Severity**: Low
- **Source**: Line 3607
- **First seen**: 2026-02-14 12:39:29 (end of session)
- **Frequency**: 1 occurrence
- **Description**: MemoryAgent's consolidation LLM call failed with `504 DEADLINE_EXCEEDED` from Gemini API. The agent still extracted 7 memories and 2 error patterns successfully (likely from cached data), but the final consolidation/summary failed.
- **Error**: `{'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}}`
- **Likely root cause**: Same as Issue 2 — Gemini API slowness on 2026-02-14. Memory consolidation happens at session end and involves summarizing the entire conversation history (31 turns, 924K input tokens). With the observed 10-30s response times, this likely exceeded the client timeout.
- **Impact**: Memory report still generated successfully (`/Users/huangzesen/.helio-agent/reports/report_20260214_121414.md`), so data loss is minimal.
- **Suggested fix**:
  1. Increase timeout for MemoryAgent consolidation call (currently uses default, probably 60s).
  2. Add retry with backoff.
  3. Consider making consolidation optional if it times out (memory extraction already succeeded).
- **Known issue?**: No.

## Low Priority Issues

### Issue 7: Budget Warnings (Planning Rounds Exhausted)
- **Severity**: Low (by design)
- **Source**: Lines 466, 570
- **First seen**: 2026-02-14 12:09:42
- **Frequency**: 2 warnings ("2 round(s) remaining", "1 round(s) remaining")
- **Description**: Planner warns when approaching maximum planning rounds (likely 5-round limit). Not an error — this is **expected behavior** for complex multi-step tasks.
- **Context**: User asked "How did scientists prove Voyager 1 left the solar system? Show me the data." — a complex request requiring discovery, data fetch, and visualization across multiple instruments and time ranges.
- **Impact**: None — planning completed successfully before hitting the limit.
- **Suggested fix**: None needed. This is working as designed.

## Recurring Patterns

### Pattern A: Viz Agent Initial Failure → Manual Retry Success
All 4 visualization tasks follow this pattern:
1. Planner creates viz task
2. VisualizationAgent invoked
3. First `render_plotly_json` call → empty `figure_json: {}`
4. Error: "data is required"
5. Duplicate call guard triggers → task fails
6. **User or orchestrator retries the same request**
7. Second attempt succeeds with valid Plotly JSON

**Example from log**:
- 12:11:05 — First attempt at "Plot Voyager 1 heliopause crossing evidence" → empty JSON → failed
- 12:11:47 — Agent enters viz_think mode → describes data
- 12:13:36 — `render_plotly_json` called again → **success** with full schema

**Hypothesis**: The think→execute workflow (commit 7ae2e7f) was implemented but may not be active for all viz tasks, OR the initial tool call bypasses the thinking phase.

**Action**: Review `visualization_agent.py` to ensure think phase is mandatory before render.

### Pattern B: High Token Usage
Token log (line 156) shows cumulative totals at end of session:
- Input: 924,153 tokens
- Output: 8,111 tokens
- Thinking: 8,365 tokens
- **Total**: 940,629 tokens (~0.94M)

**Estimated cost** (Gemini 2.5 Flash preview pricing):
- Input: 924K × $0.30/M = $0.28
- Output: 8K × $2.50/M = $0.02
- Thinking: 8K × $2.50/M = $0.02
- **Total: ~$0.32**

This is within normal range per project memory (300K–5.5M per session), but on the lower-middle end. The 33-minute session with 33 orchestrator calls (token log line 156) suggests moderate complexity.

**Concern**: The high input token count (924K) likely contributes to slow LLM response times (Issue 2). Each call re-sends the system prompt (~12K tokens) plus conversation history.

**Optimization opportunity**: Implement Gemini context caching — 90% discount on cached prompt tokens and significantly faster responses.

## Comparison: Local vs. Yuliang's Logs
Not applicable — only local logs analyzed in this report.

## Recommendations

### Immediate Actions (This Week)
1. **[Critical]** Fix Issue 1 (empty Plotly JSON): Add validation + retry loop in `visualization_agent.py`. Should be a 1-hour fix building on existing retry infrastructure from commit 7ae2e7f.
2. **[Critical]** Fix Issue 4 (Radius dimension mismatch): Add mixed-cadence detection in `fetch.py` or `master_cdf.py`. Either skip mismatched variables or interpolate. 2-3 hours.
3. **[High]** Investigate Issue 2 (slow LLM responses): Check Gemini API status dashboard, add logging for actual response times (not just warnings), consider enabling context caching.

### Short-Term (Next 2 Weeks)
4. Update `docs/known-issues.md` with Issues 1, 3, 4, 6.
5. Add dataset-specific metadata to `knowledge/missions/voyager1.json` warning about `protonDensity` NaN issue and suggesting PWS alternatives.
6. Implement context caching for Gemini API to reduce latency and cost.
7. Add retry backoff (exponential) for LLM timeouts instead of immediate retry.

### Long-Term (Next Month)
8. Refactor `DataEntry` to support multiple time axes (mixed-cadence datasets) — this will prevent Issue 4 class bugs across all missions.
9. Consider LLM provider fallback: if Gemini times out repeatedly, offer to switch to OpenAI or Anthropic (requires LLM abstraction layer Phase 4).
10. Add automated log analysis to CI: detect ERROR/WARNING spikes, flag sessions with >50% tool failures.

## Files Requiring Attention
1. `/Users/huangzesen/Documents/GitHub/helio-ai-agent/agent/visualization_agent.py` — Issue 1
2. `/Users/huangzesen/Documents/GitHub/helio-ai-agent/data_ops/fetch.py` — Issue 4
3. `/Users/huangzesen/Documents/GitHub/helio-ai-agent/data_ops/master_cdf.py` — Issue 4 (alt location)
4. `/Users/huangzesen/Documents/GitHub/helio-ai-agent/agent/llm/gemini_adapter.py` — Issue 2 (timeout handling)
5. `/Users/huangzesen/Documents/GitHub/helio-ai-agent/knowledge/missions/voyager1.json` — Issue 3 (metadata)
6. `/Users/huangzesen/Documents/GitHub/helio-ai-agent/docs/known-issues.md` — Update with new issues

---

**Next Steps**: Review this report with the team, prioritize fixes, and assign Issues 1 and 4 to the next sprint. Issues 2 and 3 are lower priority but should be tracked.
