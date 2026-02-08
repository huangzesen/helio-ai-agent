# Dataset Loading Failure Analysis & Bugfix Plan

**Date:** 2026-02-07
**Test:** `scripts/test_dataset_loading.py` — end-to-end agent pipeline on all 571 HAPI-cached datasets
**Report:** `tests/dataset_loading_report_20260207.json`
**CSV data:** `tests/fetched_data/{mission}/` (232 files)

## Executive Summary

| Status | Count | % |
|--------|-------|---|
| PASS   | 149   | 26% |
| WARN   | 269   | 47% |
| FAIL   | 153   | 27% |

- **149 datasets** fully loaded through the agent pipeline end-to-end
- **269 WARNs** are mostly expected (186 have no plottable 1D parameters)
- **153 FAILs** are agent/pipeline bugs — the agent was asked to fetch data and didn't

Token usage: 6.85M input / 171K output / 1,784 API calls (Gemini 2.5-Flash)

## WARN Breakdown (269 total — mostly expected)

| Category | Count | Fix needed? |
|----------|-------|-------------|
| No plottable 1D parameters | 186 | No — spectral/multi-dim data, expected |
| All values are NaN | 38 | Maybe — fill value or time window issue |
| Single row fetched | 11 | Maybe — cadence classification may be off |
| Other DataFrame issues | 34 | Investigate — non-float64 columns, etc. |

## FAIL Breakdown (153 total)

### Bug 1: `@`-suffixed dataset IDs not handled (78 failures, 51%)

**Datasets:** `PSP_FLD_L2_RFS_LFR@2`, `MMS1_EDI_BRST_L2_AMB@1`, `WI_H0_MFI@0`, etc.
**Missions:** PSP (57), MMS (7), STEREO_A (5), SolO (5), WIND (4)

**Root cause:** CDAWeb uses `@N` suffixes to split large datasets into sub-datasets. The MissionAgent doesn't recognize these IDs because:
1. They're not in the mission JSON's recommended datasets
2. The `@` character may confuse the LLM (37 "not found", 29 "confused about meaning", 12 "no data")
3. Some agents think `@1`+ are "support data" and refuse to fetch them

**Agent behavior:** Typically calls `browse_datasets` which returns the base dataset without the `@` suffix, then gives up or asks for clarification.

**Fix options:**
- **A (Prompt fix):** Add a note to the mission prompt explaining `@N` suffixed dataset IDs — tell the agent these are valid CDAWeb sub-datasets and should be fetched as-is
- **B (Tool fix):** In `_execute_tool` for `fetch_data`, if the dataset_id contains `@`, pass it through directly (it already works at the HAPI level)
- **C (Knowledge fix):** Include `@N` variants in `_index.json` descriptions so `browse_datasets` returns them

**Recommended:** Option A — cheapest, addresses the LLM confusion. The HAPI fetch layer already handles `@` IDs correctly.

---

### Bug 2: MissionAgent calls `browse_datasets` instead of `fetch_data` (40 failures, 26%)

**Datasets:** `SOLO_L2_MAG-RTN-LL`, `PSP_SWP_SPC_L2I`, `STA_L1_MAG_SC`, `WI_K1-RTN_MFI`, etc.
**Missions:** SolO (17), PSP (6), STEREO_A (6), MMS (6), WIND (3), ACE (0), DSCOVR (2)

**Root cause:** When the MissionAgent receives a request for a dataset not in its recommended list, it calls `browse_datasets` to look it up. After seeing the browse results, it returns a text response describing the available datasets instead of actually calling `fetch_data`. The MissionAgent uses conversational mode (not forced function calling), so it can choose to respond with text.

**Agent behavior pattern (105 of 153 failures):**
```
orchestrator: delegate_to_mission(mission_id, request)
  mission_agent: browse_datasets(mission_id)  <-- looks up datasets
  mission_agent: [returns text response]       <-- never calls fetch_data!
orchestrator: [returns the text to user]
```

**Fix options:**
- **A (Prompt fix):** Strengthen the MissionAgent system prompt: "When asked to fetch a specific dataset and parameter, ALWAYS call fetch_data directly — do NOT browse first. The user has already identified the exact dataset and parameter."
- **B (Architecture fix):** For test-style requests where dataset_id and parameter_id are already known, bypass the orchestrator→mission agent routing and call `fetch_data` directly
- **C (Forced calling):** Use `mode="ANY"` in MissionAgent.process_request() to force tool calling (currently only used in execute_task())

**Recommended:** Option A first (low-risk prompt change). If that's insufficient, Option C as a follow-up.

---

### Bug 3: No data in time window (20 failures, 13%)

**Datasets:** `AC_H3_SWI`, `AC_H4_SWI`, `PSP_FLD_L3_DUST@0`, `SOLO_LL02_RPW-TNR`, etc.

**Root cause:** The test's time window (stop_date minus 30 days, then +1 hour) lands in a gap. The agent correctly calls `fetch_data`, but HAPI returns "no data for time range" (HAPI 1201). The test only retries once with a 90-day shift, which may still miss.

The script already had retry logic for direct-fetch mode but those retries happen INSIDE the script. When going through the agent, the agent itself may try different approaches but the time window given in the prompt is fixed.

**Fix options:**
- **A (Test fix):** Add more retry windows in the prompt: "If no data is found, try 2024-06-01 to 2024-06-01T01:00:00 as a fallback"
- **B (Test fix):** Pre-validate time windows using `get_data_availability()` before sending to the agent, pick a window known to have data
- **C (Agent fix):** Teach the MissionAgent to call `get_data_availability` first and adjust the time window

**Recommended:** Option B — pre-validate in the test script.

---

### Bug 4: Attitude/orbit/key-parameter datasets not routed (14 failures, 9%)

**Datasets:** `DSCOVR_AT_DEF`, `DSCOVR_ORBIT_PRE`, `WI_AT_DEF`, `WI_OR_PRE`, `AC_K0_MFI`, `AC_K1_EPM`, etc.

**Root cause:** Two sub-issues:
1. **Attitude/orbit datasets (4):** The MissionAgent doesn't recognize attitude (`_AT_`) and orbit (`_ORBIT_`, `_OR_`) datasets as fetchable science data
2. **Key parameter datasets (10):** `K0`, `K1`, `K2` level datasets are lower-resolution summary products. The agent may not know about them or may prefer the higher-resolution `H0`/`H2` versions

**Fix options:**
- **A (Prompt fix):** Add to mission prompts: "Attitude, orbit, and key-parameter (K0/K1/K2) datasets are valid CDAWeb datasets that can be fetched with fetch_data"
- **B (No fix):** Accept that the agent prefers higher-resolution datasets — these are rarely requested by users

**Recommended:** Option A for completeness.

---

### Bug 5: COHO/HELIO merged datasets (7 failures, 5%)

**Datasets:** `OMNI_COHO1HR_MERGED_MAG_PLASMA`, `PSP_COHO1HR_MERGED_MAG_PLASMA`, `SOLO_HELIO1HR_POSITION`, etc.

**Root cause:** These are cross-mission merged datasets from the COHOWeb/HelioWeb services. They don't belong to a single mission's instrument suite, so the MissionAgent doesn't know about them.

**Fix:** Low priority — these are rarely used directly. Could add a note to the orchestrator prompt about COHO/HELIO datasets.

---

## Priority Order

| Priority | Bug | Failures Fixed | Effort | Risk |
|----------|-----|---------------|--------|------|
| **P1** | Bug 2: browse instead of fetch | ~40 | Low (prompt) | Low |
| **P2** | Bug 1: @ suffix handling | ~78 | Low (prompt) | Low |
| **P3** | Bug 3: time window gaps | ~20 | Low (test fix) | None |
| **P4** | Bug 4: attitude/orbit/KP | ~14 | Low (prompt) | Low |
| **P5** | Bug 5: COHO/HELIO | ~7 | Low (prompt) | Low |

Fixing P1+P2 alone would address **77% of failures** (118/153).

## Affected Files

| Fix | Files to modify |
|-----|----------------|
| Bug 1 (prompt) | `knowledge/prompt_builder.py` — add @-suffix note to mission prompts |
| Bug 2 (prompt) | `knowledge/prompt_builder.py` — strengthen "always call fetch_data" instruction |
| Bug 3 (test) | `scripts/test_dataset_loading.py` — pre-validate time windows |
| Bug 4 (prompt) | `knowledge/prompt_builder.py` — add attitude/orbit/KP note |
| Bug 5 (prompt) | `knowledge/prompt_builder.py` — add COHO/HELIO note |

## Re-testing

After applying fixes:
```bash
# Quick re-test on known failures
./venv/Scripts/python.exe scripts/test_dataset_loading.py --mission DSCOVR
./venv/Scripts/python.exe scripts/test_dataset_loading.py --mission ACE

# Full re-test
./venv/Scripts/python.exe scripts/test_dataset_loading.py
```

## Full Failure List

### @ Suffix Failures (78)

| Mission | Dataset ID | Parameter |
|---------|-----------|-----------|
| MMS | MMS1_EDI_BRST_L2_AMB@1 | mms1_edi_amb_gdu1_raw_count1_brst_l2 |
| MMS | MMS1_EDI_BRST_L2_Q0@1 | mms1_edi_sq_brst_l2 |
| MMS | MMS1_EDI_BRST_L2_Q0@2 | mms1_edi_sq_brst_l2 |
| MMS | MMS1_EDI_SRVY_L2_AMB-PM2@0 | mms1_edi_optics_state_srvy_l2 |
| MMS | MMS1_EDI_SRVY_L2_AMB-PM2@1 | mms1_edi_flux1_0_srvy_l2 |
| MMS | MMS1_EDI_SRVY_L2_AMB@1 | mms1_edi_amb_gdu1_raw_count1_srvy_l2 |
| MMS | MMS1_FPI_FAST_L2_DIS-MOMS@0 | mms1_dis_numberdensity_fast |
| PSP | PSP_FLD_L2_AEB@0 | psp_fld_l2_aeb_bias_flag |
| PSP | PSP_FLD_L2_DFB_DBM_DVAC@0 | psp_fld_l2_dfb_dbm_dvac1_f |
| PSP | PSP_FLD_L2_DFB_DBM_DVAC@1 | psp_fld_l2_dfb_dbm_dvac2_f |
| PSP | PSP_FLD_L2_DFB_DBM_DVDC@0 | psp_fld_l2_dfb_dbm_dvdc1_f |
| PSP | PSP_FLD_L2_DFB_DBM_VDC@0 | psp_fld_l2_dfb_dbm_vdc1_f |
| PSP | PSP_FLD_L2_DFB_WF_SCM@1 | psp_fld_l2_dfb_wf_scm_lg_sensor |
| PSP | PSP_FLD_L2_RFS_BURST@1 | psp_fld_l2_rfs_burst_auto_V1V2_gain |
| PSP | PSP_FLD_L2_RFS_BURST@2 | psp_fld_l2_rfs_burst_cross_im_V1V2_V3V4_gain |
| PSP | PSP_FLD_L2_RFS_BURST@3 | psp_fld_l2_rfs_burst_cross_re_V1V2_V3V4_gain |
| PSP | PSP_FLD_L2_RFS_BURST@4 | psp_fld_l2_rfs_burst_cross_im_V3V4_V1V2_gain |
| PSP | PSP_FLD_L2_RFS_HFR@1 ... | (57 PSP total) |
| SolO | SOLO_L2_RPW-LFR-SURV-BP2@3 | MAGNETIC_SPECTRAL_POWER_3 |
| SolO | SOLO_LL02_EPD-EPT-NORTH-RATES@1 | EPT_N_Ele_Flux |
| SolO | SOLO_LL02_EPD-HET-ASUN-RATES@4 | HET_A_H_Flux |
| SolO | SOLO_LL02_EPD-STEP-RATES@3 | STEP_Ele_Flux |
| SolO | SOLO_LL02_SWA-EAS-SS@0 | (no plottable) |
| STEREO_A | STA_L1_SEPT@0, @1 | sept_ns_ele_flux |
| STEREO_A | STA_LB_IMPACT@0, @1, @2 | SWEARate |
| WIND | WI_H0_MFI@0, @1 | BF1 |
| WIND | WI_H3-RTN_MFI@0, @1 | BRTN |

### Browse-Instead-of-Fetch Failures (40)

| Mission | Dataset ID |
|---------|-----------|
| MMS | MMS1_FPI_FAST_L2_DES-DIST, DES-MOMSAUX, DIS-MOMS, DIS-MOMSAUX |
| MMS | MMS1_FPI_SLOW_L2_DIS-DIST, MMS1_HPCA_SRVY_L2_ION |
| PSP | PSP_FLD_L2_MAG_VSO, PSP_FLD_L3_RFS_LFR_QTN |
| PSP | PSP_SWP_SPB_SF1_L2_32E, PSP_SWP_SPC_L2I |
| PSP | PSP_SWP_SPI_SF00_L2_8DX32EX8A, PSP_SWP_SPI_SF01_L2_8DX32EX8A |
| SolO | SOLO_L2_MAG-RTN-LL, MAG-RTN-LL-1-MINUTE, MAG-SRF-BURST |
| SolO | SOLO_L2_MAG-SRF-LL, MAG-SRF-NORMAL, MAG-VSO-BURST |
| SolO | SOLO_L2_MAG-VSO-NORMAL, MAG-VSO-NORMAL-1-MINUTE |
| SolO | SOLO_L2_RPW-LFR-SURV-CWF-E, RPW-TDS-SURV-HIST2D |
| SolO | SOLO_L2_SWA-EAS1-HIRES3D-DEF, SWA-EAS1-HIRES3D-DNF |
| SolO | SOLO_L2_SWA-EAS2-HIRES3D-DEF, SWA-EAS2-HIRES3D-PSD |
| SolO | SOLO_L2_SWA-HIS-HK, SWA-HIS-PHA, SWA-PAS-VDF |
| SolO | SOLO_L3_MULTI-MAG-RPW-SCM-MERGED-* (4 variants) |
| STEREO_A | STA_L1_IMPACT_BURST, IMPACT_HKP, LET, MAG_SC |
| STEREO_A | STA_L2_PLA_ALPHA_RA_1DMAX_10MIN |
| WIND | WI_H1_SWE, WI_K1-RTN_MFI |
| DSCOVR | DSCOVR_AT_DEF, DSCOVR_ORBIT_PRE |

### No-Data-in-Time-Window Failures (20)

| Mission | Dataset ID | Note |
|---------|-----------|------|
| ACE | AC_H3_SWI, AC_H4_SWI, AC_H5_SWI | Data ends 2011, window may miss |
| MMS | MMS1_EDI_BRST/SRVY_L2_EFIELD | Sparse coverage |
| MMS | MMS1_EDP_FAST_L2_DCE | Sparse coverage |
| MMS | MMS1_MEC_BRST_L2_EPHTS04D | Burst mode gaps |
| PSP | PSP_FLD_L2_DFB_DBM_DVDC@0, DFB_DBM_VDC@0 | Burst mode gaps |
| PSP | PSP_FLD_L2_DFB_WF_SCM@1 | Sparse |
| PSP | PSP_FLD_L2_RFS_BURST@1-4 | Burst mode gaps |
| PSP | PSP_FLD_L3_DUST@0, RFS_HFR@6-7, RFS_LFR@6,11 | Sparse |
| SolO | SOLO_L2_SWA-EAS2-HIRES3D-DNF, SOLO_LL02_RPW-TNR | Sparse |
| STEREO_A | STA_LB_PLA_BROWSE | Data ended |
| WIND | WI_L3-DUSTIMPACT_WAVES | Sparse |

### Attitude/Orbit/Key-Parameter Failures (14)

| Mission | Dataset ID | Type |
|---------|-----------|------|
| ACE | AC_K0_MFI, AC_K0_SWE | Key param |
| ACE | AC_K1_EPM, AC_K1_MFI, AC_K1_SWE | Key param |
| ACE | AC_K2_MFI | Key param |
| DSCOVR | DSCOVR_AT_DEF | Attitude |
| DSCOVR | DSCOVR_ORBIT_PRE | Orbit |
| STEREO_A | STA_L2_PLA_ALPHA_RA_1DMAX_10MIN | Specialized |
| WIND | WI_AT_DEF | Attitude |
| WIND | WI_K1-RTN_MFI | Key param |
| WIND | WI_OR_PRE | Orbit |

### COHO/HELIO Failures (7)

| Mission | Dataset ID |
|---------|-----------|
| OMNI | OMNI_COHO1HR_MERGED_MAG_PLASMA |
| PSP | PSP_COHO1HR_MERGED_MAG_PLASMA |
| SolO | SOLO_COHO1HR_MERGED_MAG_PLASMA, SOLO_HELIO1HR_POSITION |
| STEREO_A | STA_COHO1HR_MERGED_MAG_PLASMA |
| WIND | WI_COHO1HR_MERGED_MAG_PLASMA, WI_HELIO1HR_POSITION |
