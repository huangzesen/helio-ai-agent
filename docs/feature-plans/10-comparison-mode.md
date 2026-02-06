# Feature 10: Streamlined Comparison Mode

## Summary

Optimize the agent's handling of "compare X and Y" requests so it automatically fetches both datasets, resamples to a common cadence, computes correlation, and produces a clean overplot — all from a single natural language command.

## Motivation

"Compare PSP and ACE magnetic field" is one of the most natural things a user would say. Currently it works through the multi-step planner, but the planner doesn't know to resample to common cadence or compute correlation. This feature adds prompt guidance to make comparisons first-class.

## Approach

Like Feature 09, this is primarily **prompt engineering** + a small enhancement to the planner. No new tools are needed — the existing `fetch_data` → `custom_operation` → `plot_computed_data` pipeline already handles everything. We just need the LLM to execute the right sequence.

## Files to Modify

### 1. `agent/prompts.py` — Add comparison workflow

Add a new section to the system prompt:

```
## Comparison Workflow

When the user asks to "compare" two datasets or spacecraft:

1. **Fetch both datasets** using `fetch_data` with the same time range
2. **Resample to common cadence** if the datasets have different time resolutions:
   - Use `custom_operation` with: `result = df.resample('1min').mean().dropna(how='all')`
   - Choose the coarser cadence as the target (e.g., if one is 1s and other is 1min, use 1min)
3. **Plot together** using `plot_computed_data` with both labels comma-separated
4. **Optionally compute correlation** if the user asks "how similar" or "correlate":
   - Use `custom_operation` to compute: merge both DataFrames on time index, then `df.corr()`

### Example comparison sequence:
```
User: "Compare ACE and PSP magnetic field magnitude for January 2024"

Step 1: fetch_data(dataset_id="AC_H2_MFI", parameter_id="Magnitude", time_range="January 2024")
Step 2: fetch_data(dataset_id="PSP_FLD_L2_MAG_RTN_1MIN", parameter_id="psp_fld_l2_mag_RTN_1min", time_range="January 2024")
Step 3: custom_operation on PSP data to compute magnitude if needed
Step 4: plot_computed_data(labels="AC_H2_MFI.Magnitude,Bmag_psp", title="ACE vs PSP |B|")
```

Important: When comparing data from different spacecraft, note in your response that:
- The spacecraft may be at different distances from the Sun (especially PSP)
- Time propagation delays exist between L1 and near-Sun spacecraft
- Absolute values may differ due to radial distance effects
```

### 2. `agent/planner.py` — Improve comparison plan generation

The `create_plan_from_request` function already detects "compare" as a complex request. Enhance the planning prompt to produce better comparison plans:

In the planning prompt (inside `create_plan_from_request`), add:

```python
# Add to the planning system instruction:
COMPARISON_GUIDANCE = """
When creating a plan for comparison requests:
1. First task: Search for datasets and identify parameters
2. Fetch tasks: One fetch per dataset, using the SAME time range
3. Compute tasks: Resample to common cadence if needed, compute magnitudes if needed
4. Plot task: Use plot_computed_data with all labels to overplot
5. Summary task: Report on the comparison (values, trends, correlation if asked)

Keep the plan to 4-6 tasks maximum. Don't over-decompose.
"""
```

### 3. (Optional) Helper for cross-dataset correlation

If users frequently ask for correlation, add a convenience pattern to the `custom_operation` examples in `agent/tools.py`:

Add to the `custom_operation` description's "Common operations" list:

```
- Cross-correlation: To correlate two datasets, first fetch both, then use custom_operation
  on one with the other's data embedded. Example coming soon.
```

## Testing

This is primarily tested through end-to-end interaction:

```
Test 1: "Compare ACE and OMNI magnetic field for last week"
  → Agent fetches both, overplots them

Test 2: "How does PSP magnetic field compare to ACE for January 2024?"
  → Agent fetches both, plots, mentions radial distance difference

Test 3: "Compare solar wind speed from different spacecraft"
  → Agent asks which spacecraft, then compares
```

## Demo Script

```
You: Compare ACE and OMNI magnetic field magnitude for January 15-20, 2024
Agent: I'll fetch both datasets and plot them together.

       [fetches AC_H2_MFI.Magnitude]
       [fetches OMNI_HRO_1MIN.F]
       [plots both on same canvas]

       Here's the comparison of ACE and OMNI magnetic field magnitude
       for January 15-20, 2024. Both show similar trends, which is
       expected since OMNI includes ACE data propagated to the bow shock
       nose. The ACE values (blue) and OMNI values (orange) track closely,
       with minor differences due to the propagation algorithm.

       Peak values around Jan 17 (~18 nT) suggest a moderate solar wind
       enhancement.
```

## Notes

- This feature builds on existing capabilities — no new tools needed.
- The quality depends on the LLM following the comparison workflow correctly.
- Works best when combined with Feature 09 (data narration) for the interpretive summary.
- The multi-step planner already handles "compare" — this just makes the generated plans smarter.
- Future enhancement: add a dedicated `compare_data` tool that automates the entire workflow internally (resample + align + correlate + overplot), but the prompt-only approach works well as a starting point.
