# Feature 09: Natural Language Data Narration

## Summary

Teach the LLM to automatically describe what it sees in fetched data — trends, spikes, quiet periods, data quality — using the statistics from `describe_data` (Feature 01) or directly from `fetch_data` results.

## Motivation

This is the most "AI-feeling" feature: after fetching data, the agent says something like "The magnetic field shows elevated values (mean 12 nT vs. typical 5 nT) with a sharp spike to 35 nT around Jan 17, suggesting a CME passage." This transforms the agent from a data-plumbing tool into a scientific assistant.

## Approach

This is **pure prompt engineering** — no new tools or code changes needed. We update the system prompt to instruct the LLM to narrate data when appropriate.

**Prerequisite**: Feature 01 (Describe Data Tool) should be implemented first, as it provides the statistics the LLM needs to make meaningful observations. However, a lighter version works even without it — the LLM can comment on `fetch_data` results (which already return `num_points`, `time_min`, `time_max`).

## Files to Modify

### 1. `agent/prompts.py` — Add narration instructions

Add a new section to the system prompt:

```
## Data Narration

When you have data statistics available (from `describe_data` or after `fetch_data`), provide a brief scientific interpretation if relevant. Consider:

- **Magnitude context**: Is the mean value typical or unusual? (e.g., solar wind speed ~400 km/s is quiet, >600 km/s is fast)
- **Variability**: High std relative to mean suggests disturbed conditions
- **Data quality**: Mention if NaN percentage is significant (>5%)
- **Time coverage**: Note gaps or incomplete coverage
- **Trends**: Large difference between 25th and 75th percentiles suggests changes during the interval

### Typical Reference Values (for context)
- IMF |B| at 1 AU: 5-7 nT (quiet), >15 nT (storm)
- Solar wind speed: 300-450 km/s (slow), 500-800 km/s (fast)
- Proton density: 3-10 cm⁻³ (typical), >20 cm⁻³ (compressed)
- SYM-H: ~0 nT (quiet), < -50 nT (storm), < -100 nT (intense storm)

Keep narration brief (1-3 sentences). Don't narrate when the user is clearly just doing data plumbing (fetch-compute-plot workflows). DO narrate when:
- User explicitly asks what the data shows
- User asks for a summary or overview
- Data shows clearly unusual values
- It's the final response after completing a multi-step analysis
```

### 2. (Optional) Add reference values to describe_data output

If Feature 01 is implemented, enhance the `describe_data` handler to include typical reference ranges as a hint for the LLM:

```python
# In the describe_data handler, add context hints
REFERENCE_RANGES = {
    "nT": {"typical": "5-7 nT for IMF at 1 AU", "storm": ">15 nT"},
    "km/s": {"slow": "300-450 km/s", "fast": ">600 km/s"},
    "cm^-3": {"typical": "3-10 cm^-3"},
}

if entry.units in REFERENCE_RANGES:
    result["reference"] = REFERENCE_RANGES[entry.units]
```

## Testing

This is tested through interaction rather than unit tests:

```
Test 1: Fetch OMNI data for a known storm period
  → Agent should mention elevated values and storm-like conditions

Test 2: Fetch ACE data for a quiet period
  → Agent should note typical/quiet conditions

Test 3: Fetch data with many NaN gaps
  → Agent should mention data quality concerns

Test 4: "Describe the data" after fetching
  → Agent should provide detailed narration
```

## Demo Script

```
You: Fetch OMNI solar wind data for the Halloween storm (October 29-31, 2003)
Agent: Fetched OMNI_HRO_1MIN.flow_speed — 4,320 points from 2003-10-29 to 2003-10-31.

       The solar wind speed during this period is remarkably elevated,
       ranging from 400 to over 1,800 km/s with a mean of 920 km/s —
       well above the typical 400 km/s. This is consistent with the
       historic October 2003 "Halloween" solar storms, one of the most
       intense space weather events on record.
```

## Notes

- The narration quality depends entirely on the LLM's ability to interpret statistics. Gemini 2.5-Flash handles this well with proper prompting.
- The reference values in the prompt give the LLM scientific context it may not have from training data alone.
- Narration should be optional-feeling — the LLM should not narrate every single fetch (only when it adds value or the user asks).
- This pairs well with Feature 01 (describe_data) and Feature 10 (comparison mode).
