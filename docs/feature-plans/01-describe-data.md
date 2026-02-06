# Feature 01: Describe Data Tool

## Summary

Add a `describe_data` tool that returns statistical summaries of fetched timeseries data. The LLM uses these stats to narrate trends, anomalies, and data quality in natural language.

## Motivation

Currently users can fetch data and plot it, but there's no way to ask "what does this data look like?" without plotting. A describe tool gives the LLM context to say things like "The magnetic field ranges from 2.1 to 38.5 nT with a mean of 6.3 nT — there appears to be a spike around the middle of the interval."

## Files to Modify

### 1. `agent/tools.py` — Add tool schema

Add after the `custom_operation` tool entry:

```python
{
    "name": "describe_data",
    "description": """Get statistical summary of an in-memory timeseries. Use this when:
- User asks "what does the data look like?" or "summarize the data"
- You want to understand the data before deciding what operations to apply
- User asks about min, max, average, or data quality

Returns statistics (min, max, mean, std, percentiles, NaN count) and the LLM can narrate findings.""",
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Label of the data in memory (e.g., 'AC_H2_MFI.BGSEc')"
            }
        },
        "required": ["label"]
    }
}
```

### 2. `agent/core.py` — Add handler in `_execute_tool()`

Add a new `elif` block before the `else` clause (around line 349):

```python
elif tool_name == "describe_data":
    store = get_store()
    entry = store.get(tool_args["label"])
    if entry is None:
        return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

    df = entry.data
    stats = {}

    # Per-column statistics
    desc = df.describe(percentiles=[0.25, 0.5, 0.75])
    for col in df.columns:
        col_stats = {
            "min": float(desc.loc["min", col]),
            "max": float(desc.loc["max", col]),
            "mean": float(desc.loc["mean", col]),
            "std": float(desc.loc["std", col]),
            "25%": float(desc.loc["25%", col]),
            "50%": float(desc.loc["50%", col]),
            "75%": float(desc.loc["75%", col]),
        }
        stats[col] = col_stats

    # Global metadata
    nan_count = int(df.isna().sum().sum())
    total_points = len(df)
    time_span = str(df.index[-1] - df.index[0]) if total_points > 1 else "single point"

    # Cadence estimate (median time step)
    if total_points > 1:
        dt = df.index.to_series().diff().dropna()
        median_cadence = str(dt.median())
    else:
        median_cadence = "N/A"

    return {
        "status": "success",
        "label": entry.label,
        "units": entry.units,
        "num_points": total_points,
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "time_start": str(df.index[0]),
        "time_end": str(df.index[-1]),
        "time_span": time_span,
        "median_cadence": median_cadence,
        "nan_count": nan_count,
        "nan_percentage": round(nan_count / (total_points * len(df.columns)) * 100, 1) if total_points > 0 else 0,
        "statistics": stats,
    }
```

### 3. `agent/prompts.py` — Update system prompt

In the "Data Operations" section, add after the `list_fetched_data` bullet:

```
5. **`describe_data`** — Get statistical summary (min, max, mean, std, percentiles, NaN count, cadence) of a stored timeseries. Use this to understand data before computing or to answer user questions about data characteristics.
```

### 4. `docs/capability-summary.md` — Update tool count and table

Add row to the "Data Operations" tools table:
```
| `describe_data` | Statistical summary of in-memory data (min/max/mean/std/percentiles/NaN) |
```

Update tool count from 12 to 13.

## Testing

Add to `tests/test_store.py` or create `tests/test_describe.py`:

```python
def test_describe_data_scalar():
    """describe_data returns correct stats for scalar timeseries."""
    # Create a simple DataEntry with known values
    # Verify min, max, mean, std, nan_count are correct

def test_describe_data_vector():
    """describe_data returns per-column stats for vector data."""

def test_describe_data_with_nans():
    """NaN counting is correct."""

def test_describe_data_missing_label():
    """Returns error for unknown label."""
```

## Demo Script

```
You: Show me ACE magnetic field data for January 2024
Agent: [plots data]
You: Fetch it so I can analyze it
Agent: [fetches AC_H2_MFI.BGSEc]
You: Describe the data
Agent: The ACE magnetic field data for January 2024 contains 44,640 points
       at 1-minute cadence. The three components (Bx, By, Bz) range from
       -15.2 to 18.7 nT with a mean magnitude of about 5.1 nT. Data quality
       is good — only 0.3% NaN values. There's notably higher variance in
       the second half of the month, suggesting increased solar wind activity.
```
