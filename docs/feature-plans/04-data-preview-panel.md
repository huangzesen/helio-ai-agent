# Feature Plan: Data Preview Panel in Gradio Sidebar

## Goal

Add a data preview panel to the Gradio sidebar so users can inspect the actual DataFrame contents of any loaded dataset — not just the summary table (label, shape, points) that exists today.

## Motivation

- Users currently have no way to see the raw data without asking the agent ("describe the data") or exporting to CSV
- Scientists want to sanity-check values — "is this in nT or T?", "are there NaNs?", "what do the timestamps look like?"
- The sidebar already shows metadata (label, shape, points, units, time range) but not the actual numbers
- A preview panel closes the gap between "the agent fetched data" and "I trust what it fetched"

## Design

### UI Layout

Add two components below the existing "Data in Memory" summary table:

1. **Dropdown** (`gr.Dropdown`) — lists all labels currently in the DataStore
2. **Preview table** (`gr.Dataframe`) — shows head + tail rows of the selected DataFrame

```
Sidebar (existing):
  [Interactive Plot]
  [Data in Memory — summary table]

Sidebar (new, below summary table):
  [Select Dataset ▼ dropdown]
  [Data Preview — first/last 10 rows with timestamp index]

  [Token Usage]
  [Reset Session]
```

### Behavior

- **Dropdown choices** refresh after every agent message (same as the summary table)
- **Selecting a label** triggers a callback that reads `get_store().get(label).data` and returns a preview DataFrame
- **Preview format**: first 5 rows + `...` separator + last 5 rows, with the DatetimeIndex shown as a column (ISO 8601, truncated to seconds)
- **Default state**: dropdown empty, preview table empty
- **On reset**: dropdown cleared, preview table cleared
- **Auto-select**: after an agent message, if only one label exists, auto-select it. If new data was just added, select the newest label.

### Preview Table Format

For a vector dataset like `AC_H2_MFI.BGSEc` (shape: vector[3]):

| timestamp | col_0 | col_1 | col_2 |
|-----------|-------|-------|-------|
| 2024-01-01 00:00:00 | -3.42 | 1.17 | -0.89 |
| 2024-01-01 00:01:00 | -3.38 | 1.21 | -0.91 |
| ... | ... | ... | ... |
| 2024-01-07 23:58:00 | 2.05 | -0.44 | 1.33 |
| 2024-01-07 23:59:00 | 2.11 | -0.39 | 1.28 |

For a scalar like `ACE_Bmag`:

| timestamp | magnitude |
|-----------|-----------|
| 2024-01-01 00:00:00 | 3.71 |
| ... | ... |

Values rounded to 4 decimal places for readability.

## Files to Modify

### 1. `gradio_app.py` (~50 lines added)

**a) New helper function:**

```python
def _preview_data(label: str) -> list[list] | None:
    """Return head+tail preview rows for a DataStore label."""
    from data_ops.store import get_store
    store = get_store()
    entry = store.get(label)
    if entry is None:
        return None
    df = entry.data.copy()
    df.insert(0, "timestamp", df.index.strftime("%Y-%m-%d %H:%M:%S"))
    df = df.reset_index(drop=True)
    # Round numeric columns
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(4)
    n = len(df)
    if n <= 20:
        return df.values.tolist()
    head = df.head(10)
    tail = df.tail(10)
    sep = [["..."] * len(df.columns)]
    return head.values.tolist() + sep + tail.values.tolist()
```

**b) New helper to get dropdown choices:**

```python
def _get_label_choices() -> list[str]:
    from data_ops.store import get_store
    return [e["label"] for e in get_store().list_entries()]
```

**c) Add components in sidebar (after data_table):**

```python
label_dropdown = gr.Dropdown(
    label="Select Dataset",
    choices=[],
    interactive=True,
)
data_preview = gr.Dataframe(
    label="Data Preview",
    interactive=False,
    wrap=True,
    row_count=(0, "dynamic"),
)
```

**d) Wire dropdown change event:**

```python
label_dropdown.change(
    fn=_preview_data,
    inputs=[label_dropdown],
    outputs=[data_preview],
)
```

**e) Update `respond()` to return dropdown choices:**

Add `label_choices` to the return tuple. Update `gr.Dropdown` with new `choices` after each message.

```python
# In respond():
label_choices = _get_label_choices()
# Auto-select: newest label if new data appeared
selected = label_choices[-1] if label_choices else None
preview = _preview_data(selected) if selected else None
return history, fig, data_rows, token_text, "", gr.update(choices=label_choices, value=selected), preview
```

**f) Update `reset_session()` to clear dropdown and preview:**

```python
return [], None, [], "*Session reset*", "", gr.update(choices=[], value=None), None
```

**g) Update event wiring outputs to include new components.**

### 2. No other files need changes

The DataStore and DataEntry already expose everything needed. No backend changes required.

## Files NOT Modified

- `agent/core.py` — no changes needed
- `data_ops/store.py` — already has `get()` and `list_entries()`
- `rendering/plotly_renderer.py` — not involved
- `knowledge/prompt_builder.py` — not involved

## Testing

### Manual

```bash
python gradio_app.py --verbose
# 1. Send "Fetch ACE magnetic field data for last week"
# 2. Verify dropdown shows "AC_H2_MFI.BGSEc"
# 3. Verify preview table shows head+tail rows with timestamps
# 4. Send "Compute the magnitude, save as ACE_Bmag"
# 5. Verify dropdown now has two choices
# 6. Select "ACE_Bmag" — preview updates to scalar data
# 7. Click Reset Session — dropdown and preview clear
```

### Unit tests (optional, `tests/test_gradio_preview.py`)

- `test_preview_data_scalar` — single-column DataFrame → correct row count and columns
- `test_preview_data_vector` — 3-column DataFrame → timestamp + 3 value columns
- `test_preview_data_short` — <20 rows → no separator row
- `test_preview_data_long` — >20 rows → head(10) + separator + tail(10) = 21 rows
- `test_preview_data_missing_label` — returns None
- `test_label_choices_empty` — empty store → empty list
- `test_label_choices_populated` — store with 2 entries → 2 labels

## Estimated Effort

~1 hour. Single file change, no architectural impact.
