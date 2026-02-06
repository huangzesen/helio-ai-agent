# Feature 02: Export Data to CSV

## Summary

Add a `save_data` tool that exports in-memory timeseries to CSV files. Users can save raw or computed data for use in other programs.

## Motivation

Currently the only export is PNG images via `export_plot`. Scientists often want the underlying data in CSV/ASCII format for further analysis in Excel, MATLAB, or custom scripts. This is a frequently requested capability.

## Files to Modify

### 1. `agent/tools.py` — Add tool schema

```python
{
    "name": "save_data",
    "description": """Export an in-memory timeseries to a CSV file. Use this when:
- User asks to save, export, or download data
- User wants data in a file for external use (Excel, MATLAB, etc.)
- User wants to keep a copy of computed results

The CSV file has a datetime column (ISO 8601 UTC) followed by data columns.
If no filename is given, one is auto-generated from the label.""",
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Label of the data in memory to export"
            },
            "filename": {
                "type": "string",
                "description": "Output filename (e.g., 'ace_mag.csv'). '.csv' is appended if missing. Default: auto-generated from label."
            }
        },
        "required": ["label"]
    }
}
```

### 2. `agent/core.py` — Add handler in `_execute_tool()`

Add before the `else` clause:

```python
elif tool_name == "save_data":
    store = get_store()
    entry = store.get(tool_args["label"])
    if entry is None:
        return {"status": "error", "message": f"Label '{tool_args['label']}' not found in memory"}

    # Generate filename if not provided
    filename = tool_args.get("filename", "")
    if not filename:
        # Sanitize label for filename: replace dots and slashes
        safe_label = entry.label.replace(".", "_").replace("/", "_")
        filename = f"{safe_label}.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"

    # Ensure parent directory exists
    from pathlib import Path
    parent = Path(filename).parent
    if parent and str(parent) != "." and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    # Export with ISO 8601 timestamps
    df = entry.data.copy()
    df.index.name = "timestamp"
    df.to_csv(filename, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

    filepath = str(Path(filename).resolve())
    file_size = Path(filename).stat().st_size

    if self.verbose:
        print(f"  [DataOps] Exported '{entry.label}' to {filepath} ({file_size:,} bytes)")

    return {
        "status": "success",
        "label": entry.label,
        "filepath": filepath,
        "num_points": len(df),
        "num_columns": len(df.columns),
        "file_size_bytes": file_size,
    }
```

### 3. `agent/prompts.py` — Update system prompt

Add to the data operations workflow section:

```
- **`save_data`** — Export any in-memory timeseries to CSV. The file includes ISO 8601 timestamps and all data columns.
```

### 4. `docs/capability-summary.md` — Update

Add row to data operations table:
```
| `save_data` | Export in-memory timeseries to CSV file |
```

## Testing

```python
def test_save_data_creates_file(tmp_path):
    """save_data creates a valid CSV file."""
    # Create DataEntry, save to tmp_path / "test.csv"
    # Read back with pd.read_csv, verify contents match

def test_save_data_auto_filename():
    """Auto-generates filename from label."""

def test_save_data_appends_extension():
    """Appends .csv if missing."""

def test_save_data_missing_label():
    """Returns error for unknown label."""
```

## Edge Cases

- Labels with dots (e.g., `AC_H2_MFI.BGSEc`) → sanitize to `AC_H2_MFI_BGSEc.csv`
- Large datasets → pandas `to_csv()` handles these efficiently
- NaN values → exported as empty cells (pandas default), which most tools handle

## Demo Script

```
You: Fetch ACE magnetic field data for January 2024
Agent: [fetches data]
You: Compute the magnitude
Agent: [computes Bmag]
You: Save the magnitude to a file
Agent: Exported 'Bmag' to C:\Users\...\Bmag.csv (1.2 MB, 44,640 points, 1 column).
       The file has ISO 8601 timestamps and a 'magnitude' column in nT.
```
