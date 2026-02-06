# Feature 03: Auto-Open Exported PNG

## Summary

After `export_plot` successfully creates a PNG file, automatically open it in the system's default image viewer. On Windows this uses `os.startfile()`.

## Motivation

Currently after exporting, the user has to navigate to the file and open it manually. Auto-opening makes demos feel polished and saves users a step. This is ~5 lines of code for a noticeable UX improvement.

## Files to Modify

### 1. `agent/core.py` — Modify the `export_plot` handler

Current code (around line 244):

```python
elif tool_name == "export_plot":
    filename = tool_args["filename"]
    if not filename.endswith(".png"):
        filename += ".png"
    return self.autoplot.export_png(filename)
```

Replace with:

```python
elif tool_name == "export_plot":
    filename = tool_args["filename"]
    if not filename.endswith(".png"):
        filename += ".png"
    result = self.autoplot.export_png(filename)

    # Auto-open the exported file in default viewer
    if result.get("status") == "success":
        try:
            import os
            import platform
            filepath = result["filepath"]
            if platform.system() == "Windows":
                os.startfile(filepath)
            elif platform.system() == "Darwin":
                import subprocess
                subprocess.Popen(["open", filepath])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", filepath])
            result["auto_opened"] = True
        except Exception as e:
            # Non-fatal — file was still exported successfully
            if self.verbose:
                print(f"  [Export] Could not auto-open: {e}")
            result["auto_opened"] = False

    return result
```

### 2. (Optional) `agent/prompts.py` — Update response context

No prompt change strictly needed, but the LLM can mention that the file was opened:

In the response formatting, when `auto_opened` is True, the LLM could say "Exported and opened ace_mag.png" instead of just "Exported ace_mag.png".

## Testing

This is hard to unit-test (opens a GUI application), but you can:

```python
def test_export_plot_auto_open_flag(monkeypatch):
    """Verify auto_opened flag is set in result."""
    # Mock os.startfile to be a no-op
    monkeypatch.setattr(os, "startfile", lambda x: None)
    # Run export_plot handler
    # Assert result["auto_opened"] is True
```

## Edge Cases

- **Headless/SSH sessions**: `startfile` / `xdg-open` may fail — that's fine, it's caught and `auto_opened` is set to False. The file is still saved.
- **macOS**: Uses `open` command. Linux: uses `xdg-open`.
- **Non-fatal**: The export itself always succeeds regardless of whether auto-open works.

## Notes

- Keep this behavior always-on (no flag to disable). Users who export want to see the result.
- The `subprocess.Popen` (not `run`) ensures we don't block waiting for the viewer to close.
