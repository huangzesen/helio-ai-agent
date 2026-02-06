# Feature 06: Event Markers on Plots

## Summary

Add a tool to draw vertical line annotations at specific times on the current plot, with optional labels. This makes plots look publication-ready and helps users mark scientifically interesting features.

## Motivation

Marking events (shock crossings, CME arrivals, sector boundary crossings) on time-series plots is one of the most common tasks in heliophysics. Currently users have to do this manually in a separate tool. Adding annotation support makes the agent's output directly useful for presentations and papers.

## Files to Modify

### 1. `agent/tools.py` — Add tool schema

```python
{
    "name": "add_event_marker",
    "description": """Add a vertical line marker at a specific time on the current plot.
Use this when:
- User wants to mark a specific event (shock, CME, boundary crossing)
- User says "mark this time" or "add a line at..."
- User wants to annotate a feature they see in the plot

Multiple markers can be added by calling this tool multiple times.""",
    "parameters": {
        "type": "object",
        "properties": {
            "time": {
                "type": "string",
                "description": "Time for the vertical line (ISO 8601 format, e.g., '2024-01-15T06:30:00')"
            },
            "label": {
                "type": "string",
                "description": "Optional text label for the marker (e.g., 'Shock arrival', 'CME onset')"
            },
            "color": {
                "type": "string",
                "description": "Optional color: 'red', 'blue', 'green', 'black' (default: red)"
            }
        },
        "required": ["time"]
    }
}
```

### 2. `autoplot_bridge/commands.py` — Add annotation method

Add to the `AutoplotCommands` class:

```python
def add_annotation(self, time_str: str, label: str = "", color_name: str = "red") -> dict:
    """Add a vertical line annotation at a specific time.

    Args:
        time_str: ISO 8601 timestamp (e.g., '2024-01-15T06:30:00')
        label: Optional text label displayed near the line
        color_name: Color name ('red', 'blue', 'green', 'black')

    Returns:
        dict with status
    """
    import numpy as np

    Color = jpype.JClass("java.awt.Color")
    Units = jpype.JClass("org.das2.datum.Units")
    DatumRangeUtil = jpype.JClass("org.das2.datum.DatumRangeUtil")

    color_map = {
        "red": Color.RED,
        "blue": Color.BLUE,
        "green": Color(0, 128, 0),  # Dark green
        "black": Color.BLACK,
        "orange": Color.ORANGE,
        "purple": Color(128, 0, 128),
    }
    color = color_map.get(color_name.lower(), Color.RED)

    # Convert ISO time to Autoplot datum
    epoch_2000 = np.datetime64("2000-01-01T00:00:00", "ns")
    time_ns = np.datetime64(time_str, "ns")
    time_seconds = float((time_ns - epoch_2000).astype(np.float64) / 1e9)

    t2000 = Units.t2000
    datum = t2000.createDatum(time_seconds)

    # Access the DOM and add annotation
    dom = self.ctx.getDocumentModel()

    # Create a new annotation
    Annotation = jpype.JClass("org.autoplot.dom.Annotation")
    ann = Annotation()
    ann.setText(label if label else "")

    # Set the position using xrange (vertical line spanning the plot)
    ann.setShowArrow(False)

    # Use pointAt to set the annotation position
    # The annotation xrange property defines where the vertical line goes
    ann.setXrange(DatumRangeUtil.parseTimeRange(
        f"{time_str}/{time_str}"
    ))

    # Alternative approach: use plot annotation via ScriptContext
    # This may be more reliable depending on Autoplot version
    self._log(f"Adding marker at {time_str}" + (f" ({label})" if label else ""))

    # Add to DOM
    annotations = list(dom.getAnnotations())
    annotations.append(ann)
    dom.setAnnotations(jpype.JArray(Annotation)(annotations))

    return {
        "status": "success",
        "time": time_str,
        "label": label,
        "color": color_name,
    }
```

**Important**: The exact Autoplot DOM annotation API needs to be verified. The above is a best-effort based on the Autoplot ScriptContext documentation. Before implementing, test interactively:

```python
# In a Python session with JPype/Autoplot running:
dom = ctx.getDocumentModel()
# Explore: dir(dom), dom.getAnnotations(), etc.
# Find the correct way to add vertical line annotations
```

If the DOM annotation approach is too complex, a simpler alternative is to use Autoplot's `addPlotElement` with a vertical line dataset:

```python
def add_annotation_simple(self, time_str: str, label: str = "", color_name: str = "red") -> dict:
    """Simpler approach: draw a vertical line using a 2-point dataset."""
    # Create a dataset with 2 points at the same time, spanning y-axis
    # This is hacky but guaranteed to work
    ...
```

### 3. `agent/core.py` — Add handler

```python
elif tool_name == "add_event_marker":
    try:
        result = self.autoplot.add_annotation(
            time_str=tool_args["time"],
            label=tool_args.get("label", ""),
            color_name=tool_args.get("color", "red"),
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return result
```

### 4. `agent/prompts.py` — Add annotation guidance

```
## Annotations

Use `add_event_marker` to draw vertical lines at specific times. This is useful for:
- Marking shock crossings, CME arrivals, or boundary crossings
- Highlighting features the user points out
- Comparing events across different parameters
```

## Research Needed

**Before implementing**, investigate Autoplot's annotation API:

1. Open an interactive JPype session with Autoplot
2. Run: `dom = ctx.getDocumentModel()`
3. Explore: `dom.getAnnotations()`, `dom.getController()`
4. Check the Autoplot source or docs for `org.autoplot.dom.Annotation`
5. Try adding a vertical line annotation manually
6. Document the working approach

This research step is critical — the exact API may differ from the code sketch above.

## Testing

Testing requires a running Autoplot instance (integration test):

```python
def test_add_marker_returns_success():
    """Integration test: marker added without error."""
    # Requires: active plot in Autoplot
    # Call add_annotation with a valid time within the plot range
    # Verify status is "success"
```

## Demo Script

```
You: Plot ACE magnetic field for January 15-20, 2024
Agent: [plots data]
You: Mark the shock at January 17 around 6:30 UTC
Agent: [adds vertical line at 2024-01-17T06:30:00 labeled "Shock"]
       Marked the shock arrival at 2024-01-17T06:30 UTC on the plot.
You: Also mark the CME start at Jan 16 14:00
Agent: [adds another marker]
       Added a second marker for the CME onset at 2024-01-16T14:00 UTC.
You: Export this as annotated_plot.png
Agent: [exports with markers visible]
```
