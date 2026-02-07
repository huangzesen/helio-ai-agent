# Feature 1: Autoplot Interactive GUI Mode

## Summary

Add a `--gui` flag that launches Autoplot with its native Swing window visible, giving users live interactive plots with built-in zoom, pan, and crosshairs. Headless mode remains the default for backward compatibility.

## Motivation

Currently Autoplot runs in headless mode only — JVM starts with `-Djava.awt.headless=true`, all plots render in memory, and users must export to PNG to see anything. This wastes Autoplot's main strength: its interactive GUI. On Windows, the GUI "just works" when the headless flag is removed.

---

## Files to Modify

| File | Change |
|------|--------|
| `autoplot_bridge/connection.py` | Add `headless` param, conditionally omit `-Djava.awt.headless=true` |
| `autoplot_bridge/commands.py` | Add `gui_mode` param, new DOM methods (reset, title, axis labels, log scale, save/load session) |
| `agent/core.py` | Wire `gui_mode` through constructor, add 7 new tool handlers, skip PNG auto-open in GUI mode |
| `agent/tools.py` | Add 7 new tool schemas (15 → 22 tools) |
| `agent/prompts.py` | Append GUI-mode instructions to system prompt when active |
| `main.py` | Add `--gui` argparse flag, pass to agent |
| `tests/test_gui_mode.py` | New file: unit tests for GUI mode (headless flag, prompt, tools, reset) |
| `docs/capability-summary.md` | Document new tools and GUI mode |

---

## Implementation Steps

### Step 1: `autoplot_bridge/connection.py` — Conditional headless flag

- Add `headless: bool = True` param to `init_autoplot()` and `get_script_context()`
- Only pass `-Djava.awt.headless=true` to `jpype.startJVM()` when `headless=True`
- Store mode in module-level `_headless_mode` variable with `is_headless()` getter
- Update `__main__` test block to accept `--gui` flag

### Step 2: `autoplot_bridge/commands.py` — GUI mode + new DOM methods

- Add `gui_mode: bool = False` to `AutoplotCommands.__init__()` and `get_commands()`
- Pass `headless=(not self.gui_mode)` in the lazy `ctx` property
- Add `"display": "gui_window"` or `"headless"` to `plot_cdaweb()` and `plot_dataset()` return dicts
- Add new methods (all DOM-mutating methods should call `waitUntilIdle()` after changes to let the GUI repaint):
  - `reset()` — calls `ScriptContext.reset()`, clears `_current_uri`, `_current_time_range`, `_label_colors` + `waitUntilIdle()`
  - `set_plot_title(title)` — DOM: `dom.getPlotElements()[0].setTitle(title)` + `waitUntilIdle()`
  - `set_axis_label(axis, label)` — DOM: `plot.getYaxis().setLabel(label)` etc. + `waitUntilIdle()`
  - `toggle_log_scale(axis, enabled)` — DOM: `plot.getYaxis().setLog(enabled)` + `waitUntilIdle()`
  - `set_axis_range(axis, min, max)` — DOM: `plot.getYaxis().setRange(DatumRange)` + `waitUntilIdle()`
  - `save_session(filepath)` — `ScriptContext.save(filepath)`
  - `load_session(filepath)` — `ScriptContext.load(filepath)` + `waitUntilIdle()`

### Step 3: `agent/tools.py` — 7 new tool schemas

Add to the `TOOLS` list with category `"plotting"`:

1. **`reset_plot`** — No params. Clears the canvas.
2. **`set_plot_title`** — Params: `title` (string)
3. **`set_axis_label`** — Params: `axis` (y/z), `label` (string). Note: x-axis labels are auto-managed by Autoplot for time-series data; only y and z are exposed.
4. **`toggle_log_scale`** — Params: `axis` (y/z), `enabled` (boolean)
5. **`set_axis_range`** — Params: `axis` (y/z), `min` (number), `max` (number). Manual override of axis bounds.
6. **`save_session`** — Params: `filepath` (string, .vap extension)
7. **`load_session`** — Params: `filepath` (string, .vap extension). Restores a previously saved session.

### Step 4: `agent/core.py` — Wire gui_mode + new tool handlers

- Add `gui_mode: bool = False` to `AutoplotAgent.__init__()` and `create_agent()`
- Pass `gui_mode` to `get_system_prompt()` and `get_commands()`
- Add `elif` handlers in `_execute_tool()` for the 7 new tools (after `get_plot_info` block ~line 288)
- Modify `export_plot` handler (lines 265-283): skip `os.startfile()` auto-open when `self.gui_mode`

### Step 5: `agent/prompts.py` — GUI-aware system prompt

- Add `gui_mode: bool = False` param to `get_system_prompt()`
- When `gui_mode=True`, append a section telling the LLM:
  - Plots appear instantly in the Autoplot window
  - No need to suggest PNG export for viewing
  - New tools available: reset_plot, set_plot_title, set_axis_label, toggle_log_scale, set_axis_range, save_session, load_session
  - User may ask for interactive refinements

### Step 6: `main.py` — `--gui` CLI flag

- Add `--gui` argparse argument (after `--verbose`)
- Pass `gui_mode=args.gui` to `create_agent()`
- Print "GUI Mode: Autoplot window will appear when plotting" after welcome message

### Step 7: Documentation updates

- `docs/capability-summary.md` — Add new tools, document `--gui` flag, update tool count
- Update relevant sections of `CLAUDE.md` Commands section

---

## Propagation Flow

```
main.py --gui
  → create_agent(gui_mode=True)
    → AutoplotAgent(gui_mode=True)
      → get_system_prompt(gui_mode=True)    # LLM knows about GUI
      → get_commands(gui_mode=True)          # lazy, on first plot
        → AutoplotCommands(gui_mode=True)
          → get_script_context(headless=False)
            → init_autoplot(headless=False)
              → jpype.startJVM(jvm_path, classpath=[...])  # NO headless flag
              → ScriptContext.createApplicationModel('')    # Window appears
```

---

## New Tool Details

### reset_plot

```python
{
    "category": "plotting",
    "name": "reset_plot",
    "description": "Reset the Autoplot canvas, clearing all plots and data. Use when the user wants to start fresh or clear the current display.",
    "parameters": {"type": "object", "properties": {}, "required": []}
}
```

### set_plot_title

```python
{
    "category": "plotting",
    "name": "set_plot_title",
    "description": "Set or change the title of the current plot.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The title text to display above the plot"}
        },
        "required": ["title"]
    }
}
```

### set_axis_label

```python
{
    "category": "plotting",
    "name": "set_axis_label",
    "description": "Set a label on an axis of the current plot. Only y and z axes are supported (x-axis labels are auto-managed for time-series data).",
    "parameters": {
        "type": "object",
        "properties": {
            "axis": {"type": "string", "description": "Which axis: 'y' or 'z'"},
            "label": {"type": "string", "description": "The text label to set"}
        },
        "required": ["axis", "label"]
    }
}
```

### toggle_log_scale

```python
{
    "category": "plotting",
    "name": "toggle_log_scale",
    "description": "Enable or disable logarithmic scale on a plot axis.",
    "parameters": {
        "type": "object",
        "properties": {
            "axis": {"type": "string", "description": "Which axis: 'y' or 'z'"},
            "enabled": {"type": "boolean", "description": "True for log, False for linear"}
        },
        "required": ["axis", "enabled"]
    }
}
```

### set_axis_range

```python
{
    "category": "plotting",
    "name": "set_axis_range",
    "description": "Manually set the range of a plot axis. Useful when the user wants to zoom into a specific value range.",
    "parameters": {
        "type": "object",
        "properties": {
            "axis": {"type": "string", "description": "Which axis: 'y' or 'z'"},
            "min": {"type": "number", "description": "Minimum value for the axis"},
            "max": {"type": "number", "description": "Maximum value for the axis"}
        },
        "required": ["axis", "min", "max"]
    }
}
```

### save_session

```python
{
    "category": "plotting",
    "name": "save_session",
    "description": "Save the current Autoplot session to a .vap file for later restoration.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Output .vap file path"}
        },
        "required": ["filepath"]
    }
}
```

### load_session

```python
{
    "category": "plotting",
    "name": "load_session",
    "description": "Load a previously saved Autoplot session from a .vap file, restoring all plots and settings.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the .vap file to load"}
        },
        "required": ["filepath"]
    }
}
```

---

## GUI-Mode System Prompt Addition

When `gui_mode=True`, append to the system prompt:

```
## Interactive GUI Mode

The Autoplot window is visible to the user. Plots appear immediately in the GUI
when you call plot_data or plot_computed_data. Key differences:
- The user can already see the plot — do NOT suggest exporting to PNG for viewing
- Changes like zoom (change_time_range), axis labels, log scale, and title are reflected instantly
- Use reset_plot to clear the canvas when starting a new analysis
- Use save_session/load_session to let the user save and restore workspaces
- Say "The plot is now showing in the Autoplot window" rather than suggesting export
- The user may request interactive refinements: "make y-axis log", "label the axis", "set title", "zoom y to 0-100", "reset"
```

---

## Verification

1. **Headless mode (existing behavior)** — `python main.py "Plot ACE magnetic field for last week"` should work exactly as before.

2. **GUI mode** — `python main.py --gui`, then:
   - "Plot ACE magnetic field for last week" → Autoplot window appears with live plot
   - "title it Solar Wind Magnetic Field" → title appears on plot
   - "make the y-axis logarithmic" → toggles log scale live
   - "zoom y-axis to 0 to 50" → axis range updates
   - "reset the plot" → canvas clears
   - "save this session as my_session" → .vap file created
   - "load my_session" → session restored

3. **Unit tests** (`tests/test_gui_mode.py`):
   - Mock `jpype.startJVM` to verify headless flag presence/absence
   - Test `get_system_prompt(gui_mode=True/False)` content
   - Test tool schema count is 22
   - Test `AutoplotCommands.reset()` clears internal state
   - Test `waitUntilIdle()` is called after DOM-mutating operations

4. **Existing tests** — `python -m pytest tests/` should all pass (headless is default).

---

## Notes

- **Windows 11**: Primary target. GUI works out of the box when headless flag removed.
- **macOS caveat**: Swing needs the main thread. Add a TODO comment; not blocking for Windows.
- **JVM lifecycle**: Can only start once per process — mode is set at startup, can't switch mid-session.
- **Singleton guard**: If `get_commands()` is called with conflicting `gui_mode`, raise an error.
