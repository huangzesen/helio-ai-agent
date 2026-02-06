# Feature 07: Rich Terminal Output

## Summary

Add ANSI color-coded terminal output to make the CLI visually appealing: tool names in cyan, errors in red, data labels in green, status messages dimmed. Works on Windows 10+ (which supports ANSI), macOS, and Linux.

## Motivation

The current CLI prints everything in plain monochrome text. When the agent chains multiple tool calls, it's hard to scan the output. Color-coding different elements (tool calls vs. results vs. errors vs. agent text) makes the verbose output readable and the whole experience feel polished.

## Files to Create/Modify

### 1. New file: `agent/colors.py` — ANSI color utility

```python
"""
ANSI terminal color utilities.

All formatting is a no-op if stdout is not a TTY or if NO_COLOR env var is set.
"""

import os
import sys

# Check if colors should be enabled
def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()

ENABLED = _colors_enabled()

# Enable ANSI on Windows 10+
if ENABLED and sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

# ANSI codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"


def _wrap(code: str, text: str) -> str:
    if not ENABLED:
        return text
    return f"{code}{text}{RESET}"


def bold(text: str) -> str:
    return _wrap(BOLD, text)

def dim(text: str) -> str:
    return _wrap(DIM, text)

def red(text: str) -> str:
    return _wrap(RED, text)

def green(text: str) -> str:
    return _wrap(GREEN, text)

def yellow(text: str) -> str:
    return _wrap(YELLOW, text)

def blue(text: str) -> str:
    return _wrap(BLUE, text)

def cyan(text: str) -> str:
    return _wrap(CYAN, text)

def magenta(text: str) -> str:
    return _wrap(MAGENTA, text)


# Semantic helpers
def tool_name(name: str) -> str:
    """Format a tool name for display."""
    return cyan(name)

def label(text: str) -> str:
    """Format a data label for display."""
    return green(text)

def error(text: str) -> str:
    """Format an error message."""
    return red(text)

def success(text: str) -> str:
    """Format a success message."""
    return green(text)

def info(text: str) -> str:
    """Format an informational message."""
    return dim(text)

def header(text: str) -> str:
    """Format a section header."""
    return bold(text)
```

### 2. `main.py` — Color the welcome banner and prompt

```python
from agent.colors import bold, dim, cyan, green, yellow, header

def print_welcome():
    print(bold("=" * 60))
    print(bold("  Autoplot Natural Language Interface"))
    print(bold("=" * 60))
    print()
    print("I can help you visualize spacecraft data. Try commands like:")
    print(f"  {cyan('-')} {green('Show me Parker magnetic field data for last week')}")
    print(f"  {cyan('-')} {green('What data is available for Solar Orbiter?')}")
    print(f"  {cyan('-')} {green('Plot ACE solar wind velocity for January 2024')}")
    # ... etc
```

Change the prompt from `"You: "` to a colored version:
```python
user_input = input(f"{bold('You:')} ").strip()
```

Change the agent response prefix:
```python
print(f"{bold('Agent:')} {response}")
```

### 3. `agent/core.py` — Color verbose output

Replace the plain `print()` calls in verbose mode:

```python
from agent.colors import tool_name as fmt_tool, dim, red, green, cyan, yellow

# In _execute_tool:
if self.verbose:
    print(f"  [{fmt_tool(tool_name)}({dim(str(tool_args))})]")

# On error:
if self.verbose and result.get("status") == "error":
    print(f"  [{red('ERROR')}] {result.get('message', '')}")

# On success:
if self.verbose:
    print(f"  [{green('OK')}] {tool_name} completed")

# Plan execution:
print(f"  [{cyan('Plan')}] Step {i+1}/{len(plan.tasks)}: {task.description}")
```

### 4. `autoplot_bridge/commands.py` — Color Autoplot status messages

```python
from agent.colors import dim

def _log(self, msg: str):
    if self.verbose:
        print(dim(f"  [Autoplot] {msg}"))
        sys.stdout.flush()
```

## Design Principles

1. **Respect `NO_COLOR`**: Check the `NO_COLOR` environment variable (https://no-color.org/)
2. **TTY detection**: Don't emit ANSI when piped to a file
3. **Graceful degradation**: If ANSI doesn't work, output is still readable (just uncolored)
4. **Minimal dependency**: Pure ANSI codes, no external packages (no `rich`, no `colorama`)
5. **Semantic wrappers**: Use `tool_name()`, `error()`, `label()` instead of raw colors — easy to change the color scheme later

## Testing

```python
def test_colors_disabled_with_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    # Reimport or re-check
    assert _wrap(RED, "text") == "text"

def test_colors_wrap():
    # Direct test with ENABLED forced True
    assert _wrap(RED, "hello").startswith("\033[31m")
    assert _wrap(RED, "hello").endswith("\033[0m")
```

## Visual Preview

```
============================================================
  Autoplot Natural Language Interface
============================================================

You: Show me ACE magnetic field data for last week

  [search_datasets({"query": "ACE magnetic"})]
  [OK] search_datasets completed
  [list_parameters({"dataset_id": "AC_H2_MFI"})]
  [OK] list_parameters completed
  [plot_data({"dataset_id": "AC_H2_MFI", ...})]
  [Autoplot] Initializing ScriptContext...
  [Autoplot] Plotting URI: vap+cdaweb:ds=AC_H2_MFI&id=Magnitude...
  [Autoplot] ... Plotting (12s elapsed)
  [OK] plot_data completed

Agent: I've plotted the ACE magnetic field magnitude for the past week.
```

(Imagine: tool names in cyan, OK in green, Autoplot messages dimmed, errors in red)
