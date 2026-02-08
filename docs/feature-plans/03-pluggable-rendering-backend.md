# Feature Plan: Plotly Rendering + Pluggable Data Backend + Hackathon Readiness

Consolidates: Feature 2 (Hackathon Readiness), Feature 3 (Plotly Rendering), Feature 3 (Pluggable Data Backend).

## Goal

Switch rendering from Autoplot's Java canvas to plotly (interactive, no JVM). Make the data loading backend pluggable: HAPI (default, pure Python) or Autoplot (optional, CDF file caching for power users). Upgrade the existing Gradio web UI from static PNG snapshots to interactive plotly plots. Add Gemini-specific differentiators for hackathon submission. Keep all existing Autoplot bridge code intact for future extension.

## Terminology

- **Rendering frontend**: How plots are displayed. Plotly (now). Autoplot rendering code kept for future use.
- **Data backend**: Where data comes from. HAPI (default) or Autoplot (optional, CDF cache).

Both backends produce the same output: a pandas DataFrame in the DataStore. Plotly renders from pandas regardless of how the data was loaded.

## Motivation

- **Autoplot rendering is the #1 source of critical bugs.** 4-panel plots crash the JVM (ISSUE-01), invalid CDAWeb parameters crash the JVM (ISSUE-02). These are unrecoverable — the entire process dies.
- **The JVM is the heaviest dependency.** Java 8 + JPype 1.5.x (pinned) + Autoplot JAR. Hard to install, slow to start, hangs on shutdown (`os._exit(0)` workaround).
- **Static canvas is a poor fit for web.** Gradio currently snapshots the Java canvas to PNG — no interactivity. Plotly gives zoom/pan/hover for free in the browser.
- **Dual data paths confuse the LLM.** `plot_cdaweb` (Autoplot fetches + renders) vs `fetch_data` + `plot_stored_data` (Python fetches, Java renders). One path is simpler to reason about.
- **All data should be in Python.** Currently `plot_cdaweb` data is trapped in Java — can't describe, compute, or export it.
- **Autoplot's CDF cache is valuable.** Power users doing iterative exploration benefit from `~/.autoplot/fscache/` — repeat access is instant from local disk. This should remain available as an optional data backend.
- **Hackathon requires a web demo.** Judges can't `pip install jpype1`. A browser-based Gradio UI with `share=True` is table stakes.

## Architecture

### Current

```
User → Agent → [plot_cdaweb]      → Autoplot fetches CDF from CDAWeb → Autoplot renders (static PNG)
             → [fetch_data]       → HAPI REST API → pandas
             → [plot_stored_data] → numpy→QDataSet conversion → Autoplot renders (static PNG)
             → Gradio              snapshots PNG from Java canvas
```

### Proposed

```
User → Agent → Data Backend (pluggable) → pandas DataFrame → DataStore
                 ├── HAPI (default)         HTTP REST, pure Python, no JVM
                 └── Autoplot (optional)    JVM + CDF cache, fast repeat access
             → Plotly (always)            → render from pandas, interactive in browser
             → Gradio                     → gr.Plot(fig) interactive widget
```

### Key principles

1. **Data backends output pandas DataFrames.** The DataStore is the single handoff point. Plotly never knows where data came from.
2. **Plotly is the only active renderer.** Autoplot rendering code stays in `autoplot_bridge/` untouched for future extension, but is not wired into the default flow.
3. **The agent layer is backend-agnostic.** Same tools, prompts, and registry regardless of which data backend is active.

---

## Phase 1: Plotly Rendering (~6-8 hours)

Replace Autoplot's Java canvas with plotly for all visualization.

### New file: `rendering/plotly_renderer.py`

Stateful plotly figure that the agent manipulates through tool calls.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PlotlyRenderer:
    """Stateful plotly renderer. Replaces AutoplotCommands for visualization."""

    def __init__(self):
        self._fig = go.Figure()
        self._num_panels = 1
        self._traces_per_panel = {}  # panel_index -> [trace names]

    def get_figure(self) -> go.Figure:
        """Return the current figure for Gradio display."""
        return self._fig

    def plot(self, entries, title="", panel=-1):
        """Plot DataEntry timeseries. Vector (n,3) decomposed into components.
        For large datasets (>100k points), uses go.Scattergl (WebGL) for performance.
        """
        ...

    def set_time_range(self, start, end): ...
    def set_title(self, text): ...
    def set_axis_label(self, axis, label): ...
    def toggle_log_scale(self, axis, enabled): ...
    def set_axis_range(self, axis, min_val, max_val): ...
    def set_render_type(self, render_type, index=0): ...
    def set_color_table(self, name): ...
    def set_canvas_size(self, width, height): ...
    def export_png(self, filepath): ...
    def export_pdf(self, filepath): ...
    def reset(self): ...
    def get_state(self): ...

    # Utility methods
    def to_html(self, full_html=False) -> str:
        """Export current figure as HTML string."""
        return self._fig.to_html(full_html=full_html)

    def to_png(self, filepath: str) -> str:
        """Export current figure as PNG. Requires kaleido."""
        self._fig.write_image(filepath)
        return filepath
```

#### Method mapping

| Current Autoplot method | Plotly implementation |
|---|---|
| `plot` (single series) | `fig.add_trace(go.Scatter(x=time, y=values))` |
| `plot` (overlay) | Multiple `add_trace` calls on same figure |
| `plot` (multi-panel) | `make_subplots(rows=n, shared_xaxes=True)` |
| `set_time_range` | `fig.update_xaxes(range=[start, end])` |
| `set_title` | `fig.update_layout(title=text)` |
| `set_axis_label` | `fig.update_yaxes(title_text=label, row=panel)` |
| `toggle_log_scale` | `fig.update_yaxes(type='log'/'linear')` |
| `set_axis_range` | `fig.update_yaxes(range=[min, max])` |
| `set_render_type` | See render type mapping below |
| `set_color_table` | `fig.update_traces(colorscale=name)` |
| `set_canvas_size` | `fig.update_layout(width=w, height=h)` |
| `export_png` | `fig.write_image(path)` (requires kaleido) |
| `export_pdf` | `fig.write_image(path, format='pdf')` |
| `reset` | `self._fig = go.Figure()` |
| `get_state` | Return current trace count, axis ranges, title |

#### Render type mapping

```python
RENDER_TYPE_MAP = {
    "series":       dict(mode="lines"),
    "scatter":      dict(mode="markers"),
    "fill_to_zero": dict(fill="tozeroy"),
    "staircase":    dict(line_shape="hv"),
    "digital":      dict(mode="lines", line_shape="hv"),
    "color_scatter": "special",    # go.Scatter with marker color array
    "spectrogram":   "special",    # go.Heatmap (needs rank-2 data, Phase 6)
}
```

Specialized types (pitch_angle_distribution, events_bar, orbit) are deferred — these are niche and can be added later.

#### Key decisions

- **Stateful figure**: The renderer holds a `plotly.graph_objects.Figure` as mutable state, like Autoplot's ScriptContext singleton. Each tool call mutates it.
- **Vector decomposition**: Same as current — split (n,3) into 3 traces labeled `.x`, `.y`, `.z`.
- **Color assignment**: Use plotly's default color sequence (`plotly.colors.qualitative.Plotly`).
- **No QDataSet conversion**: Plotly works directly with pandas DatetimeIndex and numpy arrays. The entire `_numpy_to_qdataset()` path is eliminated.
- **No panel limit**: Plotly handles unlimited subplots. The 3-panel guard (ISSUE-01 workaround) is no longer needed.
- **WebGL for large data**: Use `go.Scattergl` instead of `go.Scatter` when data exceeds 100k points for smooth rendering.

### Changes to `agent/core.py`

Replace `self.autoplot` (AutoplotCommands) with `self.renderer` (PlotlyRenderer):

```python
# Before:
self._autoplot = None  # lazy-init AutoplotCommands

# After:
from rendering.plotly_renderer import PlotlyRenderer
self._renderer = PlotlyRenderer()
```

The `_dispatch_autoplot_method()` function routes to `self._renderer` methods instead of `self.autoplot` methods. The interface is the same — only the implementation changes.

### Changes to `autoplot_bridge/registry.py`

Remove `plot_cdaweb` from the registry. It becomes an internal implementation detail (fetch + plot) rather than an LLM-facing method. The remaining 15 methods map 1:1 to `PlotlyRenderer` methods.

Or: keep `plot_cdaweb` in the registry but implement it as `fetch_data` + `plot_stored_data` internally (see Phase 3).

### Changes to prompts

Update `build_autoplot_prompt()` to remove DOM/ScriptContext references and `autoplot_script` examples. The method catalog stays the same (minus `plot_cdaweb` if removed). Remove `autoplot_script` from the AutoplotAgent's tool set.

### Unit tests: `tests/test_plotly_renderer.py`

No API key or JVM needed:

| Test | What it verifies |
|------|------------------|
| `test_single_scalar_entry` | 1 trace, correct x/y data |
| `test_vector_entry_decomposition` | 3 traces for (n,3) data |
| `test_multiple_overlay` | Correct trace count for overlaid datasets |
| `test_empty_entries_raises` | ValueError on empty input |
| `test_log_scale` | yaxis type = "log" |
| `test_y_range` | yaxis range set correctly |
| `test_time_range_filter` | xaxis range set correctly |
| `test_subplot_figure` | Correct subplot count |
| `test_to_html` | Returns string with plotly div |
| `test_to_png` | File created (skip if kaleido missing) |
| `test_scattergl_large_dataset` | Uses Scattergl for >100k points |

### Files

| File | Change |
|------|--------|
| `rendering/__init__.py` | **New** — renderer factory |
| `rendering/plotly_renderer.py` | **New** — plotly rendering implementation |
| `tests/test_plotly_renderer.py` | **New** — unit tests |
| `agent/core.py` | Replace `self.autoplot` → `self._renderer`; update `_dispatch_autoplot_method()` |
| `agent/autoplot_agent.py` | Remove `autoplot_script` from tool set; update prompt |
| `agent/tools.py` | Remove `autoplot_script` tool (or gate behind a flag) |
| `knowledge/prompt_builder.py` | Simplify `build_autoplot_prompt()` — no DOM/script sections |
| `requirements.txt` | Add `plotly>=5.18.0`, `kaleido>=0.2.1`; make `jpype1` optional |
| `config.py` | Make `AUTOPLOT_JAR` optional |

### NOT changed

| File | Why |
|------|-----|
| `autoplot_bridge/*` | **Kept intact** for future extension. Not deleted, not modified. |

---

## Phase 2: Upgrade Gradio to Plotly (~2-3 hours)

The Gradio web UI already exists (`gradio_app.py`) with chat, sidebar data table, token usage, example prompts, and `--share`/`--port`/`--verbose` flags. Currently it uses **static PNG snapshots** (`gr.Image` + MD5 hash change detection + `_snapshot_plot()`). Upgrade it to use interactive plotly figures.

### Changes

Replace `gr.Image` with `gr.Plot` in the sidebar:

```python
# Before:
plot_image = gr.Image(label="Current Plot", type="filepath", ...)

# After:
plotly_plot = gr.Plot(label="Interactive Plot")
```

Replace `_snapshot_plot()` (PNG export + MD5 hash) with direct figure access:

```python
# Before:
plot_path, plot_changed = _snapshot_plot()  # exports PNG, checks MD5

# After:
fig = _agent.get_plotly_figure()  # returns go.Figure or None
```

Remove `_plot_dir`, `_last_plot_hash`, `tempfile` usage, and `hashlib` import — all related to the PNG snapshot approach.

Remove `concurrency_limit=1` — no longer needed since plotly is stateless per-figure (no JVM singleton).

Remove `os._exit(0)` shutdown workaround — no JVM to hang.

### Files

| File | Change |
|------|--------|
| `gradio_app.py` | Replace `gr.Image` → `gr.Plot`; remove PNG snapshot logic |
| `agent/core.py` | Add `get_plotly_figure()` accessor returning `self._renderer.get_figure()` |

---

## Phase 3: HAPI-Only Data (Default Backend) (~2-3 hours)

Consolidate all data fetching through HAPI. Remove the `plot_cdaweb` path where Autoplot fetches directly.

### Make `plot_cdaweb` a convenience alias

Instead of removing it, make it internally call `fetch_data` + `plot_stored_data`:

```python
# In _dispatch_autoplot_method():
elif method == "plot_cdaweb":
    # Step 1: Fetch via HAPI into DataStore
    fetch_result = self._execute_tool("fetch_data", {
        "dataset_id": args["dataset_id"],
        "parameter_id": args["parameter_id"],
        "time_range": args["time_range"],
    })
    if fetch_result.get("status") == "error":
        return fetch_result
    # Step 2: Plot from DataStore
    label = fetch_result["label"]
    return self._dispatch_autoplot_method("plot_stored_data", {"labels": label})
```

This is transparent to the LLM — `plot_cdaweb` still works as a tool, but data always flows through HAPI → pandas → plotly.

### Enrich HAPI metadata in DataEntry

Extend `data_ops/fetch.py` to capture fill values and valid ranges from HAPI `/info`:

```python
# After fetching /info for the parameter:
entry = DataEntry(
    label=label,
    data=df,
    units=param_info.get("units", ""),
    description=param_info.get("description", ""),
    source="hapi",
    metadata={
        "fill_value": param_info.get("fill", None),
        "dataset_id": dataset_id,
        "parameter_id": parameter_id,
    },
)
```

The plotly renderer can use `metadata["fill_value"]` to mask fill values and use `units` for axis labels — matching what Autoplot did automatically.

### Changes to `data_ops/store.py`

Add optional `metadata` dict to `DataEntry`:

```python
@dataclass
class DataEntry:
    label: str
    data: pd.DataFrame
    units: str = ""
    description: str = ""
    source: str = "computed"
    metadata: dict = field(default_factory=dict)  # NEW: fill_value, dataset_id, etc.
```

### Files

| File | Change |
|------|--------|
| `agent/core.py` | `plot_cdaweb` dispatches to `fetch_data` + `plot_stored_data` |
| `data_ops/fetch.py` | Capture HAPI `/info` metadata (fill value, description) |
| `data_ops/store.py` | Add `metadata` field to `DataEntry` |

---

## Phase 4: Autoplot Data Backend (Optional) (~3-4 hours)

For power users who have Autoplot installed and want CDF file caching.

### New file: `data_ops/autoplot_fetch.py`

Uses Autoplot's ScriptContext to fetch data via CDAWeb (leveraging CDF cache), then converts the result back to pandas:

```python
def fetch_via_autoplot(dataset_id: str, parameter_id: str,
                       time_min: str, time_max: str) -> dict:
    """Fetch CDAWeb data via Autoplot's CDF cache.

    Uses Autoplot's ScriptContext.getDataSet() to load data from
    a vap+cdaweb: URI. If the CDF is already in ~/.autoplot/fscache/,
    this is an instant local read. Otherwise Autoplot downloads and caches it.

    Returns the same dict format as fetch_hapi_data():
        {"data": pd.DataFrame, "units": str, "description": str}
    """
    from autoplot_bridge.connection import get_script_context
    import jpype
    import numpy as np
    import pandas as pd

    sc = get_script_context()
    uri = f"vap+cdaweb:ds={dataset_id}&id={parameter_id}&timerange={time_min}+to+{time_max}"

    # Fetch data as QDataSet (uses CDF cache)
    ds = sc.getDataSet(uri)
    sc.waitUntilIdle()

    # Convert QDataSet → pandas DataFrame
    QDataSet = jpype.JClass("org.das2.qds.QDataSet")

    dep0 = ds.property(QDataSet.DEPEND_0)
    n = int(ds.length())

    # Time: convert from Units.t2000 (seconds since 2000) to datetime64
    epoch_2000 = np.datetime64("2000-01-01T00:00:00", "ns")
    time_arr = np.array([
        epoch_2000 + np.timedelta64(int(dep0.value(i) * 1e9), "ns")
        for i in range(n)
    ], dtype="datetime64[ns]")

    # Values
    if ds.rank() == 2:
        ncols = int(ds.length(0))
        values = np.array([[ds.value(i, j) for j in range(ncols)] for i in range(n)])
        columns = [f"col_{j}" for j in range(ncols)]
    else:
        values = np.array([ds.value(i) for i in range(n)])
        columns = [parameter_id]

    df = pd.DataFrame(values, index=pd.DatetimeIndex(time_arr), columns=columns)

    # Extract units
    units_obj = ds.property(QDataSet.UNITS)
    units = str(units_obj) if units_obj else ""

    return {"data": df, "units": units, "description": ""}
```

### Wire into `data_ops/fetch.py`

Add a backend switch to the existing `fetch_data` tool handler:

```python
# In agent/core.py _execute_tool(), fetch_data handler:
if data_backend == "autoplot":
    from data_ops.autoplot_fetch import fetch_via_autoplot
    result = fetch_via_autoplot(dataset_id, parameter_id, time_min, time_max)
else:
    result = fetch_hapi_data(dataset_id, parameter_id, time_min, time_max)
```

The rest of the pipeline (DataStore, compute, plotly rendering) is identical regardless of which backend fetched the data.

### CLI flag

```bash
# Default: HAPI (no JVM)
python main.py "Show me ACE mag data"

# Power user: Autoplot data backend (CDF cache, requires JVM)
python main.py --data-backend autoplot "Show me ACE mag data"
```

### Benefit: CDF caching

First request for `AC_H2_MFI` January 2024: Autoplot downloads CDF from CDAWeb (~2-5s), caches to `~/.autoplot/fscache/`.

Second request (zoom, different parameter from same CDF): instant local read (<100ms).

This is particularly valuable for iterative exploration: "show me Bx" → "now show By" → "overlay Bz" → "zoom in to Jan 10-15".

### Files

| File | Change |
|------|--------|
| `data_ops/autoplot_fetch.py` | **New** — QDataSet → pandas conversion |
| `agent/core.py` | Backend switch in `fetch_data` handler |
| `main.py` | Add `--data-backend` CLI flag |

---

## Phase 5: HAPI-Side Data Caching (~2-3 hours)

Add caching directly to the HAPI fetch layer — no JVM needed. Simpler alternative to Phase 4 that covers 90% of the caching benefit.

### Local file cache for HAPI responses

```python
# In data_ops/fetch.py:
import hashlib
from pathlib import Path

CACHE_DIR = Path.home() / ".helio-agent" / "data_cache"

def _cache_key(dataset_id, parameter_id, time_min, time_max):
    raw = f"{dataset_id}|{parameter_id}|{time_min}|{time_max}"
    return hashlib.md5(raw.encode()).hexdigest()

def fetch_hapi_data(dataset_id, parameter_id, time_min, time_max):
    key = _cache_key(dataset_id, parameter_id, time_min, time_max)
    cache_path = CACHE_DIR / f"{key}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return {"data": df, "units": ..., "description": ...}

    # Fetch from HAPI as usual
    result = _fetch_from_hapi(dataset_id, parameter_id, time_min, time_max)

    # Cache locally
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result["data"].to_parquet(cache_path)

    return result
```

The tradeoff vs Phase 4: HAPI caches per-parameter (not per-CDF-file), so "show me Bx" then "show me By" is two cache entries, whereas Autoplot's CDF cache stores the whole file.

Phase 4 and Phase 5 are not mutually exclusive. Phase 5 is simpler and covers most use cases. Phase 4 is for users who specifically want Autoplot's CDF handling.

---

## Phase 6: Gemini Differentiators (~3-4 hours)

### 6A: Multimodal Input — "Analyze This Plot" (high impact, ~2 hours)

Let users upload a screenshot of a plot and ask questions about it.

```
User: [uploads screenshot] "What spacecraft is this data from? Can you reproduce it?"
Agent: "This looks like ACE magnetic field data from January 2024. Let me fetch and plot it..."
```

**Implementation:**
- Gradio already supports image upload in chat
- Send the image to Gemini as a `Part` with `inline_data` (Gemini is natively multimodal)
- Add a system prompt section: "When the user uploads an image of a plot, analyze it..."
- Gemini identifies the spacecraft, parameter, and time range, then calls tools to reproduce it

**Files:**

| File | Change |
|------|--------|
| `agent/core.py` | Accept image parts in `process_message()`, pass to Gemini as multimodal content |
| `gradio_app.py` | Enable `multimodal=True` in ChatInterface |
| System prompt | Add instructions for image analysis |

### 6B: Grounding with Google Search (medium impact, ~1 hour)

Use Gemini's built-in Google Search grounding for real-world context.

```
User: "What was happening with the Sun in January 2024?"
Agent: [uses Google Search grounding] "In January 2024, several X-class flares were observed..."
Agent: "Would you like me to pull PSP and ACE data from that period to see the solar wind response?"
```

**Implementation:**
- Add `google_search_retrieval` tool config to Gemini client
- The LLM decides when web context is useful (no code changes to routing)

**Files:**

| File | Change |
|------|--------|
| `agent/core.py` | Add `google_search_retrieval` tool config |
| System prompt | Add: "Use Google Search when the user asks about solar events, space weather, or context" |

### 6C: Already implemented (highlight in submission)

- **Structured JSON output**: Multi-step planner uses `response_mime_type: "application/json"`
- **Function calling**: 14 tools across 3 specialized agents

---

## Phase 7: Demo & Hackathon Polish (~2-3 hours)

### Demo Script (3 minutes)

| Time | Action | What Judges See |
|------|--------|-----------------|
| 0:00-0:20 | Title card + problem statement | "Scientists spend hours scripting. We made it conversational." |
| 0:20-0:50 | "Show me ACE magnetic field data for last week" | Chat message → interactive plotly plot appears |
| 0:50-1:20 | "Compute the magnetic field magnitude and overlay it" | Data pipeline: fetch → compute → overplot |
| 1:20-1:50 | "Now compare with Parker Solar Probe for the same period" | Multi-mission: two datasets on one plot |
| 1:50-2:20 | "Describe the data — any interesting features?" | `describe_data` → statistical summary in chat |
| 2:20-2:40 | "Export the plot and save the data as CSV" | PNG + CSV download links appear |
| 2:40-3:00 | Architecture slide + Gemini callout | 3-agent routing, function calling, 8 spacecraft |

### Devpost Page Structure

```markdown
# Helio AI Agent
> Talk to spacecraft data. Powered by Gemini.

## The Problem
Scientists spend hours writing scripts to find, download, process,
and visualize heliophysics data from NASA's spacecraft fleet.

## The Solution
A conversational AI agent that handles the entire workflow — from
"Show me Parker Solar Probe magnetic field data" to a publication-ready
plot — in seconds.

## How It Uses Gemini
- **Function calling**: 14 tools across 3 specialized agents
- **Multi-agent routing**: Orchestrator delegates to mission specialists
- **Multimodal**: Upload a plot screenshot, get it reproduced
- **Google Search grounding**: Real-time space weather context
- **JSON mode**: Structured task decomposition for complex queries

## Demo
[3-minute video link]

## Try It
[Gradio share link]
```

### Quick Wins

- **Loading indicators**: Show "Fetching ACE data..." in Gradio status panel
- **Token usage display**: Show token count in UI footer (already tracked)
- **Error recovery UX**: Friendly messages instead of tracebacks

---

## Phase 8: Spectrogram / Rank-2 Data Support (Stretch)

Currently `data_ops/fetch.py` only handles 1D HAPI parameters. Some datasets (energy spectrograms, pitch angle distributions) are rank-2.

### Extend HAPI fetch for rank-2 data

```python
# In fetch_hapi_data():
# If parameter has size=[N] where N > 1, reshape into (time, N) DataFrame
# Store with a "rank" field in DataEntry metadata
```

### Plotly spectrogram rendering

```python
# In plotly_renderer.py:
if render_type == "spectrogram":
    fig.add_trace(go.Heatmap(
        x=entry.data.index,
        y=energy_bins,     # from HAPI bins metadata
        z=entry.data.values.T,
        colorscale=color_table,
    ))
```

---

## File Change Summary

### New files

| File | Purpose |
|------|---------|
| `rendering/__init__.py` | Renderer factory |
| `rendering/plotly_renderer.py` | Plotly rendering (default) |
| `tests/test_plotly_renderer.py` | Unit tests (no API key or JVM needed) |
| `data_ops/autoplot_fetch.py` | Autoplot CDF-cached data fetching (Phase 4, optional) |

### Modified files

| File | Change |
|------|--------|
| `agent/core.py` | Replace `self.autoplot` with `self._renderer`; `plot_cdaweb` → fetch+plot; `--data-backend` flag; multimodal support; `get_plotly_figure()` accessor |
| `agent/autoplot_agent.py` | Remove `autoplot_script` from default tool set; simplify prompt |
| `agent/tools.py` | Remove or gate `autoplot_script` tool |
| `knowledge/prompt_builder.py` | Simplify `build_autoplot_prompt()` — no DOM/ScriptContext sections |
| `data_ops/fetch.py` | Add HAPI metadata capture; optional cache layer |
| `data_ops/store.py` | Add `metadata` field to `DataEntry`; add `get_last_entry_preview()` |
| `main.py` | Add `--data-backend` CLI flag |
| `gradio_app.py` | Replace `gr.Image` → `gr.Plot(fig)`; remove PNG snapshot logic; add multimodal chat |
| `requirements.txt` | Add `plotly>=5.18.0`, `kaleido>=0.2.1`; make `jpype1` optional |
| `config.py` | Make `AUTOPLOT_JAR` optional |

### Preserved (not modified, not deleted)

| File | Why kept |
|------|----------|
| `autoplot_bridge/connection.py` | Needed by Autoplot data backend (Phase 4); future rendering extension |
| `autoplot_bridge/commands.py` | Future rendering extension |
| `autoplot_bridge/registry.py` | Future rendering extension |
| `autoplot_bridge/script_runner.py` | Future rendering extension |

---

## Migration Path

### Phase 1 (~6-8 hours): Plotly rendering
1. Create `rendering/plotly_renderer.py` implementing all visualization methods
2. Create `tests/test_plotly_renderer.py` with unit tests
3. Wire `agent/core.py` to use `PlotlyRenderer` instead of `AutoplotCommands`
4. Update prompts (remove DOM/script references)
5. **Test**: Unit tests pass; existing agent test scenarios pass with plotly rendering

### Phase 2 (~2-3 hours): Upgrade Gradio to plotly
1. Replace `gr.Image` with `gr.Plot` in `gradio_app.py`
2. Remove PNG snapshot logic (`_snapshot_plot`, MD5 hash, tempfile)
3. Remove `concurrency_limit=1` and `os._exit(0)` workarounds
4. **Test**: `python gradio_app.py` shows interactive plotly plots with zoom/pan/hover

### Phase 3 (~2-3 hours): HAPI-only data by default
1. Make `plot_cdaweb` internally call `fetch_data` + `plot_stored_data`
2. Add HAPI metadata (fill values, valid ranges) to `DataEntry`
3. Plotly renderer uses metadata for axis labels and fill masking
4. **Test**: "Show me ACE mag data for last week" works end-to-end without JVM

### Phase 4 (~3-4 hours): Autoplot data backend (optional)
1. Implement `data_ops/autoplot_fetch.py` (QDataSet → pandas)
2. Add `--data-backend autoplot` CLI flag
3. **Test**: Same queries work with Autoplot data backend, leveraging CDF cache

### Phase 5 (~2-3 hours): HAPI-side caching
1. Add parquet file cache to `data_ops/fetch.py`
2. **Test**: Repeat queries load from cache instantly

### Phase 6 (~3-4 hours): Gemini differentiators
1. Add multimodal input support (image upload → Gemini analysis)
2. Add Google Search grounding
3. **Test**: Upload a plot screenshot, get it analyzed

### Phase 7 (~2-3 hours): Demo + hackathon polish
1. Record demo video with scripted scenario
2. Write Devpost page
3. Add loading indicators and token usage display

### Phase 8 (stretch): Rank-2 data + spectrograms
1. Extend HAPI fetch for multi-dimensional parameters
2. Add `go.Heatmap` rendering in plotly renderer

---

## What Gets Eliminated (Default Mode)

| Current problem | Resolution |
|---|---|
| JVM crashes on invalid parameters (ISSUE-02) | HAPI returns JSON errors, never reaches JVM |
| JVM crashes on 4+ panels (ISSUE-01) | Plotly handles unlimited subplots |
| JVM shutdown hangs (`os._exit(0)`) | No JVM in default mode |
| `_numpy_to_qdataset()` slow element-by-element loop | Plotly plots pandas directly |
| Static PNG snapshots in Gradio | Interactive plotly widget with zoom/pan/hover |
| `concurrency_limit=1` (JVM singleton) | Plotly is stateless per-figure; multi-user possible |
| Java 8 + JPype 1.5.x pinned dependency | Only needed with `--data-backend autoplot` |

## What Gets Preserved

| Feature | How |
|---|---|
| CDF file caching for fast repeat access | `--data-backend autoplot` (Phase 4) or HAPI parquet cache (Phase 5) |
| All Autoplot rendering code | Kept in `autoplot_bridge/` untouched for future extension |
| `autoplot_script` escape hatch | Available in future when Autoplot rendering is re-enabled |
| `.vap` session save/restore | Available in future when Autoplot rendering is re-enabled |

---

## CLI Interface

```bash
# Default: HAPI data + plotly rendering, no JVM needed
python main.py "Show me ACE magnetic field for last week"
python gradio_app.py

# Power user: Autoplot data backend (CDF cache, requires JVM)
python main.py --data-backend autoplot "Show me ACE mag data"
```

---

## Verification

After Phase 1-3:

1. `pip install -r requirements.txt` succeeds **without Java or JPype**
2. `python -m pytest tests/test_plotly_renderer.py` passes
3. `python main.py "Show me ACE mag data for last week"` produces an interactive plotly plot
4. `python gradio_app.py` shows interactive plots in browser with zoom/pan/hover
5. All 6 agent test scenarios pass
6. `describe_data` and `save_data` work on every dataset that was plotted (no more Java-trapped data)

After Phase 4:

7. `python main.py --data-backend autoplot "Show me ACE mag data"` uses CDF cache
8. Repeat queries with Autoplot backend are noticeably faster than HAPI

After Phase 6:

9. Uploading a plot screenshot triggers Gemini multimodal analysis
10. Asking "What solar events happened in Jan 2024?" uses Google Search grounding

## What NOT to Build

- Don't add more spacecraft or data sources (8 is enough for demo)
- Don't refactor the agent architecture (it's already good)
- Don't add authentication or multi-user support (single-user is fine for demo)
- Don't build a custom frontend (Gradio is faster and good enough)
- Don't build an MCP server — the value is the intelligence layer, not raw HAPI calls
