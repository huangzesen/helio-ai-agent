"""
Plotly-based renderer for visualization.

The ``fill_figure_data()`` function resolves ``data_label`` placeholders
in LLM-generated Plotly figure JSON, populating real data arrays.

The ``PlotlyRenderer`` class is a thin stateful wrapper providing:
- ``render_plotly_json()`` — fill data → copy into state
- ``export()``, ``reset()``, ``set_time_range()`` — structural operations
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_ops.store import DataEntry

if TYPE_CHECKING:
    from agent.time_utils import TimeRange

# Explicit layout defaults — prevent Gradio dark theme from overriding
_DEFAULT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="#2a3f5f",
    autosize=False,
)

_PANEL_HEIGHT = 300  # px per subplot panel
_DEFAULT_WIDTH = 1100  # px figure width


# ---------------------------------------------------------------------------
# RenderResult — return type of fill_figure_data
# ---------------------------------------------------------------------------

class RenderResult:
    """Result of a stateless fill_figure_data() call."""

    __slots__ = ("figure", "trace_labels", "panel_count")

    def __init__(
        self,
        figure: go.Figure,
        trace_labels: list[str],
        panel_count: int,
    ):
        self.figure = figure
        self.trace_labels = trace_labels
        self.panel_count = panel_count


# ---------------------------------------------------------------------------
# Stateless figure builder
# ---------------------------------------------------------------------------

def _extract_x_data(entry: DataEntry) -> list:
    """Extract x-axis values from a DataEntry.

    DatetimeIndex / xarray time coords → ISO 8601 strings.
    Numeric or other indices → raw values as list.
    """
    if entry.is_xarray:
        import pandas as pd
        return [pd.Timestamp(t).isoformat() for t in entry.data.coords["time"].values]
    idx = entry.data.index
    if hasattr(idx, 'tz') or str(idx.dtype).startswith('datetime'):
        return [t.isoformat() for t in idx]
    return list(idx)


# ---------------------------------------------------------------------------
# Fill function — resolves data_label placeholders in Plotly JSON
# ---------------------------------------------------------------------------

def fill_figure_data(
    fig_json: dict,
    entries: dict[str, DataEntry],
    time_range: str | None = None,
) -> RenderResult:
    """Fill data_label placeholders in a Plotly figure JSON with actual data.

    The LLM generates a Plotly figure JSON where each trace has a ``data_label``
    field instead of actual x/y/z arrays.  This function resolves those
    placeholders by looking up the corresponding DataEntry objects, extracting
    time and value arrays, and populating the trace dicts.

    Handles:
    - Time extraction (DatetimeIndex/xarray → ISO 8601 strings)
    - NaN → None conversion (Plotly requirement)
    - Spectrogram/heatmap data population (x=times, y=bins, z=values)

    For multi-column (vector) data, the LLM must emit one trace per column.
    If a single trace references multi-column data, an error is raised.

    Args:
        fig_json: Plotly figure dict with ``data`` and ``layout`` keys.
            Each trace in ``data`` must have a ``data_label`` field.
        entries: Mapping of label → DataEntry for all referenced labels.
        time_range: Optional time range string (``"start to end"``).

    Returns:
        RenderResult with the constructed go.Figure and metadata.
    """
    layout = fig_json.get("layout", {})
    raw_traces = fig_json.get("data", [])

    filled_traces: list[dict] = []
    trace_labels: list[str] = []

    for trace_dict in raw_traces:
        trace = dict(trace_dict)  # shallow copy
        label = trace.pop("data_label", None)
        if label is None:
            # Trace without data_label — pass through as-is (e.g., shapes)
            filled_traces.append(trace)
            continue

        # Resolve entry
        entry = entries.get(label)
        if entry is None:
            raise ValueError(f"data_label '{label}' not found in provided entries")

        trace_type = trace.get("type", "scatter").lower()
        is_heatmap = trace_type in ("heatmap", "heatmapgl")

        display_name = trace.get("name") or entry.description or entry.label

        if is_heatmap:
            _fill_heatmap_trace(trace, entry)
        else:
            # Reject multi-column data — LLM must emit one trace per column
            if entry.values.ndim == 2 and entry.values.shape[1] > 1:
                ncols = entry.values.shape[1]
                col_names = list(entry.data.columns) if hasattr(entry.data, 'columns') else [str(i) for i in range(ncols)]
                raise ValueError(
                    f"Entry '{entry.label}' has {ncols} columns {col_names}. "
                    f"Emit one trace per column using a separate data_label for each, "
                    f"or use custom_operation to select a single column first."
                )
            vals = entry.values.ravel() if entry.values.ndim > 1 else entry.values
            trace["x"] = _extract_x_data(entry)
            trace["y"] = [float(v) if np.isfinite(v) else None for v in vals]
            trace.setdefault("mode", "lines")

        trace["name"] = display_name
        filled_traces.append(trace)
        trace_labels.append(display_name)

    # Count panels from layout
    n_panels = sum(1 for key in layout if key.startswith("yaxis"))
    n_panels = max(n_panels, 1)
    n_columns = sum(1 for key in layout if key.startswith("xaxis"))
    n_columns = max(n_columns, 1)

    # Apply default layout settings
    width = (_DEFAULT_WIDTH if n_columns == 1
             else int(_DEFAULT_WIDTH * n_columns * 0.55))
    height = _PANEL_HEIGHT * n_panels

    defaults = dict(
        **_DEFAULT_LAYOUT,
        width=width,
        height=height,
        legend=dict(font=dict(size=11), tracegroupgap=2),
    )
    # Merge defaults under layout (user layout wins)
    merged_layout = {**defaults, **layout}

    # Apply time range constraint
    if time_range:
        parts = time_range.split(" to ")
        if len(parts) == 2:
            start, end = parts[0].strip(), parts[1].strip()
            # Apply to all xaxes
            for key in list(merged_layout.keys()):
                if key.startswith("xaxis"):
                    ax = merged_layout[key]
                    if isinstance(ax, dict):
                        ax.setdefault("range", [start, end])

    fig = go.Figure({"data": filled_traces, "layout": merged_layout})

    return RenderResult(
        figure=fig,
        trace_labels=trace_labels,
        panel_count=n_panels,
    )


def _fill_heatmap_trace(
    trace: dict,
    entry: DataEntry,
) -> None:
    """Populate a heatmap trace dict with data from a DataEntry."""
    meta = entry.metadata or {}
    times = _extract_x_data(entry)

    if entry.is_xarray:
        da = entry.data
        non_time_dims = [d for d in da.dims if d != "time"]
        if non_time_dims:
            last_dim = non_time_dims[-1]
            if last_dim in da.coords:
                bin_values = [float(v) for v in da.coords[last_dim].values]
            else:
                bin_values = list(range(da.sizes[last_dim]))
        else:
            bin_values = [0]
        z_values = da.values.astype(float)
        if z_values.ndim > 2:
            middle_axes = tuple(range(1, z_values.ndim - 1))
            z_values = np.nanmean(z_values, axis=middle_axes)
    else:
        bin_values = meta.get("bin_values")
        if bin_values is None:
            try:
                bin_values = [float(c) for c in entry.data.columns]
            except (ValueError, TypeError):
                bin_values = list(range(len(entry.data.columns)))
        z_values = entry.data.values.astype(float)

    trace["x"] = times
    trace["y"] = [float(b) for b in bin_values]
    trace["z"] = z_values.T.tolist()


class PlotlyRenderer:
    """Stateful Plotly renderer for heliophysics data visualization."""

    def __init__(self, verbose: bool = False, gui_mode: bool = False):
        self.verbose = verbose
        self.gui_mode = gui_mode
        self._figure: Optional[go.Figure] = None
        self._panel_count: int = 0
        self._current_time_range: Optional[TimeRange] = None
        self._trace_labels: list[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [PlotlyRenderer] {msg}")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Time range
    # ------------------------------------------------------------------

    def set_time_range(self, time_range: TimeRange) -> dict:
        tr_str = time_range.to_time_range_string()
        self._log(f"Setting time range: {tr_str}")
        if self._figure is not None:
            self._figure.update_xaxes(range=[time_range.start, time_range.end])
        self._current_time_range = time_range
        return {"status": "success", "time_range": tr_str}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, filepath: str, format: str = "png") -> dict:
        """Export the current plot to a file (PNG or PDF).

        Args:
            filepath: Output file path.
            format: 'png' (default) or 'pdf'.

        Returns:
            Result dict with status, filepath, and size_bytes.
        """
        # Ensure correct extension
        if format == "pdf" and not filepath.endswith(".pdf"):
            filepath += ".pdf"
        elif format == "png" and not filepath.endswith(".png"):
            filepath += ".png"

        filepath = str(Path(filepath).resolve())
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        if self._figure is None or len(self._figure.data) == 0:
            return {"status": "error",
                    "message": "No plot to export. Plot data first before exporting."}

        self._log(f"Exporting {format.upper()} to {filepath}...")
        try:
            self._figure.write_image(filepath, format=format)
        except Exception as e:
            return {"status": "error", "message": f"{format.upper()} export failed: {e}"}

        path_obj = Path(filepath)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            return {
                "status": "success",
                "filepath": str(path_obj.resolve()),
                "size_bytes": path_obj.stat().st_size,
            }
        return {"status": "error", "message": f"{format.upper()} file not created or is empty: {filepath}"}

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        self._log("Resetting canvas...")
        self._figure = None
        self._panel_count = 0
        self._current_time_range = None
        self._trace_labels.clear()
        return {"status": "success", "message": "Canvas reset."}

    def get_current_state(self) -> dict:
        tr_str = self._current_time_range.to_time_range_string() if self._current_time_range else None
        return {
            "uri": None,
            "time_range": tr_str,
            "panel_count": self._panel_count,
            "has_plot": self._figure is not None and len(self._figure.data) > 0,
            "traces": list(self._trace_labels),
        }

    # ------------------------------------------------------------------
    # Public API: render_plotly_json
    # ------------------------------------------------------------------

    def render_plotly_json(
        self,
        fig_json: dict,
        entries: dict[str, DataEntry],
    ) -> dict:
        """Create a plot from LLM-generated Plotly figure JSON.

        The LLM produces a Plotly figure dict with ``data_label`` placeholders
        in each trace.  This method resolves them via ``fill_figure_data()``,
        copies the result into renderer state, and returns basic metadata.

        Args:
            fig_json: Plotly figure dict (``data`` + ``layout``).
            entries: Mapping of label → DataEntry for all referenced labels.

        Returns:
            Result dict with status, panels, traces, display.
        """
        tr_str = (self._current_time_range.to_time_range_string()
                  if self._current_time_range else None)

        try:
            build_result = fill_figure_data(fig_json, entries, time_range=tr_str)
        except (ValueError, KeyError) as e:
            return {"status": "error", "message": str(e)}

        # Copy into stateful fields
        self._figure = build_result.figure
        self._panel_count = build_result.panel_count
        self._trace_labels = build_result.trace_labels

        # Build basic trace info for the LLM
        trace_info = []
        for i, trace in enumerate(self._figure.data):
            name = self._trace_labels[i] if i < len(self._trace_labels) else (trace.name or f"trace_{i}")
            y = trace.y
            if y is not None:
                points = len(y)
            elif hasattr(trace, 'z') and trace.z is not None:
                z_arr = np.asarray(trace.z)
                points = f"{z_arr.shape[0]}x{z_arr.shape[1]}" if z_arr.ndim == 2 else len(trace.z)
            else:
                points = 0
            trace_info.append({"name": name, "points": points})

        return {
            "status": "success",
            "panels": build_result.panel_count,
            "traces": list(build_result.trace_labels),
            "trace_info": trace_info,
            "display": "plotly",
        }

    # ------------------------------------------------------------------
    # Accessor for Gradio / external use
    # ------------------------------------------------------------------

    def get_figure(self) -> Optional[go.Figure]:
        """Return the current Plotly figure (or None if nothing plotted)."""
        return self._figure

    # ------------------------------------------------------------------
    # Serialization for session persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict | None:
        """Serialize the renderer state (figure + metadata) to a dict.

        Returns None if there is no figure to save.
        """
        if self._figure is None:
            return None
        tr = self._current_time_range
        return {
            "figure_json": self._figure.to_json(),
            "panel_count": self._panel_count,
            "time_range": tr.to_time_range_string() if tr else None,
            "trace_labels": list(self._trace_labels),
        }

    def restore_state(self, state: dict) -> None:
        """Restore renderer state from a dict produced by save_state()."""
        import plotly.io as pio

        fig_json = state.get("figure_json")
        if not fig_json:
            return

        self._figure = pio.from_json(fig_json)
        self._panel_count = state.get("panel_count", 0)
        self._trace_labels = state.get("trace_labels", [])

        tr_str = state.get("time_range")
        if tr_str:
            try:
                from agent.time_utils import parse_time_range
                self._current_time_range = parse_time_range(tr_str)
            except Exception:
                self._current_time_range = None
        else:
            self._current_time_range = None
