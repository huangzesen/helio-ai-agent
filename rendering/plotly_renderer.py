"""
Plotly-based renderer for visualization.

Provides all visualization methods dispatched by agent/core.py via the
method registry in rendering/registry.py.

State is kept in a mutable go.Figure that accumulates traces / layout
changes across calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from agent.time_utils import TimeRange
    from data_ops.store import DataEntry

# Threshold: above this many points per trace, use Scattergl (WebGL)
_GL_THRESHOLD = 100_000

# Default colour sequence (golden-ratio HSL spacing, pre-computed hex)
_DEFAULT_COLORS = [
    "#cc6633",  # hue=0.000
    "#55cc33",  # hue=0.618
    "#3384cc",  # hue=0.236
    "#a833cc",  # hue=0.854
    "#33cc98",  # hue=0.472
    "#cc3340",  # hue=0.090
    "#33cccc",  # hue=0.708
    "#ccbe33",  # hue=0.326
]


class PlotlyRenderer:
    """Stateful Plotly renderer that mirrors AutoplotCommands."""

    def __init__(self, verbose: bool = False, gui_mode: bool = False):
        self.verbose = verbose
        self.gui_mode = gui_mode
        self._figure: Optional[go.Figure] = None
        self._panel_count: int = 0
        self._current_time_range: Optional[TimeRange] = None
        self._label_colors: dict[str, str] = {}
        self._color_index: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [PlotlyRenderer] {msg}")
            sys.stdout.flush()

    def _next_color(self, label: str) -> str:
        """Return a stable colour for *label*, assigning a new one if unseen."""
        if label in self._label_colors:
            return self._label_colors[label]
        color = _DEFAULT_COLORS[self._color_index % len(_DEFAULT_COLORS)]
        self._color_index += 1
        self._label_colors[label] = color
        return color

    def _ensure_figure(self, rows: int = 1) -> go.Figure:
        """Guarantee a figure exists with at least *rows* subplot rows.

        Always uses make_subplots so that row/col args work on add_trace.
        """
        if self._figure is None or rows > self._panel_count:
            self._figure = make_subplots(
                rows=max(rows, 1), cols=1, shared_xaxes=True,
                vertical_spacing=0.06,
            )
            self._panel_count = max(rows, 1)
        return self._figure

    def _grow_panels(self, needed: int) -> go.Figure:
        """Rebuild with more rows if needed, copying existing traces."""
        if needed <= self._panel_count:
            return self._ensure_figure()

        old_fig = self._figure
        new_fig = make_subplots(
            rows=needed, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
        )

        # Copy traces from old figure, preserving row assignment
        if old_fig is not None:
            for trace in old_fig.data:
                # Plotly stores subplot info in xaxis/yaxis attributes
                new_fig.add_trace(trace, row=_row_of_trace(trace), col=1)
            # Copy layout properties we care about (title, axis labels)
            if old_fig.layout.title and old_fig.layout.title.text:
                new_fig.update_layout(title_text=old_fig.layout.title.text)

        self._figure = new_fig
        self._panel_count = needed
        return self._figure

    def _scatter_cls(self, n_points: int):
        """Return go.Scattergl for large datasets, go.Scatter otherwise."""
        return go.Scattergl if n_points > _GL_THRESHOLD else go.Scatter

    # ------------------------------------------------------------------
    # Core plot methods
    # ------------------------------------------------------------------

    def plot_dataset(
        self,
        entries: list[DataEntry],
        title: str = "",
        filename: str = "",
        index: int = -1,
    ) -> dict:
        """Plot one or more DataEntry timeseries.

        Vector entries (n, 3) are decomposed into scalar x/y/z components.
        Multiple series are overlaid by default. ``index >= 0`` targets a
        specific panel row.
        """
        if not entries:
            return {"status": "error", "message": "No entries to plot"}

        for entry in entries:
            if len(entry.time) == 0:
                return {"status": "error",
                        "message": f"Entry '{entry.label}' has no data points"}

        # Decompose vectors into scalar components
        series: list[tuple[str, np.ndarray, np.ndarray]] = []  # (label, time, values1d)
        for entry in entries:
            display_name = entry.description or entry.label
            if entry.values.ndim == 2 and entry.values.shape[1] > 1:
                comp_names = ["x", "y", "z"]
                for col in range(entry.values.shape[1]):
                    comp = comp_names[col] if col < 3 else str(col)
                    series.append((f"{display_name} ({comp})", entry.time, entry.values[:, col]))
            else:
                vals = entry.values.ravel() if entry.values.ndim > 1 else entry.values
                series.append((display_name, entry.time, vals))

        n_series = len(series)

        if index >= 0:
            # Panel-targeted mode
            needed = index + n_series
            fig = self._grow_panels(needed)
            for i, (label, time_arr, vals) in enumerate(series):
                row = index + i + 1  # plotly rows are 1-based
                Scatter = self._scatter_cls(len(vals))
                fig.add_trace(
                    Scatter(
                        x=time_arr, y=vals,
                        name=label, mode="lines",
                        line=dict(color=self._next_color(label)),
                    ),
                    row=row, col=1,
                )
                self._log(f"Panel {index + i}: '{label}'")
        else:
            # Overlay mode — all in row 1
            fig = self._ensure_figure()
            for label, time_arr, vals in series:
                Scatter = self._scatter_cls(len(vals))
                fig.add_trace(
                    Scatter(
                        x=time_arr, y=vals,
                        name=label, mode="lines",
                        line=dict(color=self._next_color(label)),
                    ),
                    row=1, col=1,
                )

        # Title
        if title:
            fig.update_layout(title_text=title)

        result: dict = {
            "status": "success",
            "labels": [s[0] for s in series],
            "num_series": n_series,
            "display": "plotly",
        }

        # Auto-export
        if filename:
            if not filename.endswith(".png"):
                filename += ".png"
            export_result = self.export_png(filename)
            if export_result.get("status") == "success":
                result["filepath"] = export_result["filepath"]
            else:
                result["export_warning"] = export_result.get("message", "Export failed")

        return result

    # ------------------------------------------------------------------
    # Axis / layout methods
    # ------------------------------------------------------------------

    def set_time_range(self, time_range: TimeRange) -> dict:
        tr_str = time_range.to_autoplot_string()
        self._log(f"Setting time range: {tr_str}")
        fig = self._ensure_figure()
        fig.update_xaxes(range=[time_range.start, time_range.end])
        self._current_time_range = time_range
        return {"status": "success", "time_range": tr_str}

    def set_plot_title(self, title: str) -> dict:
        self._log(f"Setting plot title: {title}")
        fig = self._ensure_figure()
        fig.update_layout(title_text=title)
        return {"status": "success", "title": title}

    def set_axis_label(self, axis: str, label: str) -> dict:
        self._log(f"Setting {axis}-axis label: {label}")
        if axis.lower() not in ("y", "z"):
            return {"status": "error", "message": f"Unsupported axis '{axis}'. Use 'y' or 'z'."}
        fig = self._ensure_figure()
        if axis.lower() == "y":
            fig.update_yaxes(title_text=label, row=1, col=1)
        else:
            # z-axis has no direct Plotly analog for line plots
            return {"status": "error",
                    "message": "z-axis label only applies to spectrograms, which are not yet supported in the Plotly renderer."}
        return {"status": "success", "axis": axis, "label": label}

    def toggle_log_scale(self, axis: str, enabled: bool) -> dict:
        self._log(f"Setting {axis}-axis log scale: {enabled}")
        if axis.lower() not in ("y", "z"):
            return {"status": "error", "message": f"Unsupported axis '{axis}'. Use 'y' or 'z'."}
        fig = self._ensure_figure()
        scale_type = "log" if enabled else "linear"
        if axis.lower() == "y":
            fig.update_yaxes(type=scale_type, row=1, col=1)
        return {"status": "success", "axis": axis, "log_scale": enabled}

    def set_axis_range(self, axis: str, min_val: float, max_val: float) -> dict:
        self._log(f"Setting {axis}-axis range: {min_val} to {max_val}")
        if axis.lower() not in ("y", "z"):
            return {"status": "error", "message": f"Unsupported axis '{axis}'. Use 'y' or 'z'."}
        fig = self._ensure_figure()
        if axis.lower() == "y":
            fig.update_yaxes(range=[min_val, max_val], row=1, col=1)
        return {"status": "success", "axis": axis, "min": min_val, "max": max_val}

    # ------------------------------------------------------------------
    # Render type
    # ------------------------------------------------------------------

    def set_render_type(self, render_type: str, index: int = 0) -> dict:
        """Map Autoplot render types to Plotly trace modes."""
        self._log(f"Setting render type to '{render_type}' for trace {index}")
        fig = self._ensure_figure()
        if index >= len(fig.data):
            return {"status": "error",
                    "message": f"Trace index {index} out of range (have {len(fig.data)} traces)"}

        mode_map = {
            "series": "lines",
            "scatter": "markers",
            "fill_to_zero": "lines",
            "staircase": "lines",
            "digital": "lines",
        }
        mode = mode_map.get(render_type)
        if mode is None:
            return {"status": "error",
                    "message": f"Render type '{render_type}' is not supported by the Plotly renderer."}

        trace = fig.data[index]
        trace.mode = mode

        # Fill and step shapes
        if render_type == "fill_to_zero":
            trace.fill = "tozeroy"
        elif render_type == "staircase":
            trace.line = dict(shape="hv", **(dict(color=trace.line.color) if trace.line and trace.line.color else {}))
        elif render_type == "digital":
            trace.line = dict(shape="hv", **(dict(color=trace.line.color) if trace.line and trace.line.color else {}))

        return {"status": "success", "render_type": render_type, "index": index}

    def set_color_table(self, name: str) -> dict:
        return {
            "status": "error",
            "message": "Color tables are not supported by the Plotly renderer (only for spectrograms).",
        }

    def set_canvas_size(self, width: int, height: int) -> dict:
        self._log(f"Setting canvas size to {width}x{height}")
        fig = self._ensure_figure()
        fig.update_layout(width=width, height=height)
        return {"status": "success", "width": width, "height": height}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_png(self, filepath: str) -> dict:
        filepath = str(Path(filepath).resolve())
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        if self._figure is None or len(self._figure.data) == 0:
            return {"status": "error",
                    "message": "No plot to export. Plot data first before exporting."}

        self._log(f"Exporting PNG to {filepath}...")
        try:
            self._figure.write_image(filepath, format="png")
        except Exception as e:
            return {"status": "error", "message": f"PNG export failed: {e}"}

        path_obj = Path(filepath)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            return {
                "status": "success",
                "filepath": str(path_obj.resolve()),
                "size_bytes": path_obj.stat().st_size,
            }
        return {"status": "error", "message": f"PNG file not created or is empty: {filepath}"}

    def export_pdf(self, filepath: str) -> dict:
        if not filepath.endswith(".pdf"):
            filepath += ".pdf"
        filepath = str(Path(filepath).resolve())
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        if self._figure is None or len(self._figure.data) == 0:
            return {"status": "error",
                    "message": "No plot to export. Plot data first before exporting."}

        self._log(f"Exporting PDF to {filepath}...")
        try:
            self._figure.write_image(filepath, format="pdf")
        except Exception as e:
            return {"status": "error", "message": f"PDF export failed: {e}"}

        path_obj = Path(filepath)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            return {
                "status": "success",
                "filepath": str(path_obj.resolve()),
                "size_bytes": path_obj.stat().st_size,
            }
        return {"status": "error", "message": f"PDF file not created or is empty: {filepath}"}

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        self._log("Resetting canvas...")
        self._figure = None
        self._panel_count = 0
        self._current_time_range = None
        self._label_colors.clear()
        self._color_index = 0
        return {"status": "success", "message": "Canvas reset."}

    def get_current_state(self) -> dict:
        tr_str = self._current_time_range.to_autoplot_string() if self._current_time_range else None
        return {
            "uri": None,
            "time_range": tr_str,
            "panel_count": self._panel_count,
            "has_plot": self._figure is not None and len(self._figure.data) > 0,
        }

    # ------------------------------------------------------------------
    # Not-supported passthrough stubs
    # ------------------------------------------------------------------

    def save_session(self, filepath: str) -> dict:
        return {"status": "error", "message": "Session save (.vap) is not supported by the Plotly renderer."}

    def load_session(self, filepath: str) -> dict:
        return {"status": "error", "message": "Session load (.vap) is not supported by the Plotly renderer."}

    def execute_script(self, code: str) -> dict:
        return {"status": "error",
                "message": "Direct ScriptContext/DOM scripting is not supported by the Plotly renderer. "
                           "Use execute_visualization methods instead."}

    # ------------------------------------------------------------------
    # Accessor for Gradio / external use
    # ------------------------------------------------------------------

    def get_figure(self) -> Optional[go.Figure]:
        """Return the current Plotly figure (or None if nothing plotted)."""
        return self._figure


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _row_of_trace(trace) -> int:
    """Determine which subplot row a trace belongs to (1-based).

    Plotly stores this as xaxis='x2', yaxis='y2' for row 2, etc.
    """
    yaxis = getattr(trace, "yaxis", None) or "y"
    # 'y' -> row 1, 'y2' -> row 2, 'y3' -> row 3
    suffix = yaxis.replace("y", "")
    return int(suffix) if suffix else 1
