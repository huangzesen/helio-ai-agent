"""
Plotly-based renderer for visualization.

Provides three public methods dispatched by agent/core.py:
- plot_data() — create plots (single panel overlay or multi-panel)
- style() — apply aesthetics via key-value params (no code gen)
- manage() — structural ops: export, reset, zoom, add/remove traces

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

# Maximum points per trace for interactive display.
# Above this, traces are downsampled using min-max decimation to keep
# visual features (peaks, dips) while reducing JSON payload size.
_MAX_DISPLAY_POINTS = 5_000

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

def _downsample_minmax(time_arr, val_arr, max_points: int = _MAX_DISPLAY_POINTS):
    """Downsample using min-max decimation to preserve peaks and dips.

    Splits data into buckets and keeps the min and max value from each
    bucket, preserving the visual envelope of the signal.

    Returns (time_out, val_out) as lists ready for Plotly.
    """
    n = len(val_arr)
    if n <= max_points:
        return time_arr, val_arr

    # Each bucket contributes 2 points (min + max), so use half as many buckets
    n_buckets = max_points // 2
    bucket_size = n / n_buckets

    indices = []
    for i in range(n_buckets):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        end = min(end, n)
        if start >= end:
            continue
        chunk = val_arr[start:end]
        # Handle all-NaN buckets
        finite_mask = np.isfinite(chunk)
        if not finite_mask.any():
            indices.append(start)
            continue
        idx_min = start + np.nanargmin(chunk)
        idx_max = start + np.nanargmax(chunk)
        # Add in chronological order
        if idx_min <= idx_max:
            indices.extend([idx_min, idx_max])
        else:
            indices.extend([idx_max, idx_min])

    # Deduplicate and sort
    indices = sorted(set(indices))
    return [time_arr[i] for i in indices], [val_arr[i] for i in indices]


# Explicit layout defaults — prevent Gradio dark theme from overriding
_DEFAULT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="#2a3f5f",
)

_LEGEND_MAX_LINE = 30  # max chars per line in legend


def _wrap_display_name(name: str, max_line: int = _LEGEND_MAX_LINE) -> str:
    """Wrap a long display name with <br> for multi-line Plotly legends."""
    if len(name) <= max_line:
        return name
    words = name.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_line and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "<br>".join(lines)


class PlotlyRenderer:
    """Stateful Plotly renderer for heliophysics data visualization."""

    def __init__(self, verbose: bool = False, gui_mode: bool = False):
        self.verbose = verbose
        self.gui_mode = gui_mode
        self._figure: Optional[go.Figure] = None
        self._panel_count: int = 0
        self._current_time_range: Optional[TimeRange] = None
        self._label_colors: dict[str, str] = {}
        self._color_index: int = 0
        # Trace tracking: parallel to fig.data
        self._trace_labels: list[str] = []
        self._trace_panels: list[int] = []  # 1-based row per trace

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
            self._figure.update_layout(
                **_DEFAULT_LAYOUT,
                legend=dict(font=dict(size=11), tracegroupgap=2),
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
        new_fig.update_layout(
            **_DEFAULT_LAYOUT,
            legend=dict(font=dict(size=11), tracegroupgap=2),
        )

        # Copy traces from old figure, preserving row assignment
        if old_fig is not None:
            for i, trace in enumerate(old_fig.data):
                row = self._trace_panels[i] if i < len(self._trace_panels) else _row_of_trace(trace)
                new_fig.add_trace(trace, row=row, col=1)
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
    # Internal helpers (used by plot_data)
    # ------------------------------------------------------------------

    def _add_line_traces(
        self,
        entries: list[DataEntry],
        row: int,
        fig: go.Figure,
    ) -> list[str]:
        """Add line traces for entries to a specific panel row.

        Returns the list of trace labels added.
        """
        added_labels: list[str] = []

        for entry in entries:
            display_name = entry.description or entry.label
            # Use pandas index (Timestamps with .isoformat()) not numpy
            # datetime64 (which .tolist() converts to nanosecond ints).
            time_list = [t.isoformat() for t in entry.data.index]

            # Decompose vectors into scalar components
            if entry.values.ndim == 2 and entry.values.shape[1] > 1:
                comp_names = ["x", "y", "z"]
                for col in range(entry.values.shape[1]):
                    comp = comp_names[col] if col < 3 else str(col)
                    label = f"{display_name} ({comp})"
                    val_arr = entry.values[:, col]
                    t_disp, v_disp = _downsample_minmax(
                        time_list, val_arr, _MAX_DISPLAY_POINTS,
                    )
                    v_disp = [float(v) if np.isfinite(v) else None for v in v_disp]
                    Scatter = self._scatter_cls(len(v_disp))
                    fig.add_trace(
                        Scatter(
                            x=t_disp, y=v_disp,
                            name=_wrap_display_name(label),
                            mode="lines",
                            line=dict(color=self._next_color(label)),
                        ),
                        row=row, col=1,
                    )
                    self._trace_labels.append(label)
                    self._trace_panels.append(row)
                    added_labels.append(label)
            else:
                vals = entry.values.ravel() if entry.values.ndim > 1 else entry.values
                t_disp, v_disp = _downsample_minmax(
                    time_list, vals, _MAX_DISPLAY_POINTS,
                )
                v_disp = [float(v) if np.isfinite(v) else None for v in v_disp]
                Scatter = self._scatter_cls(len(v_disp))
                fig.add_trace(
                    Scatter(
                        x=t_disp, y=v_disp,
                        name=_wrap_display_name(display_name),
                        mode="lines",
                        line=dict(color=self._next_color(display_name)),
                    ),
                    row=row, col=1,
                )
                self._trace_labels.append(display_name)
                self._trace_panels.append(row)
                added_labels.append(display_name)

        return added_labels

    def _add_spectrogram_trace(
        self,
        entry: DataEntry,
        row: int,
        fig: go.Figure,
        colorscale: str = "Viridis",
        log_y: bool = False,
        log_z: bool = False,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> str:
        """Add a spectrogram heatmap trace to a specific panel row.

        Returns the trace label.
        """
        times = [t.isoformat() for t in entry.data.index]

        meta = entry.metadata or {}
        bin_values = meta.get("bin_values")
        if bin_values is None:
            try:
                bin_values = [float(c) for c in entry.data.columns]
            except (ValueError, TypeError):
                bin_values = list(range(len(entry.data.columns)))

        z_values = entry.data.values.astype(float)
        if log_z:
            z_values = np.where(z_values > 0, np.log10(z_values), np.nan)

        z_data = z_values.T.tolist()
        bin_list = [float(b) for b in bin_values]

        colorbar_title = meta.get("value_label", "")
        if log_z and colorbar_title:
            colorbar_title = f"log\u2081\u2080({colorbar_title})"
        elif log_z:
            colorbar_title = "log\u2081\u2080(intensity)"

        label = entry.description or entry.label

        heatmap = go.Heatmap(
            x=times,
            y=bin_list,
            z=z_data,
            colorscale=colorscale,
            colorbar=dict(title=dict(text=colorbar_title)) if colorbar_title else None,
            zmin=z_min,
            zmax=z_max,
            name=label,
        )

        fig.add_trace(heatmap, row=row, col=1)
        self._trace_labels.append(label)
        self._trace_panels.append(row)

        fig.update_xaxes(type="date", row=row, col=1)

        y_axis_title = meta.get("bin_label", "")
        if y_axis_title:
            fig.update_yaxes(title_text=y_axis_title, row=row, col=1)

        if log_y:
            fig.update_yaxes(type="log", row=row, col=1)

        self._log(f"Spectrogram '{entry.label}': {len(times)} time steps x {len(bin_list)} bins")

        return label

    # ------------------------------------------------------------------
    # Public API: plot_data
    # ------------------------------------------------------------------

    def plot_data(
        self,
        entries: list[DataEntry],
        panels: list[list[str]] | None = None,
        title: str = "",
        plot_type: str = "line",
        colorscale: str = "Viridis",
        log_y: bool = False,
        log_z: bool = False,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> dict:
        """Create a fresh plot from DataEntry objects.

        Args:
            entries: All data to plot.
            panels: Panel layout, e.g. [["A","B"], ["C"]] for 2-panel.
                    None = overlay all in one panel.
            title: Plot title.
            plot_type: "line" (default) or "spectrogram".
            colorscale: Plotly colorscale for spectrogram.
            log_y: Log scale on y-axis (spectrogram).
            log_z: Log scale on z-axis (spectrogram).
            z_min: Min value for spectrogram color scale.
            z_max: Max value for spectrogram color scale.

        Returns:
            Result dict with status, panels, traces, display.
        """
        if not entries:
            return {"status": "error", "message": "No entries to plot"}

        for entry in entries:
            if len(entry.time) == 0:
                return {"status": "error",
                        "message": f"Entry '{entry.label}' has no data points"}

        # Build label -> entry lookup
        entry_map: dict[str, DataEntry] = {}
        for entry in entries:
            entry_map[entry.label] = entry

        # Reset figure for fresh plot
        self._figure = None
        self._panel_count = 0
        self._trace_labels = []
        self._trace_panels = []
        # Keep label_colors for stable coloring across plot_data calls

        all_trace_labels: list[str] = []

        if panels is not None:
            # Multi-panel mode
            n_panels = len(panels)
            fig = self._ensure_figure(rows=n_panels)

            for panel_idx, panel_labels in enumerate(panels):
                row = panel_idx + 1  # 1-based
                panel_entries = []
                for lbl in panel_labels:
                    e = entry_map.get(lbl)
                    if e is None:
                        return {"status": "error",
                                "message": f"Label '{lbl}' not found in provided entries"}
                    panel_entries.append(e)

                if plot_type == "spectrogram":
                    for e in panel_entries:
                        label = self._add_spectrogram_trace(
                            e, row, fig,
                            colorscale=colorscale, log_y=log_y, log_z=log_z,
                            z_min=z_min, z_max=z_max,
                        )
                        all_trace_labels.append(label)
                else:
                    added = self._add_line_traces(panel_entries, row, fig)
                    all_trace_labels.extend(added)
        else:
            # Overlay mode — all in row 1
            fig = self._ensure_figure(rows=1)

            if plot_type == "spectrogram":
                for e in entries:
                    label = self._add_spectrogram_trace(
                        e, 1, fig,
                        colorscale=colorscale, log_y=log_y, log_z=log_z,
                        z_min=z_min, z_max=z_max,
                    )
                    all_trace_labels.append(label)
            else:
                added = self._add_line_traces(entries, 1, fig)
                all_trace_labels.extend(added)

        # Ensure the x-axis is rendered as formatted dates
        fig.update_xaxes(type="date")

        # Apply stored time range to the new figure
        if self._current_time_range:
            fig.update_xaxes(range=[
                self._current_time_range.start.isoformat(),
                self._current_time_range.end.isoformat(),
            ])

        if title:
            fig.update_layout(title_text=title)

        result = {
            "status": "success",
            "panels": self._panel_count,
            "traces": all_trace_labels,
            "display": "plotly",
        }
        result["review"] = self._build_review_metadata()
        return result

    # ------------------------------------------------------------------
    # Public API: style
    # ------------------------------------------------------------------

    def style(
        self,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | dict | None = None,
        trace_colors: dict | None = None,
        line_styles: dict | None = None,
        log_scale: str | None = None,
        x_range: list | dict | None = None,
        y_range: list | dict | None = None,
        legend: bool | None = None,
        font_size: int | None = None,
        canvas_size: dict | None = None,
        annotations: list | None = None,
        colorscale: str | None = None,
        theme: str | None = None,
        vlines: list | None = None,
    ) -> dict:
        """Apply aesthetic changes to the current figure.

        All parameters are optional. Changes are applied in place.

        Args:
            title: Plot title.
            x_label: X-axis label (applied to bottom axis).
            y_label: Y-axis label. String (all panels) or dict {panel_num: label}.
            trace_colors: Dict mapping trace label -> color string.
            line_styles: Dict mapping trace label -> {width, dash, mode}.
            log_scale: "x", "y", "both", or "linear".
            x_range: [min, max] list, or dict {panel_num: [min, max]}.
            y_range: [min, max] list, or dict {panel_num: [min, max]}.
            legend: Show or hide legend.
            font_size: Global font size.
            canvas_size: {width: int, height: int}.
            annotations: List of {text, x, y} dicts.
            colorscale: Plotly colorscale name (for heatmap traces).
            theme: Plotly template name (e.g. "plotly_dark").
            vlines: List of {x, label, color, dash, width} dicts for vertical lines.

        Returns:
            Result dict with status.
        """
        if self._figure is None or len(self._figure.data) == 0:
            return {"status": "error",
                    "message": "No plot to style. Use plot_data first."}

        fig = self._figure

        if title is not None:
            fig.update_layout(title_text=title)

        if x_label is not None:
            # Apply to the bottom-most x-axis
            fig.update_xaxes(title_text=x_label, row=self._panel_count, col=1)

        if y_label is not None:
            if isinstance(y_label, dict):
                for panel_str, label_text in y_label.items():
                    panel = int(panel_str)
                    fig.update_yaxes(title_text=_wrap_display_name(str(label_text)), row=panel, col=1)
            else:
                wrapped = _wrap_display_name(str(y_label))
                for row in range(1, self._panel_count + 1):
                    fig.update_yaxes(title_text=wrapped, row=row, col=1)

        if trace_colors is not None:
            for trace_label, color in trace_colors.items():
                for i, tl in enumerate(self._trace_labels):
                    if tl == trace_label and i < len(fig.data):
                        trace = fig.data[i]
                        if hasattr(trace, 'line'):
                            trace.line.color = color
                        elif hasattr(trace, 'marker'):
                            trace.marker.color = color

        if line_styles is not None:
            for trace_label, style_dict in line_styles.items():
                for i, tl in enumerate(self._trace_labels):
                    if tl == trace_label and i < len(fig.data):
                        trace = fig.data[i]
                        if "width" in style_dict and hasattr(trace, 'line'):
                            trace.line.width = style_dict["width"]
                        if "dash" in style_dict and hasattr(trace, 'line'):
                            trace.line.dash = style_dict["dash"]
                        if "mode" in style_dict:
                            trace.mode = style_dict["mode"]

        if log_scale is not None:
            if log_scale == "y":
                for row in range(1, self._panel_count + 1):
                    fig.update_yaxes(type="log", row=row, col=1)
            elif log_scale == "linear":
                for row in range(1, self._panel_count + 1):
                    fig.update_yaxes(type="linear", row=row, col=1)

        if x_range is not None:
            if isinstance(x_range, dict):
                for panel_str, rng in x_range.items():
                    fig.update_xaxes(range=rng, row=int(panel_str), col=1)
            else:
                fig.update_xaxes(range=x_range)

        if y_range is not None:
            if isinstance(y_range, dict):
                for panel_str, rng in y_range.items():
                    fig.update_yaxes(range=rng, row=int(panel_str), col=1)
            else:
                for row in range(1, self._panel_count + 1):
                    fig.update_yaxes(range=y_range, row=row, col=1)

        if legend is not None:
            fig.update_layout(showlegend=legend)

        if font_size is not None:
            fig.update_layout(font=dict(size=font_size))

        if canvas_size is not None:
            fig.update_layout(
                width=canvas_size.get("width"),
                height=canvas_size.get("height"),
            )

        if annotations is not None:
            for ann in annotations:
                fig.add_annotation(
                    x=ann.get("x"),
                    y=ann.get("y"),
                    text=ann.get("text", ""),
                    showarrow=ann.get("showarrow", True),
                )

        if colorscale is not None:
            for trace in fig.data:
                if isinstance(trace, go.Heatmap):
                    trace.colorscale = colorscale

        if theme is not None:
            fig.update_layout(template=theme)

        if vlines is not None:
            for vl in vlines:
                x_val = vl.get("x")
                if x_val is None:
                    continue
                color = vl.get("color", "white")
                width = vl.get("width", 1.5)
                dash = vl.get("dash", "dash")
                label = vl.get("label")
                # Draw line across all panels
                for row in range(1, self._panel_count + 1):
                    fig.add_vline(
                        x=x_val, row=row, col=1,
                        line_width=width, line_dash=dash, line_color=color,
                    )
                # Add text annotation at top panel if label provided
                if label:
                    fig.add_annotation(
                        x=x_val, y=1.02, yref="paper",
                        text=label, showarrow=False,
                        font=dict(size=11, color=color),
                    )

        return {"status": "success", "message": "Style applied.", "display": "plotly"}

    # ------------------------------------------------------------------
    # Public API: manage
    # ------------------------------------------------------------------

    def manage(self, action: str, **kwargs) -> dict:
        """Perform structural operations on the plot.

        Args:
            action: One of "reset", "get_state", "set_time_range",
                    "export", "remove_trace", "add_trace".
            **kwargs: Action-specific parameters.

        Returns:
            Result dict with status.
        """
        if action == "reset":
            return self.reset()

        elif action == "get_state":
            return self.get_current_state()

        elif action == "set_time_range":
            time_range = kwargs.get("time_range")
            if time_range is None:
                return {"status": "error", "message": "time_range is required"}
            return self.set_time_range(time_range)

        elif action == "export":
            filename = kwargs.get("filename", "output.png")
            fmt = kwargs.get("format", "png")
            return self.export(filename, format=fmt)

        elif action == "remove_trace":
            label = kwargs.get("label")
            if label is None:
                return {"status": "error", "message": "label is required"}
            return self._remove_trace(label)

        elif action == "add_trace":
            entry = kwargs.get("entry")
            panel = kwargs.get("panel", 1)
            if entry is None:
                return {"status": "error", "message": "entry is required"}
            return self._add_trace_to_existing(entry, panel)

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    # ------------------------------------------------------------------
    # Manage helpers
    # ------------------------------------------------------------------

    def _remove_trace(self, label: str) -> dict:
        """Remove a trace by label from the current figure."""
        if self._figure is None:
            return {"status": "error", "message": "No figure to modify."}

        # Find indices to remove
        indices_to_remove = [
            i for i, tl in enumerate(self._trace_labels) if tl == label
        ]
        if not indices_to_remove:
            return {"status": "error",
                    "message": f"Trace '{label}' not found. "
                    f"Available: {', '.join(self._trace_labels)}"}

        # Rebuild fig.data tuple without removed indices
        keep = set(range(len(self._figure.data))) - set(indices_to_remove)
        new_data = [self._figure.data[i] for i in sorted(keep)]
        new_labels = [self._trace_labels[i] for i in sorted(keep)]
        new_panels = [self._trace_panels[i] for i in sorted(keep)]

        self._figure.data = tuple(new_data)
        self._trace_labels = new_labels
        self._trace_panels = new_panels

        return {"status": "success",
                "message": f"Removed trace '{label}'.",
                "remaining_traces": self._trace_labels,
                "display": "plotly"}

    def _add_trace_to_existing(self, entry: DataEntry, panel: int) -> dict:
        """Add a DataEntry to the existing figure at the specified panel."""
        if self._figure is None:
            return {"status": "error", "message": "No figure. Use plot_data first."}

        if len(entry.time) == 0:
            return {"status": "error",
                    "message": f"Entry '{entry.label}' has no data points"}

        fig = self._grow_panels(panel)
        added = self._add_line_traces([entry], panel, fig)
        fig.update_xaxes(type="date")

        return {"status": "success",
                "added_traces": added,
                "display": "plotly"}

    # ------------------------------------------------------------------
    # Time range
    # ------------------------------------------------------------------

    def set_time_range(self, time_range: TimeRange) -> dict:
        tr_str = time_range.to_time_range_string()
        self._log(f"Setting time range: {tr_str}")
        fig = self._ensure_figure()
        fig.update_xaxes(range=[time_range.start, time_range.end])
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
        self._label_colors.clear()
        self._color_index = 0
        self._trace_labels.clear()
        self._trace_panels.clear()
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
    # Review metadata (for LLM self-assessment)
    # ------------------------------------------------------------------

    def _build_review_metadata(self) -> dict:
        """Build structured review metadata from the current figure.

        Returns a dict with trace_summary, warnings, and hint that the LLM
        can inspect to self-assess plot quality and self-correct.
        """
        try:
            return self._build_review_metadata_inner()
        except Exception:
            # Graceful degradation — plot still works without review
            return {}

    def _build_review_metadata_inner(self) -> dict:
        fig = self._figure
        if fig is None or len(fig.data) == 0:
            return {}

        trace_summary = []
        # Group traces by panel for warning checks
        panel_traces: dict[int, list[dict]] = {}

        for i, trace in enumerate(fig.data):
            name = self._trace_labels[i] if i < len(self._trace_labels) else (trace.name or f"trace_{i}")
            panel = self._trace_panels[i] if i < len(self._trace_panels) else 1

            is_heatmap = isinstance(trace, go.Heatmap)

            if is_heatmap:
                z = trace.z
                if z is not None:
                    z_arr = np.asarray(z)
                    points_desc = f"{z_arr.shape[0]}x{z_arr.shape[1]}" if z_arr.ndim == 2 else str(len(z))
                else:
                    points_desc = "0"
                info = {
                    "name": name,
                    "panel": panel,
                    "points": points_desc,
                    "y_range": None,
                    "has_gaps": False,
                }
            else:
                y = trace.y
                if y is not None:
                    y_arr = np.asarray(y, dtype=float)
                    n_points = len(y_arr)
                    finite = y_arr[np.isfinite(y_arr)]
                    if len(finite) > 0:
                        y_range = [round(float(finite.min()), 4), round(float(finite.max()), 4)]
                    else:
                        y_range = None
                    has_gaps = bool(np.any(~np.isfinite(y_arr)))
                else:
                    n_points = 0
                    y_range = None
                    has_gaps = False
                info = {
                    "name": name,
                    "panel": panel,
                    "points": n_points,
                    "y_range": y_range,
                    "has_gaps": has_gaps,
                }

            trace_summary.append(info)
            panel_traces.setdefault(panel, []).append(info)

        # --- Warnings ---
        warnings = []

        for p, traces_in_panel in sorted(panel_traces.items()):
            n = len(traces_in_panel)

            # Cluttered panel
            if n > 6:
                warnings.append(
                    f"Panel {p} has {n} traces — may be hard to read (consider splitting into separate panels)"
                )

            # Resolution mismatch (only for numeric point counts)
            numeric_traces = [(t["name"], t["points"]) for t in traces_in_panel if isinstance(t["points"], int) and t["points"] > 0]
            if len(numeric_traces) >= 2:
                counts = [c for _, c in numeric_traces]
                min_c, max_c = min(counts), max(counts)
                if min_c > 0 and max_c / min_c > 10:
                    lo_name = next(n for n, c in numeric_traces if c == min_c)
                    hi_name = next(n for n, c in numeric_traces if c == max_c)
                    warnings.append(
                        f"Resolution mismatch in panel {p}: '{lo_name}' has {min_c} points vs "
                        f"'{hi_name}' has {max_c} points — consider resampling"
                    )

            # Empty panel
            if n == 0:
                warnings.append(f"Panel {p} has no traces")

        # Suspicious y-range (possible fill values)
        for info in trace_summary:
            yr = info["y_range"]
            if yr is not None and (yr[1] - yr[0]) > 1e6:
                warnings.append(
                    f"Trace '{info['name']}' has suspicious y-range [{yr[0]}, {yr[1]}] — possible fill values"
                )

        # --- Hint ---
        total_traces = len(trace_summary)
        panel_descs = []
        for p in sorted(panel_traces.keys()):
            names = [t["name"] for t in panel_traces[p]]
            panel_descs.append(f"Panel {p}: {', '.join(names)}")
        hint = f"Plot has {self._panel_count} panel(s), {total_traces} trace(s). {'. '.join(panel_descs)}."
        if warnings:
            hint += " Potential issues found — consider addressing before responding to user."

        return {
            "trace_summary": trace_summary,
            "warnings": warnings,
            "hint": hint,
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
            "label_colors": dict(self._label_colors),
            "color_index": self._color_index,
            "trace_labels": list(self._trace_labels),
            "trace_panels": list(self._trace_panels),
        }

    def restore_state(self, state: dict) -> None:
        """Restore renderer state from a dict produced by save_state()."""
        import plotly.io as pio

        fig_json = state.get("figure_json")
        if not fig_json:
            return

        self._figure = pio.from_json(fig_json)
        self._panel_count = state.get("panel_count", 0)
        self._label_colors = state.get("label_colors", {})
        self._color_index = state.get("color_index", 0)
        self._trace_labels = state.get("trace_labels", [])
        self._trace_panels = state.get("trace_panels", [])

        tr_str = state.get("time_range")
        if tr_str:
            try:
                from agent.time_utils import parse_time_range
                self._current_time_range = parse_time_range(tr_str)
            except Exception:
                self._current_time_range = None
        else:
            self._current_time_range = None


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
