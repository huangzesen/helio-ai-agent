"""
Plotly-based renderer for visualization.

Provides three public methods dispatched by agent/core.py:
- plot_data() — create plots (single panel overlay or multi-panel)
- style() — apply aesthetics via key-value params (no code gen)
- manage() — structural ops: export, reset, zoom, add/remove traces

Also provides a stateless entry point:
- build_figure_from_spec() — creates a fresh go.Figure from a spec dict
  containing _meta (data/grid layout) and operations (ordered style ops).

State is kept in a mutable go.Figure that accumulates traces / layout
changes across calls (legacy API). The stateless builder is the preferred
path for new code.
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
    autosize=False,
)

_PANEL_HEIGHT = 300  # px per subplot panel
_DEFAULT_WIDTH = 1100  # px figure width

_LEGEND_MAX_LINE = 30  # max chars per line in legend


# ---------------------------------------------------------------------------
# ColorState — extracted color assignment logic for stateless use
# ---------------------------------------------------------------------------

class ColorState:
    """Tracks label-to-color assignments for stable coloring across renders.

    This is the extracted, reusable version of PlotlyRenderer._next_color().
    """

    def __init__(
        self,
        label_colors: dict[str, str] | None = None,
        color_index: int = 0,
    ):
        self.label_colors: dict[str, str] = dict(label_colors or {})
        self.color_index: int = color_index

    def next_color(self, label: str) -> str:
        """Return a stable colour for *label*, assigning a new one if unseen."""
        if label in self.label_colors:
            return self.label_colors[label]
        color = _DEFAULT_COLORS[self.color_index % len(_DEFAULT_COLORS)]
        self.color_index += 1
        self.label_colors[label] = color
        return color

    def to_dict(self) -> dict:
        return {
            "label_colors": dict(self.label_colors),
            "color_index": self.color_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ColorState:
        return cls(
            label_colors=d.get("label_colors", {}),
            color_index=d.get("color_index", 0),
        )


# ---------------------------------------------------------------------------
# RenderResult — return type of build_figure_from_spec
# ---------------------------------------------------------------------------

class RenderResult:
    """Result of a stateless build_figure_from_spec() call."""

    __slots__ = ("figure", "color_state", "trace_labels", "trace_panels",
                 "panel_count", "column_count")

    def __init__(
        self,
        figure: go.Figure,
        color_state: ColorState,
        trace_labels: list[str],
        trace_panels: list[tuple[int, int]],
        panel_count: int,
        column_count: int,
    ):
        self.figure = figure
        self.color_state = color_state
        self.trace_labels = trace_labels
        self.trace_panels = trace_panels
        self.panel_count = panel_count
        self.column_count = column_count


# ---------------------------------------------------------------------------
# Stateless figure builder
# ---------------------------------------------------------------------------

def _scatter_cls_static(n_points: int):
    """Return go.Scattergl for large datasets, go.Scatter otherwise."""
    return go.Scattergl if n_points > _GL_THRESHOLD else go.Scatter


def _add_line_traces_static(
    entries: list[DataEntry],
    row: int,
    fig: go.Figure,
    color_state: ColorState,
    trace_labels: list[str],
    trace_panels: list[tuple[int, int]],
    col: int = 1,
) -> list[str]:
    """Add line traces — stateless version of PlotlyRenderer._add_line_traces."""
    added_labels: list[str] = []

    for entry in entries:
        display_name = entry.description or entry.label
        if entry.is_xarray:
            import pandas as pd
            time_list = [pd.Timestamp(t).isoformat() for t in entry.data.coords["time"].values]
        else:
            time_list = [t.isoformat() for t in entry.data.index]

        if entry.values.ndim == 2 and entry.values.shape[1] > 1:
            comp_names = ["x", "y", "z"]
            for comp_col in range(entry.values.shape[1]):
                comp = comp_names[comp_col] if comp_col < 3 else str(comp_col)
                label = f"{display_name} ({comp})"
                val_arr = entry.values[:, comp_col]
                t_disp, v_disp = _downsample_minmax(time_list, val_arr, _MAX_DISPLAY_POINTS)
                v_disp = [float(v) if np.isfinite(v) else None for v in v_disp]
                Scatter = _scatter_cls_static(len(v_disp))
                fig.add_trace(
                    Scatter(
                        x=t_disp, y=v_disp,
                        name=_wrap_display_name(label),
                        mode="lines",
                        line=dict(color=color_state.next_color(label)),
                    ),
                    row=row, col=col,
                )
                trace_labels.append(label)
                trace_panels.append((row, col))
                added_labels.append(label)
        else:
            vals = entry.values.ravel() if entry.values.ndim > 1 else entry.values
            t_disp, v_disp = _downsample_minmax(time_list, vals, _MAX_DISPLAY_POINTS)
            v_disp = [float(v) if np.isfinite(v) else None for v in v_disp]
            Scatter = _scatter_cls_static(len(v_disp))
            fig.add_trace(
                Scatter(
                    x=t_disp, y=v_disp,
                    name=_wrap_display_name(display_name),
                    mode="lines",
                    line=dict(color=color_state.next_color(display_name)),
                ),
                row=row, col=col,
            )
            trace_labels.append(display_name)
            trace_panels.append((row, col))
            added_labels.append(display_name)

    return added_labels


def _add_spectrogram_trace_static(
    entry: DataEntry,
    row: int,
    fig: go.Figure,
    trace_labels: list[str],
    trace_panels: list[tuple[int, int]],
    col: int = 1,
    colorscale: str = "Viridis",
    log_y: bool = False,
    log_z: bool = False,
    z_min: float | None = None,
    z_max: float | None = None,
) -> str:
    """Add a spectrogram heatmap — stateless version of PlotlyRenderer._add_spectrogram_trace."""
    meta = entry.metadata or {}

    if entry.is_xarray:
        import pandas as pd
        da = entry.data
        times = [pd.Timestamp(t).isoformat() for t in da.coords["time"].values]
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
        times = [t.isoformat() for t in entry.data.index]
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

    subplot_ref = fig.get_subplot(row, col)
    if subplot_ref and hasattr(subplot_ref, 'yaxis'):
        domain = subplot_ref.yaxis.domain
        cb_y = (domain[0] + domain[1]) / 2
        cb_len = domain[1] - domain[0]
    else:
        cb_y = 0.5
        cb_len = 1.0

    colorbar_cfg = dict(y=cb_y, len=cb_len, yanchor="middle")
    if colorbar_title:
        colorbar_cfg["title"] = dict(text=colorbar_title)

    heatmap = go.Heatmap(
        x=times, y=bin_list, z=z_data,
        colorscale=colorscale, colorbar=colorbar_cfg,
        zmin=z_min, zmax=z_max, name=label,
    )
    fig.add_trace(heatmap, row=row, col=col)
    trace_labels.append(label)
    trace_panels.append((row, col))

    fig.update_xaxes(type="date", row=row, col=col)
    y_axis_title = meta.get("bin_label", "")
    if y_axis_title:
        fig.update_yaxes(title_text=y_axis_title, row=row, col=col)
    if log_y:
        fig.update_yaxes(type="log", row=row, col=col)

    return label


def _validate_spectrogram_entry_static(entry: DataEntry) -> str | None:
    """Return error message if entry is unsuitable for spectrogram, else None."""
    if entry.is_xarray:
        non_time_dims = [d for d in entry.data.dims if d != "time"]
        if not non_time_dims:
            return (
                f"Entry '{entry.label}' is scalar (no non-time dimensions). "
                "Spectrograms require 2D data (time x bins). "
                "Use panel_types to set this panel to 'line'."
            )
    else:
        if len(entry.data.columns) <= 1:
            return (
                f"Entry '{entry.label}' is scalar (1 column). "
                "Spectrograms require 2D data (time x bins). "
                "Use panel_types to set this panel to 'line'."
            )
    return None


def _resolve_column_sublabel_static(
    label: str, entry_map: dict[str, DataEntry],
) -> DataEntry | None:
    """Try to resolve 'PARENT.column' by selecting a column from a parent entry."""
    parts = label.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent_label = ".".join(parts[:i])
        col_name = ".".join(parts[i:])
        parent = entry_map.get(parent_label)
        if parent is not None and not parent.is_xarray and col_name in parent.data.columns:
            return DataEntry(
                label=label,
                data=parent.data[[col_name]],
                units=parent.units,
                description=(
                    f"{parent.description} [{col_name}]"
                    if parent.description else col_name
                ),
                source=parent.source,
                metadata=parent.metadata,
            )
    return None


def build_figure_from_spec(
    spec: dict,
    entries: list[DataEntry],
    color_state: ColorState | None = None,
    time_range: str | None = None,
) -> RenderResult | dict:
    """Build a fresh go.Figure from an operation-based spec.

    This is the main stateless entry point for the rendering pipeline.
    It reads ``_meta`` to create the subplot grid and traces, then applies
    each operation in ``spec["operations"]`` via the operation registry.

    The spec can contain either:
    - The new operation-based format: ``_meta`` + ``operations``
    - The legacy flat format (labels, panels, title, etc.)

    Args:
        spec: Plot specification dict.
        entries: DataEntry objects referenced by the spec labels.
        color_state: Optional ColorState for stable cross-render coloring.
        time_range: Optional time range string to apply to x-axis.

    Returns:
        RenderResult on success, or a dict with status='error' on failure.
    """
    if not entries:
        return {"status": "error", "message": "No entries to plot"}

    for entry in entries:
        if len(entry.time) == 0:
            return {"status": "error",
                    "message": f"Entry '{entry.label}' has no data points"}

    if color_state is None:
        color_state = ColorState()

    # Determine if this is the new operation-based format or legacy
    meta = spec.get("_meta", {})
    operations = spec.get("operations", [])

    if meta:
        # New format: _meta + operations
        return _build_from_meta(meta, operations, entries, color_state, time_range)
    else:
        # Legacy format: flat spec with labels, panels, etc.
        return _build_from_legacy(spec, entries, color_state, time_range)


def _build_from_meta(
    meta: dict,
    operations: list[dict],
    entries: list[DataEntry],
    color_state: ColorState,
    time_range: str | None,
) -> RenderResult | dict:
    """Build figure from _meta + operations format."""
    panels = meta.get("panels")
    panel_types = meta.get("panel_types")
    columns = max(meta.get("columns", 1), 1)

    # Build entry map
    entry_map: dict[str, DataEntry] = {}
    for entry in entries:
        entry_map[entry.label] = entry

    # Determine panel layout
    if panels:
        n_panels = len(panels)
    else:
        n_panels = 1
        panels = None

    # Resolve effective per-panel types
    default_type = "line"
    if panel_types is not None and panels is not None:
        if len(panel_types) != len(panels):
            return {"status": "error",
                    "message": f"panel_types length ({len(panel_types)}) "
                               f"must match panels length ({len(panels)})"}
        effective_types = list(panel_types)
    else:
        effective_types = [default_type] * n_panels

    # Create subplot grid
    subplot_kwargs: dict = dict(
        rows=max(n_panels, 1), cols=columns, shared_xaxes=True,
        vertical_spacing=0.06,
    )
    if columns > 1:
        subplot_kwargs["horizontal_spacing"] = 0.08
    fig = make_subplots(**subplot_kwargs)
    width = _DEFAULT_WIDTH if columns == 1 else int(_DEFAULT_WIDTH * columns * 0.55)
    fig.update_layout(
        **_DEFAULT_LAYOUT,
        width=width,
        height=_PANEL_HEIGHT * max(n_panels, 1),
        legend=dict(font=dict(size=11), tracegroupgap=2),
    )

    trace_labels: list[str] = []
    trace_panels: list[tuple[int, int]] = []
    all_trace_labels: list[str] = []

    spectro_kwargs = {}  # can extend later from meta

    if panels is not None:
        for panel_idx, panel_labels in enumerate(panels):
            row = panel_idx + 1
            ptype = effective_types[panel_idx]
            panel_entries = []
            for lbl in panel_labels:
                e = entry_map.get(lbl)
                if e is None:
                    e = _resolve_column_sublabel_static(lbl, entry_map)
                if e is None:
                    return {"status": "error",
                            "message": f"Label '{lbl}' not found in provided entries"}
                panel_entries.append(e)

            if columns > 1:
                n_entries = len(panel_entries)
                per_col = max(1, (n_entries + columns - 1) // columns)
                for col_idx in range(columns):
                    start = col_idx * per_col
                    end = min(start + per_col, n_entries)
                    col_entries = panel_entries[start:end]
                    if not col_entries:
                        continue
                    c = col_idx + 1
                    result = _dispatch_traces_static(
                        col_entries, row, fig, ptype, color_state,
                        trace_labels, trace_panels, col=c, **spectro_kwargs,
                    )
                    if isinstance(result, dict):
                        return result
                    all_trace_labels.extend(result)
            else:
                result = _dispatch_traces_static(
                    panel_entries, row, fig, ptype, color_state,
                    trace_labels, trace_panels, **spectro_kwargs,
                )
                if isinstance(result, dict):
                    return result
                all_trace_labels.extend(result)
    else:
        ptype = effective_types[0]
        if columns > 1:
            n_entries = len(entries)
            per_col = max(1, (n_entries + columns - 1) // columns)
            for col_idx in range(columns):
                start = col_idx * per_col
                end = min(start + per_col, n_entries)
                col_entries = entries[start:end]
                if not col_entries:
                    continue
                c = col_idx + 1
                result = _dispatch_traces_static(
                    col_entries, 1, fig, ptype, color_state,
                    trace_labels, trace_panels, col=c, **spectro_kwargs,
                )
                if isinstance(result, dict):
                    return result
                all_trace_labels.extend(result)
        else:
            result = _dispatch_traces_static(
                entries, 1, fig, ptype, color_state,
                trace_labels, trace_panels, **spectro_kwargs,
            )
            if isinstance(result, dict):
                return result
            all_trace_labels.extend(result)

    fig.update_xaxes(type="date")

    if time_range:
        # Parse simple "start to end" format
        parts = time_range.split(" to ")
        if len(parts) == 2:
            fig.update_xaxes(range=[parts[0].strip(), parts[1].strip()])

    # Apply operations
    if operations:
        from rendering.operations import get_default_registry
        registry = get_default_registry()
        trace_label_map = {label: i for i, label in enumerate(trace_labels)}
        for op_dict in operations:
            registry.resolve(op_dict, fig, trace_label_map)

    return RenderResult(
        figure=fig,
        color_state=color_state,
        trace_labels=trace_labels,
        trace_panels=trace_panels,
        panel_count=max(n_panels, 1),
        column_count=columns,
    )


def _build_from_legacy(
    spec: dict,
    entries: list[DataEntry],
    color_state: ColorState,
    time_range: str | None,
) -> RenderResult | dict:
    """Build figure from the legacy flat spec format (for backward compat).

    Converts the flat spec into _meta + operations and delegates.
    """
    # Build _meta from flat spec
    meta: dict = {}
    if "panels" in spec:
        meta["panels"] = spec["panels"]
    if "panel_types" in spec:
        meta["panel_types"] = spec["panel_types"]
    if "columns" in spec:
        meta["columns"] = spec["columns"]

    # Build operations from style fields
    operations: list[dict] = []

    if spec.get("title"):
        operations.append({"op": "set_title", "text": spec["title"]})
    if spec.get("x_label"):
        operations.append({"op": "set_x_label", "text": spec["x_label"]})
    if spec.get("y_label"):
        y_label = spec["y_label"]
        if isinstance(y_label, dict):
            for panel_str, label_text in y_label.items():
                operations.append({
                    "op": "set_y_label",
                    "panel": int(panel_str),
                    "text": str(label_text),
                })
        else:
            operations.append({"op": "set_y_label", "panel": 1, "text": str(y_label)})
    if spec.get("theme"):
        operations.append({"op": "set_theme", "theme": spec["theme"]})
    if spec.get("font_size"):
        operations.append({"op": "set_font_size", "size": spec["font_size"]})
    if spec.get("canvas_size"):
        cs = spec["canvas_size"]
        operations.append({
            "op": "set_canvas_size",
            "width": cs.get("width", _DEFAULT_WIDTH),
            "height": cs.get("height", _PANEL_HEIGHT),
        })
    if "legend" in spec:
        operations.append({"op": "set_legend", "show": spec["legend"]})
    if spec.get("x_range"):
        operations.append({"op": "set_x_range", "range": spec["x_range"]})
    if spec.get("y_range"):
        y_range = spec["y_range"]
        if isinstance(y_range, dict):
            for panel_str, rng in y_range.items():
                operations.append({
                    "op": "set_y_range",
                    "panel": int(panel_str),
                    "range": rng,
                })
        elif isinstance(y_range, list) and len(y_range) == 2:
            operations.append({"op": "set_y_range", "panel": 1, "range": y_range})
    if spec.get("log_scale"):
        log_scale = spec["log_scale"]
        if isinstance(log_scale, dict):
            for panel_str, scale_type in log_scale.items():
                operations.append({
                    "op": "set_y_scale",
                    "panel": int(panel_str),
                    "scale": scale_type,
                })
        elif log_scale == "y":
            operations.append({"op": "set_y_scale", "panel": 1, "scale": "log"})
        elif log_scale == "linear":
            operations.append({"op": "set_y_scale", "panel": 1, "scale": "linear"})
    if spec.get("trace_colors"):
        for trace_label, color in spec["trace_colors"].items():
            operations.append({
                "op": "set_trace_color",
                "trace": trace_label,
                "color": color,
            })
    if spec.get("line_styles"):
        for trace_label, style_dict in spec["line_styles"].items():
            op = {"op": "set_line_style", "trace": trace_label}
            if "width" in style_dict:
                op["width"] = style_dict["width"]
            if "dash" in style_dict:
                op["dash"] = style_dict["dash"]
            operations.append(op)
    if spec.get("vlines"):
        for vl in spec["vlines"]:
            if vl.get("x") is not None:
                operations.append({
                    "op": "add_vline",
                    "x": vl["x"],
                    "color": vl.get("color", "red"),
                    "width": vl.get("width", 1.5),
                    "dash": vl.get("dash", "solid"),
                    "label": vl.get("label", ""),
                })
    if spec.get("vrects"):
        for vr in spec["vrects"]:
            if vr.get("x0") is not None and vr.get("x1") is not None:
                operations.append({
                    "op": "add_vrect",
                    "x0": vr["x0"],
                    "x1": vr["x1"],
                    "color": vr.get("color", "rgba(135,206,250,0.3)"),
                    "opacity": vr.get("opacity", 0.3),
                    "label": vr.get("label", ""),
                })
    if spec.get("annotations"):
        for ann in spec["annotations"]:
            operations.append({
                "op": "add_annotation",
                "text": ann.get("text", ""),
                "x": ann.get("x"),
                "y": ann.get("y"),
                "showarrow": ann.get("showarrow", True),
            })

    # Carry over spectrogram params into meta if present
    # (for legacy specs that use plot_type="spectrogram")
    if spec.get("plot_type"):
        # Not an operation — handle via panel_types
        if not meta.get("panel_types"):
            n_panels = len(meta["panels"]) if "panels" in meta else 1
            meta["panel_types"] = [spec["plot_type"]] * n_panels

    effective_time_range = time_range or spec.get("time_range")

    return _build_from_meta(meta, operations, entries, color_state, effective_time_range)


def _dispatch_traces_static(
    entries: list[DataEntry],
    row: int,
    fig: go.Figure,
    panel_type: str,
    color_state: ColorState,
    trace_labels: list[str],
    trace_panels: list[tuple[int, int]],
    col: int = 1,
    **spectro_kwargs,
) -> list[str] | dict:
    """Add traces for a panel — stateless version of _dispatch_panel_traces."""
    if panel_type == "spectrogram":
        labels: list[str] = []
        for e in entries:
            err = _validate_spectrogram_entry_static(e)
            if err is not None:
                return {"status": "error", "message": err}
            label = _add_spectrogram_trace_static(
                e, row, fig, trace_labels, trace_panels,
                col=col, **spectro_kwargs,
            )
            labels.append(label)
        return labels
    else:
        return _add_line_traces_static(
            entries, row, fig, color_state, trace_labels, trace_panels, col=col,
        )


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
        self._column_count: int = 1
        self._current_time_range: Optional[TimeRange] = None
        self._label_colors: dict[str, str] = {}
        self._color_index: int = 0
        # Trace tracking: parallel to fig.data
        self._trace_labels: list[str] = []
        self._trace_panels: list[tuple[int, int]] = []  # (row, col) per trace
        # Current plot spec for spec-based rendering
        self._current_plot_spec: dict = {}

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

    def _panel_to_rowcol(self, panel: int) -> tuple[int, int]:
        """Convert a flat 1-based panel number to (row, col) using row-major indexing.

        For a grid with C columns: panel 1→(1,1), 2→(1,2), 3→(2,1), 4→(2,2).
        When columns=1, panel N → (N, 1) — backward compatible.
        """
        cols = max(self._column_count, 1)
        row = (panel - 1) // cols + 1
        col = (panel - 1) % cols + 1
        return (row, col)

    def _ensure_figure(self, rows: int = 1, cols: int = 1,
                        column_titles: list[str] | None = None) -> go.Figure:
        """Guarantee a figure exists with at least *rows* subplot rows and *cols* columns.

        Always uses make_subplots so that row/col args work on add_trace.
        """
        cols = max(cols, 1)
        if self._figure is None or rows > self._panel_count or cols > self._column_count:
            subplot_kwargs: dict = dict(
                rows=max(rows, 1), cols=cols, shared_xaxes=True,
                vertical_spacing=0.06,
            )
            if cols > 1:
                subplot_kwargs["horizontal_spacing"] = 0.08
            if column_titles and len(column_titles) == cols:
                subplot_kwargs["column_titles"] = column_titles
            self._figure = make_subplots(**subplot_kwargs)
            self._panel_count = max(rows, 1)
            self._column_count = cols
            width = _DEFAULT_WIDTH if cols == 1 else int(_DEFAULT_WIDTH * cols * 0.55)
            self._figure.update_layout(
                **_DEFAULT_LAYOUT,
                width=width,
                height=_PANEL_HEIGHT * self._panel_count,
                legend=dict(font=dict(size=11), tracegroupgap=2),
            )
        return self._figure

    def _grow_panels(self, needed_rows: int, needed_cols: int | None = None) -> go.Figure:
        """Rebuild with more rows if needed, copying existing traces."""
        if needed_cols is None:
            needed_cols = self._column_count
        if needed_rows <= self._panel_count and needed_cols <= self._column_count:
            return self._ensure_figure()

        old_fig = self._figure
        cols = max(needed_cols, self._column_count)
        subplot_kwargs: dict = dict(
            rows=needed_rows, cols=cols, shared_xaxes=True,
            vertical_spacing=0.06,
        )
        if cols > 1:
            subplot_kwargs["horizontal_spacing"] = 0.08
        new_fig = make_subplots(**subplot_kwargs)
        width = _DEFAULT_WIDTH if cols == 1 else int(_DEFAULT_WIDTH * cols * 0.55)
        new_fig.update_layout(
            **_DEFAULT_LAYOUT,
            width=width,
            height=_PANEL_HEIGHT * needed_rows,
            legend=dict(font=dict(size=11), tracegroupgap=2),
        )

        # Copy traces from old figure, preserving (row, col) assignment
        if old_fig is not None:
            for i, trace in enumerate(old_fig.data):
                if i < len(self._trace_panels):
                    row, col = self._trace_panels[i]
                else:
                    row = _row_of_trace(trace)
                    col = 1
                new_fig.add_trace(trace, row=row, col=col)
            # Copy layout properties we care about (title, axis labels)
            if old_fig.layout.title and old_fig.layout.title.text:
                new_fig.update_layout(title_text=old_fig.layout.title.text)

        self._figure = new_fig
        self._panel_count = needed_rows
        self._column_count = cols
        return self._figure

    def _scatter_cls(self, n_points: int):
        """Return go.Scattergl for large datasets, go.Scatter otherwise."""
        return go.Scattergl if n_points > _GL_THRESHOLD else go.Scatter

    @staticmethod
    def _resolve_column_sublabel(
        label: str, entry_map: dict[str, "DataEntry"]
    ) -> "DataEntry | None":
        """Try to resolve 'PARENT.column' by selecting a column from a parent entry.

        Splits the label progressively (right to left) to find a parent entry
        whose DataFrame contains the trailing part as a column name.
        Returns a new DataEntry with just that column, or None.
        """
        parts = label.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_label = ".".join(parts[:i])
            col_name = ".".join(parts[i:])
            parent = entry_map.get(parent_label)
            if parent is not None and not parent.is_xarray and col_name in parent.data.columns:
                return DataEntry(
                    label=label,
                    data=parent.data[[col_name]],
                    units=parent.units,
                    description=(
                        f"{parent.description} [{col_name}]"
                        if parent.description else col_name
                    ),
                    source=parent.source,
                    metadata=parent.metadata,
                )
        return None

    # ------------------------------------------------------------------
    # Internal helpers (used by plot_data)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_spectrogram_entry(entry: DataEntry) -> str | None:
        """Return error message if entry is unsuitable for spectrogram, else None."""
        if entry.is_xarray:
            non_time_dims = [d for d in entry.data.dims if d != "time"]
            if not non_time_dims:
                return (
                    f"Entry '{entry.label}' is scalar (no non-time dimensions). "
                    "Spectrograms require 2D data (time x bins). "
                    "Use panel_types to set this panel to 'line'."
                )
        else:
            if len(entry.data.columns) <= 1:
                return (
                    f"Entry '{entry.label}' is scalar (1 column). "
                    "Spectrograms require 2D data (time x bins). "
                    "Use panel_types to set this panel to 'line'."
                )
        return None

    def _dispatch_panel_traces(
        self,
        entries: list[DataEntry],
        row: int,
        fig: go.Figure,
        panel_type: str,
        col: int = 1,
        **spectro_kwargs,
    ) -> list[str] | dict:
        """Add traces for a single panel cell, dispatching by panel_type.

        Returns list of trace labels on success, or an error dict.
        """
        if panel_type == "spectrogram":
            labels: list[str] = []
            for e in entries:
                err = self._validate_spectrogram_entry(e)
                if err is not None:
                    return {"status": "error", "message": err}
                label = self._add_spectrogram_trace(
                    e, row, fig, col=col, **spectro_kwargs,
                )
                labels.append(label)
            return labels
        else:
            return self._add_line_traces(entries, row, fig, col=col)

    def _add_line_traces(
        self,
        entries: list[DataEntry],
        row: int,
        fig: go.Figure,
        col: int = 1,
    ) -> list[str]:
        """Add line traces for entries to a specific panel cell.

        Returns the list of trace labels added.
        """
        added_labels: list[str] = []

        for entry in entries:
            display_name = entry.description or entry.label
            # Use pandas index (Timestamps with .isoformat()) not numpy
            # datetime64 (which .tolist() converts to nanosecond ints).
            if entry.is_xarray:
                import pandas as pd
                time_list = [pd.Timestamp(t).isoformat() for t in entry.data.coords["time"].values]
            else:
                time_list = [t.isoformat() for t in entry.data.index]

            # Decompose vectors into scalar components
            if entry.values.ndim == 2 and entry.values.shape[1] > 1:
                comp_names = ["x", "y", "z"]
                for comp_col in range(entry.values.shape[1]):
                    comp = comp_names[comp_col] if comp_col < 3 else str(comp_col)
                    label = f"{display_name} ({comp})"
                    val_arr = entry.values[:, comp_col]
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
                        row=row, col=col,
                    )
                    self._trace_labels.append(label)
                    self._trace_panels.append((row, col))
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
                    row=row, col=col,
                )
                self._trace_labels.append(display_name)
                self._trace_panels.append((row, col))
                added_labels.append(display_name)

        return added_labels

    def _add_spectrogram_trace(
        self,
        entry: DataEntry,
        row: int,
        fig: go.Figure,
        col: int = 1,
        colorscale: str = "Viridis",
        log_y: bool = False,
        log_z: bool = False,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> str:
        """Add a spectrogram heatmap trace to a specific panel cell.

        Handles both DataFrame entries (time x columns) and xarray DataArray
        entries (2D: time x bins, or 3D+: averaged over middle dims).

        Returns the trace label.
        """
        meta = entry.metadata or {}

        if entry.is_xarray:
            import pandas as pd
            da = entry.data
            # Convert time coordinates to isoformat strings
            times = [pd.Timestamp(t).isoformat() for t in da.coords["time"].values]

            # Find the non-time dimension(s) for bin values
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
            # If 3D+, average over middle dims to get (time, bins)
            if z_values.ndim > 2:
                # Axes between time (0) and last non-time dim (-1)
                middle_axes = tuple(range(1, z_values.ndim - 1))
                z_values = np.nanmean(z_values, axis=middle_axes)
        else:
            # DataFrame path
            times = [t.isoformat() for t in entry.data.index]

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

        # Constrain colorbar to the panel's y-domain so it doesn't span
        # the full figure height in multi-panel layouts.
        subplot_ref = fig.get_subplot(row, col)
        if subplot_ref and hasattr(subplot_ref, 'yaxis'):
            domain = subplot_ref.yaxis.domain
            cb_y = (domain[0] + domain[1]) / 2
            cb_len = domain[1] - domain[0]
        else:
            cb_y = 0.5
            cb_len = 1.0

        colorbar_cfg = dict(y=cb_y, len=cb_len, yanchor="middle")
        if colorbar_title:
            colorbar_cfg["title"] = dict(text=colorbar_title)

        heatmap = go.Heatmap(
            x=times,
            y=bin_list,
            z=z_data,
            colorscale=colorscale,
            colorbar=colorbar_cfg,
            zmin=z_min,
            zmax=z_max,
            name=label,
        )

        fig.add_trace(heatmap, row=row, col=col)
        self._trace_labels.append(label)
        self._trace_panels.append((row, col))

        fig.update_xaxes(type="date", row=row, col=col)

        y_axis_title = meta.get("bin_label", "")
        if y_axis_title:
            fig.update_yaxes(title_text=y_axis_title, row=row, col=col)

        if log_y:
            fig.update_yaxes(type="log", row=row, col=col)

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
        panel_types: list[str] | None = None,
        colorscale: str = "Viridis",
        log_y: bool = False,
        log_z: bool = False,
        z_min: float | None = None,
        z_max: float | None = None,
        columns: int = 1,
        column_titles: list[str] | None = None,
    ) -> dict:
        """Create a fresh plot from DataEntry objects.

        Args:
            entries: All data to plot.
            panels: Panel layout, e.g. [["A","B"], ["C"]] for 2-panel.
                    None = overlay all in one panel.
            title: Plot title.
            plot_type: Default plot type: "line" or "spectrogram".
            panel_types: Per-panel plot type, parallel to panels array.
                         E.g. ["spectrogram", "line", "line"]. Omit to use
                         plot_type for all panels.
            colorscale: Plotly colorscale for spectrogram.
            log_y: Log scale on y-axis (spectrogram).
            log_z: Log scale on z-axis (spectrogram).
            z_min: Min value for spectrogram color scale.
            z_max: Max value for spectrogram color scale.
            columns: Number of columns for grid layout (default 1).
                     Use 2 for side-by-side epoch comparison.
            column_titles: Column header labels (e.g. ['Jan 2020', 'Oct 2024']).

        Returns:
            Result dict with status, panels, traces, display, columns.
        """
        if not entries:
            return {"status": "error", "message": "No entries to plot"}

        for entry in entries:
            if len(entry.time) == 0:
                return {"status": "error",
                        "message": f"Entry '{entry.label}' has no data points"}

        columns = max(columns, 1)

        # Resolve effective per-panel types
        if panel_types is not None and panels is not None:
            if len(panel_types) != len(panels):
                return {"status": "error",
                        "message": f"panel_types length ({len(panel_types)}) "
                                   f"must match panels length ({len(panels)})"}
            effective_types = list(panel_types)
        else:
            n = len(panels) if panels is not None else 1
            effective_types = [plot_type] * n

        spectro_kwargs = dict(
            colorscale=colorscale, log_y=log_y, log_z=log_z,
            z_min=z_min, z_max=z_max,
        )

        # Build label -> entry lookup
        entry_map: dict[str, DataEntry] = {}
        for entry in entries:
            entry_map[entry.label] = entry

        # Reset figure for fresh plot
        self._figure = None
        self._panel_count = 0
        self._column_count = 1
        self._trace_labels = []
        self._trace_panels = []
        # Keep label_colors for stable coloring across plot_data calls

        all_trace_labels: list[str] = []

        if panels is not None:
            # Multi-panel mode
            n_panels = len(panels)
            fig = self._ensure_figure(rows=n_panels, cols=columns,
                                      column_titles=column_titles)

            for panel_idx, panel_labels in enumerate(panels):
                row = panel_idx + 1  # 1-based
                ptype = effective_types[panel_idx]

                # Resolve entries for this row
                panel_entries = []
                for lbl in panel_labels:
                    e = entry_map.get(lbl)
                    if e is None:
                        e = self._resolve_column_sublabel(lbl, entry_map)
                    if e is None:
                        return {"status": "error",
                                "message": f"Label '{lbl}' not found in provided entries"}
                    panel_entries.append(e)

                if columns > 1:
                    # Distribute this row's entries across columns
                    n_entries = len(panel_entries)
                    per_col = max(1, (n_entries + columns - 1) // columns)
                    for col_idx in range(columns):
                        start = col_idx * per_col
                        end = min(start + per_col, n_entries)
                        col_entries = panel_entries[start:end]
                        if not col_entries:
                            continue
                        c = col_idx + 1  # 1-based
                        result = self._dispatch_panel_traces(
                            col_entries, row, fig, ptype, col=c,
                            **spectro_kwargs,
                        )
                        if isinstance(result, dict):
                            return result  # error
                        all_trace_labels.extend(result)
                else:
                    result = self._dispatch_panel_traces(
                        panel_entries, row, fig, ptype,
                        **spectro_kwargs,
                    )
                    if isinstance(result, dict):
                        return result  # error
                    all_trace_labels.extend(result)
        else:
            # Overlay mode — all in row 1
            ptype = effective_types[0]
            fig = self._ensure_figure(rows=1, cols=columns,
                                      column_titles=column_titles)

            if columns > 1:
                # Distribute entries across columns
                n_entries = len(entries)
                per_col = max(1, (n_entries + columns - 1) // columns)
                for col_idx in range(columns):
                    start = col_idx * per_col
                    end = min(start + per_col, n_entries)
                    col_entries = entries[start:end]
                    if not col_entries:
                        continue
                    c = col_idx + 1
                    result = self._dispatch_panel_traces(
                        col_entries, 1, fig, ptype, col=c,
                        **spectro_kwargs,
                    )
                    if isinstance(result, dict):
                        return result  # error
                    all_trace_labels.extend(result)
            else:
                result = self._dispatch_panel_traces(
                    entries, 1, fig, ptype,
                    **spectro_kwargs,
                )
                if isinstance(result, dict):
                    return result  # error
                all_trace_labels.extend(result)

        # Ensure the x-axis is rendered as formatted dates
        fig.update_xaxes(type="date")

        # Apply stored time range to the new figure (only for single-column)
        if self._current_time_range and columns == 1:
            fig.update_xaxes(range=[
                self._current_time_range.start.isoformat(),
                self._current_time_range.end.isoformat(),
            ])

        if title:
            fig.update_layout(title_text=title)

        result = {
            "status": "success",
            "panels": self._panel_count,
            "columns": self._column_count,
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
        log_scale: str | dict | None = None,
        x_range: list | dict | None = None,
        y_range: list | dict | None = None,
        legend: bool | None = None,
        font_size: int | None = None,
        canvas_size: dict | None = None,
        annotations: list | None = None,
        colorscale: str | None = None,
        theme: str | None = None,
        vlines: list | None = None,
        vrects: list | None = None,
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
            vrects: List of {x0, x1, label, color, opacity} dicts for highlighted time ranges.

        Returns:
            Result dict with status.
        """
        if self._figure is None or len(self._figure.data) == 0:
            return {"status": "error",
                    "message": "No plot to style. Use plot_data first."}

        fig = self._figure
        warnings = []

        if title is not None:
            fig.update_layout(title_text=title)

        if x_label is not None:
            # Apply to the bottom row, all columns
            for c in range(1, self._column_count + 1):
                fig.update_xaxes(title_text=x_label, row=self._panel_count, col=c)

        if y_label is not None:
            if isinstance(y_label, dict):
                for panel_str, label_text in y_label.items():
                    row, col = self._panel_to_rowcol(int(panel_str))
                    fig.update_yaxes(title_text=_wrap_display_name(str(label_text)), row=row, col=col)
            else:
                wrapped = _wrap_display_name(str(y_label))
                for row in range(1, self._panel_count + 1):
                    for c in range(1, self._column_count + 1):
                        fig.update_yaxes(title_text=wrapped, row=row, col=c)

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
            if isinstance(log_scale, dict):
                for panel_str, scale_type in log_scale.items():
                    axis_type = "log" if scale_type in ("log", "y") else "linear"
                    row, col = self._panel_to_rowcol(int(panel_str))
                    fig.update_yaxes(type=axis_type, row=row, col=col)
            elif isinstance(log_scale, int):
                # Integer = panel number to apply log scale to
                row, col = self._panel_to_rowcol(log_scale)
                fig.update_yaxes(type="log", row=row, col=col)
            elif log_scale == "y":
                for row in range(1, self._panel_count + 1):
                    for c in range(1, self._column_count + 1):
                        fig.update_yaxes(type="log", row=row, col=c)
            elif log_scale == "linear":
                for row in range(1, self._panel_count + 1):
                    for c in range(1, self._column_count + 1):
                        fig.update_yaxes(type="linear", row=row, col=c)
            else:
                warnings.append(
                    f"Unrecognized log_scale value '{log_scale}'. "
                    "Use 'y', 'linear', or a dict like {{'4': 'log', '5': 'log'}}."
                )

        if x_range is not None:
            if isinstance(x_range, dict):
                for panel_str, rng in x_range.items():
                    row, col = self._panel_to_rowcol(int(panel_str))
                    fig.update_xaxes(range=rng, row=row, col=col)
            else:
                fig.update_xaxes(range=x_range)

        if y_range is not None:
            if isinstance(y_range, dict):
                for panel_str, rng in y_range.items():
                    if isinstance(rng, list) and len(rng) == 2:
                        row, col = self._panel_to_rowcol(int(panel_str))
                        fig.update_yaxes(range=rng, row=row, col=col)
                    elif rng:
                        warnings.append(f"y_range for panel {panel_str} must be [min, max], got {rng}")
            elif isinstance(y_range, list) and len(y_range) == 2:
                for row in range(1, self._panel_count + 1):
                    for c in range(1, self._column_count + 1):
                        fig.update_yaxes(range=y_range, row=row, col=c)
            elif isinstance(y_range, list) and len(y_range) == 0:
                pass  # empty list — skip silently
            else:
                warnings.append(f"y_range must be [min, max], got {y_range}")

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
            # Detect background color to avoid invisible vlines
            bg = (fig.layout.plot_bgcolor or "white").lower().strip()
            bg_is_light = bg in (
                "white", "#fff", "#ffffff", "#e5ecf6", "rgb(255,255,255)",
            )
            skipped = 0
            drawn = 0
            for vl in vlines:
                x_val = vl.get("x")
                if x_val is None:
                    skipped += 1
                    continue
                drawn += 1
                color = vl.get("color", "red")
                # Override white/near-white colors on light backgrounds
                if bg_is_light and color.lower().strip() in (
                    "white", "#fff", "#ffffff", "rgb(255,255,255)",
                ):
                    color = "red"
                width = vl.get("width", 1.5)
                dash = vl.get("dash", "solid")
                label = vl.get("label")
                # Draw line across all panels and columns
                for row in range(1, self._panel_count + 1):
                    for c in range(1, self._column_count + 1):
                        fig.add_vline(
                            x=x_val, row=row, col=c,
                            line_width=width, line_dash=dash, line_color=color,
                        )
                # Add text annotation at top panel if label provided
                if label:
                    fig.add_annotation(
                        x=x_val, y=1.02, xref="x", yref="paper",
                        text=label, showarrow=False,
                        font=dict(size=11, color=color),
                    )
            if skipped > 0:
                total = skipped + drawn
                warnings.append(
                    f"{skipped} of {total} vlines skipped — each vline requires "
                    f"an 'x' field with a timestamp string, e.g. "
                    f'vlines=[{{"x": "2024-01-15T12:00:00"}}]'
                )

        if vrects is not None:
            skipped = 0
            drawn = 0
            for vr in vrects:
                x0 = vr.get("x0")
                x1 = vr.get("x1")
                if x0 is None or x1 is None:
                    skipped += 1
                    continue
                drawn += 1
                color = vr.get("color", "rgba(135,206,250,0.3)")
                opacity = vr.get("opacity", 0.3)
                label = vr.get("label")
                for row in range(1, self._panel_count + 1):
                    for c in range(1, self._column_count + 1):
                        fig.add_vrect(
                            x0=x0, x1=x1, row=row, col=c,
                            fillcolor=color, opacity=opacity,
                            line_width=0, layer="below",
                        )
                if label:
                    import pandas as pd
                    try:
                        mid_x = pd.Timestamp(x0) + (pd.Timestamp(x1) - pd.Timestamp(x0)) / 2
                        fig.add_annotation(
                            x=mid_x.isoformat(), y=1.02,
                            xref="x", yref="paper",
                            text=label, showarrow=False,
                            font=dict(size=11),
                        )
                    except Exception:
                        pass  # skip label if timestamps can't be parsed
            if skipped > 0:
                total = skipped + drawn
                warnings.append(
                    f"{skipped} of {total} vrects skipped — each vrect requires "
                    f"'x0' and 'x1' fields with timestamp strings, e.g. "
                    f'vrects=[{{"x0": "2024-01-10", "x1": "2024-01-15"}}]'
                )

        # Merge applied style fields into the current spec
        style_fields = {
            "title": title, "x_label": x_label, "y_label": y_label,
            "trace_colors": trace_colors, "line_styles": line_styles,
            "log_scale": log_scale, "x_range": x_range, "y_range": y_range,
            "legend": legend, "font_size": font_size, "canvas_size": canvas_size,
            "annotations": annotations, "colorscale": colorscale, "theme": theme,
            "vlines": vlines, "vrects": vrects,
        }
        for k, v in style_fields.items():
            if v is not None:
                self._current_plot_spec[k] = v

        result = {"status": "success", "message": "Style applied.", "display": "plotly"}
        if warnings:
            result["warnings"] = warnings
        return result

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

        row, col = self._panel_to_rowcol(panel)
        fig = self._grow_panels(row)
        added = self._add_line_traces([entry], row, fig, col=col)
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
        self._column_count = 1
        self._current_time_range = None
        self._label_colors.clear()
        self._color_index = 0
        self._trace_labels.clear()
        self._trace_panels.clear()
        self._current_plot_spec = {}
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

    def get_current_spec(self) -> dict:
        """Return a copy of the current plot spec.

        Returns an empty dict if no spec-based render has occurred.
        """
        return dict(self._current_plot_spec)

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
        # Group traces by (row, col) for warning checks
        panel_traces: dict[tuple[int, int], list[dict]] = {}

        for i, trace in enumerate(fig.data):
            name = self._trace_labels[i] if i < len(self._trace_labels) else (trace.name or f"trace_{i}")
            rc = self._trace_panels[i] if i < len(self._trace_panels) else (1, 1)

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
                    "row": rc[0],
                    "col": rc[1],
                    "panel": rc[0] if self._column_count == 1 else None,
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
                    "row": rc[0],
                    "col": rc[1],
                    "panel": rc[0] if self._column_count == 1 else None,
                    "points": n_points,
                    "y_range": y_range,
                    "has_gaps": has_gaps,
                }

            trace_summary.append(info)
            panel_traces.setdefault(rc, []).append(info)

        # --- Warnings ---
        warnings = []

        for rc, traces_in_panel in sorted(panel_traces.items()):
            n = len(traces_in_panel)
            panel_label = f"Panel ({rc[0]},{rc[1]})" if self._column_count > 1 else f"Panel {rc[0]}"

            # Cluttered panel
            if n > 6:
                warnings.append(
                    f"{panel_label} has {n} traces — may be hard to read (consider splitting into separate panels)"
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
                        f"Resolution mismatch in {panel_label}: '{lo_name}' has {min_c} points vs "
                        f"'{hi_name}' has {max_c} points — consider resampling"
                    )

            # Empty panel
            if n == 0:
                warnings.append(f"{panel_label} has no traces")

            # Invisible traces (data points exist but all are NaN — nothing renders)
            invisible = [
                t["name"] for t in traces_in_panel
                if isinstance(t["points"], int) and t["points"] > 0 and t["y_range"] is None
            ]
            if invisible:
                names_str = ", ".join(f"'{nm}'" for nm in invisible)
                if len(invisible) == n:
                    warnings.append(
                        f"{panel_label} appears empty — all traces have only NaN/missing data: {names_str}"
                    )
                else:
                    warnings.append(
                        f"{panel_label} has invisible traces (all NaN/missing data): {names_str}"
                    )

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
        for rc in sorted(panel_traces.keys()):
            names = [t["name"] for t in panel_traces[rc]]
            if self._column_count > 1:
                panel_descs.append(f"Panel ({rc[0]},{rc[1]}): {', '.join(names)}")
            else:
                panel_descs.append(f"Panel {rc[0]}: {', '.join(names)}")
        hint = f"Plot has {self._panel_count} panel(s), {total_traces} trace(s). {'. '.join(panel_descs)}."
        if self._column_count > 1:
            hint = f"Plot has {self._panel_count} row(s) x {self._column_count} column(s), {total_traces} trace(s). {'. '.join(panel_descs)}."
        if warnings:
            hint += " Potential issues found — consider addressing before responding to user."

        # --- Figure sizing ---
        current_width = _DEFAULT_WIDTH if self._column_count == 1 else int(_DEFAULT_WIDTH * self._column_count * 0.55)
        current_height = _PANEL_HEIGHT * self._panel_count

        has_spectrogram = any(isinstance(t, go.Heatmap) for t in fig.data)
        if has_spectrogram:
            rec_height = max(400, _PANEL_HEIGHT * self._panel_count)
            rec_width = 1200 if self._column_count == 1 else int(1200 * self._column_count * 0.55)
            reason = f"{self._panel_count} panel(s) with spectrogram"
        elif self._panel_count >= 4:
            rec_height = 250 * self._panel_count
            rec_width = current_width
            reason = f"{self._panel_count} panels — compact spacing"
        elif self._column_count > 1:
            rec_height = current_height
            rec_width = current_width
            reason = f"{self._panel_count} row(s) x {self._column_count} column(s) grid"
        else:
            rec_height = current_height
            rec_width = current_width
            reason = f"{self._panel_count} panel(s), {total_traces} trace(s) — default sizing"

        result_dict = {
            "trace_summary": trace_summary,
            "warnings": warnings,
            "hint": hint,
            "figure_size": {"width": current_width, "height": current_height},
            "sizing_recommendation": {"width": rec_width, "height": rec_height, "reason": reason},
        }
        if self._current_time_range:
            result_dict["current_time_range"] = self._current_time_range.to_time_range_string()
        return result_dict

    # ------------------------------------------------------------------
    # Public API: render_from_spec (unified pipeline spec)
    # ------------------------------------------------------------------

    def render_from_spec(self, spec: dict, entries: list[DataEntry]) -> dict:
        """Create a complete plot from a unified spec dict.

        Supports two spec formats:
        1. **Operation-based** (new): ``_meta`` + ``operations`` list
        2. **Legacy flat**: labels, panels, title, y_label, etc.

        Both formats are routed through ``build_figure_from_spec()``, which
        builds a fresh figure each time. The result is copied into this
        renderer's stateful fields (bridge code for backward compat).

        Args:
            spec: Unified plot specification dict.
            entries: DataEntry objects referenced by the spec labels.

        Returns:
            Result dict with status, panels, traces, display — same shape
            as plot_data().
        """
        # Build a ColorState from our current label_colors
        cs = ColorState(
            label_colors=dict(self._label_colors),
            color_index=self._color_index,
        )

        tr_str = (self._current_time_range.to_time_range_string()
                  if self._current_time_range else None)

        build_result = build_figure_from_spec(spec, entries, color_state=cs,
                                              time_range=tr_str)

        if isinstance(build_result, dict):
            return build_result  # error dict

        # Copy stateless result into our stateful fields (bridge)
        self._figure = build_result.figure
        self._panel_count = build_result.panel_count
        self._column_count = build_result.column_count
        self._trace_labels = build_result.trace_labels
        self._trace_panels = build_result.trace_panels
        self._label_colors = build_result.color_state.label_colors
        self._color_index = build_result.color_state.color_index
        self._current_plot_spec = dict(spec)

        result = {
            "status": "success",
            "panels": build_result.panel_count,
            "columns": build_result.column_count,
            "traces": list(build_result.trace_labels),
            "display": "plotly",
        }
        result["review"] = self._build_review_metadata()
        return result

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
            "column_count": self._column_count,
            "time_range": tr.to_time_range_string() if tr else None,
            "label_colors": dict(self._label_colors),
            "color_index": self._color_index,
            "trace_labels": list(self._trace_labels),
            "trace_panels": [{"row": r, "col": c} for r, c in self._trace_panels],
            "plot_spec": dict(self._current_plot_spec),
        }

    def restore_state(self, state: dict) -> None:
        """Restore renderer state from a dict produced by save_state()."""
        import plotly.io as pio

        fig_json = state.get("figure_json")
        if not fig_json:
            return

        self._figure = pio.from_json(fig_json)
        self._panel_count = state.get("panel_count", 0)
        self._column_count = state.get("column_count", 1)
        self._label_colors = state.get("label_colors", {})
        self._color_index = state.get("color_index", 0)
        self._trace_labels = state.get("trace_labels", [])

        # Deserialize trace_panels: new format is [{"row": r, "col": c}, ...],
        # legacy format is [int, ...] (flat row numbers).
        raw_panels = state.get("trace_panels", [])
        panels: list[tuple[int, int]] = []
        for item in raw_panels:
            if isinstance(item, dict):
                panels.append((item["row"], item["col"]))
            elif isinstance(item, int):
                panels.append((item, 1))
            else:
                panels.append((1, 1))
        self._trace_panels = panels

        self._current_plot_spec = state.get("plot_spec", {})

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
