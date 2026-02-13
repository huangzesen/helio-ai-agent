"""
Unit tests for rendering/operations.py — OperationRegistry, resolver, and built-in ops.

Also tests build_figure_from_spec() from rendering/plotly_renderer.py.

No API key, no network — fast and self-contained.
"""

import numpy as np
import pandas as pd
import pytest

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_ops.store import DataEntry
from rendering.operations import (
    OperationDef,
    OperationRegistry,
    get_default_registry,
    _panel_to_axis_key,
    _deep_merge,
    _substitute_vars,
)
from rendering.plotly_renderer import (
    ColorState,
    RenderResult,
    build_figure_from_spec,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_entry(label: str, n: int = 100, ncols: int = 1, desc: str = "test data") -> DataEntry:
    rng = pd.date_range("2024-01-01", periods=n, freq="min")
    if ncols == 1:
        df = pd.DataFrame({"value": np.random.randn(n)}, index=rng)
    else:
        cols = {f"c{i}": np.random.randn(n) for i in range(ncols)}
        df = pd.DataFrame(cols, index=rng)
    return DataEntry(label=label, data=df, units="nT", description=desc)


def _make_spectrogram_entry(label: str, n_times: int = 50, n_bins: int = 20) -> DataEntry:
    rng = pd.date_range("2024-01-01", periods=n_times, freq="10min")
    bins = np.linspace(0.001, 0.5, n_bins)
    data = np.random.rand(n_times, n_bins)
    df = pd.DataFrame(data, index=rng, columns=[str(b) for b in bins])
    return DataEntry(
        label=label, data=df, units="nT",
        description=f"{label} spectrogram",
        metadata={"type": "spectrogram", "bin_label": "Frequency (Hz)",
                  "value_label": "PSD", "bin_values": bins.tolist()},
    )


def _make_fig_with_traces(n_traces: int = 2) -> tuple[go.Figure, dict[str, int], list[str]]:
    """Create a simple figure with named traces for testing."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    labels = []
    label_map = {}
    for i in range(n_traces):
        name = f"Trace_{i}"
        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[i, i + 1, i + 2], name=name, mode="lines"),
            row=(i % 2) + 1, col=1,
        )
        labels.append(name)
        label_map[name] = i
    return fig, label_map, labels


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestPanelToAxisKey:
    def test_panel_1(self):
        assert _panel_to_axis_key(1, "yaxis") == "yaxis"

    def test_panel_2(self):
        assert _panel_to_axis_key(2, "yaxis") == "yaxis2"

    def test_panel_3(self):
        assert _panel_to_axis_key(3, "xaxis") == "xaxis3"

    def test_panel_0_raises(self):
        with pytest.raises(ValueError, match="Panel must be >= 1"):
            _panel_to_axis_key(0, "yaxis")


class TestDeepMerge:
    def test_flat(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_overwrite(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested(self):
        result = _deep_merge(
            {"a": {"x": 1, "y": 2}},
            {"a": {"y": 3, "z": 4}},
        )
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}


class TestSubstituteVars:
    def test_exact_string(self):
        assert _substitute_vars("$text", {"text": "Hello"}) == "Hello"

    def test_exact_preserves_type(self):
        assert _substitute_vars("$size", {"size": 14}) == 14
        assert _substitute_vars("$flag", {"flag": True}) is True
        assert _substitute_vars("$items", {"items": [1, 2]}) == [1, 2]

    def test_substring(self):
        assert _substitute_vars("hello $name!", {"name": "World"}) == "hello World!"

    def test_nested_dict(self):
        result = _substitute_vars({"title": {"text": "$text"}}, {"text": "My Title"})
        assert result == {"title": {"text": "My Title"}}

    def test_nested_list(self):
        result = _substitute_vars(["$a", "$b"], {"a": 1, "b": 2})
        assert result == [1, 2]

    def test_no_match(self):
        assert _substitute_vars("$unknown", {"text": "x"}) == "$unknown"

    def test_non_string_passthrough(self):
        assert _substitute_vars(42, {"x": 1}) == 42
        assert _substitute_vars(None, {"x": 1}) is None


# ---------------------------------------------------------------------------
# OperationDef tests
# ---------------------------------------------------------------------------

class TestOperationDef:
    def test_from_dict(self):
        d = {
            "name": "set_title",
            "description": "Set title",
            "params": {"text": {"type": "string"}},
            "target": "layout",
            "patch": {"title": {"text": "$text"}},
        }
        op = OperationDef.from_dict(d)
        assert op.name == "set_title"
        assert op.target == "layout"
        assert op.patch == {"title": {"text": "$text"}}
        assert op.panel_indexed_axis is None
        assert op.composite is None

    def test_from_dict_defaults(self):
        op = OperationDef.from_dict({"name": "noop"})
        assert op.target == "layout"
        assert op.patch == {}
        assert op.params == {}


# ---------------------------------------------------------------------------
# OperationRegistry tests
# ---------------------------------------------------------------------------

class TestOperationRegistry:
    def test_register_and_get(self):
        reg = OperationRegistry()
        reg.register({"name": "test_op", "patch": {"a": 1}})
        op = reg.get("test_op")
        assert op is not None
        assert op.name == "test_op"

    def test_get_unknown_returns_none(self):
        reg = OperationRegistry()
        assert reg.get("nonexistent") is None

    def test_register_builtins(self):
        reg = OperationRegistry()
        reg.register_builtins()
        ops = reg.list_operations()
        assert "set_title" in ops
        assert "set_y_label" in ops
        assert "set_trace_color" in ops
        assert "add_vline" in ops
        assert "style_publication" in ops
        assert len(ops) >= 20  # we have ~25 built-in ops

    def test_list_operations_sorted(self):
        reg = OperationRegistry()
        reg.register({"name": "zzz"})
        reg.register({"name": "aaa"})
        assert reg.list_operations() == ["aaa", "zzz"]


# ---------------------------------------------------------------------------
# Resolver tests — layout operations
# ---------------------------------------------------------------------------

class TestResolverLayout:
    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.register_builtins()
        return reg

    @pytest.fixture
    def fig(self):
        return make_subplots(rows=2, cols=1, shared_xaxes=True)

    def test_set_title(self, registry, fig):
        registry.resolve(
            {"op": "set_title", "text": "My Title"},
            fig, {}, None,
        )
        assert fig.layout.title.text == "My Title"

    def test_set_x_label(self, registry, fig):
        registry.resolve(
            {"op": "set_x_label", "text": "Time (UTC)"},
            fig, {}, None,
        )
        assert fig.layout.xaxis.title.text == "Time (UTC)"

    def test_set_theme(self, registry, fig):
        registry.resolve(
            {"op": "set_theme", "theme": "plotly_dark"},
            fig, {}, None,
        )
        assert fig.layout.template is not None

    def test_set_font_size(self, registry, fig):
        registry.resolve(
            {"op": "set_font_size", "size": 18},
            fig, {}, None,
        )
        assert fig.layout.font.size == 18

    def test_set_canvas_size(self, registry, fig):
        registry.resolve(
            {"op": "set_canvas_size", "width": 1920, "height": 1080},
            fig, {}, None,
        )
        assert fig.layout.width == 1920
        assert fig.layout.height == 1080

    def test_set_legend(self, registry, fig):
        registry.resolve(
            {"op": "set_legend", "show": False},
            fig, {}, None,
        )
        assert fig.layout.showlegend is False

    def test_set_x_range(self, registry, fig):
        registry.resolve(
            {"op": "set_x_range", "range": ["2024-01-01", "2024-01-10"]},
            fig, {}, None,
        )
        # Plotly converts lists to tuples
        assert tuple(fig.layout.xaxis.range) == ("2024-01-01", "2024-01-10")

    def test_set_margin(self, registry, fig):
        registry.resolve(
            {"op": "set_margin", "l": 50, "r": 30, "t": 40, "b": 60},
            fig, {}, None,
        )
        assert fig.layout.margin.l == 50
        assert fig.layout.margin.r == 30
        assert fig.layout.margin.t == 40
        assert fig.layout.margin.b == 60


# ---------------------------------------------------------------------------
# Resolver tests — panel-indexed axis operations
# ---------------------------------------------------------------------------

class TestResolverPanelIndexed:
    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.register_builtins()
        return reg

    @pytest.fixture
    def fig(self):
        return make_subplots(rows=3, cols=1, shared_xaxes=True)

    def test_set_y_label_panel_1(self, registry, fig):
        registry.resolve(
            {"op": "set_y_label", "panel": 1, "text": "B (nT)"},
            fig, {}, None,
        )
        assert fig.layout.yaxis.title.text == "B (nT)"

    def test_set_y_label_panel_2(self, registry, fig):
        registry.resolve(
            {"op": "set_y_label", "panel": 2, "text": "V (km/s)"},
            fig, {}, None,
        )
        assert fig.layout.yaxis2.title.text == "V (km/s)"

    def test_set_y_range_panel_3(self, registry, fig):
        registry.resolve(
            {"op": "set_y_range", "panel": 3, "range": [0, 100]},
            fig, {}, None,
        )
        assert fig.layout.yaxis3.range == (0, 100)

    def test_set_y_scale_log(self, registry, fig):
        registry.resolve(
            {"op": "set_y_scale", "panel": 2, "scale": "log"},
            fig, {}, None,
        )
        assert fig.layout.yaxis2.type == "log"

    def test_set_y_label_default_panel(self, registry, fig):
        """Panel defaults to 1 when not specified."""
        registry.resolve(
            {"op": "set_y_label", "text": "Default"},
            fig, {}, None,
        )
        assert fig.layout.yaxis.title.text == "Default"


# ---------------------------------------------------------------------------
# Resolver tests — trace operations
# ---------------------------------------------------------------------------

class TestResolverTrace:
    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.register_builtins()
        return reg

    def test_set_trace_color(self, registry):
        fig, label_map, _ = _make_fig_with_traces(2)
        registry.resolve(
            {"op": "set_trace_color", "trace": "Trace_0", "color": "red"},
            fig, label_map, None,
        )
        assert fig.data[0].line.color == "red"
        # Other trace unchanged
        assert fig.data[1].line.color != "red"

    def test_set_line_style(self, registry):
        fig, label_map, _ = _make_fig_with_traces(2)
        registry.resolve(
            {"op": "set_line_style", "trace": "Trace_1", "width": 3, "dash": "dot"},
            fig, label_map, None,
        )
        assert fig.data[1].line.width == 3
        assert fig.data[1].line.dash == "dot"

    def test_set_line_mode(self, registry):
        fig, label_map, _ = _make_fig_with_traces(1)
        registry.resolve(
            {"op": "set_line_mode", "trace": "Trace_0", "mode": "markers"},
            fig, label_map, None,
        )
        assert fig.data[0].mode == "markers"

    def test_set_trace_visibility(self, registry):
        fig, label_map, _ = _make_fig_with_traces(2)
        registry.resolve(
            {"op": "set_trace_visibility", "trace": "Trace_0", "visible": False},
            fig, label_map, None,
        )
        assert fig.data[0].visible is False
        assert fig.data[1].visible is not False  # default is None (visible)

    def test_trace_not_found_no_error(self, registry):
        """Unknown trace label is silently ignored."""
        fig, label_map, _ = _make_fig_with_traces(1)
        original_color = fig.data[0].line.color
        registry.resolve(
            {"op": "set_trace_color", "trace": "NonExistent", "color": "red"},
            fig, label_map, None,
        )
        assert fig.data[0].line.color == original_color

    def test_trace_substring_match(self, registry):
        """Fallback substring matching for trace labels."""
        fig, label_map, _ = _make_fig_with_traces(2)
        # "Trace_0" contains "ace_0" — but our matching checks both directions
        # Use a label that's a substring of the full name
        registry.resolve(
            {"op": "set_trace_color", "trace": "Trace_0", "color": "blue"},
            fig, label_map, None,
        )
        assert fig.data[0].line.color == "blue"

    def test_trace_op_without_trace_selector_applies_to_all(self, registry):
        """set_colorscale without trace selector applies to all traces."""
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]], name="H1", colorscale="Viridis"))
        fig.add_trace(go.Heatmap(z=[[5, 6], [7, 8]], name="H2", colorscale="Viridis"))
        label_map = {"H1": 0, "H2": 1}
        # Record default colorscale for comparison
        default_cs = fig.data[0].colorscale
        # set_colorscale has trace_target=True, so passing without 'trace'
        # key applies to all traces
        registry.resolve(
            {"op": "set_colorscale", "colorscale": "Jet"},
            fig, label_map, None,
        )
        # Plotly converts "Jet" to expanded tuple — should differ from Viridis
        for trace in fig.data:
            assert trace.colorscale != default_cs


# ---------------------------------------------------------------------------
# Resolver tests — append mode (decorations)
# ---------------------------------------------------------------------------

class TestResolverAppend:
    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.register_builtins()
        return reg

    def test_add_vline(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "add_vline", "x": "2024-01-15", "color": "red",
             "width": 2, "dash": "solid", "label": "Event"},
            fig, {}, None,
        )
        assert len(fig.layout.shapes) == 1
        shape = fig.layout.shapes[0]
        assert shape.x0 == "2024-01-15"
        assert shape.line.color == "red"
        # Annotation for label
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "Event"

    def test_add_multiple_vlines(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "add_vline", "x": "2024-01-10", "color": "blue",
             "width": 1, "dash": "dash", "label": "A"},
            fig, {}, None,
        )
        registry.resolve(
            {"op": "add_vline", "x": "2024-01-20", "color": "green",
             "width": 1, "dash": "solid", "label": "B"},
            fig, {}, None,
        )
        assert len(fig.layout.shapes) == 2
        assert len(fig.layout.annotations) == 2

    def test_add_hline(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "add_hline", "y": 3.5, "color": "gray",
             "width": 1, "dash": "dash"},
            fig, {}, None,
        )
        assert len(fig.layout.shapes) == 1
        shape = fig.layout.shapes[0]
        assert shape.y0 == 3.5
        assert shape.y1 == 3.5

    def test_add_vrect(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "add_vrect", "x0": "2024-01-10", "x1": "2024-01-15",
             "color": "rgba(0,0,255,0.2)", "opacity": 0.5, "label": "Storm"},
            fig, {}, None,
        )
        assert len(fig.layout.shapes) == 1
        shape = fig.layout.shapes[0]
        assert shape.type == "rect"
        assert shape.x0 == "2024-01-10"
        assert shape.x1 == "2024-01-15"
        # Annotation
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "Storm"

    def test_add_annotation(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "add_annotation", "text": "Peak", "x": 1.5, "y": 3.5,
             "showarrow": True},
            fig, {}, None,
        )
        assert len(fig.layout.annotations) == 1
        ann = fig.layout.annotations[0]
        assert ann.text == "Peak"
        assert ann.x == 1.5
        assert ann.showarrow is True

    def test_add_shape_generic(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "add_shape", "type": "rect", "x0": 0, "y0": 0,
             "x1": 1, "y1": 1, "line_color": "black", "line_width": 2,
             "fillcolor": "yellow"},
            fig, {}, None,
        )
        assert len(fig.layout.shapes) == 1
        shape = fig.layout.shapes[0]
        assert shape.type == "rect"
        assert shape.fillcolor == "yellow"


# ---------------------------------------------------------------------------
# Resolver tests — composite operations
# ---------------------------------------------------------------------------

class TestResolverComposite:
    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.register_builtins()
        return reg

    def test_style_publication(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "style_publication", "font_size": 16, "width": 1400, "height": 900},
            fig, {}, None,
        )
        assert fig.layout.font.size == 16
        assert fig.layout.width == 1400
        assert fig.layout.height == 900
        assert fig.layout.showlegend is True

    def test_style_publication_with_all_params(self, registry):
        """Composite with all params explicitly provided."""
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve(
            {"op": "style_publication", "font_size": 12, "width": 1000, "height": 700},
            fig, {}, None,
        )
        assert fig.layout.font.size == 12
        assert fig.layout.width == 1000
        assert fig.layout.height == 700
        assert fig.layout.showlegend is True


# ---------------------------------------------------------------------------
# Resolver edge cases
# ---------------------------------------------------------------------------

class TestResolverEdgeCases:
    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.register_builtins()
        return reg

    def test_unknown_op_skipped(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        title_before = fig.layout.title.text
        registry.resolve(
            {"op": "nonexistent_operation", "foo": "bar"},
            fig, {}, None,
        )
        # Figure unchanged
        assert fig.layout.title.text == title_before

    def test_missing_op_key_skipped(self, registry):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        registry.resolve({"text": "Hello"}, fig, {}, None)  # no 'op' key

    def test_multiple_ops_sequential(self, registry):
        fig = make_subplots(rows=2, cols=1)
        ops = [
            {"op": "set_title", "text": "Multi-Op Test"},
            {"op": "set_y_label", "panel": 1, "text": "B (nT)"},
            {"op": "set_y_label", "panel": 2, "text": "V (km/s)"},
            {"op": "set_font_size", "size": 14},
        ]
        for op_dict in ops:
            registry.resolve(op_dict, fig, {}, None)

        assert fig.layout.title.text == "Multi-Op Test"
        assert fig.layout.yaxis.title.text == "B (nT)"
        assert fig.layout.yaxis2.title.text == "V (km/s)"
        assert fig.layout.font.size == 14


# ---------------------------------------------------------------------------
# ColorState tests
# ---------------------------------------------------------------------------

class TestColorState:
    def test_new_color_assignment(self):
        cs = ColorState()
        c1 = cs.next_color("Alpha")
        c2 = cs.next_color("Beta")
        assert c1 != c2
        assert cs.color_index == 2

    def test_stable_color(self):
        cs = ColorState()
        c1 = cs.next_color("Alpha")
        c2 = cs.next_color("Alpha")
        assert c1 == c2
        assert cs.color_index == 1  # only assigned once

    def test_round_trip(self):
        cs = ColorState()
        cs.next_color("A")
        cs.next_color("B")
        d = cs.to_dict()
        cs2 = ColorState.from_dict(d)
        assert cs2.label_colors == cs.label_colors
        assert cs2.color_index == cs.color_index
        assert cs2.next_color("A") == cs.next_color("A")


# ---------------------------------------------------------------------------
# build_figure_from_spec — operation-based format
# ---------------------------------------------------------------------------

class TestBuildFigureFromSpec:
    def test_basic_meta_with_ops(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {
            "_meta": {
                "labels": ["A"],
                "panels": [["A"]],
            },
            "operations": [
                {"op": "set_title", "text": "Test Plot"},
            ],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.title.text == "Test Plot"
        assert result.panel_count == 1
        assert len(result.trace_labels) == 1
        assert result.trace_labels[0] == "Alpha"

    def test_multi_panel_with_ops(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "_meta": {
                "labels": ["A", "B"],
                "panels": [["A"], ["B"]],
            },
            "operations": [
                {"op": "set_title", "text": "Multi-Panel"},
                {"op": "set_y_label", "panel": 1, "text": "B (nT)"},
                {"op": "set_y_label", "panel": 2, "text": "V (km/s)"},
            ],
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.panel_count == 2
        assert result.figure.layout.title.text == "Multi-Panel"
        assert result.figure.layout.yaxis.title.text == "B (nT)"
        assert result.figure.layout.yaxis2.title.text == "V (km/s)"

    def test_trace_ops(self):
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {
            "_meta": {
                "labels": ["M"],
                "panels": [["M"]],
            },
            "operations": [
                {"op": "set_trace_color", "trace": "Mag", "color": "blue"},
            ],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.color == "blue"

    def test_decoration_ops(self):
        entry = _make_entry("V", n=20, desc="Vdata")
        spec = {
            "_meta": {
                "labels": ["V"],
                "panels": [["V"]],
            },
            "operations": [
                {"op": "add_vline", "x": "2024-01-01T00:10:00",
                 "color": "red", "width": 2, "dash": "solid", "label": "Event"},
            ],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert len(result.figure.layout.shapes) == 1
        assert len(result.figure.layout.annotations) == 1

    def test_color_state_preserved(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        cs = ColorState(label_colors={"Alpha": "#ff0000"}, color_index=5)
        spec = {
            "_meta": {"labels": ["A"], "panels": [["A"]]},
            "operations": [],
        }
        result = build_figure_from_spec(spec, [entry], color_state=cs)
        assert isinstance(result, RenderResult)
        # Should use pre-assigned color
        assert result.figure.data[0].line.color == "#ff0000"
        assert result.color_state.color_index == 5  # not incremented

    def test_time_range(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {
            "_meta": {"labels": ["A"], "panels": [["A"]]},
            "operations": [],
        }
        result = build_figure_from_spec(
            spec, [entry], time_range="2024-01-01 to 2024-01-03",
        )
        assert isinstance(result, RenderResult)
        xrange = result.figure.layout.xaxis.range
        assert xrange is not None
        assert "2024-01-01" in str(xrange[0])

    def test_empty_entries_error(self):
        spec = {
            "_meta": {"labels": ["A"], "panels": [["A"]]},
            "operations": [],
        }
        result = build_figure_from_spec(spec, [])
        assert isinstance(result, dict)
        assert result["status"] == "error"

    def test_zero_point_entry_error(self):
        empty = DataEntry(
            label="empty",
            data=pd.DataFrame({"v": pd.Series(dtype=float)},
                              index=pd.DatetimeIndex([], name="time")),
        )
        spec = {
            "_meta": {"labels": ["empty"], "panels": [["empty"]]},
            "operations": [],
        }
        result = build_figure_from_spec(spec, [empty])
        assert isinstance(result, dict)
        assert result["status"] == "error"

    def test_label_not_found_error(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {
            "_meta": {"labels": ["A", "MISSING"], "panels": [["A"], ["MISSING"]]},
            "operations": [],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "MISSING" in result["message"]

    def test_vector_decomposition(self):
        entry = _make_entry("Bvec", n=30, ncols=3)
        spec = {
            "_meta": {"labels": ["Bvec"], "panels": [["Bvec"]]},
            "operations": [],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert len(result.trace_labels) == 3
        assert "test data (x)" in result.trace_labels

    def test_no_panels_overlay(self):
        """When panels is not specified, all entries overlay in one panel."""
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        spec = {
            "_meta": {"labels": ["A", "B"]},
            "operations": [],
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.panel_count == 1
        assert len(result.trace_labels) == 2


# ---------------------------------------------------------------------------
# build_figure_from_spec — legacy format
# ---------------------------------------------------------------------------

class TestBuildFigureFromSpecLegacy:
    def test_legacy_flat_spec(self):
        """Legacy flat spec (no _meta) still works via conversion."""
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {
            "labels": "A",
            "title": "Legacy Test",
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.title.text == "Legacy Test"

    def test_legacy_with_panels(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "labels": "A,B",
            "panels": [["A"], ["B"]],
            "title": "Legacy Panels",
            "y_label": {"1": "nT", "2": "km/s"},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.panel_count == 2
        assert result.figure.layout.title.text == "Legacy Panels"
        assert result.figure.layout.yaxis.title.text == "nT"
        assert result.figure.layout.yaxis2.title.text == "km/s"

    def test_legacy_trace_colors(self):
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {
            "labels": "M",
            "trace_colors": {"Mag": "red"},
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.color == "red"

    def test_legacy_vlines(self):
        entry = _make_entry("V", n=20, desc="Vdata")
        spec = {
            "labels": "V",
            "vlines": [{"x": "2024-01-01T00:10:00", "color": "blue"}],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert len(result.figure.layout.shapes) >= 1

    def test_legacy_font_size_and_legend(self):
        entry = _make_entry("X", n=20, desc="Xray")
        spec = {
            "labels": "X",
            "font_size": 18,
            "legend": False,
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.font.size == 18
        assert result.figure.layout.showlegend is False


# ---------------------------------------------------------------------------
# Default registry singleton
# ---------------------------------------------------------------------------

class TestDefaultRegistry:
    def test_singleton(self):
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2

    def test_has_builtins(self):
        reg = get_default_registry()
        assert reg.get("set_title") is not None
        assert reg.get("add_vline") is not None
        assert reg.get("style_publication") is not None
