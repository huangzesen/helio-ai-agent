"""
Unit tests for build_figure_from_spec() semantic spec format.

Tests the semantic spec keys: panels, title, y_label, trace_colors,
margin, trace_visibility, trace_mode, etc.

No API key, no network — fast and self-contained.
"""

import numpy as np
import pandas as pd
import pytest

import plotly.graph_objects as go

from data_ops.store import DataEntry
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
# build_figure_from_spec — layout semantic fields
# ---------------------------------------------------------------------------

class TestLayoutSemanticFields:
    def test_set_title(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "title": "My Title"}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.title.text == "My Title"

    def test_set_y_label_panel_1(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "y_label": "B (nT)"}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis.title.text == "B (nT)"

    def test_set_y_label_panel_2(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "panels": [["A"], ["B"]],
            "y_label": {"1": "B (nT)", "2": "V (km/s)"},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis.title.text == "B (nT)"
        assert result.figure.layout.yaxis2.title.text == "V (km/s)"

    def test_set_theme(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "theme": "plotly_dark"}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.template is not None

    def test_set_font_size(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "font_size": 18}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.font.size == 18

    def test_set_canvas_size(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "canvas_size": {"width": 1920, "height": 1080}}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.width == 1920
        assert result.figure.layout.height == 1080

    def test_set_legend(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "legend": False}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.showlegend is False

    def test_set_x_range(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "x_range": ["2024-01-01", "2024-01-10"]}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert tuple(result.figure.layout.xaxis.range) == ("2024-01-01", "2024-01-10")

    def test_set_y_range_panel_indexed(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        e3 = _make_entry("C", n=30, desc="Gamma")
        spec = {
            "panels": [["A"], ["B"], ["C"]],
            "y_range": {"3": [0, 100]},
        }
        result = build_figure_from_spec(spec, [e1, e2, e3])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis3.range == (0, 100)

    def test_set_y_scale_log(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "panels": [["A"], ["B"]],
            "log_scale": {"2": "log"},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis2.type == "log"

    def test_set_margin(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {
            "panels": [["A"]],
            "margin": {"l": 50, "r": 30, "t": 40, "b": 60},
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.margin.l == 50
        assert result.figure.layout.margin.r == 30
        assert result.figure.layout.margin.t == 40
        assert result.figure.layout.margin.b == 60

    def test_empty_spec_is_fine(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]]}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.panel_count == 1

    def test_multiple_layout_fields(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "panels": [["A"], ["B"]],
            "title": "Multi-Field",
            "y_label": {"1": "B (nT)", "2": "V (km/s)"},
            "font_size": 14,
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.title.text == "Multi-Field"
        assert result.figure.layout.yaxis.title.text == "B (nT)"
        assert result.figure.layout.yaxis2.title.text == "V (km/s)"
        assert result.figure.layout.font.size == 14


# ---------------------------------------------------------------------------
# build_figure_from_spec — trace semantic fields
# ---------------------------------------------------------------------------

class TestTraceSemanticFields:
    def test_set_trace_color(self):
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {"panels": [["M"]], "trace_colors": {"Mag": "blue"}}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.color == "blue"

    def test_set_line_style(self):
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {
            "panels": [["M"]],
            "line_styles": {"Mag": {"width": 3, "dash": "dot"}},
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.width == 3
        assert result.figure.data[0].line.dash == "dot"

    def test_set_trace_visibility(self):
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        spec = {
            "panels": [["A", "B"]],
            "trace_visibility": {"Alpha": False},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].visible is False
        assert result.figure.data[1].visible is not False

    def test_set_trace_mode(self):
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {"panels": [["M"]], "trace_mode": {"Mag": "markers"}}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].mode == "markers"

    def test_trace_not_found_no_error(self):
        """Unknown trace label in trace_colors is silently ignored."""
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {"panels": [["M"]], "trace_colors": {"NonExistent": "red"}}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        # Original color unchanged (assigned by ColorState)
        assert result.figure.data[0].line.color != "red"

    def test_trace_substring_match(self):
        """Traces matched by substring (trace label contains selector)."""
        entry = _make_entry("M", n=20, desc="Magnetic Field Bx")
        spec = {"panels": [["M"]], "trace_colors": {"Bx": "red"}}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.color == "red"

    def test_empty_spec_no_trace_patches(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]]}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)


# ---------------------------------------------------------------------------
# build_figure_from_spec — core behavior
# ---------------------------------------------------------------------------

class TestBuildFigureFromSpec:
    def test_basic_spec(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]], "title": "Test Plot"}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.title.text == "Test Plot"
        assert result.panel_count == 1
        assert len(result.trace_labels) == 1
        assert result.trace_labels[0] == "Alpha"

    def test_multi_panel(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "panels": [["A"], ["B"]],
            "title": "Multi-Panel",
            "y_label": {"1": "B (nT)", "2": "V (km/s)"},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.panel_count == 2
        assert result.figure.layout.title.text == "Multi-Panel"
        assert result.figure.layout.yaxis.title.text == "B (nT)"
        assert result.figure.layout.yaxis2.title.text == "V (km/s)"

    def test_color_state_preserved(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        cs = ColorState(label_colors={"Alpha": "#ff0000"}, color_index=5)
        spec = {"panels": [["A"]]}
        result = build_figure_from_spec(spec, [entry], color_state=cs)
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.color == "#ff0000"
        assert result.color_state.color_index == 5

    def test_time_range(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"panels": [["A"]]}
        result = build_figure_from_spec(
            spec, [entry], time_range="2024-01-01 to 2024-01-03",
        )
        assert isinstance(result, RenderResult)
        xrange = result.figure.layout.xaxis.range
        assert xrange is not None
        assert "2024-01-01" in str(xrange[0])

    def test_empty_entries_error(self):
        spec = {"panels": [["A"]]}
        result = build_figure_from_spec(spec, [])
        assert isinstance(result, dict)
        assert result["status"] == "error"

    def test_zero_point_entry_error(self):
        empty = DataEntry(
            label="empty",
            data=pd.DataFrame({"v": pd.Series(dtype=float)},
                              index=pd.DatetimeIndex([], name="time")),
        )
        spec = {"panels": [["empty"]]}
        result = build_figure_from_spec(spec, [empty])
        assert isinstance(result, dict)
        assert result["status"] == "error"

    def test_label_not_found_error(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {"panels": [["A"], ["MISSING"]]}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "MISSING" in result["message"]

    def test_vector_decomposition(self):
        entry = _make_entry("Bvec", n=30, ncols=3)
        spec = {"panels": [["Bvec"]]}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert len(result.trace_labels) == 3
        assert "test data (x)" in result.trace_labels

    def test_no_panels_overlay(self):
        """When panels is not specified, all entries overlay in one panel."""
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        spec = {}
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.panel_count == 1
        assert len(result.trace_labels) == 2

    def test_flat_spec_with_labels(self):
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"labels": "A", "title": "Test"}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.title.text == "Test"

    def test_with_panels_and_labels(self):
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "labels": "A,B",
            "panels": [["A"], ["B"]],
            "title": "Panels",
            "y_label": {"1": "nT", "2": "km/s"},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.panel_count == 2
        assert result.figure.layout.title.text == "Panels"
        assert result.figure.layout.yaxis.title.text == "nT"
        assert result.figure.layout.yaxis2.title.text == "km/s"

    def test_trace_colors(self):
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {"labels": "M", "trace_colors": {"Mag": "red"}}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.color == "red"

    def test_vlines(self):
        entry = _make_entry("V", n=20, desc="Vdata")
        spec = {
            "labels": "V",
            "vlines": [{"x": "2024-01-01T00:10:00", "color": "blue"}],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert len(result.figure.layout.shapes) >= 1

    def test_font_size_and_legend(self):
        entry = _make_entry("X", n=20, desc="Xray")
        spec = {"labels": "X", "font_size": 18, "legend": False}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.font.size == 18
        assert result.figure.layout.showlegend is False

    def test_vrects(self):
        entry = _make_entry("V", n=20, desc="Vdata")
        spec = {
            "labels": "V",
            "vrects": [{"x0": "2024-01-10", "x1": "2024-01-15",
                         "color": "rgba(0,0,255,0.2)", "label": "Storm"}],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert len(result.figure.layout.shapes) >= 1
        shape = result.figure.layout.shapes[0]
        assert shape.type == "rect"

    def test_annotations(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {
            "labels": "A",
            "annotations": [{"text": "Event", "x": "2024-01-01", "y": 5}],
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert len(result.figure.layout.annotations) >= 1
        assert result.figure.layout.annotations[0].text == "Event"

    def test_log_scale_y(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {"labels": "A", "log_scale": "y"}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis.type == "log"

    def test_log_scale_dict(self):
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        spec = {
            "labels": "A,B",
            "panels": [["A"], ["B"]],
            "log_scale": {"2": "log"},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis2.type == "log"

    def test_canvas_size(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {"labels": "A", "canvas_size": {"width": 1920, "height": 1080}}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.width == 1920
        assert result.figure.layout.height == 1080

    def test_x_range(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {"labels": "A", "x_range": ["2024-01-01", "2024-01-10"]}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert tuple(result.figure.layout.xaxis.range) == ("2024-01-01", "2024-01-10")

    def test_y_range_list(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {"labels": "A", "y_range": [0, 100]}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis.range == (0, 100)

    def test_y_range_dict(self):
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        spec = {
            "labels": "A,B",
            "panels": [["A"], ["B"]],
            "y_range": {"2": [0, 50]},
        }
        result = build_figure_from_spec(spec, [e1, e2])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.yaxis2.range == (0, 50)

    def test_line_styles(self):
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {
            "labels": "M",
            "line_styles": {"Mag": {"width": 3, "dash": "dot"}},
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.width == 3
        assert result.figure.data[0].line.dash == "dot"

    def test_theme(self):
        entry = _make_entry("A", n=20, desc="Alpha")
        spec = {"labels": "A", "theme": "plotly_white"}
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.template is not None

    def test_combined_layout_and_traces(self):
        """Both layout and trace style fields are applied together."""
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {
            "panels": [["M"]],
            "title": "Combined",
            "font_size": 16,
            "trace_colors": {"Mag": "red"},
            "line_styles": {"Mag": {"width": 2}},
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.layout.title.text == "Combined"
        assert result.figure.layout.font.size == 16
        assert result.figure.data[0].line.color == "red"
        assert result.figure.data[0].line.width == 2

    def test_vector_trace_colors(self):
        """Trace colors work with vector-decomposed trace labels like 'desc (x)'."""
        entry = _make_entry("Bvec", n=30, ncols=3, desc="B")
        spec = {
            "panels": [["Bvec"]],
            "trace_colors": {
                "B (x)": "red",
                "B (z)": "blue",
            },
        }
        result = build_figure_from_spec(spec, [entry])
        assert isinstance(result, RenderResult)
        assert result.figure.data[0].line.color == "red"    # B (x)
        assert result.figure.data[2].line.color == "blue"   # B (z)
