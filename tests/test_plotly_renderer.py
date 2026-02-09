"""
Unit tests for the Plotly renderer.

No API key, no JVM, no network — fast and self-contained.
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry
from rendering.plotly_renderer import PlotlyRenderer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def renderer():
    return PlotlyRenderer(verbose=False)


def _make_entry(label: str, n: int = 100, ncols: int = 1, desc: str = "test data") -> DataEntry:
    """Create a synthetic DataEntry for testing."""
    rng = pd.date_range("2024-01-01", periods=n, freq="min")
    if ncols == 1:
        df = pd.DataFrame({"value": np.random.randn(n)}, index=rng)
    else:
        cols = {f"c{i}": np.random.randn(n) for i in range(ncols)}
        df = pd.DataFrame(cols, index=rng)
    return DataEntry(label=label, data=df, units="nT", description=desc)


# ---------------------------------------------------------------------------
# plot_data
# ---------------------------------------------------------------------------

class TestPlotData:
    def test_plot_single_scalar(self, renderer):
        entry = _make_entry("mag", n=50)
        result = renderer.plot_data([entry])
        assert result["status"] == "success"
        assert result["panels"] == 1
        assert result["traces"] == ["test data"]
        fig = renderer.get_figure()
        assert fig is not None
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 50

    def test_plot_vector_decomposition(self, renderer):
        entry = _make_entry("Bvec", n=30, ncols=3)
        result = renderer.plot_data([entry])
        assert result["status"] == "success"
        assert len(result["traces"]) == 3
        assert result["traces"] == ["test data (x)", "test data (y)", "test data (z)"]
        fig = renderer.get_figure()
        assert len(fig.data) == 3

    def test_plot_multiple_overlay(self, renderer):
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        result = renderer.plot_data([e1, e2])
        assert result["status"] == "success"
        assert len(result["traces"]) == 2
        assert result["panels"] == 1
        fig = renderer.get_figure()
        assert len(fig.data) == 2

    def test_empty_entries_error(self, renderer):
        result = renderer.plot_data([])
        assert result["status"] == "error"

    def test_empty_data_error(self, renderer):
        empty = DataEntry(
            label="empty",
            data=pd.DataFrame({"v": pd.Series(dtype=float)},
                              index=pd.DatetimeIndex([], name="time")),
        )
        result = renderer.plot_data([empty])
        assert result["status"] == "error"
        assert "no data points" in result["message"]

    def test_multi_panel(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        result = renderer.plot_data(
            [e1, e2],
            panels=[["A"], ["B"]],
        )
        assert result["status"] == "success"
        assert result["panels"] == 2
        assert len(result["traces"]) == 2
        fig = renderer.get_figure()
        assert len(fig.data) == 2

    def test_multi_panel_overlay_in_panel(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        e3 = _make_entry("C", n=10, desc="Gamma")
        result = renderer.plot_data(
            [e1, e2, e3],
            panels=[["A", "B"], ["C"]],
        )
        assert result["status"] == "success"
        assert result["panels"] == 2
        assert len(result["traces"]) == 3

    def test_fresh_figure_each_call(self, renderer):
        e1 = _make_entry("A", n=10)
        renderer.plot_data([e1])
        fig1_data_count = len(renderer.get_figure().data)

        e2 = _make_entry("B", n=10)
        renderer.plot_data([e2])
        fig2_data_count = len(renderer.get_figure().data)

        # Second plot_data should create fresh figure, not accumulate
        assert fig1_data_count == 1
        assert fig2_data_count == 1

    def test_trace_label_tracking(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        renderer.plot_data([e1, e2])
        assert renderer._trace_labels == ["Alpha", "Beta"]
        assert renderer._trace_panels == [1, 1]

    def test_with_title(self, renderer):
        entry = _make_entry("mag", n=10)
        result = renderer.plot_data([entry], title="My Plot")
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.title.text == "My Plot"

    def test_panel_label_not_found(self, renderer):
        e1 = _make_entry("A", n=10)
        result = renderer.plot_data([e1], panels=[["A"], ["MISSING"]])
        assert result["status"] == "error"
        assert "MISSING" in result["message"]


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

class TestStyle:
    def test_style_title(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(title="New Title")
        assert result["status"] == "success"
        assert renderer.get_figure().layout.title.text == "New Title"

    def test_style_y_label_string(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(y_label="B (nT)")
        assert result["status"] == "success"
        assert renderer.get_figure().layout.yaxis.title.text == "B (nT)"

    def test_style_y_label_dict(self, renderer):
        e1 = _make_entry("A", n=10)
        e2 = _make_entry("B", n=10)
        renderer.plot_data([e1, e2], panels=[["A"], ["B"]])
        result = renderer.style(y_label={"1": "Panel 1", "2": "Panel 2"})
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.yaxis.title.text == "Panel 1"
        assert fig.layout.yaxis2.title.text == "Panel 2"

    def test_style_trace_colors(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        renderer.plot_data([e1])
        result = renderer.style(trace_colors={"Alpha": "red"})
        assert result["status"] == "success"
        assert renderer.get_figure().data[0].line.color == "red"

    def test_style_log_scale_y(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(log_scale="y")
        assert result["status"] == "success"
        assert renderer.get_figure().layout.yaxis.type == "log"

    def test_style_canvas_size(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(canvas_size={"width": 1920, "height": 1080})
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.width == 1920
        assert fig.layout.height == 1080

    def test_style_font_size(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(font_size=18)
        assert result["status"] == "success"
        assert renderer.get_figure().layout.font.size == 18

    def test_style_annotations(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(annotations=[{"text": "Event", "x": "2024-01-01", "y": 5}])
        assert result["status"] == "success"
        assert len(renderer.get_figure().layout.annotations) == 1

    def test_style_no_figure_error(self, renderer):
        result = renderer.style(title="No Plot")
        assert result["status"] == "error"
        assert "No plot" in result["message"]

    def test_style_legend(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(legend=False)
        assert result["status"] == "success"
        assert renderer.get_figure().layout.showlegend is False

    def test_style_theme(self, renderer):
        renderer.plot_data([_make_entry("x")])
        result = renderer.style(theme="plotly_dark")
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# manage
# ---------------------------------------------------------------------------

class TestManage:
    def test_reset(self, renderer):
        renderer.plot_data([_make_entry("x")])
        assert renderer.get_figure() is not None
        result = renderer.manage("reset")
        assert result["status"] == "success"
        assert renderer.get_figure() is None
        assert renderer._trace_labels == []
        assert renderer._trace_panels == []

    def test_get_state(self, renderer):
        renderer.plot_data([_make_entry("x", desc="Alpha")])
        result = renderer.manage("get_state")
        assert result["has_plot"] is True
        assert result["traces"] == ["Alpha"]

    def test_get_state_empty(self, renderer):
        result = renderer.manage("get_state")
        assert result["has_plot"] is False
        assert result["traces"] == []

    def test_remove_trace(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        renderer.plot_data([e1, e2])
        assert len(renderer.get_figure().data) == 2

        result = renderer.manage("remove_trace", label="Alpha")
        assert result["status"] == "success"
        assert len(renderer.get_figure().data) == 1
        assert renderer._trace_labels == ["Beta"]

    def test_remove_trace_not_found(self, renderer):
        renderer.plot_data([_make_entry("A", desc="Alpha")])
        result = renderer.manage("remove_trace", label="Nonexistent")
        assert result["status"] == "error"

    def test_add_trace(self, renderer):
        renderer.plot_data([_make_entry("A", n=10, desc="Alpha")])
        e2 = _make_entry("B", n=10, desc="Beta")
        result = renderer.manage("add_trace", entry=e2, panel=1)
        assert result["status"] == "success"
        assert len(renderer.get_figure().data) == 2
        assert "Beta" in renderer._trace_labels

    def test_unknown_action(self, renderer):
        result = renderer.manage("nonexistent")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

class TestTimeRange:
    def test_set_time_range(self, renderer):
        from agent.time_utils import parse_time_range
        renderer.plot_data([_make_entry("x")])
        tr = parse_time_range("2024-01-01 to 2024-01-07")
        result = renderer.set_time_range(tr)
        assert result["status"] == "success"
        assert "2024-01-01" in result["time_range"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_png(self, renderer, tmp_path):
        renderer.plot_data([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_png_no_plot(self, renderer, tmp_path):
        filepath = str(tmp_path / "empty.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "error"

    def test_export_pdf(self, renderer, tmp_path):
        renderer.plot_data([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.pdf")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_default_format_is_png(self, renderer, tmp_path):
        renderer.plot_data([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output")
        result = renderer.export(filepath)
        assert result["status"] == "success"
        assert result["filepath"].endswith(".png")

    def test_export_adds_extension(self, renderer, tmp_path):
        renderer.plot_data([_make_entry("x", n=10)])
        filepath = str(tmp_path / "noext")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["filepath"].endswith(".pdf")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestState:
    def test_reset_clears_state(self, renderer):
        renderer.plot_data([_make_entry("x")])
        assert renderer.get_figure() is not None

        result = renderer.reset()
        assert result["status"] == "success"
        assert renderer.get_figure() is None
        assert renderer._panel_count == 0
        assert renderer._trace_labels == []
        assert renderer._trace_panels == []

    def test_get_current_state_empty(self, renderer):
        state = renderer.get_current_state()
        assert state["has_plot"] is False
        assert state["time_range"] is None
        assert state["traces"] == []

    def test_get_current_state_after_plot(self, renderer):
        renderer.plot_data([_make_entry("x", desc="Alpha")])
        state = renderer.get_current_state()
        assert state["has_plot"] is True
        assert state["traces"] == ["Alpha"]
