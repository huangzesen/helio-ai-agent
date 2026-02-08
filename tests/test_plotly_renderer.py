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


def _make_entry(label: str, n: int = 100, ncols: int = 1) -> DataEntry:
    """Create a synthetic DataEntry for testing."""
    rng = pd.date_range("2024-01-01", periods=n, freq="min")
    if ncols == 1:
        df = pd.DataFrame({"value": np.random.randn(n)}, index=rng)
    else:
        cols = {f"c{i}": np.random.randn(n) for i in range(ncols)}
        df = pd.DataFrame(cols, index=rng)
    return DataEntry(label=label, data=df, units="nT", description="test data")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class TestPlotDataset:
    def test_plot_single_scalar(self, renderer):
        entry = _make_entry("mag", n=50)
        result = renderer.plot_dataset([entry])
        assert result["status"] == "success"
        assert result["num_series"] == 1
        assert result["labels"] == ["test data"]  # uses description for display
        fig = renderer.get_figure()
        assert fig is not None
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 50

    def test_plot_vector_decomposition(self, renderer):
        entry = _make_entry("Bvec", n=30, ncols=3)
        result = renderer.plot_dataset([entry])
        assert result["status"] == "success"
        assert result["num_series"] == 3
        assert result["labels"] == ["test data (x)", "test data (y)", "test data (z)"]
        fig = renderer.get_figure()
        assert len(fig.data) == 3

    def test_plot_multiple_overlay(self, renderer):
        e1 = _make_entry("A", n=20)
        e2 = _make_entry("B", n=20)
        result = renderer.plot_dataset([e1, e2])
        assert result["status"] == "success"
        assert result["num_series"] == 2
        fig = renderer.get_figure()
        assert len(fig.data) == 2

    def test_empty_entries_error(self, renderer):
        result = renderer.plot_dataset([])
        assert result["status"] == "error"

    def test_empty_data_error(self, renderer):
        empty = DataEntry(
            label="empty",
            data=pd.DataFrame({"v": pd.Series(dtype=float)},
                              index=pd.DatetimeIndex([], name="time")),
        )
        result = renderer.plot_dataset([empty])
        assert result["status"] == "error"
        assert "no data points" in result["message"]

    def test_panel_targeted(self, renderer):
        e1 = _make_entry("top", n=10)
        e2 = _make_entry("bottom", n=10)
        renderer.plot_dataset([e1], index=0)
        result = renderer.plot_dataset([e2], index=1)
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert len(fig.data) == 2


# ---------------------------------------------------------------------------
# Axis / Layout
# ---------------------------------------------------------------------------

class TestAxisLayout:
    def test_set_title(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_plot_title("My Plot")
        assert result["status"] == "success"
        assert renderer.get_figure().layout.title.text == "My Plot"

    def test_set_axis_label(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_axis_label("y", "Magnetic Field (nT)")
        assert result["status"] == "success"

    def test_set_axis_label_bad_axis(self, renderer):
        result = renderer.set_axis_label("x", "Time")
        assert result["status"] == "error"

    def test_toggle_log_scale(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.toggle_log_scale("y", True)
        assert result["status"] == "success"
        assert result["log_scale"] is True

    def test_set_axis_range(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_axis_range("y", -10.0, 10.0)
        assert result["status"] == "success"
        assert result["min"] == -10.0
        assert result["max"] == 10.0

    def test_set_canvas_size(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_canvas_size(1920, 1080)
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.width == 1920
        assert fig.layout.height == 1080


# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

class TestTimeRange:
    def test_set_time_range(self, renderer):
        from agent.time_utils import parse_time_range
        renderer.plot_dataset([_make_entry("x")])
        tr = parse_time_range("2024-01-01 to 2024-01-07")
        result = renderer.set_time_range(tr)
        assert result["status"] == "success"
        assert "2024-01-01" in result["time_range"]


# ---------------------------------------------------------------------------
# Render type
# ---------------------------------------------------------------------------

class TestRenderType:
    def test_set_render_type_scatter(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_render_type("scatter", index=0)
        assert result["status"] == "success"
        assert renderer.get_figure().data[0].mode == "markers"

    def test_set_render_type_fill_to_zero(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_render_type("fill_to_zero", index=0)
        assert result["status"] == "success"
        assert renderer.get_figure().data[0].fill == "tozeroy"

    def test_set_render_type_staircase(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_render_type("staircase", index=0)
        assert result["status"] == "success"
        assert renderer.get_figure().data[0].line.shape == "hv"

    def test_set_render_type_unsupported(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_render_type("spectrogram", index=0)
        assert result["status"] == "error"

    def test_set_render_type_bad_index(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        result = renderer.set_render_type("scatter", index=99)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_png(self, renderer, tmp_path):
        renderer.plot_dataset([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.png")
        result = renderer.export_png(filepath)
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_png_no_plot(self, renderer, tmp_path):
        filepath = str(tmp_path / "empty.png")
        result = renderer.export_png(filepath)
        assert result["status"] == "error"

    def test_export_pdf(self, renderer, tmp_path):
        renderer.plot_dataset([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.pdf")
        result = renderer.export_pdf(filepath)
        assert result["status"] == "success"
        assert result["size_bytes"] > 0


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestState:
    def test_reset_clears_state(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        assert renderer.get_figure() is not None

        result = renderer.reset()
        assert result["status"] == "success"
        assert renderer.get_figure() is None
        assert renderer._panel_count == 0

    def test_get_current_state_empty(self, renderer):
        state = renderer.get_current_state()
        assert state["has_plot"] is False
        assert state["time_range"] is None

    def test_get_current_state_after_plot(self, renderer):
        renderer.plot_dataset([_make_entry("x")])
        state = renderer.get_current_state()
        assert state["has_plot"] is True


# ---------------------------------------------------------------------------
# Not-supported methods
# ---------------------------------------------------------------------------

class TestNotSupported:
    def test_save_session(self, renderer):
        result = renderer.save_session("test.vap")
        assert result["status"] == "error"

    def test_load_session(self, renderer):
        result = renderer.load_session("test.vap")
        assert result["status"] == "error"

    def test_execute_script(self, renderer):
        result = renderer.execute_script("sc.plot(0, 'vap+...')")
        assert result["status"] == "error"

    def test_set_color_table(self, renderer):
        result = renderer.set_color_table("matlab_jet")
        assert result["status"] == "error"
