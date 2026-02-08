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
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_png(self, renderer, tmp_path):
        renderer.plot_dataset([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_png_no_plot(self, renderer, tmp_path):
        filepath = str(tmp_path / "empty.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "error"

    def test_export_pdf(self, renderer, tmp_path):
        renderer.plot_dataset([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.pdf")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_default_format_is_png(self, renderer, tmp_path):
        renderer.plot_dataset([_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output")
        result = renderer.export(filepath)
        assert result["status"] == "success"
        assert result["filepath"].endswith(".png")

    def test_export_adds_extension(self, renderer, tmp_path):
        renderer.plot_dataset([_make_entry("x", n=10)])
        filepath = str(tmp_path / "noext")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["filepath"].endswith(".pdf")


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
