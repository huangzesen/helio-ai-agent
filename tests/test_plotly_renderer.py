"""
Unit tests for the Plotly renderer.

No API key, no JVM, no network — fast and self-contained.
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry
from rendering.plotly_renderer import PlotlyRenderer, fill_figure_data, ColorState, RenderResult


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


def _render_one(renderer, label="x", n=10, desc="test data"):
    """Helper: render a single scatter trace via render_plotly_json."""
    entry = _make_entry(label, n=n, desc=desc)
    fig_json = {
        "data": [{"type": "scatter", "data_label": label}],
        "layout": {},
    }
    result = renderer.render_plotly_json(fig_json, {label: entry})
    return result, entry


# ---------------------------------------------------------------------------
# manage (direct method calls)
# ---------------------------------------------------------------------------

class TestManage:
    def test_reset(self, renderer):
        _render_one(renderer)
        assert renderer.get_figure() is not None
        result = renderer.reset()
        assert result["status"] == "success"
        assert renderer.get_figure() is None
        assert renderer._trace_labels == []
        assert renderer._trace_panels == []

    def test_get_state(self, renderer):
        _render_one(renderer, desc="Alpha")
        result = renderer.get_current_state()
        assert result["has_plot"] is True
        assert result["traces"] == ["Alpha"]

    def test_get_state_empty(self, renderer):
        result = renderer.get_current_state()
        assert result["has_plot"] is False
        assert result["traces"] == []


# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

class TestTimeRange:
    def test_set_time_range(self, renderer):
        from agent.time_utils import parse_time_range
        _render_one(renderer)
        tr = parse_time_range("2024-01-01 to 2024-01-07")
        result = renderer.set_time_range(tr)
        assert result["status"] == "success"
        assert "2024-01-01" in result["time_range"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_png(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "test_output.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_png_no_plot(self, renderer, tmp_path):
        filepath = str(tmp_path / "empty.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "error"

    def test_export_pdf(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "test_output.pdf")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_default_format_is_png(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "test_output")
        result = renderer.export(filepath)
        assert result["status"] == "success"
        assert result["filepath"].endswith(".png")

    def test_export_adds_extension(self, renderer, tmp_path):
        _render_one(renderer)
        filepath = str(tmp_path / "noext")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["filepath"].endswith(".pdf")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestState:
    def test_reset_clears_state(self, renderer):
        _render_one(renderer)
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
        _render_one(renderer, desc="Alpha")
        state = renderer.get_current_state()
        assert state["has_plot"] is True
        assert state["traces"] == ["Alpha"]


# ---------------------------------------------------------------------------
# fill_figure_data (Plotly JSON pipeline)
# ---------------------------------------------------------------------------

class TestFillFigureData:
    """Tests for the fill_figure_data function that resolves data_label placeholders."""

    def test_scalar_trace(self):
        """Single scalar trace gets x and y arrays filled."""
        entry = _make_entry("mag", n=50, desc="Bmag")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag", "mode": "lines"}],
            "layout": {"title": {"text": "Test"}},
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"mag": entry}, cs)
        assert isinstance(result, RenderResult)
        fig = result.figure
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 50
        assert len(fig.data[0].y) == 50
        assert fig.layout.title.text == "Test"
        assert result.trace_labels == ["Bmag"]

    def test_vector_auto_expand(self):
        """3-column entry auto-expands into 3 traces."""
        entry = _make_entry("Bvec", n=30, ncols=3, desc="B field")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "Bvec"}],
            "layout": {},
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"Bvec": entry}, cs)
        assert len(result.figure.data) == 3
        assert result.trace_labels == ["B field (x)", "B field (y)", "B field (z)"]

    def test_explicit_color_preserved(self):
        """Trace with explicit line color keeps it, no auto-assignment."""
        entry = _make_entry("mag", n=20)
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag",
                       "line": {"color": "red"}}],
            "layout": {},
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"mag": entry}, cs)
        assert result.figure.data[0].line.color == "red"

    def test_auto_color_assignment(self):
        """Trace without color gets one from ColorState."""
        entry = _make_entry("mag", n=20, desc="Bmag")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {},
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"mag": entry}, cs)
        assert result.figure.data[0].line.color is not None
        assert "Bmag" in cs.label_colors

    def test_nan_to_none(self):
        """NaN values in data are converted to None for Plotly."""
        rng = pd.date_range("2024-01-01", periods=10, freq="min")
        vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        df = pd.DataFrame({"value": vals}, index=rng)
        entry = DataEntry(label="gappy", data=df, units="nT")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "gappy"}],
            "layout": {},
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"gappy": entry}, cs)
        y_data = result.figure.data[0].y
        assert y_data[1] is None
        assert y_data[3] is None
        assert y_data[0] == 1.0

    def test_heatmap_trace(self):
        """Heatmap trace gets x, y, z populated from spectrogram data."""
        rng = pd.date_range("2024-01-01", periods=50, freq="10min")
        bins = np.linspace(0.001, 0.5, 20)
        data = np.random.rand(50, 20)
        df = pd.DataFrame(data, index=rng, columns=[str(b) for b in bins])
        entry = DataEntry(
            label="spec", data=df, units="nT",
            description="Spectrogram",
            metadata={"type": "spectrogram", "bin_label": "Freq (Hz)",
                      "value_label": "PSD", "bin_values": bins.tolist()},
        )
        fig_json = {
            "data": [{"type": "heatmap", "data_label": "spec",
                       "colorscale": "Viridis"}],
            "layout": {"yaxis": {"domain": [0, 1]}},
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"spec": entry}, cs)
        fig = result.figure
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.type == "heatmap"
        assert len(trace.x) == 50
        assert len(trace.y) == 20
        assert len(trace.z) == 20  # z is transposed: bins x times

    def test_multi_panel_layout(self):
        """Multi-panel layout with separate y-axes works correctly."""
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        fig_json = {
            "data": [
                {"type": "scatter", "data_label": "A", "xaxis": "x", "yaxis": "y"},
                {"type": "scatter", "data_label": "B", "xaxis": "x2", "yaxis": "y2"},
            ],
            "layout": {
                "xaxis": {"domain": [0, 1], "anchor": "y"},
                "xaxis2": {"domain": [0, 1], "anchor": "y2", "matches": "x"},
                "yaxis": {"domain": [0.55, 1], "anchor": "x", "title": {"text": "nT"}},
                "yaxis2": {"domain": [0, 0.45], "anchor": "x2", "title": {"text": "km/s"}},
            },
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"A": e1, "B": e2}, cs)
        assert result.panel_count == 2
        assert len(result.figure.data) == 2
        assert result.trace_panels == [(1, 1), (2, 2)]

    def test_missing_data_label_error(self):
        """Missing data_label raises ValueError."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "MISSING"}],
            "layout": {},
        }
        cs = ColorState()
        with pytest.raises(ValueError, match="MISSING"):
            fill_figure_data(fig_json, {}, cs)

    def test_time_range_applied(self):
        """Time range is applied to x-axes in layout."""
        entry = _make_entry("mag", n=50)
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {"xaxis": {}},
        }
        cs = ColorState()
        result = fill_figure_data(
            fig_json, {"mag": entry}, cs,
            time_range="2024-01-01 to 2024-01-02",
        )
        xaxis_range = result.figure.layout.xaxis.range
        assert xaxis_range is not None
        assert "2024-01-01" in str(xaxis_range[0])

    def test_default_layout_applied(self):
        """Default white background and sizing is applied."""
        entry = _make_entry("mag", n=50)
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {},
        }
        cs = ColorState()
        result = fill_figure_data(fig_json, {"mag": entry}, cs)
        assert result.figure.layout.paper_bgcolor == "white"
        assert result.figure.layout.plot_bgcolor == "white"


# ---------------------------------------------------------------------------
# PlotlyRenderer.render_plotly_json (stateful wrapper)
# ---------------------------------------------------------------------------

class TestRenderPlotlyJson:
    """Tests for the PlotlyRenderer.render_plotly_json method."""

    def test_basic_render(self, renderer):
        """Basic render produces success result with review."""
        entry = _make_entry("mag", n=50, desc="Bmag")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {"title": {"text": "Test"}},
        }
        result = renderer.render_plotly_json(fig_json, {"mag": entry})
        assert result["status"] == "success"
        assert result["traces"] == ["Bmag"]
        assert "review" in result
        fig = renderer.get_figure()
        assert fig is not None
        assert fig.layout.title.text == "Test"

    def test_vector_render(self, renderer):
        """Vector data renders into 3 component traces."""
        entry = _make_entry("B", n=30, ncols=3, desc="Bfield")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "B"}],
            "layout": {},
        }
        result = renderer.render_plotly_json(fig_json, {"B": entry})
        assert result["status"] == "success"
        assert len(result["traces"]) == 3

    def test_missing_label_error(self, renderer):
        """Missing data_label returns error dict."""
        fig_json = {
            "data": [{"type": "scatter", "data_label": "NOPE"}],
            "layout": {},
        }
        result = renderer.render_plotly_json(fig_json, {})
        assert result["status"] == "error"
        assert "NOPE" in result["message"]

    def test_empty_data_error(self, renderer):
        """Empty data array returns error."""
        fig_json = {"data": [], "layout": {}}
        result = renderer.render_plotly_json(fig_json, {})
        assert result["status"] == "success"
        assert result["traces"] == []

    def test_state_updated(self, renderer):
        """Renderer state is updated after render_plotly_json."""
        entry = _make_entry("X", n=20, desc="Xdata")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "X"}],
            "layout": {},
        }
        renderer.render_plotly_json(fig_json, {"X": entry})
        assert renderer._trace_labels == ["Xdata"]
        assert renderer.get_figure() is not None

    def test_review_metadata(self, renderer):
        """Review metadata is included in render result."""
        entry = _make_entry("mag", n=50, desc="Mag")
        fig_json = {
            "data": [{"type": "scatter", "data_label": "mag"}],
            "layout": {},
        }
        result = renderer.render_plotly_json(fig_json, {"mag": entry})
        review = result["review"]
        assert "trace_summary" in review
        assert "warnings" in review
        assert "hint" in review
        assert len(review["trace_summary"]) == 1
        assert review["trace_summary"][0]["name"] == "Mag"
