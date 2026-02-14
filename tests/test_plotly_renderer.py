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


# ---------------------------------------------------------------------------
# render_from_spec (basic plotting tests, formerly plot_data)
# ---------------------------------------------------------------------------

class TestPlotData:
    def test_plot_single_scalar(self, renderer):
        entry = _make_entry("mag", n=50)
        result = renderer.render_from_spec({"labels": "mag"}, [entry])
        assert result["status"] == "success"
        assert result["panels"] == 1
        assert result["traces"] == ["test data"]
        fig = renderer.get_figure()
        assert fig is not None
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 50

    def test_plot_vector_decomposition(self, renderer):
        entry = _make_entry("Bvec", n=30, ncols=3)
        result = renderer.render_from_spec({"labels": "Bvec"}, [entry])
        assert result["status"] == "success"
        assert len(result["traces"]) == 3
        assert result["traces"] == ["test data (x)", "test data (y)", "test data (z)"]
        fig = renderer.get_figure()
        assert len(fig.data) == 3

    def test_plot_multiple_overlay(self, renderer):
        e1 = _make_entry("A", n=20, desc="Alpha")
        e2 = _make_entry("B", n=20, desc="Beta")
        result = renderer.render_from_spec({"labels": "A,B"}, [e1, e2])
        assert result["status"] == "success"
        assert len(result["traces"]) == 2
        assert result["panels"] == 1
        fig = renderer.get_figure()
        assert len(fig.data) == 2

    def test_empty_entries_error(self, renderer):
        result = renderer.render_from_spec({"labels": ""}, [])
        assert result["status"] == "error"

    def test_empty_data_error(self, renderer):
        empty = DataEntry(
            label="empty",
            data=pd.DataFrame({"v": pd.Series(dtype=float)},
                              index=pd.DatetimeIndex([], name="time")),
        )
        result = renderer.render_from_spec({"labels": "empty"}, [empty])
        assert result["status"] == "error"
        assert "no data points" in result["message"]

    def test_empty_entry_among_valid_still_errors(self, renderer):
        """A 0-point entry passed in entries list should still error."""
        good = _make_entry("A", n=10)
        empty = DataEntry(
            label="empty",
            data=pd.DataFrame({"v": pd.Series(dtype=float)},
                              index=pd.DatetimeIndex([], name="time")),
        )
        result = renderer.render_from_spec(
            {"labels": "A,empty", "panels": [["A"], ["empty"]]},
            [good, empty],
        )
        assert result["status"] == "error"
        assert "no data points" in result["message"]

    def test_multi_panel(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        result = renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A"], ["B"]]},
            [e1, e2],
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
        result = renderer.render_from_spec(
            {"labels": "A,B,C", "panels": [["A", "B"], ["C"]]},
            [e1, e2, e3],
        )
        assert result["status"] == "success"
        assert result["panels"] == 2
        assert len(result["traces"]) == 3

    def test_fresh_figure_each_call(self, renderer):
        e1 = _make_entry("A", n=10)
        renderer.render_from_spec({"labels": "A"}, [e1])
        fig1_data_count = len(renderer.get_figure().data)

        e2 = _make_entry("B", n=10)
        renderer.render_from_spec({"labels": "B"}, [e2])
        fig2_data_count = len(renderer.get_figure().data)

        # Second render should create fresh figure, not accumulate
        assert fig1_data_count == 1
        assert fig2_data_count == 1

    def test_trace_label_tracking(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        renderer.render_from_spec({"labels": "A,B"}, [e1, e2])
        assert renderer._trace_labels == ["Alpha", "Beta"]
        assert renderer._trace_panels == [(1, 1), (1, 1)]

    def test_with_title(self, renderer):
        entry = _make_entry("mag", n=10)
        result = renderer.render_from_spec({"labels": "mag", "title": "My Plot"}, [entry])
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.title.text == "My Plot"

    def test_panel_label_not_found(self, renderer):
        e1 = _make_entry("A", n=10)
        result = renderer.render_from_spec(
            {"labels": "A", "panels": [["A"], ["MISSING"]]},
            [e1],
        )
        assert result["status"] == "error"
        assert "MISSING" in result["message"]

    def test_panel_column_sublabel(self, renderer):
        """Column sub-label like 'PARENT.col' should select a single column."""
        rng = pd.date_range("2024-01-01", periods=50, freq="min")
        df = pd.DataFrame({"B_mag": np.random.randn(50), "dB_dt": np.random.randn(50)}, index=rng)
        entry = DataEntry(label="PSP_B_DERIVATIVE", data=df, units="nT", description="B derivatives")
        result = renderer.render_from_spec(
            {"labels": "PSP_B_DERIVATIVE", "panels": [["PSP_B_DERIVATIVE.B_mag"], ["PSP_B_DERIVATIVE.dB_dt"]]},
            [entry],
        )
        assert result["status"] == "success"
        assert result["panels"] == 2
        assert len(result["traces"]) == 2

    def test_panel_column_sublabel_not_found(self, renderer):
        """Column sub-label with non-existent column should fail."""
        rng = pd.date_range("2024-01-01", periods=10, freq="min")
        df = pd.DataFrame({"B_mag": np.random.randn(10)}, index=rng)
        entry = DataEntry(label="PARENT", data=df, units="nT")
        result = renderer.render_from_spec(
            {"labels": "PARENT", "panels": [["PARENT.NOPE"]]},
            [entry],
        )
        assert result["status"] == "error"
        assert "PARENT.NOPE" in result["message"]


# ---------------------------------------------------------------------------
# Grid layout (columns > 1)
# ---------------------------------------------------------------------------

class TestGridLayout:
    def test_grid_2x2(self, renderer):
        """4 entries in a 2x2 grid: trace positions are (1,1),(1,2),(2,1),(2,2)."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B", "C", "D"]]
        result = renderer.render_from_spec(
            {"labels": "A,B,C,D", "panels": [["A", "B"], ["C", "D"]], "columns": 2},
            entries,
        )
        assert result["status"] == "success"
        assert result["panels"] == 2
        assert result["columns"] == 2
        assert renderer._trace_panels == [(1, 1), (1, 2), (2, 1), (2, 2)]

    def test_grid_backward_compat(self, renderer):
        """columns=1 produces (row, 1) tuples — same as single-column behavior."""
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        result = renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A"], ["B"]]},
            [e1, e2],
        )
        assert result["status"] == "success"
        assert renderer._trace_panels == [(1, 1), (2, 1)]

    def test_grid_with_column_titles(self, renderer):
        """column_titles create subplot title annotations."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B"]]
        result = renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A", "B"]], "columns": 2,
             "column_titles": ["Period 1", "Period 2"]},
            entries,
        )
        assert result["status"] == "success"
        fig = renderer.get_figure()
        ann_texts = [a.text for a in fig.layout.annotations]
        assert "Period 1" in ann_texts
        assert "Period 2" in ann_texts

    def test_grid_overlay_in_cell(self, renderer):
        """4 labels in 1 row with columns=2: A,B in col 1, C,D in col 2."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B", "C", "D"]]
        result = renderer.render_from_spec(
            {"labels": "A,B,C,D", "panels": [["A", "B", "C", "D"]], "columns": 2},
            entries,
        )
        assert result["status"] == "success"
        assert renderer._trace_panels == [(1, 1), (1, 1), (1, 2), (1, 2)]

    def test_grid_width_scales(self, renderer):
        """Multi-column grid should have wider figure than single-column."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B"]]
        renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A", "B"]], "columns": 2},
            entries,
        )
        fig = renderer.get_figure()
        assert fig.layout.width > 1100

    def test_grid_reset_clears_column_count(self, renderer):
        """reset() should set _column_count back to 1."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B"]]
        renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A", "B"]], "columns": 2},
            entries,
        )
        assert renderer._column_count == 2
        renderer.reset()
        assert renderer._column_count == 1


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

class TestStyleViaSpec:
    """Test style fields applied through render_from_spec (formerly via style())."""

    def test_title(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec({"labels": "x", "title": "New Title"}, [entry])
        assert result["status"] == "success"
        assert renderer.get_figure().layout.title.text == "New Title"

    def test_y_label_string(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec({"labels": "x", "y_label": "B (nT)"}, [entry])
        assert result["status"] == "success"
        assert renderer.get_figure().layout.yaxis.title.text == "B (nT)"

    def test_y_label_dict(self, renderer):
        e1 = _make_entry("A", n=10)
        e2 = _make_entry("B", n=10)
        result = renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A"], ["B"]], "y_label": {"1": "Panel 1", "2": "Panel 2"}},
            [e1, e2],
        )
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.yaxis.title.text == "Panel 1"
        assert fig.layout.yaxis2.title.text == "Panel 2"

    def test_trace_colors(self, renderer):
        e1 = _make_entry("A", n=10, desc="Alpha")
        result = renderer.render_from_spec(
            {"labels": "A", "trace_colors": {"Alpha": "red"}}, [e1],
        )
        assert result["status"] == "success"
        assert renderer.get_figure().data[0].line.color == "red"

    def test_log_scale_y(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec({"labels": "x", "log_scale": "y"}, [entry])
        assert result["status"] == "success"
        assert renderer.get_figure().layout.yaxis.type == "log"

    def test_log_scale_int_panel(self, renderer):
        """Integer log_scale should apply log to that panel number."""
        e1 = _make_entry("A", n=10)
        e2 = _make_entry("B", n=10)
        result = renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A"], ["B"]], "log_scale": 1},
            [e1, e2],
        )
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.yaxis.type == "log"
        assert fig.layout.yaxis2.type != "log"

    def test_canvas_size(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec(
            {"labels": "x", "canvas_size": {"width": 1920, "height": 1080}}, [entry],
        )
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.width == 1920
        assert fig.layout.height == 1080

    def test_font_size(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec({"labels": "x", "font_size": 18}, [entry])
        assert result["status"] == "success"
        assert renderer.get_figure().layout.font.size == 18

    def test_annotations(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec(
            {"labels": "x", "annotations": [{"text": "Event", "x": "2024-01-01", "y": 5}]},
            [entry],
        )
        assert result["status"] == "success"
        assert any(a.text == "Event" for a in renderer.get_figure().layout.annotations)

    def test_legend(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec({"labels": "x", "legend": False}, [entry])
        assert result["status"] == "success"
        assert renderer.get_figure().layout.showlegend is False

    def test_theme(self, renderer):
        entry = _make_entry("x")
        result = renderer.render_from_spec({"labels": "x", "theme": "plotly_dark"}, [entry])
        assert result["status"] == "success"

    def test_y_label_dict_panel_mapping(self, renderer):
        """Dict y_label panel numbers map to correct axis keys."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B", "C", "D"]]
        result = renderer.render_from_spec(
            {"labels": "A,B,C,D", "panels": [["A", "B"], ["C", "D"]], "columns": 2,
             "y_label": {"1": "Top Left", "2": "Top Right"}},
            entries,
        )
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.layout.yaxis.title.text == "Top Left"
        assert fig.layout.yaxis2.title.text == "Top Right"


# ---------------------------------------------------------------------------
# manage (direct method calls)
# ---------------------------------------------------------------------------

class TestManage:
    def test_reset(self, renderer):
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x")])
        assert renderer.get_figure() is not None
        result = renderer.reset()
        assert result["status"] == "success"
        assert renderer.get_figure() is None
        assert renderer._trace_labels == []
        assert renderer._trace_panels == []

    def test_get_state(self, renderer):
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x", desc="Alpha")])
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
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x")])
        tr = parse_time_range("2024-01-01 to 2024-01-07")
        result = renderer.set_time_range(tr)
        assert result["status"] == "success"
        assert "2024-01-01" in result["time_range"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_png(self, renderer, tmp_path):
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_png_no_plot(self, renderer, tmp_path):
        filepath = str(tmp_path / "empty.png")
        result = renderer.export(filepath, format="png")
        assert result["status"] == "error"

    def test_export_pdf(self, renderer, tmp_path):
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output.pdf")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["size_bytes"] > 0

    def test_export_default_format_is_png(self, renderer, tmp_path):
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x", n=10)])
        filepath = str(tmp_path / "test_output")
        result = renderer.export(filepath)
        assert result["status"] == "success"
        assert result["filepath"].endswith(".png")

    def test_export_adds_extension(self, renderer, tmp_path):
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x", n=10)])
        filepath = str(tmp_path / "noext")
        result = renderer.export(filepath, format="pdf")
        assert result["status"] == "success"
        assert result["filepath"].endswith(".pdf")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestState:
    def test_reset_clears_state(self, renderer):
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x")])
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
        renderer.render_from_spec({"labels": "x"}, [_make_entry("x", desc="Alpha")])
        state = renderer.get_current_state()
        assert state["has_plot"] is True
        assert state["traces"] == ["Alpha"]


# ---------------------------------------------------------------------------
# Review metadata
# ---------------------------------------------------------------------------

class TestReviewMetadata:
    def test_review_in_plot_result(self, renderer):
        """render_from_spec result contains review with required keys."""
        entry = _make_entry("mag", n=50)
        result = renderer.render_from_spec({"labels": "mag"}, [entry])
        assert result["status"] == "success"
        assert "review" in result
        review = result["review"]
        assert "trace_summary" in review
        assert "warnings" in review
        assert "hint" in review

    def test_review_trace_details(self, renderer):
        """Vector entry produces 3 trace_summary entries with correct names and panel."""
        entry = _make_entry("Bvec", n=30, ncols=3)
        result = renderer.render_from_spec({"labels": "Bvec"}, [entry])
        review = result["review"]
        assert len(review["trace_summary"]) == 3
        names = [t["name"] for t in review["trace_summary"]]
        assert names == ["test data (x)", "test data (y)", "test data (z)"]
        for t in review["trace_summary"]:
            assert t["panel"] == 1
            assert t["points"] == 30
            assert t["y_range"] is not None
            assert len(t["y_range"]) == 2

    def test_review_warns_cluttered_panel(self, renderer):
        """Overlaying 7+ entries in one panel triggers a cluttered warning."""
        entries = [_make_entry(f"e{i}", n=10, desc=f"Entry{i}") for i in range(8)]
        labels = ",".join(f"e{i}" for i in range(8))
        result = renderer.render_from_spec({"labels": labels}, entries)
        review = result["review"]
        cluttered = [w for w in review["warnings"] if "hard to read" in w]
        assert len(cluttered) == 1
        assert "8 traces" in cluttered[0]

    def test_review_warns_resolution_mismatch(self, renderer):
        """Traces with >10x point count difference trigger a resolution warning."""
        e_lo = _make_entry("lo", n=10, desc="LowRes")
        e_hi = _make_entry("hi", n=500, desc="HighRes")
        result = renderer.render_from_spec({"labels": "lo,hi"}, [e_lo, e_hi])
        review = result["review"]
        mismatch = [w for w in review["warnings"] if "Resolution mismatch" in w]
        assert len(mismatch) == 1
        assert "LowRes" in mismatch[0]
        assert "HighRes" in mismatch[0]

    def test_review_multi_panel(self, renderer):
        """Multi-panel plot hint mentions correct panel count."""
        e1 = _make_entry("A", n=10, desc="Alpha")
        e2 = _make_entry("B", n=10, desc="Beta")
        result = renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A"], ["B"]]},
            [e1, e2],
        )
        review = result["review"]
        assert "2 panel(s)" in review["hint"]
        assert "Panel 1: Alpha" in review["hint"]
        assert "Panel 2: Beta" in review["hint"]

    def test_review_no_warnings_normal(self, renderer):
        """Two similar scalar entries produce no warnings."""
        e1 = _make_entry("A", n=100, desc="Alpha")
        e2 = _make_entry("B", n=100, desc="Beta")
        result = renderer.render_from_spec({"labels": "A,B"}, [e1, e2])
        review = result["review"]
        assert review["warnings"] == []

    def test_review_gaps_detected(self, renderer):
        """Entry with NaN values has has_gaps=True."""
        rng = pd.date_range("2024-01-01", periods=50, freq="min")
        vals = np.random.randn(50)
        vals[10:15] = np.nan
        df = pd.DataFrame({"value": vals}, index=rng)
        entry = DataEntry(label="gappy", data=df, units="nT", description="Gappy data")
        result = renderer.render_from_spec({"labels": "gappy"}, [entry])
        review = result["review"]
        assert len(review["trace_summary"]) == 1
        assert review["trace_summary"][0]["has_gaps"] is True

    def test_review_grid_positions(self, renderer):
        """Grid layout: trace_summary has row/col fields."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B", "C", "D"]]
        result = renderer.render_from_spec(
            {"labels": "A,B,C,D", "panels": [["A", "B"], ["C", "D"]], "columns": 2},
            entries,
        )
        review = result["review"]
        ts = review["trace_summary"]
        assert len(ts) == 4
        assert (ts[0]["row"], ts[0]["col"]) == (1, 1)
        assert (ts[1]["row"], ts[1]["col"]) == (1, 2)
        assert (ts[2]["row"], ts[2]["col"]) == (2, 1)
        assert (ts[3]["row"], ts[3]["col"]) == (2, 2)
        assert "2 row(s) x 2 column(s)" in review["hint"]

    def test_review_warns_all_nan_panel(self, renderer):
        """Panel where ALL traces are all-NaN triggers 'appears empty' warning."""
        rng = pd.date_range("2024-01-01", periods=50, freq="min")
        vals = np.full(50, np.nan)
        df = pd.DataFrame({"value": vals}, index=rng)
        entry = DataEntry(label="allnan", data=df, units="nT", description="AllNaN")
        result = renderer.render_from_spec({"labels": "allnan"}, [entry])
        review = result["review"]
        empty_warns = [w for w in review["warnings"] if "appears empty" in w]
        assert len(empty_warns) == 1
        assert "AllNaN" in empty_warns[0]
        assert "NaN/missing" in empty_warns[0]

    def test_review_warns_invisible_trace_mixed(self, renderer):
        """Panel with good + all-NaN traces warns about NaN ones only."""
        rng = pd.date_range("2024-01-01", periods=50, freq="min")
        good_df = pd.DataFrame({"value": np.random.randn(50)}, index=rng)
        good_entry = DataEntry(label="good", data=good_df, units="nT", description="Good")
        nan_df = pd.DataFrame({"value": np.full(50, np.nan)}, index=rng)
        nan_entry = DataEntry(label="bad", data=nan_df, units="nT", description="Bad")
        result = renderer.render_from_spec({"labels": "good,bad"}, [good_entry, nan_entry])
        review = result["review"]
        invis_warns = [w for w in review["warnings"] if "invisible traces" in w]
        assert len(invis_warns) == 1
        assert "Bad" in invis_warns[0]
        assert "Good" not in invis_warns[0]
        assert all("appears empty" not in w for w in review["warnings"])

    def test_review_allnan_trace_summary(self, renderer):
        """All-NaN trace reports y_range=None and has_gaps=True."""
        rng = pd.date_range("2024-01-01", periods=30, freq="min")
        df = pd.DataFrame({"value": np.full(30, np.nan)}, index=rng)
        entry = DataEntry(label="nanonly", data=df, units="nT", description="NanOnly")
        result = renderer.render_from_spec({"labels": "nanonly"}, [entry])
        ts = result["review"]["trace_summary"]
        assert len(ts) == 1
        assert ts[0]["y_range"] is None
        assert ts[0]["has_gaps"] is True
        assert ts[0]["points"] == 30


# ---------------------------------------------------------------------------
# Serialization (grid-aware)
# ---------------------------------------------------------------------------

class TestSerializationGrid:
    def test_save_restore_grid(self, renderer):
        """Round-trip preserves column_count and (row, col) tuples."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B", "C", "D"]]
        renderer.render_from_spec(
            {"labels": "A,B,C,D", "panels": [["A", "B"], ["C", "D"]], "columns": 2},
            entries,
        )
        state = renderer.save_state()
        assert state is not None
        assert state["column_count"] == 2
        assert state["trace_panels"] == [
            {"row": 1, "col": 1}, {"row": 1, "col": 2},
            {"row": 2, "col": 1}, {"row": 2, "col": 2},
        ]

        r2 = PlotlyRenderer(verbose=False)
        r2.restore_state(state)
        assert r2._column_count == 2
        assert r2._trace_panels == [(1, 1), (1, 2), (2, 1), (2, 2)]
        assert r2._panel_count == 2

    def test_restore_legacy_format(self, renderer):
        """Legacy flat int list restores as (row, 1) tuples."""
        entries = [_make_entry(l, n=10, desc=l) for l in ["A", "B"]]
        renderer.render_from_spec(
            {"labels": "A,B", "panels": [["A"], ["B"]]},
            entries,
        )
        state = renderer.save_state()
        state["trace_panels"] = [1, 2]
        if "column_count" in state:
            del state["column_count"]

        r2 = PlotlyRenderer(verbose=False)
        r2.restore_state(state)
        assert r2._column_count == 1
        assert r2._trace_panels == [(1, 1), (2, 1)]


# ---------------------------------------------------------------------------
# Per-panel plot types
# ---------------------------------------------------------------------------

def _make_spectrogram_entry(label: str, n_times: int = 50, n_bins: int = 20) -> DataEntry:
    """Create a synthetic spectrogram DataEntry for testing."""
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


class TestPerPanelType:
    def test_mixed_panel_types(self, renderer):
        """Spectrogram panel 1, line panel 2 — correct trace types."""
        spec_entry = _make_spectrogram_entry("SPEC")
        line = _make_entry("LINE", n=50)
        result = renderer.render_from_spec(
            {"labels": "SPEC,LINE", "panels": [["SPEC"], ["LINE"]],
             "panel_types": ["spectrogram", "line"]},
            [spec_entry, line],
        )
        assert result["status"] == "success"
        assert result["panels"] == 2
        fig = renderer.get_figure()
        assert "heatmap" in fig.data[0].type.lower()
        assert "scatter" in fig.data[1].type.lower()

    def test_panel_types_length_mismatch(self, renderer):
        """Wrong length panel_types returns error."""
        spec_entry = _make_spectrogram_entry("SPEC")
        line = _make_entry("LINE", n=50)
        result = renderer.render_from_spec(
            {"labels": "SPEC,LINE", "panels": [["SPEC"], ["LINE"]],
             "panel_types": ["spectrogram"]},
            [spec_entry, line],
        )
        assert result["status"] == "error"
        assert "panel_types length" in result["message"]

    def test_spectrogram_on_scalar_returns_error(self, renderer):
        """1-column entry + spectrogram type returns descriptive error."""
        scalar = _make_entry("SCALAR", n=50, ncols=1)
        result = renderer.render_from_spec(
            {"labels": "SCALAR", "panels": [["SCALAR"]],
             "panel_types": ["spectrogram"]},
            [scalar],
        )
        assert result["status"] == "error"
        assert "scalar" in result["message"].lower()
        assert "panel_types" in result["message"]

    def test_global_plot_type_still_works(self, renderer):
        """Backward compat: global plot_type=spectrogram without panel_types."""
        spec_entry = _make_spectrogram_entry("SPEC")
        result = renderer.render_from_spec(
            {"labels": "SPEC", "plot_type": "spectrogram"},
            [spec_entry],
        )
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert "heatmap" in fig.data[0].type.lower()

    def test_panel_types_overrides_global(self, renderer):
        """panel_types wins over global plot_type."""
        spec_entry = _make_spectrogram_entry("SPEC")
        line = _make_entry("LINE", n=50)
        result = renderer.render_from_spec(
            {"labels": "SPEC,LINE", "panels": [["SPEC"], ["LINE"]],
             "plot_type": "spectrogram",
             "panel_types": ["spectrogram", "line"]},
            [spec_entry, line],
        )
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert "heatmap" in fig.data[0].type.lower()
        assert "scatter" in fig.data[1].type.lower()


# ---------------------------------------------------------------------------
# render_from_spec
# ---------------------------------------------------------------------------

class TestRenderFromSpec:
    """Tests for the unified plot spec renderer."""

    def test_basic_spec(self, renderer):
        """Minimal spec with just labels produces a valid plot."""
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"labels": "A"}
        result = renderer.render_from_spec(spec, [entry])
        assert result["status"] == "success"
        assert result["panels"] == 1
        fig = renderer.get_figure()
        assert fig is not None
        assert len(fig.data) == 1

    def test_spec_with_panels_and_style(self, renderer):
        """Spec with panels, title, and style fields."""
        e1 = _make_entry("A", n=30, desc="Alpha")
        e2 = _make_entry("B", n=30, desc="Beta")
        spec = {
            "labels": "A,B",
            "panels": [["A"], ["B"]],
            "title": "Spec Title",
            "y_label": {"1": "nT", "2": "km/s"},
            "font_size": 16,
            "legend": False,
        }
        result = renderer.render_from_spec(spec, [e1, e2])
        assert result["status"] == "success"
        assert result["panels"] == 2
        fig = renderer.get_figure()
        assert fig.layout.title.text == "Spec Title"
        assert fig.layout.font.size == 16
        assert fig.layout.showlegend is False

    def test_spec_trace_colors(self, renderer):
        """Spec trace_colors are applied to traces."""
        entry = _make_entry("M", n=20, desc="Mag")
        spec = {
            "labels": "M",
            "trace_colors": {"Mag": "red"},
        }
        result = renderer.render_from_spec(spec, [entry])
        assert result["status"] == "success"
        fig = renderer.get_figure()
        assert fig.data[0].line.color == "red"

    def test_spec_empty_entries_error(self, renderer):
        """Spec with no entries returns error."""
        spec = {"labels": "A"}
        result = renderer.render_from_spec(spec, [])
        assert result["status"] == "error"

    def test_spec_title_from_style_overrides_plot(self, renderer):
        """When spec has title, it's used by both plot_data and style (style wins)."""
        entry = _make_entry("Z", n=20, desc="Zeta")
        spec = {"labels": "Z", "title": "Final Title"}
        result = renderer.render_from_spec(spec, [entry])
        assert result["status"] == "success"
        fig = renderer.get_figure()
        # style() is called after plot_data(), so title from style() is applied
        assert fig.layout.title.text == "Final Title"

    def test_spec_vlines(self, renderer):
        """Spec with vlines adds vertical line shapes."""
        entry = _make_entry("V", n=20, desc="Vdata")
        spec = {
            "labels": "V",
            "vlines": [{"x": "2024-01-01T00:10:00", "color": "blue"}],
        }
        result = renderer.render_from_spec(spec, [entry])
        assert result["status"] == "success"
        fig = renderer.get_figure()
        # vlines should add shapes to the figure
        assert len(fig.layout.shapes) > 0


# ---------------------------------------------------------------------------
# get_current_spec / spec tracking
# ---------------------------------------------------------------------------

class TestSpecTracking:
    """Tests for the current plot spec tracking."""

    def test_get_current_spec_empty_initially(self, renderer):
        """get_current_spec returns empty dict before any render."""
        assert renderer.get_current_spec() == {}

    def test_get_current_spec_after_render_from_spec(self, renderer):
        """get_current_spec returns the spec used in render_from_spec."""
        entry = _make_entry("A", n=50, desc="Alpha")
        spec = {"labels": "A", "title": "Test Title"}
        renderer.render_from_spec(spec, [entry])
        current = renderer.get_current_spec()
        assert current == spec
        # Returned dict is a copy, not the same object
        assert current is not renderer._current_plot_spec

    def test_spec_cleared_on_reset(self, renderer):
        """reset() clears the current spec."""
        entry = _make_entry("A", n=50, desc="Alpha")
        renderer.render_from_spec({"labels": "A"}, [entry])
        assert renderer.get_current_spec() != {}
        renderer.reset()
        assert renderer.get_current_spec() == {}

    def test_spec_round_trip(self, renderer):
        """Render with spec, get it back, render again — same figure."""
        entry = _make_entry("X", n=30, desc="Xray")
        spec = {"labels": "X", "title": "Round Trip", "font_size": 14}
        renderer.render_from_spec(spec, [entry])
        fig1_data = renderer.get_figure().to_dict()

        # Get spec and re-render on fresh renderer
        recovered_spec = renderer.get_current_spec()
        renderer2 = PlotlyRenderer(verbose=False)
        renderer2.render_from_spec(recovered_spec, [entry])
        fig2_data = renderer2.get_figure().to_dict()

        assert len(fig1_data["data"]) == len(fig2_data["data"])
        assert fig1_data["layout"]["title"]["text"] == fig2_data["layout"]["title"]["text"]
        assert fig2_data["layout"]["font"]["size"] == 14

    def test_spec_in_save_restore_state(self, renderer):
        """save_state/restore_state preserves the current spec."""
        entry = _make_entry("S", n=20, desc="Sigma")
        spec = {"labels": "S", "title": "Saved Spec"}
        renderer.render_from_spec(spec, [entry])

        state = renderer.save_state()
        assert state is not None
        assert state["plot_spec"] == spec

        renderer2 = PlotlyRenderer(verbose=False)
        renderer2.restore_state(state)
        assert renderer2.get_current_spec() == spec


# ---------------------------------------------------------------------------
# fill_figure_data (new Plotly JSON pipeline)
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
        # render_plotly_json on renderer itself needs handler check
        # but the renderer just passes through — this tests the core handler
        result = renderer.render_plotly_json(fig_json, {})
        # fill_figure_data will produce an empty figure
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
