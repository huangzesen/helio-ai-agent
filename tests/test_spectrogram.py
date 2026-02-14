"""
Tests for spectrogram support — computation, storage, and rendering.

Run with: python -m pytest tests/test_spectrogram.py -v
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry, DataStore, get_store, reset_store
from data_ops.custom_ops import (
    execute_spectrogram_computation,
    run_spectrogram_computation,
    validate_pandas_code,
)
from rendering.plotly_renderer import PlotlyRenderer
from rendering.registry import get_method, validate_args


@pytest.fixture(autouse=True)
def clean_store():
    """Reset the global store before each test."""
    reset_store()
    yield
    reset_store()


def _make_timeseries(n=1024, freq_s=1.0):
    """Create a simple sinusoidal timeseries for spectrogram testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{freq_s}s")
    t = np.arange(n) * freq_s
    # 0.1 Hz sine wave + noise
    values = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.random.randn(n)
    return pd.DataFrame({"signal": values}, index=idx)


def _make_spectrogram_entry(n_times=50, n_bins=20, label="test_spec"):
    """Create a spectrogram DataEntry for testing rendering."""
    idx = pd.date_range("2024-01-01", periods=n_times, freq="10min")
    bins = np.linspace(0.001, 0.5, n_bins)
    data = np.random.rand(n_times, n_bins)
    df = pd.DataFrame(data, index=idx, columns=[str(b) for b in bins])
    return DataEntry(
        label=label,
        data=df,
        units="nT",
        description="Test spectrogram",
        source="computed",
        metadata={
            "type": "spectrogram",
            "bin_label": "Frequency (Hz)",
            "value_label": "PSD (nT²/Hz)",
            "bin_values": bins.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Spectrogram computation tests
# ---------------------------------------------------------------------------


class TestExecuteSpectrogramComputation:
    def test_basic_fft(self):
        """Simple FFT-based spectrogram code should produce valid output."""
        df = _make_timeseries(n=512, freq_s=1.0)
        code = """
vals = df.iloc[:, 0].dropna().values
dt = df.index.to_series().diff().dt.total_seconds().median()
fs = 1.0 / dt
f, t_seg, Sxx = signal.spectrogram(vals, fs=fs, nperseg=64, noverlap=32)
times = pd.to_datetime(df.index[0]) + pd.to_timedelta(t_seg, unit='s')
result = pd.DataFrame(Sxx.T, index=times, columns=[str(freq) for freq in f])
"""
        result = execute_spectrogram_computation(df, code)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) > 0
        assert len(result.columns) > 0
        # Columns should be parseable as floats (frequency values)
        for col in result.columns:
            float(col)  # Should not raise

    def test_scipy_namespace_available(self):
        """The signal module from scipy should be available in the namespace."""
        df = _make_timeseries(n=256)
        code = """
vals = df.iloc[:, 0].values
f, Pxx = signal.welch(vals, fs=1.0, nperseg=64)
result = pd.DataFrame({'PSD': Pxx}, index=pd.date_range(df.index[0], periods=len(f), freq='1s'))
"""
        result = execute_spectrogram_computation(df, code)
        assert isinstance(result, pd.DataFrame)
        assert "PSD" in result.columns

    def test_validation_error_no_result(self):
        """Code without result assignment should be caught by validation."""
        df = _make_timeseries(n=100)
        code = "x = df.mean()"
        with pytest.raises(ValueError, match="Code must assign to 'result'"):
            run_spectrogram_computation(df, code)

    def test_validation_error_import(self):
        """Imports should be blocked by validation."""
        df = _make_timeseries(n=100)
        code = "import os; result = df"
        with pytest.raises(ValueError, match="Imports are not allowed"):
            run_spectrogram_computation(df, code)

    def test_runtime_error_bad_code(self):
        """Code that raises an exception should return RuntimeError."""
        df = _make_timeseries(n=100)
        code = "result = 1 / 0"
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            execute_spectrogram_computation(df, code)

    def test_result_must_be_dataframe(self):
        """Non-DataFrame result should raise ValueError."""
        df = _make_timeseries(n=100)
        code = "result = 42"
        with pytest.raises(ValueError, match="must be a DataFrame"):
            execute_spectrogram_computation(df, code)

    def test_result_must_have_datetime_index(self):
        """Result without DatetimeIndex should raise ValueError."""
        df = _make_timeseries(n=100)
        code = "result = pd.DataFrame({'a': [1, 2, 3]})"
        with pytest.raises(ValueError, match="DatetimeIndex"):
            execute_spectrogram_computation(df, code)


# ---------------------------------------------------------------------------
# DataEntry metadata tests
# ---------------------------------------------------------------------------


class TestDataEntryMetadata:
    def test_summary_spectrogram_shape(self):
        """Spectrogram entries should show 'spectrogram[N bins]' in summary."""
        entry = _make_spectrogram_entry(n_times=50, n_bins=20)
        s = entry.summary()
        assert s["shape"] == "spectrogram[20 bins]"
        assert s["metadata"]["type"] == "spectrogram"

    def test_summary_no_metadata(self):
        """Entries without metadata should still produce normal summaries."""
        idx = pd.date_range("2024-01-01", periods=10, freq="1s")
        df = pd.DataFrame({"value": np.random.randn(10)}, index=idx)
        entry = DataEntry(label="test", data=df, units="nT")
        s = entry.summary()
        assert s["shape"] == "scalar"
        assert "metadata" not in s

    def test_metadata_persistence(self, tmp_path):
        """Metadata should survive save/load cycle."""
        store = DataStore()
        entry = _make_spectrogram_entry(n_times=10, n_bins=5, label="spec1")
        store.put(entry)

        # Save
        store.save_to_directory(tmp_path)

        # Load into fresh store
        store2 = DataStore()
        count = store2.load_from_directory(tmp_path)
        assert count == 1

        loaded = store2.get("spec1")
        assert loaded is not None
        assert loaded.metadata is not None
        assert loaded.metadata["type"] == "spectrogram"
        assert loaded.metadata["bin_label"] == "Frequency (Hz)"
        assert loaded.metadata["value_label"] == "PSD (nT²/Hz)"
        assert len(loaded.metadata["bin_values"]) == 5

    def test_metadata_persistence_none(self, tmp_path):
        """Entries with no metadata should save/load cleanly."""
        store = DataStore()
        idx = pd.date_range("2024-01-01", periods=10, freq="1s")
        df = pd.DataFrame({"value": np.random.randn(10)}, index=idx)
        entry = DataEntry(label="no_meta", data=df)
        store.put(entry)

        store.save_to_directory(tmp_path)

        store2 = DataStore()
        count = store2.load_from_directory(tmp_path)
        assert count == 1

        loaded = store2.get("no_meta")
        assert loaded is not None
        assert loaded.metadata is None


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


class TestPlotSpectrogram:
    def test_basic_heatmap(self):
        """render_from_spec with plot_type='spectrogram' should create a Heatmap trace."""
        renderer = PlotlyRenderer()
        entry = _make_spectrogram_entry()
        result = renderer.render_from_spec(
            {"labels": entry.label, "plot_type": "spectrogram"}, [entry])
        assert result["status"] == "success"
        assert result["display"] == "plotly"

        fig = renderer.get_figure()
        assert fig is not None
        assert len(fig.data) == 1
        assert "heatmap" in fig.data[0].type.lower()

    def test_log_y(self):
        """log_y should be applied correctly."""
        renderer = PlotlyRenderer()
        entry = _make_spectrogram_entry()
        result = renderer.render_from_spec(
            {"labels": entry.label, "plot_type": "spectrogram",
             "log_y": True}, [entry])
        assert result["status"] == "success"

        fig = renderer.get_figure()
        yaxis = fig.layout.yaxis
        assert yaxis.type == "log"

    def test_colorscale(self):
        """Custom colorscale should be set on the trace."""
        renderer = PlotlyRenderer()
        entry = _make_spectrogram_entry()
        renderer.render_from_spec(
            {"labels": entry.label, "plot_type": "spectrogram",
             "colorscale": "Jet"}, [entry])

        fig = renderer.get_figure()
        assert fig.data[0].colorscale is not None

    def test_title(self):
        """Title parameter should set the figure title."""
        renderer = PlotlyRenderer()
        entry = _make_spectrogram_entry()
        renderer.render_from_spec(
            {"labels": entry.label, "plot_type": "spectrogram",
             "title": "Test Title"}, [entry])

        fig = renderer.get_figure()
        assert fig.layout.title.text == "Test Title"

    def test_panel_targeting(self):
        """panels parameter should target a specific panel row."""
        renderer = PlotlyRenderer()
        entry = _make_spectrogram_entry()
        result = renderer.render_from_spec(
            {"labels": entry.label, "panels": [[], [entry.label]],
             "plot_type": "spectrogram"}, [entry])
        assert result["status"] == "success"

        fig = renderer.get_figure()
        assert len(fig.data) >= 1

    def test_z_range(self):
        """z_min and z_max should be set on the trace."""
        renderer = PlotlyRenderer()
        entry = _make_spectrogram_entry()
        renderer.render_from_spec(
            {"labels": entry.label, "plot_type": "spectrogram",
             "z_min": 0.1, "z_max": 10.0}, [entry])

        fig = renderer.get_figure()
        assert fig.data[0].zmin == 0.1
        assert fig.data[0].zmax == 10.0

    def test_empty_entry(self):
        """Empty entry should return an error."""
        renderer = PlotlyRenderer()
        idx = pd.DatetimeIndex([], name="time")
        df = pd.DataFrame(dtype=float, index=idx)
        entry = DataEntry(
            label="empty_spec",
            data=df,
            metadata={"type": "spectrogram", "bin_values": []},
        )
        result = renderer.render_from_spec(
            {"labels": "empty_spec", "plot_type": "spectrogram"}, [entry])
        assert result["status"] == "error"

    def test_scalar_rejected_as_spectrogram(self):
        """Scalar (1-column) entry with plot_type='spectrogram' returns error."""
        renderer = PlotlyRenderer()
        idx = pd.date_range("2024-01-01", periods=100, freq="min")
        df = pd.DataFrame({"value": np.random.randn(100)}, index=idx)
        entry = DataEntry(label="scalar_data", data=df, units="nT",
                          description="Scalar timeseries")
        result = renderer.render_from_spec(
            {"labels": "scalar_data", "plot_type": "spectrogram"}, [entry])
        assert result["status"] == "error"
        assert "scalar" in result["message"].lower()
        assert "panel_types" in result["message"]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistryPlotSpectrogram:
    def test_update_plot_spec_exists(self):
        """Spectrograms are handled via update_plot_spec with plot_type in spec."""
        method = get_method("update_plot_spec")
        assert method is not None
        param_names = [p["name"] for p in method["parameters"]]
        assert "spec" in param_names

    def test_validate_required_args(self):
        """Missing required 'spec' should produce an error."""
        errors = validate_args("update_plot_spec", {})
        assert any("spec" in e for e in errors)

    def test_validate_valid_args(self):
        """Valid args should pass validation."""
        errors = validate_args("update_plot_spec", {"spec": {"labels": "test_spec"}})
        assert errors == []


# ---------------------------------------------------------------------------
# Tool handler integration test
# ---------------------------------------------------------------------------


class TestComputeSpectrogramToolHandler:
    def test_end_to_end(self):
        """Simulate the full compute_spectrogram tool execution flow."""
        store = get_store()

        # Put a source timeseries in the store
        df = _make_timeseries(n=512, freq_s=1.0)
        source = DataEntry(
            label="source_ts",
            data=df,
            units="nT",
            description="Test timeseries",
            source="cdf",
        )
        store.put(source)

        # Simulate what core.py handler does
        code = """
vals = df.iloc[:, 0].dropna().values
dt = df.index.to_series().diff().dt.total_seconds().median()
fs = 1.0 / dt
f, t_seg, Sxx = signal.spectrogram(vals, fs=fs, nperseg=64, noverlap=32)
times = pd.to_datetime(df.index[0]) + pd.to_timedelta(t_seg, unit='s')
result = pd.DataFrame(Sxx.T, index=times, columns=[str(freq) for freq in f])
"""
        result_df = run_spectrogram_computation(source.data, code)

        metadata = {
            "type": "spectrogram",
            "bin_label": "Frequency (Hz)",
            "value_label": "PSD (nT²/Hz)",
        }
        try:
            bin_values = [float(c) for c in result_df.columns]
            metadata["bin_values"] = bin_values
        except (ValueError, TypeError):
            metadata["bin_values"] = list(range(len(result_df.columns)))

        entry = DataEntry(
            label="source_spec",
            data=result_df,
            units=source.units,
            description="Power spectrogram",
            source="computed",
            metadata=metadata,
        )
        store.put(entry)

        # Verify the stored entry
        loaded = store.get("source_spec")
        assert loaded is not None
        assert loaded.metadata["type"] == "spectrogram"
        assert len(loaded.metadata["bin_values"]) > 0

        s = loaded.summary()
        assert "spectrogram" in s["shape"]

        # Verify we can plot it
        renderer = PlotlyRenderer()
        result = renderer.render_from_spec(
            {"labels": loaded.label, "plot_type": "spectrogram"}, [loaded])
        assert result["status"] == "success"
