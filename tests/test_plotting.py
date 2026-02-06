"""
Tests for data_ops.plotting — matplotlib timeseries rendering.

Run with: python -m pytest tests/test_plotting.py
"""

import os
import numpy as np
import pytest

from data_ops.store import DataEntry
from data_ops.plotting import plot_timeseries


def _make_entry(label="test", n=100, vector=False, units="nT"):
    """Helper to create a DataEntry for testing."""
    t0 = np.datetime64("2024-01-01", "ns")
    time = t0 + np.arange(n) * np.timedelta64(1, "s")
    if vector:
        values = np.random.randn(n, 3)
    else:
        values = np.random.randn(n)
    return DataEntry(
        label=label, time=time, values=values,
        units=units, description="test", source="computed",
    )


class TestPlotTimeseries:
    def test_single_scalar(self, tmp_path):
        entry = _make_entry("Bmag", n=50)
        filepath = plot_timeseries([entry], filename=str(tmp_path / "test.png"))
        assert os.path.exists(filepath)
        assert filepath.endswith(".png")
        assert os.path.getsize(filepath) > 0

    def test_single_vector(self, tmp_path):
        entry = _make_entry("B", n=50, vector=True)
        filepath = plot_timeseries([entry], filename=str(tmp_path / "vec.png"))
        assert os.path.exists(filepath)

    def test_multiple_overlay(self, tmp_path):
        e1 = _make_entry("A", n=50)
        e2 = _make_entry("B", n=50)
        filepath = plot_timeseries([e1, e2], filename=str(tmp_path / "multi.png"))
        assert os.path.exists(filepath)

    def test_custom_title(self, tmp_path):
        entry = _make_entry("X", n=20)
        filepath = plot_timeseries(
            [entry], title="Custom Title", filename=str(tmp_path / "titled.png")
        )
        assert os.path.exists(filepath)

    def test_auto_filename(self, tmp_path):
        entry = _make_entry("Y", n=20)
        # Change to tmp_path so auto-generated file lands there
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            filepath = plot_timeseries([entry])
            assert os.path.exists(filepath)
            assert filepath.endswith(".png")
        finally:
            os.chdir(old_cwd)

    def test_empty_entries_raises(self):
        with pytest.raises(ValueError, match="No entries"):
            plot_timeseries([])

    def test_png_extension_added(self, tmp_path):
        entry = _make_entry("Z", n=20)
        filepath = plot_timeseries([entry], filename=str(tmp_path / "noext"))
        assert filepath.endswith(".png")
