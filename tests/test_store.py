"""
Tests for data_ops.store — DataEntry and DataStore.

Run with: python -m pytest tests/test_store.py
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry, DataStore, get_store, reset_store


@pytest.fixture(autouse=True)
def clean_store():
    """Reset the global store before each test."""
    reset_store()
    yield
    reset_store()


def _make_entry(label="test", n=10, vector=False):
    """Helper to create a DataEntry for testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1s")
    if vector:
        data = pd.DataFrame(np.random.randn(n, 3), index=idx, columns=["x", "y", "z"])
    else:
        data = pd.DataFrame(np.random.randn(n), index=idx, columns=["value"])
    return DataEntry(
        label=label,
        data=data,
        units="nT",
        description="test entry",
        source="computed",
    )


class TestDataEntry:
    def test_summary_scalar(self):
        entry = _make_entry("Bmag", n=100, vector=False)
        s = entry.summary()
        assert s["label"] == "Bmag"
        assert s["num_points"] == 100
        assert s["shape"] == "scalar"
        assert s["units"] == "nT"
        assert s["source"] == "computed"
        assert s["time_min"] is not None
        assert s["time_max"] is not None

    def test_summary_vector(self):
        entry = _make_entry("B", n=50, vector=True)
        s = entry.summary()
        assert s["shape"] == "vector[3]"
        assert s["num_points"] == 50

    def test_summary_empty(self):
        empty_df = pd.DataFrame(dtype=np.float64)
        empty_df.index = pd.DatetimeIndex([], name="time")
        entry = DataEntry(label="empty", data=empty_df)
        s = entry.summary()
        assert s["num_points"] == 0
        assert s["time_min"] is None
        assert s["time_max"] is None

    def test_backward_compat_time(self):
        entry = _make_entry("A", n=5)
        t = entry.time
        assert isinstance(t, np.ndarray)
        assert np.issubdtype(t.dtype, np.datetime64)
        assert len(t) == 5

    def test_backward_compat_values_scalar(self):
        entry = _make_entry("A", n=5, vector=False)
        v = entry.values
        assert isinstance(v, np.ndarray)
        assert v.ndim == 1
        assert len(v) == 5

    def test_backward_compat_values_vector(self):
        entry = _make_entry("A", n=5, vector=True)
        v = entry.values
        assert isinstance(v, np.ndarray)
        assert v.ndim == 2
        assert v.shape == (5, 3)


class TestDataStore:
    def test_put_and_get(self):
        store = DataStore()
        entry = _make_entry("A")
        store.put(entry)
        assert store.get("A") is entry
        assert store.get("B") is None

    def test_has(self):
        store = DataStore()
        store.put(_make_entry("A"))
        assert store.has("A")
        assert not store.has("B")

    def test_overwrite(self):
        store = DataStore()
        store.put(_make_entry("A", n=10))
        store.put(_make_entry("A", n=20))
        assert len(store.get("A").data) == 20

    def test_remove(self):
        store = DataStore()
        store.put(_make_entry("A"))
        assert store.remove("A") is True
        assert store.get("A") is None
        assert store.remove("A") is False

    def test_list_entries(self):
        store = DataStore()
        store.put(_make_entry("A"))
        store.put(_make_entry("B"))
        entries = store.list_entries()
        assert len(entries) == 2
        labels = {e["label"] for e in entries}
        assert labels == {"A", "B"}

    def test_clear(self):
        store = DataStore()
        store.put(_make_entry("A"))
        store.put(_make_entry("B"))
        store.clear()
        assert len(store) == 0

    def test_len(self):
        store = DataStore()
        assert len(store) == 0
        store.put(_make_entry("A"))
        assert len(store) == 1


class TestGetStore:
    def test_singleton(self):
        s1 = get_store()
        s2 = get_store()
        assert s1 is s2

    def test_reset(self):
        s1 = get_store()
        s1.put(_make_entry("X"))
        reset_store()
        s2 = get_store()
        assert s2 is not s1
        assert len(s2) == 0
