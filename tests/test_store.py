"""
Tests for data_ops.store — DataEntry and DataStore.

Run with: python -m pytest tests/test_store.py
"""

import numpy as np
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
    t0 = np.datetime64("2024-01-01", "ns")
    time = t0 + np.arange(n) * np.timedelta64(1, "s")
    if vector:
        values = np.random.randn(n, 3)
    else:
        values = np.random.randn(n)
    return DataEntry(
        label=label,
        time=time,
        values=values,
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
        entry = DataEntry(
            label="empty",
            time=np.array([], dtype="datetime64[ns]"),
            values=np.array([], dtype=np.float64),
        )
        s = entry.summary()
        assert s["num_points"] == 0
        assert s["time_min"] is None
        assert s["time_max"] is None


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
        assert len(store.get("A").time) == 20

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
