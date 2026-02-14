"""
Tests for data_ops.store — DataEntry and DataStore.

Run with: python -m pytest tests/test_store.py
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.store import DataEntry, DataStore, get_store, reset_store, build_source_map, describe_sources


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


    def test_summary_non_timeseries(self):
        """Non-timeseries entry uses index_min/index_max instead of time_min/time_max."""
        df = pd.DataFrame({"flux": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        entry = DataEntry(
            label="events",
            data=df,
            units="",
            description="Event catalog",
            source="created",
            is_timeseries=False,
        )
        s = entry.summary()
        assert s["label"] == "events"
        assert s["num_points"] == 3
        assert "time_min" not in s
        assert "time_max" not in s
        assert s["index_min"] == "10"
        assert s["index_max"] == "30"

    def test_summary_non_timeseries_empty(self):
        """Empty non-timeseries entry should have None index_min/index_max."""
        df = pd.DataFrame({"val": pd.Series(dtype=float)})
        entry = DataEntry(label="empty_nt", data=df, is_timeseries=False)
        s = entry.summary()
        assert s["index_min"] is None
        assert s["index_max"] is None
        assert "time_min" not in s

    def test_is_timeseries_default_true(self):
        """is_timeseries defaults to True."""
        entry = _make_entry("A")
        assert entry.is_timeseries is True

    def test_summary_includes_is_timeseries_true(self):
        """Timeseries summary dict includes is_timeseries: True."""
        entry = _make_entry("Bmag", n=10)
        s = entry.summary()
        assert s["is_timeseries"] is True

    def test_summary_includes_is_timeseries_false(self):
        """Non-timeseries summary dict includes is_timeseries: False."""
        df = pd.DataFrame({"flux": [1.0, 2.0]}, index=[10, 20])
        entry = DataEntry(label="events", data=df, is_timeseries=False)
        s = entry.summary()
        assert s["is_timeseries"] is False


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


class TestBuildSourceMap:
    def test_single_label(self):
        store = DataStore()
        store.put(_make_entry("DATASET.BR", n=10))
        sources, err = build_source_map(store, ["DATASET.BR"])
        assert err is None
        assert "df_BR" in sources
        assert len(sources["df_BR"]) == 10

    def test_three_scalars(self):
        store = DataStore()
        store.put(_make_entry("DATASET.BR", n=10))
        store.put(_make_entry("DATASET.BT", n=10))
        store.put(_make_entry("DATASET.BN", n=10))
        sources, err = build_source_map(store, ["DATASET.BR", "DATASET.BT", "DATASET.BN"])
        assert err is None
        assert set(sources.keys()) == {"df_BR", "df_BT", "df_BN"}

    def test_missing_label(self):
        store = DataStore()
        store.put(_make_entry("DATASET.BR", n=10))
        sources, err = build_source_map(store, ["DATASET.BR", "DATASET.MISSING"])
        assert sources is None
        assert "DATASET.MISSING" in err

    def test_duplicate_suffix(self):
        store = DataStore()
        store.put(_make_entry("A.val", n=10))
        store.put(_make_entry("B.val", n=10))
        sources, err = build_source_map(store, ["A.val", "B.val"])
        assert sources is None
        assert "Duplicate" in err

    def test_no_dot_label(self):
        store = DataStore()
        store.put(_make_entry("Bmag", n=10))
        sources, err = build_source_map(store, ["Bmag"])
        assert err is None
        assert "df_Bmag" in sources


class TestDescribeSources:
    def test_basic_summary(self):
        store = DataStore()
        idx = pd.date_range("2024-01-01", periods=100, freq="1min")
        data = pd.DataFrame({"BR": np.random.randn(100)}, index=idx)
        data.iloc[50:60] = np.nan  # 10% NaN
        store.put(DataEntry(label="DATASET.BR", data=data, units="nT"))

        info = describe_sources(store, ["DATASET.BR"])
        assert "df_BR" in info
        summary = info["df_BR"]
        assert summary["label"] == "DATASET.BR"
        assert summary["points"] == 100
        assert len(summary["columns"]) == 1
        assert summary["nan_pct"] == 10.0
        assert len(summary["time_range"]) == 2
        assert summary["cadence"] != ""

    def test_missing_label_skipped(self):
        store = DataStore()
        info = describe_sources(store, ["NONEXISTENT"])
        assert info == {}

    def test_non_timeseries_summary(self):
        """Non-timeseries entries use index_range instead of time_range/cadence."""
        store = DataStore()
        df = pd.DataFrame({"flux": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        entry = DataEntry(label="EVENTS.catalog", data=df, is_timeseries=False)
        store.put(entry)
        info = describe_sources(store, ["EVENTS.catalog"])
        assert "df_catalog" in info
        summary = info["df_catalog"]
        assert "time_range" not in summary
        assert "cadence" not in summary
        assert summary["index_range"] == ["10", "30"]


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_is_timeseries_persists(self, tmp_path):
        """is_timeseries flag should survive save/load cycle."""
        store = DataStore()
        # Timeseries entry
        ts_entry = _make_entry("ts_data", n=5)
        assert ts_entry.is_timeseries is True
        store.put(ts_entry)
        # Non-timeseries entry
        df = pd.DataFrame({"val": [1.0, 2.0]}, index=[0, 1])
        nt_entry = DataEntry(label="nt_data", data=df, is_timeseries=False)
        store.put(nt_entry)

        store.save_to_directory(tmp_path)

        store2 = DataStore()
        count = store2.load_from_directory(tmp_path)
        assert count == 2

        loaded_ts = store2.get("ts_data")
        assert loaded_ts.is_timeseries is True
        loaded_nt = store2.get("nt_data")
        assert loaded_nt.is_timeseries is False


# ---------------------------------------------------------------------------
# Concurrent access tests
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    """Verify DataStore thread-safety under concurrent access."""

    def test_concurrent_put_get(self):
        """10 threads doing put/get simultaneously should not corrupt state."""
        import threading

        store = DataStore()
        errors = []

        def worker(thread_id):
            try:
                for i in range(20):
                    label = f"thread_{thread_id}_entry_{i}"
                    entry = _make_entry(label=label, n=5)
                    store.put(entry)
                    retrieved = store.get(label)
                    if retrieved is None:
                        errors.append(f"{label} was None after put")
                    elif retrieved.label != label:
                        errors.append(f"{label} had wrong label: {retrieved.label}")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent errors: {errors}"
        assert len(store) == 200  # 10 threads × 20 entries

    def test_concurrent_list_entries(self):
        """list_entries should not crash while other threads modify the store."""
        import threading

        store = DataStore()
        errors = []

        def writer():
            try:
                for i in range(50):
                    store.put(_make_entry(label=f"w_{i}", n=3))
            except Exception as e:
                errors.append(f"Writer error: {e}")

        def reader():
            try:
                for _ in range(50):
                    store.list_entries()
            except Exception as e:
                errors.append(f"Reader error: {e}")

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent errors: {errors}"
