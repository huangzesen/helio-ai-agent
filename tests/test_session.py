"""
Tests for agent.session — SessionManager and DataStore persistence.

Run with: python -m pytest tests/test_session.py -v
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.session import SessionManager
from data_ops.store import DataEntry, DataStore


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for session storage."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sm(tmp_dir):
    """Provide a SessionManager backed by a temp directory."""
    return SessionManager(base_dir=tmp_dir)


def _make_store_with_entries() -> DataStore:
    """Create a DataStore with sample scalar and vector entries."""
    store = DataStore()
    idx = pd.date_range("2024-01-01", periods=100, freq="1min")

    scalar_df = pd.DataFrame(np.random.randn(100), index=idx, columns=["value"])
    store.put(DataEntry(
        label="AC_H2_MFI.Magnitude",
        data=scalar_df,
        units="nT",
        description="Magnetic field magnitude",
        source="cdf",
    ))

    vector_df = pd.DataFrame(
        np.random.randn(100, 3), index=idx, columns=["Bx", "By", "Bz"]
    )
    store.put(DataEntry(
        label="AC_H2_MFI.BGSEc",
        data=vector_df,
        units="nT",
        description="Magnetic field vector",
        source="cdf",
    ))

    return store


class TestSessionManager:
    def test_create_session(self, sm, tmp_dir):
        """create_session creates directory + metadata."""
        sid = sm.create_session("gemini-test")

        session_dir = tmp_dir / sid
        assert session_dir.exists()
        assert (session_dir / "metadata.json").exists()
        assert (session_dir / "history.json").exists()
        assert (session_dir / "data").is_dir()

        with open(session_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["id"] == sid
        assert meta["model"] == "gemini-test"
        assert meta["turn_count"] == 0

    def test_list_sessions_sorted(self, sm):
        """list_sessions returns sorted by updated_at descending."""
        sid1 = sm.create_session("model-a")
        time.sleep(0.05)  # ensure distinct timestamps
        sid2 = sm.create_session("model-b")

        sessions = sm.list_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["id"] == sid2
        assert sessions[1]["id"] == sid1

    def test_delete_session(self, sm, tmp_dir):
        """delete_session removes the directory."""
        sid = sm.create_session("model")
        assert (tmp_dir / sid).exists()

        result = sm.delete_session(sid)
        assert result is True
        assert not (tmp_dir / sid).exists()

        # Deleting non-existent returns False
        assert sm.delete_session("nonexistent") is False

    def test_get_most_recent_session(self, sm):
        """get_most_recent_session returns the latest session."""
        sid1 = sm.create_session("model")
        time.sleep(0.05)
        sid2 = sm.create_session("model")

        assert sm.get_most_recent_session() == sid2

    def test_get_most_recent_empty(self, sm):
        """get_most_recent_session returns None when no sessions exist."""
        assert sm.get_most_recent_session() is None

    def test_save_and_load_history(self, sm):
        """Round-trip Content dicts through JSON."""
        sid = sm.create_session("model")
        store = DataStore()

        history = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
            {"role": "user", "parts": [{"text": "Show me ACE data"}]},
            {
                "role": "model",
                "parts": [
                    {"function_call": {"name": "search_datasets", "args": {"query": "ACE"}}},
                ],
            },
        ]

        sm.save_session(sid, history, store, {"turn_count": 2})

        loaded_history, data_dir, meta, _fig = sm.load_session(sid)
        assert len(loaded_history) == 4
        assert loaded_history[0]["role"] == "user"
        assert loaded_history[0]["parts"][0]["text"] == "Hello"
        assert loaded_history[3]["parts"][0]["function_call"]["name"] == "search_datasets"
        assert meta["turn_count"] == 2

    def test_save_and_load_dataframes(self, sm):
        """Round-trip DataFrames through pickle, verify values match."""
        sid = sm.create_session("model")
        store = _make_store_with_entries()

        # Save
        sm.save_session(sid, [], store)

        # Load into a fresh store
        _, data_dir, _, _ = sm.load_session(sid)
        restored_store = DataStore()
        count = restored_store.load_from_directory(data_dir)

        assert count == 2
        assert len(restored_store) == 2

        # Check scalar entry
        orig = store.get("AC_H2_MFI.Magnitude")
        rest = restored_store.get("AC_H2_MFI.Magnitude")
        assert rest is not None
        assert rest.units == orig.units
        assert rest.description == orig.description
        assert rest.source == orig.source
        pd.testing.assert_frame_equal(orig.data, rest.data)

        # Check vector entry
        orig_v = store.get("AC_H2_MFI.BGSEc")
        rest_v = restored_store.get("AC_H2_MFI.BGSEc")
        assert rest_v is not None
        pd.testing.assert_frame_equal(orig_v.data, rest_v.data)

    def test_save_updates_metadata(self, sm):
        """save_session merges metadata_updates into metadata.json."""
        sid = sm.create_session("model")
        store = DataStore()

        sm.save_session(sid, [], store, {
            "turn_count": 5,
            "last_message_preview": "Show me data",
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
        })

        _, _, meta, _ = sm.load_session(sid)
        assert meta["turn_count"] == 5
        assert meta["last_message_preview"] == "Show me data"
        assert meta["token_usage"]["input_tokens"] == 100

    def test_load_nonexistent_raises(self, sm):
        """load_session raises FileNotFoundError for missing session."""
        with pytest.raises(FileNotFoundError):
            sm.load_session("nonexistent_session_id")

    def test_save_nonexistent_raises(self, sm):
        """save_session raises FileNotFoundError for missing session dir."""
        store = DataStore()
        with pytest.raises(FileNotFoundError):
            sm.save_session("nonexistent", [], store)


class TestCleanupEmptySessions:
    def test_cleanup_removes_empty_sessions(self, tmp_dir):
        """Empty sessions (0 turns, no history) are removed by cleanup."""
        sm = SessionManager(base_dir=tmp_dir)
        sid_empty = sm.create_session("model")
        # Create a session with actual content
        sid_full = sm.create_session("model")
        sm.save_session(sid_full, [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi!"}]},
        ], DataStore(), {"turn_count": 1})

        # Before cleanup, both exist
        assert len(sm.list_sessions()) == 2

        deleted = sm.cleanup_empty_sessions()
        assert deleted == 1
        sessions = sm.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid_full

    def test_cleanup_skips_excluded_ids(self, tmp_dir):
        """Sessions in exclude_ids are not deleted even if empty."""
        sm = SessionManager(base_dir=tmp_dir)
        sid = sm.create_session("model")

        deleted = sm.cleanup_empty_sessions(exclude_ids={sid})
        assert deleted == 0
        assert len(sm.list_sessions()) == 1

    def test_cleanup_explicit_call(self, tmp_dir):
        """Explicit cleanup_empty_sessions() removes empties, keeps populated."""
        sm = SessionManager(base_dir=tmp_dir)
        sid_empty = sm.create_session("model")
        sid_full = sm.create_session("model")
        sm.save_session(sid_full, [
            {"role": "user", "parts": [{"text": "Hello"}]},
        ], DataStore(), {"turn_count": 1})

        assert len(sm.list_sessions()) == 2
        sm.cleanup_empty_sessions()
        sessions = sm.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid_full

    def test_cleanup_keeps_sessions_with_history(self, tmp_dir):
        """Sessions with history but turn_count=0 (stale metadata) are kept."""
        sm = SessionManager(base_dir=tmp_dir)
        sid = sm.create_session("model")
        # Save history but don't update turn_count
        sm.save_session(sid, [
            {"role": "user", "parts": [{"text": "test"}]},
        ], DataStore())

        deleted = sm.cleanup_empty_sessions()
        assert deleted == 0


class TestDataStorePersistence:
    def test_save_empty_store(self, tmp_dir):
        """Saving an empty store writes an empty index."""
        store = DataStore()
        data_dir = tmp_dir / "data"
        store.save_to_directory(data_dir)

        assert (data_dir / "_index.json").exists()
        with open(data_dir / "_index.json") as f:
            index = json.load(f)
        assert index == {}

    def test_load_empty_directory(self, tmp_dir):
        """Loading from a directory with no index returns 0."""
        store = DataStore()
        assert store.load_from_directory(tmp_dir) == 0

    def test_load_missing_pkl(self, tmp_dir):
        """Gracefully skip entries whose pkl file is missing."""
        data_dir = tmp_dir / "data"
        data_dir.mkdir()

        index = {"missing_label": {"filename": "missing.pkl", "units": "nT",
                                    "description": "", "source": "cdf"}}
        with open(data_dir / "_index.json", "w") as f:
            json.dump(index, f)

        store = DataStore()
        count = store.load_from_directory(data_dir)
        assert count == 0
        assert len(store) == 0

    def test_label_with_special_chars(self, tmp_dir):
        """Labels with special characters are sanitized for filenames."""
        store = DataStore()
        idx = pd.date_range("2024-01-01", periods=10, freq="1s")
        df = pd.DataFrame(np.random.randn(10), index=idx, columns=["v"])
        store.put(DataEntry(
            label="DS/PARAM:special*chars",
            data=df,
            units="",
            source="computed",
        ))

        data_dir = tmp_dir / "data"
        store.save_to_directory(data_dir)

        # Verify the pkl file has safe name
        with open(data_dir / "_index.json") as f:
            index = json.load(f)
        assert "DS/PARAM:special*chars" in index
        filename = index["DS/PARAM:special*chars"]["filename"]
        assert "/" not in filename
        assert ":" not in filename
        assert "*" not in filename

        # Round-trip
        store2 = DataStore()
        count = store2.load_from_directory(data_dir)
        assert count == 1
        entry = store2.get("DS/PARAM:special*chars")
        assert entry is not None
        pd.testing.assert_frame_equal(df, entry.data)

    def test_corrupted_index(self, tmp_dir):
        """Corrupted _index.json doesn't crash (raises json.JSONDecodeError)."""
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        with open(data_dir / "_index.json", "w") as f:
            f.write("{corrupted json")

        store = DataStore()
        with pytest.raises(json.JSONDecodeError):
            store.load_from_directory(data_dir)
