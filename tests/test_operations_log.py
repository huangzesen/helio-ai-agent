"""Tests for data_ops.operations_log."""

import json
import threading

import pytest

from data_ops.operations_log import OperationsLog, get_operations_log, reset_operations_log


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global singleton before/after each test."""
    reset_operations_log()
    yield
    reset_operations_log()


class TestOperationsLog:
    """Unit tests for the OperationsLog class."""

    def test_record_creates_entry(self):
        log = OperationsLog()
        rec = log.record(
            tool="fetch_data",
            args={"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            outputs=["AC_H2_MFI.BGSEc"],
        )
        assert rec["id"] == "op_001"
        assert rec["tool"] == "fetch_data"
        assert rec["status"] == "success"
        assert rec["outputs"] == ["AC_H2_MFI.BGSEc"]
        assert rec["inputs"] == []
        assert rec["error"] is None
        assert "timestamp" in rec

    def test_id_auto_increments(self):
        log = OperationsLog()
        r1 = log.record(tool="fetch_data", args={}, outputs=["a"])
        r2 = log.record(tool="custom_operation", args={}, outputs=["b"])
        r3 = log.record(tool="store_dataframe", args={}, outputs=["c"])
        assert r1["id"] == "op_001"
        assert r2["id"] == "op_002"
        assert r3["id"] == "op_003"

    def test_record_with_inputs_and_error(self):
        log = OperationsLog()
        rec = log.record(
            tool="custom_operation",
            args={"code": "bad code"},
            inputs=["AC_H2_MFI.BGSEc"],
            outputs=[],
            status="error",
            error="Execution error: name 'bad' is not defined",
        )
        assert rec["status"] == "error"
        assert rec["inputs"] == ["AC_H2_MFI.BGSEc"]
        assert rec["outputs"] == []
        assert "not defined" in rec["error"]

    def test_get_records_returns_copy(self):
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        records = log.get_records()
        assert len(records) == 1
        # Mutating the returned list shouldn't affect internal state
        records.clear()
        assert len(log.get_records()) == 1

    def test_len(self):
        log = OperationsLog()
        assert len(log) == 0
        log.record(tool="fetch_data", args={}, outputs=["a"])
        assert len(log) == 1
        log.record(tool="fetch_data", args={}, outputs=["b"])
        assert len(log) == 2

    def test_clear(self):
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        log.record(tool="fetch_data", args={}, outputs=["b"])
        assert len(log) == 2
        log.clear()
        assert len(log) == 0
        # Counter should reset too
        rec = log.record(tool="fetch_data", args={}, outputs=["c"])
        assert rec["id"] == "op_001"

    def test_json_roundtrip(self, tmp_path):
        log = OperationsLog()
        log.record(
            tool="fetch_data",
            args={"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc"},
            outputs=["AC_H2_MFI.BGSEc"],
        )
        log.record(
            tool="custom_operation",
            args={"code": "result = df.mean()", "output_label": "mean"},
            inputs=["AC_H2_MFI.BGSEc"],
            outputs=["mean"],
        )

        path = tmp_path / "operations.json"
        log.save_to_file(path)

        # Verify the JSON file is valid
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["tool"] == "fetch_data"
        assert data[1]["tool"] == "custom_operation"

        # Load into a fresh log
        log2 = OperationsLog()
        count = log2.load_from_file(path)
        assert count == 2
        records = log2.get_records()
        assert records[0]["id"] == "op_001"
        assert records[1]["id"] == "op_002"
        assert records[1]["inputs"] == ["AC_H2_MFI.BGSEc"]

    def test_counter_resumes_after_load(self, tmp_path):
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["a"])
        log.record(tool="fetch_data", args={}, outputs=["b"])
        log.record(tool="fetch_data", args={}, outputs=["c"])

        path = tmp_path / "operations.json"
        log.save_to_file(path)

        log2 = OperationsLog()
        log2.load_from_file(path)
        # Next record should be op_004
        rec = log2.record(tool="fetch_data", args={}, outputs=["d"])
        assert rec["id"] == "op_004"

    def test_load_from_records(self):
        records = [
            {"id": "op_001", "tool": "fetch_data", "args": {}, "outputs": ["a"],
             "inputs": [], "status": "success", "error": None, "timestamp": "2026-01-01T00:00:00+00:00"},
            {"id": "op_002", "tool": "custom_operation", "args": {}, "outputs": ["b"],
             "inputs": ["a"], "status": "success", "error": None, "timestamp": "2026-01-01T00:01:00+00:00"},
        ]
        log = OperationsLog()
        count = log.load_from_records(records)
        assert count == 2
        assert len(log) == 2
        # Counter resumes
        rec = log.record(tool="fetch_data", args={}, outputs=["c"])
        assert rec["id"] == "op_003"

    def test_thread_safety(self):
        log = OperationsLog()
        n_threads = 10
        n_per_thread = 50
        errors = []

        def worker(thread_id):
            try:
                for i in range(n_per_thread):
                    log.record(
                        tool="fetch_data",
                        args={"thread": thread_id, "i": i},
                        outputs=[f"t{thread_id}_{i}"],
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(log) == n_threads * n_per_thread

        # All IDs should be unique
        records = log.get_records()
        ids = [r["id"] for r in records]
        assert len(set(ids)) == len(ids)


class TestSingleton:
    """Tests for the module-level singleton helpers."""

    def test_get_operations_log_returns_same_instance(self):
        log1 = get_operations_log()
        log2 = get_operations_log()
        assert log1 is log2

    def test_reset_creates_new_instance(self):
        log1 = get_operations_log()
        log1.record(tool="fetch_data", args={}, outputs=["a"])
        reset_operations_log()
        log2 = get_operations_log()
        assert log2 is not log1
        assert len(log2) == 0
