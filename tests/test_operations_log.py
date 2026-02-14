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


class TestGetPipeline:
    """Tests for OperationsLog.get_pipeline()."""

    def test_basic_chain(self):
        """fetch → compute → render produces correct pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={"dataset_id": "AC_H2_MFI"}, outputs=["Bx"])
        log.record(
            tool="custom_operation",
            args={"code": "mag = ..."},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure": {}},
            inputs=["Bmag"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "custom_operation", "render_plotly_json"]

    def test_superseded_computation_keeps_last(self):
        """When a label is produced twice, only the last producer is kept."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation",
            args={"code": "wrong"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        log.record(
            tool="custom_operation",
            args={"code": "correct"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        # Should have fetch + last custom_operation only
        assert len(pipeline) == 2
        assert pipeline[0]["tool"] == "fetch_data"
        assert pipeline[1]["args"]["code"] == "correct"

    def test_dedup_skips_excluded(self):
        """fetch_data with already_loaded=true is skipped; real fetch is used."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={"dataset_id": "AC_H2_MFI"}, outputs=["Bx"])
        log.record(
            tool="fetch_data",
            args={"dataset_id": "AC_H2_MFI", "already_loaded": True},
            outputs=["Bx"],
        )
        pipeline = log.get_pipeline({"Bx"})
        # The dedup record is ignored during producer selection, so the
        # real fetch (op_001) is the last producer and appears in the pipeline.
        assert len(pipeline) == 1
        assert pipeline[0]["tool"] == "fetch_data"
        assert pipeline[0]["args"].get("already_loaded") is None

    def test_error_records_excluded(self):
        """Error records are never included in the pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation",
            args={"code": "bad"},
            inputs=["Bx"],
            outputs=[],
            status="error",
            error="Execution error",
        )
        log.record(
            tool="custom_operation",
            args={"code": "good"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        assert len(pipeline) == 2
        assert all(r["status"] == "success" for r in pipeline)

    def test_transitive_input_resolution(self):
        """A → B → C: requesting C pulls in A and B."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(
            tool="custom_operation", args={"code": "B = f(A)"}, inputs=["A"], outputs=["B"]
        )
        log.record(
            tool="custom_operation", args={"code": "C = f(B)"}, inputs=["B"], outputs=["C"]
        )
        pipeline = log.get_pipeline({"C"})
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "custom_operation", "custom_operation"]
        assert pipeline[0]["outputs"] == ["A"]
        assert pipeline[1]["outputs"] == ["B"]
        assert pipeline[2]["outputs"] == ["C"]

    def test_render_plotly_json_included(self):
        """The last successful render_plotly_json is included even if not a label producer."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json",
            args={"figure": {"data": []}},
            inputs=["Bx"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx"})
        tools = [r["tool"] for r in pipeline]
        assert "render_plotly_json" in tools

    def test_only_last_render_included(self):
        """Multiple render_plotly_json calls — only the last successful one is kept."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="render_plotly_json",
            args={"figure": {"version": 1}},
            inputs=["Bx"],
            outputs=[],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure": {"version": 2}},
            inputs=["Bx"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx"})
        renders = [r for r in pipeline if r["tool"] == "render_plotly_json"]
        assert len(renders) == 1
        assert renders[0]["args"]["figure"]["version"] == 2

    def test_empty_labels_returns_empty(self):
        """Empty final_labels produces an empty pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        assert log.get_pipeline(set()) == []

    def test_empty_log_returns_empty(self):
        """Empty log produces an empty pipeline regardless of labels."""
        log = OperationsLog()
        assert log.get_pipeline({"Bx"}) == []

    def test_manage_plot_reset_excluded(self):
        """manage_plot with action=reset is excluded from pipeline."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="manage_plot",
            args={"action": "reset"},
            inputs=[],
            outputs=[],
        )
        log.record(
            tool="render_plotly_json",
            args={"figure": {}},
            inputs=["Bx"],
            outputs=[],
        )
        pipeline = log.get_pipeline({"Bx"})
        tools = [r["tool"] for r in pipeline]
        assert "manage_plot" not in tools

    def test_render_inputs_resolved_transitively(self):
        """render_plotly_json inputs trigger transitive resolution."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(
            tool="custom_operation", args={}, inputs=["A"], outputs=["B"]
        )
        # Final labels don't include A or B, but render needs B
        log.record(
            tool="render_plotly_json",
            args={"figure": {}},
            inputs=["B"],
            outputs=[],
        )
        # Only ask for labels that aren't produced by render
        pipeline = log.get_pipeline({"A"})
        # Should include fetch(A), compute(B), render — because render references B
        tools = [r["tool"] for r in pipeline]
        assert "fetch_data" in tools
        assert "custom_operation" in tools
        assert "render_plotly_json" in tools

    def test_dedup_does_not_shadow_real_fetch(self):
        """A dedup fetch after the real fetch must not drop the real one."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={"dataset_id": "X"}, outputs=["Bx"])
        log.record(
            tool="fetch_data",
            args={"dataset_id": "X", "already_loaded": True},
            outputs=["Bx"],
        )
        log.record(
            tool="custom_operation",
            args={"code": "mag"},
            inputs=["Bx"],
            outputs=["Bmag"],
        )
        pipeline = log.get_pipeline({"Bx", "Bmag"})
        tools = [r["tool"] for r in pipeline]
        assert tools == ["fetch_data", "custom_operation"]
        # The fetch in the pipeline is the real one, not the dedup
        assert pipeline[0]["args"].get("already_loaded") is None

    def test_chronological_order_preserved(self):
        """Pipeline records are in the same order as the original log."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(tool="fetch_data", args={}, outputs=["B"])
        log.record(
            tool="custom_operation", args={}, inputs=["A", "B"], outputs=["C"]
        )
        pipeline = log.get_pipeline({"A", "B", "C"})
        ids = [r["id"] for r in pipeline]
        assert ids == sorted(ids)


class TestGetPipelineMermaid:
    """Tests for OperationsLog.get_pipeline_mermaid()."""

    def test_basic_flowchart(self):
        """fetch → compute → render produces valid Mermaid with edges."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx"])
        log.record(
            tool="custom_operation", args={}, inputs=["Bx"], outputs=["Bmag"]
        )
        log.record(
            tool="render_plotly_json", args={}, inputs=["Bmag"], outputs=[]
        )
        mermaid = log.get_pipeline_mermaid({"Bx", "Bmag"})
        assert mermaid.startswith("graph TD")
        # Nodes present
        assert 'op_001["fetch\\nBx"]' in mermaid
        assert 'op_002["compute\\nBmag"]' in mermaid
        assert 'op_003["plot"]' in mermaid
        # Edges present
        assert "op_001 -->|Bx| op_002" in mermaid
        assert "op_002 -->|Bmag| op_003" in mermaid

    def test_empty_pipeline_returns_empty_string(self):
        log = OperationsLog()
        assert log.get_pipeline_mermaid(set()) == ""

    def test_multiple_outputs(self):
        """A record with multiple outputs shows them comma-separated."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["Bx", "By", "Bz"])
        mermaid = log.get_pipeline_mermaid({"Bx", "By", "Bz"})
        assert "Bx, By, Bz" in mermaid

    def test_multiple_inputs(self):
        """A record with multiple inputs gets an edge from each producer."""
        log = OperationsLog()
        log.record(tool="fetch_data", args={}, outputs=["A"])
        log.record(tool="fetch_data", args={}, outputs=["B"])
        log.record(
            tool="custom_operation", args={}, inputs=["A", "B"], outputs=["C"]
        )
        mermaid = log.get_pipeline_mermaid({"A", "B", "C"})
        assert "op_001 -->|A| op_003" in mermaid
        assert "op_002 -->|B| op_003" in mermaid


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
