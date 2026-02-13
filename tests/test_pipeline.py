"""Tests for agent/pipeline.py — pipeline data structures, persistence, recording, and execution."""

import json
import pytest
from pathlib import Path

from agent.pipeline import (
    Pipeline, PipelineStep, PipelineVariable,
    PipelineStore, PipelineRecorder, PipelineExecutor,
    _slugify, _substitute_variables, merge_plot_steps,
)


# ---- Helpers ----

def _make_step(step_id=1, tool_name="fetch_data", tool_args=None, intent="test",
               produces=None, depends_on=None, critical=True):
    return PipelineStep(
        step_id=step_id,
        tool_name=tool_name,
        tool_args=tool_args or {"dataset_id": "AC_H2_MFI", "time_range": "$TIME_RANGE"},
        intent=intent,
        produces=produces or [],
        depends_on=depends_on or [],
        critical=critical,
    )


def _make_pipeline(pipeline_id="test-pipeline", name="Test Pipeline", steps=None, variables=None):
    if steps is None:
        steps = [_make_step()]
    if variables is None:
        variables = {
            "$TIME_RANGE": PipelineVariable(
                type="time_range",
                description="Date range",
                default="2026-01-01 to 2026-01-31",
            )
        }
    return Pipeline(
        id=pipeline_id,
        name=name,
        description="A test pipeline",
        steps=steps,
        variables=variables,
    )


# ---- Slugify ----

class TestSlugify:
    def test_basic(self):
        assert _slugify("ACE B-field Overview") == "ace-b-field-overview"

    def test_special_chars(self):
        assert _slugify("My Pipeline!!! (v2)") == "my-pipeline-v2"

    def test_empty(self):
        assert _slugify("") == "pipeline"

    def test_whitespace(self):
        assert _slugify("  spaces  ") == "spaces"


# ---- Variable substitution ----

class TestSubstituteVariables:
    def test_string(self):
        result = _substitute_variables("$TIME_RANGE", {"$TIME_RANGE": "2026-01-01 to 2026-01-31"})
        assert result == "2026-01-01 to 2026-01-31"

    def test_partial_string(self):
        result = _substitute_variables("range: $TIME_RANGE end", {"$TIME_RANGE": "2026-01-01 to 2026-01-31"})
        assert result == "range: 2026-01-01 to 2026-01-31 end"

    def test_dict(self):
        result = _substitute_variables(
            {"time_range": "$TIME_RANGE", "dataset": "AC_H2_MFI"},
            {"$TIME_RANGE": "2026-01-01 to 2026-01-31"},
        )
        assert result == {"time_range": "2026-01-01 to 2026-01-31", "dataset": "AC_H2_MFI"}

    def test_list(self):
        result = _substitute_variables(
            ["$TIME_RANGE", "other"],
            {"$TIME_RANGE": "2026-01-01 to 2026-01-31"},
        )
        assert result == ["2026-01-01 to 2026-01-31", "other"]

    def test_nested(self):
        result = _substitute_variables(
            {"a": {"b": "$TIME_RANGE"}},
            {"$TIME_RANGE": "2026-01-01 to 2026-01-31"},
        )
        assert result == {"a": {"b": "2026-01-01 to 2026-01-31"}}

    def test_no_match(self):
        result = _substitute_variables("no vars here", {"$TIME_RANGE": "2026-01-01 to 2026-01-31"})
        assert result == "no vars here"

    def test_scalar_passthrough(self):
        assert _substitute_variables(42, {"$TIME_RANGE": "x"}) == 42
        assert _substitute_variables(True, {"$TIME_RANGE": "x"}) is True
        assert _substitute_variables(None, {"$TIME_RANGE": "x"}) is None

    def test_multiple_variables(self):
        result = _substitute_variables(
            "$MISSION data for $TIME_RANGE",
            {"$MISSION": "ACE", "$TIME_RANGE": "2026-01-01 to 2026-01-31"},
        )
        assert result == "ACE data for 2026-01-01 to 2026-01-31"


# ---- PipelineStep ----

class TestPipelineStep:
    def test_round_trip(self):
        step = _make_step(produces=["AC_H2_MFI.BGSEc"], depends_on=[1, 2])
        d = step.to_dict()
        restored = PipelineStep.from_dict(d)
        assert restored.step_id == step.step_id
        assert restored.tool_name == step.tool_name
        assert restored.tool_args == step.tool_args
        assert restored.intent == step.intent
        assert restored.produces == step.produces
        assert restored.depends_on == step.depends_on
        assert restored.critical == step.critical

    def test_defaults(self):
        d = {"step_id": 1, "tool_name": "fetch_data", "tool_args": {}}
        step = PipelineStep.from_dict(d)
        assert step.intent == ""
        assert step.produces == []
        assert step.depends_on == []
        assert step.critical is True


# ---- Pipeline ----

class TestPipeline:
    def test_round_trip(self):
        p = _make_pipeline()
        d = p.to_dict()
        restored = Pipeline.from_dict(d)
        assert restored.id == p.id
        assert restored.name == p.name
        assert len(restored.steps) == len(p.steps)
        assert "$TIME_RANGE" in restored.variables
        assert restored.variables["$TIME_RANGE"].default == "2026-01-01 to 2026-01-31"

    def test_to_llm_context(self):
        p = _make_pipeline()
        ctx = p.to_llm_context()
        assert "Test Pipeline" in ctx
        assert "$TIME_RANGE" in ctx
        assert "fetch_data" in ctx
        assert "Step 1:" in ctx

    def test_empty_pipeline(self):
        p = Pipeline(id="empty", name="Empty", description="No steps")
        d = p.to_dict()
        restored = Pipeline.from_dict(d)
        assert restored.steps == []
        assert restored.variables == {}


# ---- PipelineStore ----

class TestPipelineStore:
    @pytest.fixture
    def store(self, tmp_path):
        return PipelineStore(base_dir=tmp_path)

    def test_save_and_load(self, store):
        p = _make_pipeline()
        store.save(p)
        loaded = store.load("test-pipeline")
        assert loaded is not None
        assert loaded.name == "Test Pipeline"
        assert len(loaded.steps) == 1
        assert loaded.created_at  # auto-set
        assert loaded.updated_at  # auto-set

    def test_load_nonexistent(self, store):
        assert store.load("nonexistent") is None

    def test_overwrite(self, store):
        p = _make_pipeline()
        store.save(p)
        p.description = "Updated"
        store.save(p)
        loaded = store.load("test-pipeline")
        assert loaded.description == "Updated"

    def test_delete(self, store):
        p = _make_pipeline()
        store.save(p)
        assert store.delete("test-pipeline") is True
        assert store.load("test-pipeline") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nonexistent") is False

    def test_list_pipelines(self, store):
        store.save(_make_pipeline("alpha", "Alpha"))
        store.save(_make_pipeline("beta", "Beta"))
        listing = store.list_pipelines()
        assert len(listing) == 2
        ids = {p["id"] for p in listing}
        assert ids == {"alpha", "beta"}
        # Check fields
        for p in listing:
            assert "name" in p
            assert "step_count" in p
            assert "variables" in p

    def test_list_empty(self, store):
        assert store.list_pipelines() == []


# ---- PipelineRecorder ----

class TestPipelineRecorder:
    def test_records_recordable_tools(self):
        rec = PipelineRecorder()
        rec.record("fetch_data", {"dataset_id": "AC"}, {"status": "success"})
        rec.record("update_plot_spec", {"spec": {"labels": "AC"}}, {"status": "success"})
        assert len(rec) == 2

    def test_ignores_non_recordable_tools(self):
        rec = PipelineRecorder()
        rec.record("search_datasets", {"query": "ACE"}, {"status": "success"})
        rec.record("delegate_to_mission", {"mission_id": "ACE"}, {"status": "success"})
        rec.record("list_parameters", {"dataset_id": "AC"}, {"status": "success"})
        assert len(rec) == 0

    def test_ignores_errors(self):
        rec = PipelineRecorder()
        rec.record("fetch_data", {"dataset_id": "AC"}, {"status": "error", "message": "not found"})
        assert len(rec) == 0

    def test_get_recording_returns_copy(self):
        rec = PipelineRecorder()
        rec.record("fetch_data", {"x": 1}, {"status": "success"})
        recording = rec.get_recording()
        recording.clear()
        assert len(rec) == 1  # original unaffected

    def test_clear(self):
        rec = PipelineRecorder()
        rec.record("fetch_data", {"x": 1}, {"status": "success"})
        rec.clear()
        assert len(rec) == 0

    def test_deep_copies_args(self):
        rec = PipelineRecorder()
        args = {"nested": {"key": "value"}}
        rec.record("fetch_data", args, {"status": "success"})
        args["nested"]["key"] = "mutated"
        assert rec.get_recording()[0]["tool_args"]["nested"]["key"] == "value"


# ---- PipelineExecutor ----

class TestPipelineExecutor:
    def _make_executor(self, tool_results=None):
        """Create an executor with a mock tool function.

        Args:
            tool_results: dict mapping tool_name to result dict,
                          or a callable (tool_name, tool_args) -> result
        """
        if tool_results is None:
            tool_results = {}
        if callable(tool_results):
            return PipelineExecutor(tool_executor=tool_results)

        def mock_tool(tool_name, tool_args):
            if tool_name in tool_results:
                return tool_results[tool_name]
            return {"status": "success"}

        return PipelineExecutor(tool_executor=mock_tool)

    def test_simple_execution(self):
        executor = self._make_executor()
        pipeline = _make_pipeline()
        result = executor.execute(pipeline)
        assert result["status"] == "success"
        assert result["steps_completed"] == 1
        assert result["steps_failed"] == 0

    def test_variable_substitution(self):
        captured = {}

        def mock_tool(name, args):
            captured.update(args)
            return {"status": "success"}

        executor = self._make_executor(mock_tool)
        pipeline = _make_pipeline()
        result = executor.execute(pipeline, variable_overrides={"$TIME_RANGE": "2026-06-01 to 2026-06-30"})
        assert result["status"] == "success"
        assert captured["time_range"] == "2026-06-01 to 2026-06-30"

    def test_default_variable(self):
        captured = {}

        def mock_tool(name, args):
            captured.update(args)
            return {"status": "success"}

        executor = self._make_executor(mock_tool)
        pipeline = _make_pipeline()
        result = executor.execute(pipeline)
        assert captured["time_range"] == "2026-01-01 to 2026-01-31"  # default

    def test_critical_failure_skips_dependents(self):
        steps = [
            _make_step(step_id=1, tool_name="fetch_data", critical=True),
            _make_step(step_id=2, tool_name="custom_operation", depends_on=[1], critical=True),
            _make_step(step_id=3, tool_name="plot_data", depends_on=[2], critical=False),
        ]
        pipeline = _make_pipeline(steps=steps)
        executor = self._make_executor({"fetch_data": {"status": "error", "message": "timeout"}})
        result = executor.execute(pipeline)
        assert result["status"] == "failed"
        assert result["steps_completed"] == 0
        assert result["steps_failed"] == 1
        assert result["steps_skipped"] == 2

    def test_non_critical_failure_continues(self):
        steps = [
            _make_step(step_id=1, tool_name="plot_data", critical=False),
            _make_step(step_id=2, tool_name="style_plot", tool_args={},
                       depends_on=[], critical=False),
        ]
        pipeline = _make_pipeline(steps=steps)
        executor = self._make_executor({"plot_data": {"status": "error", "message": "oops"}})
        result = executor.execute(pipeline)
        # plot_data failed but non-critical, style_plot doesn't depend on it so runs
        assert result["steps_completed"] == 1
        assert result["steps_failed"] == 1
        assert result["status"] == "partial"

    def test_partial_status(self):
        steps = [
            _make_step(step_id=1, tool_name="fetch_data", critical=True),
            _make_step(step_id=2, tool_name="plot_data", depends_on=[], critical=True),
        ]
        pipeline = _make_pipeline(steps=steps)
        executor = self._make_executor({"fetch_data": {"status": "error", "message": "fail"}})
        result = executor.execute(pipeline)
        assert result["status"] == "partial"
        assert result["steps_completed"] == 1  # plot_data succeeds (no dependency on fetch)
        assert result["steps_failed"] == 1

    def test_exception_in_tool(self):
        def exploding_tool(name, args):
            raise RuntimeError("boom")

        executor = self._make_executor(exploding_tool)
        pipeline = _make_pipeline()
        result = executor.execute(pipeline)
        assert result["status"] == "failed"
        assert result["steps_failed"] == 1
        assert "boom" in result["step_results"][0]["error"]

    def test_multi_step_success(self):
        steps = [
            _make_step(step_id=1, tool_name="fetch_data", produces=["A"]),
            _make_step(step_id=2, tool_name="custom_operation",
                       tool_args={"source": "A"}, depends_on=[1], produces=["B"]),
            _make_step(step_id=3, tool_name="plot_data",
                       tool_args={"labels": "A,B"}, depends_on=[1, 2]),
            _make_step(step_id=4, tool_name="style_plot",
                       tool_args={"title": "Test"}, depends_on=[3], critical=False),
        ]
        pipeline = _make_pipeline(steps=steps)
        executor = self._make_executor()
        result = executor.execute(pipeline)
        assert result["status"] == "success"
        assert result["steps_completed"] == 4

    def test_variables_used_in_result(self):
        executor = self._make_executor()
        pipeline = _make_pipeline()
        result = executor.execute(pipeline, variable_overrides={"$TIME_RANGE": "custom"})
        assert result["variables_used"]["$TIME_RANGE"] == "custom"

    def test_empty_pipeline(self):
        pipeline = _make_pipeline(steps=[])
        executor = self._make_executor()
        result = executor.execute(pipeline)
        assert result["status"] == "success"
        assert result["steps_completed"] == 0

    def test_transitive_skip(self):
        """If step 1 fails (critical), step 2 (depends on 1) is skipped,
        and step 3 (depends on 2, also critical) is also skipped."""
        steps = [
            _make_step(step_id=1, tool_name="fetch_data", critical=True),
            _make_step(step_id=2, tool_name="custom_operation", depends_on=[1], critical=True),
            _make_step(step_id=3, tool_name="plot_data", depends_on=[2], critical=True),
        ]
        pipeline = _make_pipeline(steps=steps)
        executor = self._make_executor({"fetch_data": {"status": "error", "message": "fail"}})
        result = executor.execute(pipeline)
        assert result["steps_failed"] == 1
        assert result["steps_skipped"] == 2
        assert result["step_results"][1]["status"] == "skipped"
        assert result["step_results"][2]["status"] == "skipped"


# ---- merge_plot_steps ----

class TestMergePlotSteps:
    def test_no_plot_steps_unchanged(self):
        """Non-plot steps pass through with renumbered IDs."""
        steps = [
            _make_step(step_id=1, tool_name="fetch_data"),
            _make_step(step_id=2, tool_name="custom_operation", depends_on=[1]),
        ]
        merged = merge_plot_steps(steps)
        assert len(merged) == 2
        assert merged[0].tool_name == "fetch_data"
        assert merged[1].tool_name == "custom_operation"
        assert merged[1].depends_on == [1]

    def test_plot_data_alone_becomes_render_spec(self):
        """A lone plot_data (no following style_plot) becomes render_spec."""
        steps = [
            _make_step(step_id=1, tool_name="plot_data",
                       tool_args={"labels": "A,B", "title": "Test"}),
        ]
        merged = merge_plot_steps(steps)
        assert len(merged) == 1
        assert merged[0].tool_name == "render_spec"
        assert merged[0].tool_args["spec"]["labels"] == "A,B"
        assert merged[0].tool_args["spec"]["title"] == "Test"

    def test_plot_data_plus_style_merged(self):
        """plot_data followed by style_plot merged into single render_spec."""
        steps = [
            _make_step(step_id=1, tool_name="plot_data",
                       tool_args={"labels": "A", "panels": [["A"]]}),
            _make_step(step_id=2, tool_name="style_plot",
                       tool_args={"title": "My Title", "font_size": 14},
                       depends_on=[1]),
        ]
        merged = merge_plot_steps(steps)
        assert len(merged) == 1
        step = merged[0]
        assert step.tool_name == "render_spec"
        spec = step.tool_args["spec"]
        assert spec["labels"] == "A"
        assert spec["panels"] == [["A"]]
        assert spec["title"] == "My Title"
        assert spec["font_size"] == 14

    def test_plot_data_plus_multiple_style_merged(self):
        """plot_data + multiple style_plot steps merged into one render_spec."""
        steps = [
            _make_step(step_id=1, tool_name="plot_data",
                       tool_args={"labels": "A"}),
            _make_step(step_id=2, tool_name="style_plot",
                       tool_args={"title": "T1"}, depends_on=[1]),
            _make_step(step_id=3, tool_name="style_plot",
                       tool_args={"font_size": 16, "legend": True}, depends_on=[2]),
        ]
        merged = merge_plot_steps(steps)
        assert len(merged) == 1
        spec = merged[0].tool_args["spec"]
        assert spec["labels"] == "A"
        assert spec["title"] == "T1"
        assert spec["font_size"] == 16
        assert spec["legend"] is True

    def test_style_overrides_plot_field(self):
        """When plot_data and style_plot set the same field, style_plot wins."""
        steps = [
            _make_step(step_id=1, tool_name="plot_data",
                       tool_args={"labels": "A", "title": "Original"}),
            _make_step(step_id=2, tool_name="style_plot",
                       tool_args={"title": "Styled"}, depends_on=[1]),
        ]
        merged = merge_plot_steps(steps)
        assert merged[0].tool_args["spec"]["title"] == "Styled"

    def test_mixed_steps_preserve_order(self):
        """fetch → compute → plot_data + style_plot → result is 3 steps."""
        steps = [
            _make_step(step_id=1, tool_name="fetch_data"),
            _make_step(step_id=2, tool_name="custom_operation", depends_on=[1]),
            _make_step(step_id=3, tool_name="plot_data",
                       tool_args={"labels": "A"}, depends_on=[2]),
            _make_step(step_id=4, tool_name="style_plot",
                       tool_args={"title": "T"}, depends_on=[3]),
        ]
        merged = merge_plot_steps(steps)
        assert len(merged) == 3
        assert merged[0].tool_name == "fetch_data"
        assert merged[0].step_id == 1
        assert merged[1].tool_name == "custom_operation"
        assert merged[1].step_id == 2
        assert merged[2].tool_name == "render_spec"
        assert merged[2].step_id == 3

    def test_depends_on_remapped(self):
        """depends_on references updated after merging shifts IDs."""
        steps = [
            _make_step(step_id=1, tool_name="fetch_data"),
            _make_step(step_id=2, tool_name="plot_data",
                       tool_args={"labels": "A"}, depends_on=[1]),
            _make_step(step_id=3, tool_name="style_plot",
                       tool_args={"title": "T"}, depends_on=[2]),
        ]
        merged = merge_plot_steps(steps)
        assert len(merged) == 2
        # render_spec depends on fetch_data (step 1), not on the old plot_data (2)
        assert merged[1].depends_on == [1]

    def test_internal_deps_removed(self):
        """Dependencies within a merged group are excluded from the result."""
        steps = [
            _make_step(step_id=1, tool_name="plot_data",
                       tool_args={"labels": "A"}),
            _make_step(step_id=2, tool_name="style_plot",
                       tool_args={"title": "T"}, depends_on=[1]),
        ]
        merged = merge_plot_steps(steps)
        # The style_plot's dep on plot_data is internal to the group
        assert merged[0].depends_on == []

    def test_intent_combined(self):
        """Intents from merged steps are joined."""
        steps = [
            _make_step(step_id=1, tool_name="plot_data",
                       tool_args={"labels": "A"}, intent="Create plot"),
            _make_step(step_id=2, tool_name="style_plot",
                       tool_args={"title": "T"}, intent="Style axes"),
        ]
        merged = merge_plot_steps(steps)
        assert "Create plot" in merged[0].intent
        assert "Style axes" in merged[0].intent

    def test_empty_steps(self):
        """Empty input returns empty output."""
        assert merge_plot_steps([]) == []

    def test_two_separate_plot_groups(self):
        """Two plot_data groups separated by a non-plot step stay separate."""
        steps = [
            _make_step(step_id=1, tool_name="plot_data",
                       tool_args={"labels": "A"}),
            _make_step(step_id=2, tool_name="style_plot",
                       tool_args={"title": "T1"}, depends_on=[1]),
            _make_step(step_id=3, tool_name="custom_operation",
                       tool_args={"code": "x"}),
            _make_step(step_id=4, tool_name="plot_data",
                       tool_args={"labels": "B"}),
            _make_step(step_id=5, tool_name="style_plot",
                       tool_args={"title": "T2"}, depends_on=[4]),
        ]
        merged = merge_plot_steps(steps)
        assert len(merged) == 3
        assert merged[0].tool_name == "render_spec"
        assert merged[0].tool_args["spec"]["labels"] == "A"
        assert merged[1].tool_name == "custom_operation"
        assert merged[2].tool_name == "render_spec"
        assert merged[2].tool_args["spec"]["labels"] == "B"
