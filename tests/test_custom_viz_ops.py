"""
Tests for rendering.custom_viz_ops — Plotly code sandbox.

Run with: python -m pytest tests/test_custom_viz_ops.py
"""

import pytest
import plotly.graph_objects as go

from rendering.custom_viz_ops import (
    validate_plotly_code,
    execute_custom_visualization,
    run_custom_visualization,
)


# ── Validator Tests ──────────────────────────────────────────────────────────


class TestValidatePlotlyCode:
    def test_valid_title_update(self):
        assert validate_plotly_code('fig.update_layout(title_text="Test")') == []

    def test_valid_multiline(self):
        code = 'fig.update_layout(title_text="T")\nfig.update_yaxes(title_text="Y")'
        assert validate_plotly_code(code) == []

    def test_valid_add_hline(self):
        assert validate_plotly_code('fig.add_hline(y=0, line_dash="dash")') == []

    def test_valid_trace_mode(self):
        assert validate_plotly_code('fig.data[0].mode = "markers"') == []

    def test_no_result_required(self):
        # Should pass without assigning to 'result'
        assert validate_plotly_code("x = 42") == []

    def test_reject_import(self):
        violations = validate_plotly_code("import os")
        assert any("Import" in v for v in violations)

    def test_reject_from_import(self):
        violations = validate_plotly_code("from os import path")
        assert any("Import" in v for v in violations)

    def test_reject_exec(self):
        violations = validate_plotly_code('exec("x=1")')
        assert any("exec" in v for v in violations)

    def test_reject_eval(self):
        violations = validate_plotly_code('eval("1+1")')
        assert any("eval" in v for v in violations)

    def test_reject_dunder_access(self):
        violations = validate_plotly_code("fig.__class__")
        assert any("__class__" in v for v in violations)

    def test_reject_syntax_error(self):
        violations = validate_plotly_code("fig.update_layout(")
        assert any("Syntax" in v for v in violations)


# ── Executor Tests ───────────────────────────────────────────────────────────


class TestExecuteCustomVisualization:
    def test_update_title(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        execute_custom_visualization(fig, 'fig.update_layout(title_text="Hello")')
        assert fig.layout.title.text == "Hello"

    def test_add_hline(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        execute_custom_visualization(fig, 'fig.add_hline(y=0)')
        # add_hline adds a shape to the layout
        assert len(fig.layout.shapes) > 0

    def test_change_trace_mode(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], mode="lines"))
        execute_custom_visualization(fig, 'fig.data[0].mode = "markers"')
        assert fig.data[0].mode == "markers"

    def test_canvas_resize(self):
        fig = go.Figure()
        execute_custom_visualization(fig, "fig.update_layout(width=1920, height=1080)")
        assert fig.layout.width == 1920
        assert fig.layout.height == 1080

    def test_log_scale(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        execute_custom_visualization(fig, 'fig.update_yaxes(type="log")')
        assert fig.layout.yaxis.type == "log"

    def test_axis_label(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        execute_custom_visualization(fig, 'fig.update_yaxes(title_text="B (nT)")')
        assert fig.layout.yaxis.title.text == "B (nT)"

    def test_go_available(self):
        """The go (graph_objects) module should be available in namespace."""
        fig = go.Figure()
        execute_custom_visualization(fig, "fig.add_trace(go.Scatter(x=[1], y=[1]))")
        assert len(fig.data) == 1

    def test_np_available(self):
        """numpy should be available in namespace."""
        fig = go.Figure()
        execute_custom_visualization(fig, "x = np.array([1,2,3])\nfig.add_trace(go.Scatter(x=x, y=x))")
        assert len(fig.data) == 1

    def test_runtime_error_caught(self):
        fig = go.Figure()
        with pytest.raises(RuntimeError, match="Execution error"):
            execute_custom_visualization(fig, "fig.nonexistent_method()")

    def test_fig_is_mutated_in_place(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        original_id = id(fig)
        execute_custom_visualization(fig, 'fig.update_layout(title_text="Mutated")')
        assert id(fig) == original_id
        assert fig.layout.title.text == "Mutated"


# ── Integration Tests (run_custom_visualization) ─────────────────────────────


class TestRunCustomVisualization:
    def test_end_to_end_success(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        run_custom_visualization(fig, 'fig.update_layout(title_text="Test")')
        assert fig.layout.title.text == "Test"

    def test_validation_rejection(self):
        fig = go.Figure()
        with pytest.raises(ValueError, match="validation failed"):
            run_custom_visualization(fig, "import os")

    def test_execution_error_propagation(self):
        fig = go.Figure()
        with pytest.raises(RuntimeError, match="Execution error"):
            run_custom_visualization(fig, "fig.nonexistent()")
