"""
Tests for autoplot_bridge.script_runner — AST validator and sandboxed executor.

Validation tests are pure Python (no JVM needed).
Execution tests mock _build_namespace to avoid JPype/JVM dependency.

Run with: python -m pytest tests/test_script_runner.py
"""

from unittest.mock import MagicMock, patch

import pytest

from autoplot_bridge.script_runner import (
    validate_autoplot_code,
    execute_autoplot_script,
    run_autoplot_script,
    _format_java_exception,
)


# ── Validator Tests ──────────────────────────────────────────────────────────


class TestValidateAutoplotCode:
    """Test AST validation of Autoplot script code."""

    # --- Valid code ---

    def test_valid_simple_script(self):
        assert validate_autoplot_code("sc.plot(0, 'vap+cdaweb:ds=AC_H2_MFI&id=Magnitude')") == []

    def test_valid_dom_access(self):
        assert validate_autoplot_code("dom.getPlots(0).setTitle('Test')") == []

    def test_valid_color_usage(self):
        assert validate_autoplot_code("c = Color.RED") == []

    def test_valid_color_constructor(self):
        assert validate_autoplot_code("c = Color(255, 0, 0)") == []

    def test_valid_no_result_required(self):
        """Unlike custom_ops, result assignment is NOT required."""
        code = "sc.plot(0, 'uri')\nsc.waitUntilIdle()"
        assert validate_autoplot_code(code) == []

    def test_valid_with_result(self):
        code = "result = dom.getPlots().length"
        assert validate_autoplot_code(code) == []

    def test_valid_multiline(self):
        code = """plot = dom.getPlots(0)
plot.setTitle('Panel 1')
pe = dom.getPlotElements(0)
style = pe.getStyle()
style.setColor(Color.BLUE)"""
        assert validate_autoplot_code(code) == []

    def test_valid_loop(self):
        code = """for i in range(3):
    dom.getPlots(i).setTitle(str(i))"""
        assert validate_autoplot_code(code) == []

    def test_valid_string_formatting(self):
        code = "result = f'{dom.getPlots().length} panels'"
        assert validate_autoplot_code(code) == []

    def test_valid_datum_range(self):
        code = "tr = DatumRangeUtil.parseTimeRange('2024-01-01 to 2024-01-07')"
        assert validate_autoplot_code(code) == []

    def test_valid_units_lookup(self):
        code = "u = Units.lookupUnits('nT')"
        assert validate_autoplot_code(code) == []

    # --- Rejected code ---

    def test_reject_import(self):
        violations = validate_autoplot_code("import os")
        assert any("Import" in v for v in violations)

    def test_reject_from_import(self):
        violations = validate_autoplot_code("from java.io import File")
        assert any("Import" in v for v in violations)

    def test_reject_exec(self):
        violations = validate_autoplot_code("exec('x=1')")
        assert any("exec" in v for v in violations)

    def test_reject_eval(self):
        violations = validate_autoplot_code("result = eval('1+1')")
        assert any("eval" in v for v in violations)

    def test_reject_open(self):
        violations = validate_autoplot_code("f = open('/etc/passwd')")
        assert any("open" in v for v in violations)

    def test_reject_dunder(self):
        violations = validate_autoplot_code("x = sc.__class__")
        assert any("__class__" in v for v in violations)

    def test_reject_jclass(self):
        violations = validate_autoplot_code("Runtime = JClass('java.lang.Runtime')")
        assert any("JClass" in v for v in violations)

    def test_reject_jpype(self):
        violations = validate_autoplot_code("jpype.JClass('java.io.File')")
        assert any("jpype" in v for v in violations)

    def test_reject_os(self):
        violations = validate_autoplot_code("os.system('rm -rf /')")
        assert any("'os'" in v for v in violations)

    def test_reject_sys(self):
        violations = validate_autoplot_code("sys.exit(1)")
        assert any("'sys'" in v for v in violations)

    def test_reject_subprocess(self):
        violations = validate_autoplot_code("subprocess.run(['ls'])")
        assert any("subprocess" in v for v in violations)

    def test_reject_global(self):
        violations = validate_autoplot_code("global x")
        assert any("global" in v for v in violations)

    def test_reject_syntax_error(self):
        violations = validate_autoplot_code("sc.plot(")
        assert any("Syntax" in v for v in violations)

    def test_reject_async(self):
        violations = validate_autoplot_code("async def f(): pass")
        assert any("Async" in v or "async" in v for v in violations)

    def test_reject_socket(self):
        violations = validate_autoplot_code("socket.connect(('host', 80))")
        assert any("socket" in v for v in violations)

    def test_reject_shutil(self):
        violations = validate_autoplot_code("shutil.rmtree('/tmp')")
        assert any("shutil" in v for v in violations)

    def test_reject_pathlib(self):
        violations = validate_autoplot_code("pathlib.Path('/etc/passwd').read_text()")
        assert any("pathlib" in v for v in violations)


# ── Execution Tests (mocked) ────────────────────────────────────────────────


def _make_mock_namespace():
    """Create a mock namespace that _build_namespace would return."""
    mock_sc = MagicMock(name="ScriptContext")
    mock_dom = MagicMock(name="Application")
    mock_store = MagicMock(name="DataStore")
    return {
        "sc": mock_sc,
        "dom": mock_dom,
        "Color": MagicMock(name="Color"),
        "RenderType": MagicMock(name="RenderType"),
        "DatumRangeUtil": MagicMock(name="DatumRangeUtil"),
        "DatumRange": MagicMock(name="DatumRange"),
        "Units": MagicMock(name="Units"),
        "DDataSet": MagicMock(name="DDataSet"),
        "QDataSet": MagicMock(name="QDataSet"),
        "store": mock_store,
    }


class TestExecuteAutoplotScript:
    """Test execution of Autoplot scripts with mocked JVM."""

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_simple_execution(self, mock_build_ns):
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        result = execute_autoplot_script("x = 1 + 2", ctx)
        assert result["status"] == "success"
        assert result["output"] == ""

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_result_capture(self, mock_build_ns):
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        result = execute_autoplot_script("result = 42", ctx)
        assert result["status"] == "success"
        assert result["result"] == "42"

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_print_capture(self, mock_build_ns):
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        result = execute_autoplot_script("print('hello world')", ctx)
        assert result["status"] == "success"
        assert "hello world" in result["output"]

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_error_returns_dict(self, mock_build_ns):
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        result = execute_autoplot_script("x = 1 / 0", ctx)
        assert result["status"] == "error"
        assert "division by zero" in result["message"]

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_namespace_has_sc_and_dom(self, mock_build_ns):
        ns = _make_mock_namespace()
        mock_build_ns.return_value = ns
        ctx = MagicMock()

        # Verify sc and dom are accessible
        result = execute_autoplot_script("result = str(type(sc))", ctx)
        assert result["status"] == "success"
        assert result["result"] is not None

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_store_accessible(self, mock_build_ns):
        ns = _make_mock_namespace()
        mock_build_ns.return_value = ns
        ctx = MagicMock()

        result = execute_autoplot_script("result = str(type(store))", ctx)
        assert result["status"] == "success"
        assert result["result"] is not None

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_no_result_is_fine(self, mock_build_ns):
        """Scripts that don't assign to result should still succeed."""
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        result = execute_autoplot_script("x = 42", ctx)
        assert result["status"] == "success"
        assert "result" not in result

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_multiline_with_print_and_result(self, mock_build_ns):
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        code = "x = 10\nprint(f'x = {x}')\nresult = x * 2"
        result = execute_autoplot_script(code, ctx)
        assert result["status"] == "success"
        assert "x = 10" in result["output"]
        assert result["result"] == "20"

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_error_output_preserved(self, mock_build_ns):
        """Print output before an error should still be captured."""
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        code = "print('before error')\nx = 1 / 0"
        result = execute_autoplot_script(code, ctx)
        assert result["status"] == "error"
        assert "before error" in result["output"]

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_builtins_available(self, mock_build_ns):
        """Safe builtins like len, range, str should be available."""
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        code = "result = str(len(range(5)))"
        result = execute_autoplot_script(code, ctx)
        assert result["status"] == "success"
        assert result["result"] == "5"


# ── Integration Tests (run_autoplot_script) ──────────────────────────────────


class TestRunAutoplotScript:
    """Test the combined validate + execute convenience function."""

    def test_validation_rejection(self):
        ctx = MagicMock()
        result = run_autoplot_script("import os\nos.system('rm -rf /')", ctx)
        assert result["status"] == "error"
        assert "validation failed" in result["message"]

    @patch("autoplot_bridge.script_runner._build_namespace")
    def test_valid_code_executes(self, mock_build_ns):
        mock_build_ns.return_value = _make_mock_namespace()
        ctx = MagicMock()

        result = run_autoplot_script("result = 'hello'", ctx)
        assert result["status"] == "success"
        assert result["result"] == "hello"

    def test_multiple_violations_reported(self):
        ctx = MagicMock()
        result = run_autoplot_script("import os\nexec('bad')\nJClass('x')", ctx)
        assert result["status"] == "error"
        # Should report all violations
        assert "Import" in result["message"]
        assert "exec" in result["message"]
        assert "JClass" in result["message"]


# ── Helper Tests ─────────────────────────────────────────────────────────────


class TestFormatJavaException:
    def test_plain_python_exception(self):
        e = ValueError("test error")
        assert "test error" in _format_java_exception(e)

    def test_with_java_exception_attr(self):
        e = Exception("wrapper")
        java_exc = MagicMock()
        java_exc.getMessage.return_value = "Java error message"
        type(java_exc).__name__ = "NullPointerException"
        e.java_exception = java_exc
        result = _format_java_exception(e)
        assert "NullPointerException" in result
        assert "Java error message" in result

    def test_without_java_attrs(self):
        """Plain exception with no java_exception or __cause__ falls back to str()."""
        e = RuntimeError("plain error")
        result = _format_java_exception(e)
        assert "plain error" in result
