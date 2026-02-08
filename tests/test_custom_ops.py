"""
Tests for data_ops.custom_ops — AST validator and sandboxed executor.

Run with: python -m pytest tests/test_custom_ops.py
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.custom_ops import (
    validate_pandas_code,
    execute_custom_operation,
    run_custom_operation,
)


def _make_time(n=100, start="2024-01-01", cadence_s=60):
    """Create a DatetimeIndex with fixed cadence."""
    return pd.date_range(start, periods=n, freq=f"{cadence_s}s")


def _make_df(values, index, columns=None):
    """Create a DataFrame from values and DatetimeIndex."""
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return pd.DataFrame(values, index=index, columns=columns or ["value"])
    return pd.DataFrame(values, index=index, columns=columns)


# ── Validator Tests ──────────────────────────────────────────────────────────


class TestValidatePandasCode:
    def test_valid_simple_operation(self):
        assert validate_pandas_code("result = df * 2") == []

    def test_valid_multiline(self):
        code = "mean = df.mean()\nresult = df - mean"
        assert validate_pandas_code(code) == []

    def test_valid_numpy_operation(self):
        assert validate_pandas_code("result = np.log10(df.abs())") == []

    def test_valid_rolling(self):
        assert validate_pandas_code("result = df.rolling(10, center=True, min_periods=1).mean()") == []

    def test_valid_interpolate(self):
        assert validate_pandas_code("result = df.interpolate(method='linear')") == []

    def test_valid_clip(self):
        assert validate_pandas_code("result = df.clip(lower=-50, upper=50)") == []

    def test_valid_complex_multiline(self):
        code = "z = (df - df.mean()) / df.std()\nmask = z.abs() < 3\nresult = df[mask].reindex(df.index)"
        assert validate_pandas_code(code) == []

    def test_reject_no_result_assignment(self):
        violations = validate_pandas_code("x = df * 2")
        assert any("result" in v for v in violations)

    def test_reject_import(self):
        violations = validate_pandas_code("import os\nresult = df")
        assert any("Import" in v for v in violations)

    def test_reject_from_import(self):
        violations = validate_pandas_code("from os import path\nresult = df")
        assert any("Import" in v for v in violations)

    def test_reject_exec(self):
        violations = validate_pandas_code("exec('x=1')\nresult = df")
        assert any("exec" in v for v in violations)

    def test_reject_eval(self):
        violations = validate_pandas_code("result = eval('df * 2')")
        assert any("eval" in v for v in violations)

    def test_reject_open(self):
        violations = validate_pandas_code("open('test.txt')\nresult = df")
        assert any("open" in v for v in violations)

    def test_reject_dunder_access(self):
        violations = validate_pandas_code("result = df.__class__")
        assert any("__class__" in v for v in violations)

    def test_reject_global(self):
        violations = validate_pandas_code("global x\nresult = df")
        assert any("global" in v for v in violations)

    def test_reject_nonlocal(self):
        violations = validate_pandas_code("nonlocal x\nresult = df")
        assert any("global/nonlocal" in v.lower() or "nonlocal" in v.lower() for v in violations)

    def test_reject_syntax_error(self):
        violations = validate_pandas_code("result = df +")
        assert any("Syntax" in v for v in violations)

    def test_reject_async(self):
        violations = validate_pandas_code("async def f(): pass\nresult = df")
        assert any("Async" in v or "async" in v for v in violations)

    def test_require_result_false_allows_no_assignment(self):
        violations = validate_pandas_code("x = 42", require_result=False)
        assert violations == []

    def test_require_result_false_still_blocks_imports(self):
        violations = validate_pandas_code("import os", require_result=False)
        assert any("Import" in v for v in violations)

    def test_require_result_false_still_blocks_exec(self):
        violations = validate_pandas_code("exec('x=1')", require_result=False)
        assert any("exec" in v for v in violations)

    def test_require_result_false_still_blocks_dunder(self):
        violations = validate_pandas_code("x = obj.__class__", require_result=False)
        assert any("__class__" in v for v in violations)

    def test_require_result_default_is_true(self):
        violations = validate_pandas_code("x = 42")
        assert any("result" in v for v in violations)


# ── Executor Tests ───────────────────────────────────────────────────────────


class TestExecuteCustomOperation:
    def test_multiply(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        result = execute_custom_operation(df, "result = df * 2")
        np.testing.assert_allclose(result.values.squeeze(), [2.0, 4.0, 6.0, 8.0, 10.0])

    def test_normalize(self):
        idx = _make_time(5)
        df = _make_df(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), idx)
        result = execute_custom_operation(df, "result = (df - df.mean()) / df.std()")
        # Normalized data should have mean ~0 and std ~1 (using ddof=1 to match pandas)
        np.testing.assert_allclose(result.values.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.values.std(ddof=1), 1.0, atol=1e-10)

    def test_clip(self):
        idx = _make_time(5)
        df = _make_df(np.array([-100.0, -5.0, 0.0, 5.0, 100.0]), idx)
        result = execute_custom_operation(df, "result = df.clip(lower=-10, upper=10)")
        np.testing.assert_allclose(result.values.squeeze(), [-10.0, -5.0, 0.0, 5.0, 10.0])

    def test_numpy_function(self):
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 10.0, 100.0]), idx)
        result = execute_custom_operation(df, "result = np.log10(df)")
        np.testing.assert_allclose(result.values.squeeze(), [0.0, 1.0, 2.0])

    def test_series_to_dataframe_conversion(self):
        idx = _make_time(3)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        result = execute_custom_operation(df, "result = df['a']")
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1

    def test_datetime_index_preserved(self):
        idx = _make_time(5)
        df = _make_df(np.arange(5, dtype=float), idx)
        result = execute_custom_operation(df, "result = df * 2")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result.index) == 5

    def test_source_not_mutated(self):
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        original_values = df.values.copy()
        execute_custom_operation(df, "result = df * 0")
        np.testing.assert_allclose(df.values, original_values)

    def test_no_result_error(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="did not assign"):
            execute_custom_operation(df, "x = df * 2")

    def test_runtime_error(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(RuntimeError, match="Execution error"):
            execute_custom_operation(df, "result = df / undefined_var")

    def test_non_dataframe_error(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="DataFrame or Series"):
            execute_custom_operation(df, "result = 42")

    def test_lost_index_error(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            execute_custom_operation(df, "result = pd.DataFrame({'a': [1, 2, 3]})")

    def test_multiline_code(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        code = "mean = df.mean()\nstd = df.std()\nresult = (df - mean) / std"
        result = execute_custom_operation(df, code)
        assert len(result) == 5

    def test_vector_dataframe(self):
        idx = _make_time(5)
        df = pd.DataFrame(
            np.ones((5, 3)), index=idx, columns=["x", "y", "z"]
        )
        result = execute_custom_operation(df, "result = df * 3")
        np.testing.assert_allclose(result.values, np.ones((5, 3)) * 3)


# ── Integration Tests (run_custom_operation) ─────────────────────────────────


class TestRunCustomOperation:
    def test_end_to_end_success(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 4.0, 9.0, 16.0, 25.0]), idx)
        result = run_custom_operation(df, "result = np.sqrt(df)")
        np.testing.assert_allclose(result.values.squeeze(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_validation_rejection(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="validation failed"):
            run_custom_operation(df, "import os\nresult = df")

    def test_execution_error_propagation(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(RuntimeError, match="Execution error"):
            run_custom_operation(df, "result = df.nonexistent_method()")


# ── Replacement Tests: Can custom_operation replicate the 5 hardcoded ops? ──


class TestReplaceHardcodedOps:
    """Verify that custom_operation can replicate each of the 5 dedicated tools."""

    def test_replaces_compute_magnitude(self):
        """magnitude = sqrt(x^2 + y^2 + z^2)"""
        idx = _make_time(3)
        df = pd.DataFrame(
            {"x": [3.0, 0.0, 1.0], "y": [4.0, 0.0, 2.0], "z": [0.0, 5.0, 2.0]},
            index=idx,
        )
        code = "result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [5.0, 5.0, 3.0])

    def test_replaces_compute_arithmetic_add(self):
        """Element-wise addition of two aligned DataFrames."""
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        # Simulate: df is "a", we embed "b" values in code
        code = "b = pd.DataFrame([10.0, 20.0, 30.0], index=df.index, columns=df.columns)\nresult = df + b"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [11.0, 22.0, 33.0])

    def test_replaces_compute_arithmetic_divide_with_nan(self):
        """Division with zero handling."""
        idx = _make_time(3)
        df = _make_df(np.array([10.0, 20.0, 30.0]), idx)
        code = "divisor = pd.DataFrame([2.0, 0.0, 5.0], index=df.index, columns=df.columns)\nresult = (df / divisor).replace([np.inf, -np.inf], np.nan)"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.iloc[0, 0], 5.0)
        assert np.isnan(result.iloc[1, 0])
        np.testing.assert_allclose(result.iloc[2, 0], 6.0)

    def test_replaces_compute_running_average(self):
        """Centered rolling mean with min_periods=1."""
        idx = _make_time(5)
        df = _make_df(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), idx)
        code = "result = df.rolling(3, center=True, min_periods=1).mean()"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.iloc[0, 0], 15.0)   # avg(10,20)
        np.testing.assert_allclose(result.iloc[2, 0], 30.0)   # avg(20,30,40)
        np.testing.assert_allclose(result.iloc[4, 0], 45.0)   # avg(40,50)

    def test_replaces_compute_resample(self):
        """Downsample by bin-averaging at fixed cadence."""
        idx = _make_time(100, cadence_s=1)
        df = _make_df(np.arange(100, dtype=np.float64), idx)
        code = "result = df.resample('10s').mean().dropna(how='all')"
        result = run_custom_operation(df, code)
        assert len(result) == 10
        np.testing.assert_allclose(result.iloc[0, 0], 4.5)  # mean(0..9)

    def test_replaces_compute_delta_difference(self):
        """Differences: df.diff().iloc[1:]"""
        idx = _make_time(5, cadence_s=60)
        df = _make_df(np.array([10.0, 12.0, 15.0, 11.0, 14.0]), idx)
        code = "result = df.diff().iloc[1:]"
        result = run_custom_operation(df, code)
        assert len(result) == 4
        np.testing.assert_allclose(result.values.squeeze(), [2.0, 3.0, -4.0, 3.0])

    def test_replaces_compute_delta_derivative(self):
        """Time derivative: dv/dt in units per second."""
        idx = _make_time(3, cadence_s=60)
        df = _make_df(np.array([0.0, 60.0, 180.0]), idx)
        code = "dv = df.diff().iloc[1:]\ndt_s = df.index.to_series().diff().dt.total_seconds().iloc[1:]\nresult = dv.div(dt_s, axis=0)"
        result = run_custom_operation(df, code)
        np.testing.assert_allclose(result.values.squeeze(), [1.0, 2.0])
