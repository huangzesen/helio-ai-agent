"""
Tests for data_ops.operations — pandas-based timeseries operations.

Run with: python -m pytest tests/test_operations.py
"""

import numpy as np
import pandas as pd
import pytest

from data_ops.operations import (
    compute_magnitude,
    compute_arithmetic,
    compute_running_average,
    compute_resample,
    compute_delta,
)


def _make_time(n=100, start="2024-01-01", cadence_s=60):
    """Create a DatetimeIndex with fixed cadence."""
    return pd.date_range(start, periods=n, freq=f"{cadence_s}s")


def _make_df(values, index, columns=None):
    """Create a DataFrame from values and DatetimeIndex."""
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return pd.DataFrame(values, index=index, columns=columns or ["value"])
    return pd.DataFrame(values, index=index, columns=columns)


class TestComputeMagnitude:
    def test_basic(self):
        idx = _make_time(2)
        df = _make_df(
            np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]),
            idx, columns=["x", "y", "z"],
        )
        result = compute_magnitude(df)
        np.testing.assert_allclose(result["magnitude"].values, [5.0, 5.0])

    def test_nan_propagates(self):
        idx = _make_time(2)
        df = _make_df(
            np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]]),
            idx, columns=["x", "y", "z"],
        )
        result = compute_magnitude(df)
        assert np.isnan(result["magnitude"].iloc[0])
        np.testing.assert_allclose(result["magnitude"].iloc[1], np.sqrt(14))

    def test_wrong_shape_1_col(self):
        idx = _make_time(3)
        df = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        with pytest.raises(ValueError, match="requires 3 columns"):
            compute_magnitude(df)

    def test_wrong_shape_2_cols(self):
        idx = _make_time(5)
        df = pd.DataFrame(np.ones((5, 2)), index=idx, columns=["a", "b"])
        with pytest.raises(ValueError, match="requires 3 columns"):
            compute_magnitude(df)


class TestComputeArithmetic:
    def test_add(self):
        idx = _make_time(3)
        a = _make_df(np.array([1.0, 2.0, 3.0]), idx)
        b = _make_df(np.array([10.0, 20.0, 30.0]), idx)
        result = compute_arithmetic(a, b, "+")
        np.testing.assert_allclose(result.values.squeeze(), [11.0, 22.0, 33.0])

    def test_subtract(self):
        idx = _make_time(2)
        a = _make_df(np.array([10.0, 20.0]), idx)
        b = _make_df(np.array([3.0, 5.0]), idx)
        result = compute_arithmetic(a, b, "-")
        np.testing.assert_allclose(result.values.squeeze(), [7.0, 15.0])

    def test_multiply(self):
        idx = _make_time(2)
        a = _make_df(np.array([2.0, 3.0]), idx)
        b = _make_df(np.array([4.0, 5.0]), idx)
        result = compute_arithmetic(a, b, "*")
        np.testing.assert_allclose(result.values.squeeze(), [8.0, 15.0])

    def test_divide(self):
        idx = _make_time(2)
        a = _make_df(np.array([10.0, 6.0]), idx)
        b = _make_df(np.array([2.0, 3.0]), idx)
        result = compute_arithmetic(a, b, "/")
        np.testing.assert_allclose(result.values.squeeze(), [5.0, 2.0])

    def test_divide_by_zero_gives_nan(self):
        idx = _make_time(2)
        a = _make_df(np.array([1.0, 2.0]), idx)
        b = _make_df(np.array([0.0, 0.0]), idx)
        result = compute_arithmetic(a, b, "/")
        assert np.all(np.isnan(result.values))

    def test_unknown_operation(self):
        idx = _make_time(3)
        a = _make_df(np.ones(3), idx)
        b = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="Unknown operation"):
            compute_arithmetic(a, b, "^")

    def test_vector_arithmetic(self):
        idx = _make_time(5)
        a = pd.DataFrame(np.ones((5, 3)), index=idx, columns=["x", "y", "z"])
        b = pd.DataFrame(np.ones((5, 3)) * 2, index=idx, columns=["x", "y", "z"])
        result = compute_arithmetic(a, b, "+")
        np.testing.assert_allclose(result.values, np.ones((5, 3)) * 3)


class TestComputeRunningAverage:
    def test_basic(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        smoothed = compute_running_average(df, 3)
        assert len(smoothed) == 5
        # Center point should be exact mean of window
        np.testing.assert_allclose(smoothed.iloc[2, 0], 3.0)

    def test_edges(self):
        idx = _make_time(5)
        df = _make_df(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), idx)
        smoothed = compute_running_average(df, 3)
        # First point: average of [10, 20] = 15
        np.testing.assert_allclose(smoothed.iloc[0, 0], 15.0)
        # Last point: average of [40, 50] = 45
        np.testing.assert_allclose(smoothed.iloc[4, 0], 45.0)

    def test_even_window_forced_odd(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
        smoothed_even = compute_running_average(df, 4)
        smoothed_odd = compute_running_average(df, 5)
        # window=4 should become window=5
        np.testing.assert_allclose(smoothed_even.values, smoothed_odd.values)

    def test_handles_nan(self):
        idx = _make_time(5)
        df = _make_df(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), idx)
        smoothed = compute_running_average(df, 3)
        # Center point at index 1: nanmean([1, nan, 3]) = 2
        np.testing.assert_allclose(smoothed.iloc[1, 0], 2.0)

    def test_window_size_1(self):
        idx = _make_time(3)
        df = _make_df(np.array([10.0, 20.0, 30.0]), idx)
        smoothed = compute_running_average(df, 1)
        np.testing.assert_allclose(smoothed.values.squeeze(), [10.0, 20.0, 30.0])

    def test_multi_column(self):
        idx = _make_time(5)
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=idx,
        )
        smoothed = compute_running_average(df, 3)
        assert smoothed.shape == (5, 2)
        np.testing.assert_allclose(smoothed["a"].iloc[2], 3.0)
        np.testing.assert_allclose(smoothed["b"].iloc[2], 30.0)

    def test_rejects_zero_window(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="window_size"):
            compute_running_average(df, 0)


class TestComputeResample:
    def test_basic_downsample(self):
        # 100 points at 1-second cadence, resample to 10-second bins
        idx = _make_time(100, cadence_s=1)
        df = _make_df(np.arange(100, dtype=np.float64), idx)
        result = compute_resample(df, 10)
        assert len(result) == 10
        # First bin: mean of [0..9] = 4.5
        np.testing.assert_allclose(result.iloc[0, 0], 4.5)

    def test_vector_downsample(self):
        idx = _make_time(20, cadence_s=1)
        df = pd.DataFrame(np.ones((20, 3)), index=idx, columns=["x", "y", "z"])
        result = compute_resample(df, 10)
        assert result.shape[1] == 3

    def test_rejects_zero_cadence(self):
        idx = _make_time(10)
        df = _make_df(np.ones(10), idx)
        with pytest.raises(ValueError, match="cadence_seconds"):
            compute_resample(df, 0)

    def test_rejects_negative_cadence(self):
        idx = _make_time(10)
        df = _make_df(np.ones(10), idx)
        with pytest.raises(ValueError, match="cadence_seconds"):
            compute_resample(df, -5)


class TestComputeDelta:
    def test_difference(self):
        idx = _make_time(5, cadence_s=60)
        df = _make_df(np.array([10.0, 12.0, 15.0, 11.0, 14.0]), idx)
        result = compute_delta(df, "difference")
        assert len(result) == 4
        np.testing.assert_allclose(result.values.squeeze(), [2.0, 3.0, -4.0, 3.0])

    def test_derivative(self):
        idx = _make_time(3, cadence_s=60)
        df = _make_df(np.array([0.0, 60.0, 180.0]), idx)
        result = compute_delta(df, "derivative")
        # dv/dt: (60-0)/60 = 1.0, (180-60)/60 = 2.0
        np.testing.assert_allclose(result.values.squeeze(), [1.0, 2.0])

    def test_vector_difference(self):
        idx = _make_time(3)
        df = pd.DataFrame(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 10.0, 10.0]],
            index=idx, columns=["x", "y", "z"],
        )
        result = compute_delta(df, "difference")
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result.iloc[0].values, [3.0, 3.0, 3.0])

    def test_timestamps_use_later_point(self):
        idx = _make_time(3, cadence_s=100)
        df = _make_df(np.array([0.0, 1.0, 2.0]), idx)
        result = compute_delta(df, "difference")
        # Timestamps should be the later point of each pair
        assert result.index[0] == idx[1]
        assert result.index[1] == idx[2]

    def test_unknown_mode(self):
        idx = _make_time(3)
        df = _make_df(np.ones(3), idx)
        with pytest.raises(ValueError, match="mode"):
            compute_delta(df, "unknown")

    def test_too_few_points(self):
        idx = _make_time(1)
        df = _make_df(np.ones(1), idx)
        with pytest.raises(ValueError, match="at least 2"):
            compute_delta(df, "difference")
