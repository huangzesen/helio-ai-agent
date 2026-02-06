"""
Tests for data_ops.operations — pure numpy timeseries operations.

Run with: python -m pytest tests/test_operations.py
"""

import numpy as np
import pytest

from data_ops.operations import (
    compute_magnitude,
    compute_arithmetic,
    compute_running_average,
    compute_resample,
    compute_delta,
)


def _make_time(n=100, start="2024-01-01", cadence_s=60):
    """Create a datetime64[ns] array with fixed cadence."""
    t0 = np.datetime64(start, "ns")
    return t0 + np.arange(n, dtype=np.int64) * np.timedelta64(cadence_s, "s")


class TestComputeMagnitude:
    def test_basic(self):
        v = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        result = compute_magnitude(v)
        np.testing.assert_allclose(result, [5.0, 5.0])

    def test_nan_propagates(self):
        v = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
        result = compute_magnitude(v)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], np.sqrt(14))

    def test_wrong_shape_1d(self):
        with pytest.raises(ValueError, match="requires.*\\(n, 3\\)"):
            compute_magnitude(np.array([1.0, 2.0, 3.0]))

    def test_wrong_shape_2d_wrong_cols(self):
        with pytest.raises(ValueError, match="requires.*\\(n, 3\\)"):
            compute_magnitude(np.ones((5, 2)))


class TestComputeArithmetic:
    def test_add(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(compute_arithmetic(a, b, "+"), [11.0, 22.0, 33.0])

    def test_subtract(self):
        a = np.array([10.0, 20.0])
        b = np.array([3.0, 5.0])
        np.testing.assert_allclose(compute_arithmetic(a, b, "-"), [7.0, 15.0])

    def test_multiply(self):
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        np.testing.assert_allclose(compute_arithmetic(a, b, "*"), [8.0, 15.0])

    def test_divide(self):
        a = np.array([10.0, 6.0])
        b = np.array([2.0, 3.0])
        np.testing.assert_allclose(compute_arithmetic(a, b, "/"), [5.0, 2.0])

    def test_divide_by_zero_gives_nan(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        result = compute_arithmetic(a, b, "/")
        assert np.all(np.isnan(result))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_arithmetic(np.ones(3), np.ones(4), "+")

    def test_unknown_operation(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            compute_arithmetic(np.ones(3), np.ones(3), "^")

    def test_vector_arithmetic(self):
        a = np.ones((5, 3))
        b = np.ones((5, 3)) * 2
        result = compute_arithmetic(a, b, "+")
        np.testing.assert_allclose(result, np.ones((5, 3)) * 3)


class TestComputeRunningAverage:
    def test_basic(self):
        time = _make_time(5)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t_out, smoothed = compute_running_average(time, values, 3)
        assert len(smoothed) == 5
        # Center point should be exact mean of window
        np.testing.assert_allclose(smoothed[2], 3.0)

    def test_edges(self):
        time = _make_time(5)
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        _, smoothed = compute_running_average(time, values, 3)
        # First point: average of [10, 20] = 15
        np.testing.assert_allclose(smoothed[0], 15.0)
        # Last point: average of [40, 50] = 45
        np.testing.assert_allclose(smoothed[4], 45.0)

    def test_even_window_forced_odd(self):
        time = _make_time(5)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, smoothed_even = compute_running_average(time, values, 4)
        _, smoothed_odd = compute_running_average(time, values, 5)
        # window=4 should become window=5
        np.testing.assert_allclose(smoothed_even, smoothed_odd)

    def test_handles_nan(self):
        time = _make_time(5)
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        _, smoothed = compute_running_average(time, values, 3)
        # Center point at index 1: nanmean([1, nan, 3]) = 2
        np.testing.assert_allclose(smoothed[1], 2.0)

    def test_window_size_1(self):
        time = _make_time(3)
        values = np.array([10.0, 20.0, 30.0])
        _, smoothed = compute_running_average(time, values, 1)
        np.testing.assert_allclose(smoothed, values)

    def test_rejects_2d(self):
        time = _make_time(3)
        values = np.ones((3, 3))
        with pytest.raises(ValueError, match="1D"):
            compute_running_average(time, values, 3)

    def test_rejects_zero_window(self):
        time = _make_time(3)
        values = np.ones(3)
        with pytest.raises(ValueError, match="window_size"):
            compute_running_average(time, values, 0)


class TestComputeResample:
    def test_basic_downsample(self):
        # 100 points at 1-second cadence, resample to 10-second bins
        time = _make_time(100, cadence_s=1)
        values = np.arange(100, dtype=np.float64)
        new_time, new_values = compute_resample(time, values, 10)
        assert len(new_time) == 10
        # First bin: mean of [0..9] = 4.5
        np.testing.assert_allclose(new_values[0], 4.5)

    def test_vector_downsample(self):
        time = _make_time(20, cadence_s=1)
        values = np.ones((20, 3))
        new_time, new_values = compute_resample(time, values, 10)
        assert new_values.ndim == 2
        assert new_values.shape[1] == 3

    def test_rejects_zero_cadence(self):
        with pytest.raises(ValueError, match="cadence_seconds"):
            compute_resample(_make_time(10), np.ones(10), 0)

    def test_rejects_negative_cadence(self):
        with pytest.raises(ValueError, match="cadence_seconds"):
            compute_resample(_make_time(10), np.ones(10), -5)


class TestComputeDelta:
    def test_difference(self):
        time = _make_time(5, cadence_s=60)
        values = np.array([10.0, 12.0, 15.0, 11.0, 14.0])
        mid_time, dv = compute_delta(time, values, "difference")
        assert len(mid_time) == 4
        np.testing.assert_allclose(dv, [2.0, 3.0, -4.0, 3.0])

    def test_derivative(self):
        time = _make_time(3, cadence_s=60)
        values = np.array([0.0, 60.0, 180.0])
        mid_time, dv = compute_delta(time, values, "derivative")
        # dv/dt: (60-0)/60 = 1.0, (180-60)/60 = 2.0
        np.testing.assert_allclose(dv, [1.0, 2.0])

    def test_vector_difference(self):
        time = _make_time(3)
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 10.0, 10.0]])
        _, dv = compute_delta(time, values, "difference")
        assert dv.shape == (2, 3)
        np.testing.assert_allclose(dv[0], [3.0, 3.0, 3.0])

    def test_midpoint_timestamps(self):
        time = _make_time(3, cadence_s=100)
        values = np.array([0.0, 1.0, 2.0])
        mid_time, _ = compute_delta(time, values, "difference")
        # Midpoint between first two timestamps
        expected_mid = time[0] + np.timedelta64(50, "s")
        assert mid_time[0] == expected_mid

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            compute_delta(_make_time(3), np.ones(3), "unknown")

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_delta(_make_time(1), np.ones(1), "difference")
