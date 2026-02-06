"""
Pure numpy timeseries operations.

All functions take numpy arrays as input and return numpy arrays.
No dependency on DataStore — the agent/core.py layer handles storage.
"""

import numpy as np


def compute_magnitude(values: np.ndarray) -> np.ndarray:
    """Compute vector magnitude: sqrt(x² + y² + z²).

    Uses np.sum (not np.nansum) so NaN in any component propagates to the result.

    Args:
        values: Array of shape (n, 3) — vector components.

    Returns:
        1D array of shape (n,) — magnitudes.

    Raises:
        ValueError: If input is not a 2D array with 3 columns.
    """
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"compute_magnitude requires (n, 3) array, got shape {values.shape}"
        )
    return np.sqrt(np.sum(values ** 2, axis=1))


def compute_arithmetic(
    values_a: np.ndarray,
    values_b: np.ndarray,
    operation: str,
) -> np.ndarray:
    """Element-wise arithmetic between two same-shape arrays.

    For division, inf values (from divide-by-zero) are replaced with np.nan.

    Args:
        values_a: First operand array.
        values_b: Second operand array (must match shape of values_a).
        operation: One of "+", "-", "*", "/".

    Returns:
        Result array with same shape as inputs.

    Raises:
        ValueError: If shapes don't match or operation is unknown.
    """
    if values_a.shape != values_b.shape:
        raise ValueError(
            f"Shape mismatch: {values_a.shape} vs {values_b.shape}. "
            "Use compute_resample to align time series first."
        )

    if operation == "+":
        return values_a + values_b
    elif operation == "-":
        return values_a - values_b
    elif operation == "*":
        return values_a * values_b
    elif operation == "/":
        with np.errstate(divide="ignore", invalid="ignore"):
            result = values_a / values_b
        result[~np.isfinite(result)] = np.nan
        return result
    else:
        raise ValueError(f"Unknown operation '{operation}'. Use +, -, *, or /.")


def compute_running_average(
    time: np.ndarray,
    values: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered moving average using np.nanmean to skip gaps.

    Window size is forced to be odd so the result is centered. Partial windows
    at the edges use whatever points are available.

    Args:
        time: 1D datetime64 array of shape (n,).
        values: 1D scalar array of shape (n,).
        window_size: Number of points in the averaging window (forced odd).

    Returns:
        Tuple of (time, smoothed_values) with same length as input.

    Raises:
        ValueError: If values is not 1D or window_size < 1.
    """
    if values.ndim != 1:
        raise ValueError(
            f"compute_running_average requires 1D (scalar) array, got shape {values.shape}"
        )
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    # Force odd window
    if window_size % 2 == 0:
        window_size += 1

    n = len(values)
    half = window_size // 2
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.nanmean(values[lo:hi])

    return time.copy(), result


def compute_resample(
    time: np.ndarray,
    values: np.ndarray,
    cadence_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample by bin-averaging at fixed cadence.

    Bins are aligned to the start of the data. Bin-center timestamps are used.
    Empty bins (no data points) are skipped in the output.

    Works on both scalar (n,) and vector (n,k) data.

    Args:
        time: 1D datetime64[ns] array.
        values: Scalar (n,) or vector (n, k) array.
        cadence_seconds: Bin width in seconds.

    Returns:
        Tuple of (new_time, new_values) with one entry per non-empty bin.

    Raises:
        ValueError: If cadence_seconds <= 0.
    """
    if cadence_seconds <= 0:
        raise ValueError(f"cadence_seconds must be > 0, got {cadence_seconds}")

    # Convert time to float seconds from epoch for binning
    t0 = time[0]
    dt_ns = (time - t0).astype(np.float64)  # nanoseconds
    dt_s = dt_ns / 1e9  # seconds

    cadence_ns = np.timedelta64(int(cadence_seconds * 1e9), "ns")

    # Compute bin indices
    bin_indices = (dt_s / cadence_seconds).astype(np.int64)
    unique_bins = np.unique(bin_indices)

    is_vector = values.ndim == 2

    new_times = []
    new_values = []

    for b in unique_bins:
        mask = bin_indices == b
        count = np.sum(mask)
        if count == 0:
            continue

        # Bin-center timestamp
        bin_start = t0 + np.timedelta64(int(b * cadence_seconds * 1e9), "ns")
        bin_center = bin_start + cadence_ns // 2
        new_times.append(bin_center)

        if is_vector:
            new_values.append(np.nanmean(values[mask], axis=0))
        else:
            new_values.append(np.nanmean(values[mask]))

    new_time = np.array(new_times, dtype="datetime64[ns]")
    if is_vector:
        new_vals = np.array(new_values, dtype=np.float64)
    else:
        new_vals = np.array(new_values, dtype=np.float64)

    return new_time, new_vals


def compute_delta(
    time: np.ndarray,
    values: np.ndarray,
    mode: str = "difference",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute differences or time derivatives.

    Output has n-1 points. Timestamps are placed at midpoints between
    consecutive original timestamps.

    Args:
        time: 1D datetime64[ns] array.
        values: Scalar (n,) or vector (n, k) array.
        mode: "difference" for Δv, "derivative" for Δv/Δt (units/second).

    Returns:
        Tuple of (mid_time, delta_values) with n-1 points.

    Raises:
        ValueError: If mode is unknown or array has fewer than 2 points.
    """
    if mode not in ("difference", "derivative"):
        raise ValueError(f"mode must be 'difference' or 'derivative', got '{mode}'")
    if len(time) < 2:
        raise ValueError("Need at least 2 points to compute delta")

    # Midpoint timestamps
    half_dt = (time[1:] - time[:-1]) // 2
    mid_time = time[:-1] + half_dt

    # Value differences
    dv = np.diff(values, axis=0).astype(np.float64)

    if mode == "derivative":
        dt_seconds = (time[1:] - time[:-1]).astype(np.float64) / 1e9
        # Guard against zero dt
        dt_seconds[dt_seconds == 0] = np.nan
        if dv.ndim == 2:
            dv = dv / dt_seconds[:, np.newaxis]
        else:
            dv = dv / dt_seconds

    return mid_time, dv
