"""
Pandas-based timeseries operations.

All functions take and return pd.DataFrame with DatetimeIndex.
No dependency on DataStore — the agent/core.py layer handles storage.
"""

import numpy as np
import pandas as pd


def compute_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vector magnitude: sqrt(x² + y² + z²).

    NaN in any component propagates to the result.

    Args:
        df: DataFrame with exactly 3 columns (vector components).

    Returns:
        Single-column DataFrame with magnitudes.

    Raises:
        ValueError: If input does not have 3 columns.
    """
    if len(df.columns) != 3:
        raise ValueError(
            f"compute_magnitude requires 3 columns, got {len(df.columns)}"
        )
    mag = df.pow(2).sum(axis=1, skipna=False).pow(0.5)
    return mag.to_frame(name="magnitude")


def compute_arithmetic(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    operation: str,
) -> pd.DataFrame:
    """Element-wise arithmetic between two DataFrames.

    For division, inf values (from divide-by-zero) are replaced with NaN.
    DataFrames are aligned on their index.

    Args:
        df_a: First operand DataFrame.
        df_b: Second operand DataFrame.
        operation: One of "+", "-", "*", "/".

    Returns:
        Result DataFrame with same shape.

    Raises:
        ValueError: If operation is unknown.
    """
    ops = {
        "+": pd.DataFrame.add,
        "-": pd.DataFrame.sub,
        "*": pd.DataFrame.mul,
        "/": pd.DataFrame.div,
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation '{operation}'. Use +, -, *, or /.")
    result = ops[operation](df_a, df_b)
    if operation == "/":
        result = result.replace([np.inf, -np.inf], np.nan)
    return result


def compute_running_average(
    df: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    """Centered moving average using rolling mean (skips NaN).

    Window size is forced to be odd so the result is centered. Partial windows
    at the edges use whatever points are available (min_periods=1).

    Args:
        df: DataFrame with DatetimeIndex and one or more columns.
        window_size: Number of points in the averaging window (forced odd).

    Returns:
        DataFrame with same shape, smoothed values.

    Raises:
        ValueError: If window_size < 1.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if window_size % 2 == 0:
        window_size += 1
    return df.rolling(window_size, center=True, min_periods=1).mean()


def compute_resample(
    df: pd.DataFrame,
    cadence_seconds: float,
) -> pd.DataFrame:
    """Downsample by bin-averaging at fixed cadence.

    Uses pandas resample with the given cadence. Empty bins are dropped.

    Args:
        df: DataFrame with DatetimeIndex.
        cadence_seconds: Bin width in seconds.

    Returns:
        Resampled DataFrame with one entry per non-empty bin.

    Raises:
        ValueError: If cadence_seconds <= 0.
    """
    if cadence_seconds <= 0:
        raise ValueError(f"cadence_seconds must be > 0, got {cadence_seconds}")
    return df.resample(f"{cadence_seconds}s").mean().dropna(how="all")


def compute_delta(
    df: pd.DataFrame,
    mode: str = "difference",
) -> pd.DataFrame:
    """Compute differences or time derivatives.

    Output has n-1 points. Timestamps use the later point of each pair.

    Args:
        df: DataFrame with DatetimeIndex.
        mode: "difference" for Δv, "derivative" for Δv/Δt (units/second).

    Returns:
        DataFrame with n-1 points.

    Raises:
        ValueError: If mode is unknown or DataFrame has fewer than 2 rows.
    """
    if mode not in ("difference", "derivative"):
        raise ValueError(f"mode must be 'difference' or 'derivative', got '{mode}'")
    if len(df) < 2:
        raise ValueError("Need at least 2 points to compute delta")

    result = df.diff().iloc[1:]
    if mode == "derivative":
        dt_s = df.index.to_series().diff().dt.total_seconds().iloc[1:]
        result = result.div(dt_s, axis=0)
    return result
