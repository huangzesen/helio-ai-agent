"""
Data fetcher — pulls timeseries from CDAWeb into pandas DataFrames.

Supports two backends:
  - HAPI CSV (default) via CDAWeb's /hapi/data endpoint
  - CDF file download via CDAWeb's REST API + cdflib

The active backend is controlled by config.DATA_BACKEND ("hapi" or "cdf").
On startup, check_hapi_status() can probe HAPI availability and trigger
automatic fallback to the CDF backend.
"""

import io

import numpy as np
import pandas as pd
import requests

from knowledge.hapi_client import HAPI_BASE, get_dataset_info


def fetch_hapi_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
) -> dict:
    """Fetch timeseries data from the CDAWeb HAPI /data endpoint.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").
        parameter_id: Parameter name (e.g., "BGSEc").
        time_min: ISO start time (e.g., "2024-01-15T00:00:00Z").
        time_max: ISO end time (e.g., "2024-01-16T00:00:00Z").

    Returns:
        Dict with keys:
            data: pd.DataFrame with DatetimeIndex and float64 columns
            units: str
            description: str
            fill_value: original fill value (for reference)

    Raises:
        requests.HTTPError: If the HAPI request fails.
        ValueError: If the response contains no data.
    """
    # Get metadata for units, fill value, and size
    info = get_dataset_info(dataset_id)
    param_meta = _find_parameter_meta(info, parameter_id)

    units = param_meta.get("units", "")
    description = param_meta.get("description", "")
    fill_value = param_meta.get("fill", None)

    # Fetch CSV data
    resp = requests.get(
        f"{HAPI_BASE}/data",
        params={
            "id": dataset_id,
            "parameters": parameter_id,
            "time.min": time_min,
            "time.max": time_max,
            "format": "csv",
        },
        timeout=120,
    )
    resp.raise_for_status()

    text = resp.text.strip()
    if not text:
        raise ValueError(
            f"No data returned for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    # HAPI returns JSON (not CSV) when there's no data or an error,
    # even with format=csv and HTTP 200. Detect and handle this.
    if text.startswith("{"):
        import json
        try:
            hapi_resp = json.loads(text)
            status = hapi_resp.get("status", {})
            msg = status.get("message", "Unknown HAPI error")
            code = status.get("code", 0)
        except json.JSONDecodeError:
            msg = "Unexpected non-CSV response from HAPI server"
            code = 0
        raise ValueError(
            f"No data for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}: {msg} (HAPI {code})"
        )

    # Parse CSV with pandas: column 0 = ISO timestamp, columns 1+ = data
    df = pd.read_csv(
        io.StringIO(text),
        header=None,
        index_col=0,
        parse_dates=[0],
    )
    del text  # free raw response text early
    df.index.name = "time"

    if len(df) == 0:
        raise ValueError(
            f"No data rows parsed for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    # Ensure float64 dtype (in-place, column by column)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace fill values with NaN (in-place)
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            df.replace(fill_f, np.nan, inplace=True)
        except (ValueError, TypeError):
            pass

    return {
        "data": df,
        "units": units,
        "description": description,
        "fill_value": fill_value,
    }


def check_hapi_status(timeout: float = 2) -> bool:
    """Check whether the CDAWeb HAPI service is reachable.

    Sends a lightweight GET to the /hapi/capabilities endpoint.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        True if HAPI responded with HTTP 200, False otherwise.
    """
    try:
        resp = requests.get(f"{HAPI_BASE}/capabilities", timeout=timeout)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout, requests.RequestException):
        return False


def fetch_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
) -> dict:
    """Route data fetching to the configured backend (HAPI or CDF).

    Reads DATA_BACKEND from config to decide which path to use.
    Same return format regardless of backend.
    """
    import config
    if config.DATA_BACKEND == "cdf":
        from data_ops.fetch_cdf import fetch_cdf_data
        return fetch_cdf_data(dataset_id, parameter_id, time_min, time_max)
    return fetch_hapi_data(dataset_id, parameter_id, time_min, time_max)


def _find_parameter_meta(info: dict, parameter_id: str) -> dict:
    """Find metadata for a specific parameter in a HAPI info response.

    Args:
        info: Full HAPI /info response dict.
        parameter_id: Parameter name to find.

    Returns:
        Parameter metadata dict.

    Raises:
        ValueError: If the parameter is not found.
    """
    for p in info.get("parameters", []):
        if p.get("name") == parameter_id:
            return p
    available = [p.get("name") for p in info.get("parameters", [])]
    raise ValueError(
        f"Parameter '{parameter_id}' not found. Available: {available}"
    )
