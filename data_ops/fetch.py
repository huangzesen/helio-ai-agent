"""
HAPI data fetcher — pulls timeseries from CDAWeb into pandas DataFrames.

Reuses HAPI_BASE and get_dataset_info() from knowledge/hapi_client.py
for metadata (units, size, fill value). Fetches actual data via the
HAPI /data endpoint in CSV format.
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
    df.index.name = "time"

    if len(df) == 0:
        raise ValueError(
            f"No data rows parsed for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    # Ensure float64 dtype
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace fill values with NaN
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            df = df.replace(fill_f, np.nan)
        except (ValueError, TypeError):
            pass

    return {
        "data": df,
        "units": units,
        "description": description,
        "fill_value": fill_value,
    }


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
