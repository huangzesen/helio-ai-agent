"""
HAPI data fetcher — pulls timeseries from CDAWeb into numpy arrays.

Reuses HAPI_BASE and get_dataset_info() from knowledge/hapi_client.py
for metadata (units, size, fill value). Fetches actual data via the
HAPI /data endpoint in CSV format.
"""

import io

import numpy as np
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
            time: np.ndarray of datetime64[ns]
            values: np.ndarray — (n,) for scalars, (n,k) for vectors
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
    size = param_meta.get("size", None)
    if size is None:
        size = [1]
    elif isinstance(size, int):
        size = [size]
    num_columns = size[0]

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

    # Parse CSV: column 0 = ISO timestamp, columns 1+ = data
    times = []
    data_rows = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        times.append(parts[0].strip())
        row = [float(v.strip().strip('"')) for v in parts[1:]]
        data_rows.append(row)

    if not times:
        raise ValueError(
            f"No data rows parsed for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    time_arr = np.array(times, dtype="datetime64[ns]")
    values_arr = np.array(data_rows, dtype=np.float64)

    # Replace fill values with NaN
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            values_arr[values_arr == fill_f] = np.nan
        except (ValueError, TypeError):
            pass

    # Squeeze scalar params from (n, 1) to (n,)
    if num_columns == 1 and values_arr.ndim == 2 and values_arr.shape[1] == 1:
        values_arr = values_arr.squeeze(axis=1)

    return {
        "time": time_arr,
        "values": values_arr,
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
