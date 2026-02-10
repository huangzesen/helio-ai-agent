"""
CDF file download backend — fetches data from CDAWeb via direct CDF file download.

Alternative to the HAPI CSV path. Downloads CDF files from CDAWeb's REST API,
caches them locally in cdaweb_data/, and reads parameters using cdflib.

Produces the same output format as fetch_hapi_data() so callers don't need changes.
"""

from pathlib import Path
from urllib.parse import urlparse

import cdflib
import numpy as np
import pandas as pd
import requests

import logging

from knowledge.hapi_client import get_dataset_info

logger = logging.getLogger("helio-agent")

CDAWEB_REST_BASE = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cdaweb_data"


def fetch_cdf_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
) -> dict:
    """Fetch timeseries data by downloading CDF files from CDAWeb.

    Same signature and return format as fetch_hapi_data().

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").
        parameter_id: Parameter name (e.g., "BGSEc").
        time_min: ISO start time (e.g., "2024-01-15T00:00:00Z").
        time_max: ISO end time (e.g., "2024-01-16T00:00:00Z").

    Returns:
        Dict with keys: data (DataFrame), units, description, fill_value.

    Raises:
        ValueError: If no data is available.
        requests.HTTPError: If a download fails.
    """
    # Get metadata from local HAPI cache
    info = get_dataset_info(dataset_id)
    param_meta = _find_parameter_meta(info, parameter_id)

    units = param_meta.get("units", "")
    description = param_meta.get("description", "")
    fill_value = param_meta.get("fill", None)
    param_size = param_meta.get("size", [1])
    if isinstance(param_size, list):
        param_size = param_size[0] if param_size else 1

    # Discover CDF files covering the time range
    file_list = _get_cdf_file_list(dataset_id, time_min, time_max)
    logger.debug(f"[CDF] Found {len(file_list)} files for {dataset_id} "
                 f"({time_min} to {time_max})")

    # Download and read each file
    frames = []
    for file_info in file_list:
        local_path = _download_cdf_file(file_info["url"], CACHE_DIR)
        df = _read_cdf_parameter(local_path, parameter_id, param_size)
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No data extracted for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    # Concatenate and trim to requested time range
    df = pd.concat(frames)
    df.sort_index(inplace=True)

    # Remove duplicates (overlapping files)
    df = df[~df.index.duplicated(keep="first")]

    # Trim to requested time range (strip 'Z' for timezone-naive index)
    t_start = time_min.rstrip("Z")
    t_stop = time_max.rstrip("Z")
    df = df.loc[t_start:t_stop]

    if len(df) == 0:
        raise ValueError(
            f"No data rows for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    # Ensure float64 dtype (CDF often stores float32; match HAPI path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    # Replace fill values with NaN
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            df.replace(fill_f, np.nan, inplace=True)
        except (ValueError, TypeError):
            pass

    logger.debug(f"[CDF] {dataset_id}/{parameter_id}: {len(df)} rows, "
                 f"{len(df.columns)} columns")

    return {
        "data": df,
        "units": units,
        "description": description,
        "fill_value": fill_value,
    }


def _find_parameter_meta(info: dict, parameter_id: str) -> dict:
    """Find metadata for a specific parameter in a HAPI info response."""
    for p in info.get("parameters", []):
        if p.get("name") == parameter_id:
            return p
    available = [p.get("name") for p in info.get("parameters", [])]
    raise ValueError(
        f"Parameter '{parameter_id}' not found. Available: {available}"
    )


def _get_cdf_file_list(
    dataset_id: str, time_min: str, time_max: str
) -> list[dict]:
    """Query CDAWeb REST API for CDF file URLs covering a time range.

    Args:
        dataset_id: CDAWeb dataset ID.
        time_min: ISO start time.
        time_max: ISO end time.

    Returns:
        List of dicts with 'url', 'start_time', 'end_time', 'size' keys.
    """
    # Convert ISO times to CDAWeb format: YYYYMMDDTHHmmSSZ
    start_str = _iso_to_cdaweb_time(time_min)
    stop_str = _iso_to_cdaweb_time(time_max)

    url = (f"{CDAWEB_REST_BASE}/datasets/{dataset_id}"
           f"/orig_data/{start_str},{stop_str}")

    logger.debug(f"[CDF] Querying file list: {url}")
    resp = requests.get(
        url,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()

    # Navigate the response structure
    file_descs = (data.get("FileDescription")
                  or data.get("FileDescriptionList", {}).get("FileDescription")
                  or [])

    if not file_descs:
        raise ValueError(
            f"No CDF files found for {dataset_id} "
            f"in range {time_min} to {time_max}"
        )

    result = []
    for fd in file_descs:
        file_url = fd.get("Name", "")
        if not file_url:
            continue
        result.append({
            "url": file_url,
            "start_time": fd.get("StartTime", ""),
            "end_time": fd.get("EndTime", ""),
            "size": fd.get("Length", 0),
        })

    return result


def _iso_to_cdaweb_time(iso_time: str) -> str:
    """Convert ISO 8601 time string to CDAWeb REST API format.

    '2024-01-15T00:00:00Z' -> '20240115T000000Z'
    """
    # Strip common ISO separators
    t = iso_time.replace("-", "").replace(":", "")
    # Ensure trailing Z
    if not t.endswith("Z"):
        t += "Z"
    return t


def _download_cdf_file(url: str, cache_base: Path) -> Path:
    """Download a CDF file, using local cache if available.

    Preserves CDAWeb directory structure under cache_base.

    Args:
        url: Full URL to the CDF file.
        cache_base: Local directory for cached files.

    Returns:
        Path to the local CDF file.
    """
    # Extract relative path from URL
    parsed = urlparse(url)
    path = parsed.path  # e.g., /sp_phys/data/ace/mag/.../file.cdf

    # Find the part after 'sp_phys/data/'
    marker = "sp_phys/data/"
    idx = path.find(marker)
    if idx >= 0:
        rel_path = path[idx + len(marker):]
    else:
        # Fallback: use filename only
        rel_path = Path(parsed.path).name

    local_path = cache_base / rel_path

    # Skip download if cached
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.debug(f"[CDF] Cache hit: {local_path}")
        return local_path

    # Download
    logger.info(f"[CDF] Downloading: {Path(rel_path).name}")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    local_path.write_bytes(resp.content)
    size_mb = len(resp.content) / (1024 * 1024)
    logger.debug(f"[CDF] Downloaded {size_mb:.1f} MB -> {local_path}")

    return local_path


def _read_cdf_parameter(
    cdf_path: Path, parameter_id: str, param_size: int
) -> pd.DataFrame:
    """Extract one parameter from a CDF file.

    Args:
        cdf_path: Path to local CDF file.
        parameter_id: CDF variable name to read.
        param_size: Expected number of components (1 for scalar, 3 for vector).

    Returns:
        DataFrame with DatetimeIndex named 'time' and integer column names
        (1, 2, 3...) matching HAPI CSV column naming.
    """
    cdf = cdflib.CDF(str(cdf_path))
    info = cdf.cdf_info()

    # Find the epoch variable
    epoch_var = _find_epoch_variable(cdf, info)
    epoch_data = cdf.varget(epoch_var)
    times = cdflib.cdfepoch.to_datetime(epoch_data)

    # Read the parameter data
    try:
        param_data = cdf.varget(parameter_id)
    except Exception as e:
        all_vars = info.zVariables + info.rVariables
        raise ValueError(
            f"Variable '{parameter_id}' not found in {cdf_path.name}. "
            f"Available: {all_vars}"
        ) from e

    # Build DataFrame with integer column names matching HAPI CSV convention
    if param_data.ndim == 1:
        # Scalar parameter
        df = pd.DataFrame({1: param_data}, index=times)
    else:
        # Vector/multi-component parameter
        ncols = param_data.shape[1] if param_data.ndim > 1 else 1
        columns = {i + 1: param_data[:, i] for i in range(ncols)}
        df = pd.DataFrame(columns, index=times)

    df.index.name = "time"
    return df


def _find_epoch_variable(cdf: cdflib.CDF, info) -> str:
    """Find the epoch/time variable in a CDF file.

    Looks for common epoch variable names, then falls back to checking
    variable types.

    Args:
        cdf: Open CDF file object.
        info: CDF info from cdf.cdf_info().

    Returns:
        Name of the epoch variable.

    Raises:
        ValueError: If no epoch variable is found.
    """
    all_vars = info.zVariables + info.rVariables

    # Check common names first
    for name in ["Epoch", "EPOCH", "epoch", "Epoch1"]:
        if name in all_vars:
            return name

    # Fall back: look for CDF epoch data types
    for var_name in all_vars:
        try:
            var_info = cdf.varinq(var_name)
            # CDF epoch types: CDF_EPOCH (31), CDF_EPOCH16 (32), CDF_TIME_TT2000 (33)
            if var_info.Data_Type_Description in (
                "CDF_EPOCH", "CDF_EPOCH16", "CDF_TIME_TT2000"
            ):
                return var_name
        except Exception:
            continue

    raise ValueError(
        f"No epoch variable found in CDF file. Variables: {all_vars}"
    )
