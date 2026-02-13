"""
CDF file download backend — fetches data from CDAWeb via direct CDF file download.

Downloads CDF files from CDAWeb's REST API, caches them locally in cdaweb_data/,
and reads parameters using cdflib. Multi-day requests download CDF files in
parallel using a thread pool.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import cdflib
import numpy as np
import pandas as pd
import requests
import xarray as xr

import logging

from knowledge.metadata_client import get_dataset_info

logger = logging.getLogger("helio-agent")

from agent.logging import tagged

CDAWEB_REST_BASE = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cdaweb_data"

_WARN_THRESHOLD_BYTES = 500 * 1024 * 1024   # 500 MB
_BLOCK_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GB


# CDF variable data types to skip (epoch/time and character/metadata types)
_EPOCH_TYPES = {"CDF_EPOCH", "CDF_EPOCH16", "CDF_TIME_TT2000"}
_SKIP_TYPES = _EPOCH_TYPES | {"CDF_CHAR", "CDF_UCHAR"}


def list_cdf_variables(dataset_id: str) -> list[dict]:
    """List data variables for a CDAWeb dataset.

    Uses the metadata resolution chain (local cache → Master CDF).
    Note: master CDF metadata may not perfectly match actual data CDF files
    (CDF versions can diverge). If a parameter listed here is missing from
    data files, fetch_cdf_data will return an error with the actual available
    variables so the agent can self-correct.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        List of dicts with keys: name, description, units, size.
    """
    info = get_dataset_info(dataset_id)

    result = []
    for param in info.get("parameters", []):
        if param.get("name", "").lower() == "time":
            continue
        size = param.get("size", [1])
        result.append({
            "name": param["name"],
            "description": param.get("description", ""),
            "units": param.get("units", ""),
            "size": size,
        })

    logger.debug(f"[CDF] Listed {len(result)} data variables for {dataset_id}")
    return result


def fetch_cdf_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
    force: bool = False,
) -> dict:
    """Fetch timeseries data by downloading CDF files from CDAWeb.

    Same signature and return format as fetch_data().

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
    # Get metadata from local cache if available
    cdf_native = False
    info = get_dataset_info(dataset_id)
    try:
        param_meta = _find_parameter_meta(info, parameter_id)
        units = param_meta.get("units", "")
        description = param_meta.get("description", "")
        fill_value = param_meta.get("fill", None)
    except ValueError:
        # Parameter not in metadata cache — it's a CDF-native variable name.
        # We'll extract metadata from the first CDF file below.
        cdf_native = True
        units = ""
        description = ""
        fill_value = None

    # Discover CDF files covering the time range
    file_list = _get_cdf_file_list(dataset_id, time_min, time_max)
    logger.debug(f"[CDF] Found {len(file_list)} files for {dataset_id} "
                 f"({time_min} to {time_max})")

    # Check download size before proceeding
    download_bytes, total_bytes, n_cached, n_to_download = _check_download_size(
        file_list, CACHE_DIR
    )

    if download_bytes > _BLOCK_THRESHOLD_BYTES and not force:
        size_mb = download_bytes / 1e6
        return {
            "status": "confirmation_required",
            "download_mb": round(size_mb),
            "n_files": n_to_download,
            "n_cached": n_cached,
            "dataset_id": dataset_id,
            "message": (
                f"This request requires downloading {size_mb:.0f} MB "
                f"({n_to_download} files) from CDAWeb. "
                f"Do you want to proceed?"
            ),
        }

    if download_bytes > _WARN_THRESHOLD_BYTES:
        logger.warning(
            f"[CDF] Large download: {download_bytes / 1e6:.0f} MB ({n_to_download} files) for {dataset_id}. "
            f"Consider narrowing the time range.",
            extra=tagged("progress"),
        )

    # Download and read CDF files (parallel when enabled and multiple files)
    from config import PARALLEL_FETCH, PARALLEL_MAX_WORKERS
    use_parallel = PARALLEL_FETCH and len(file_list) > 1
    max_workers = min(len(file_list), PARALLEL_MAX_WORKERS, 6)
    frames = []
    validmin = None
    validmax = None

    if use_parallel:
        logger.info(f"[CDF] Downloading {len(file_list)} files in parallel "
                    f"(max_workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_and_read, fi["url"], parameter_id, CACHE_DIR): idx
                for idx, fi in enumerate(file_list)
            }
            results_by_idx: dict[int, tuple[Path, pd.DataFrame | xr.DataArray]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                results_by_idx[idx] = future.result()
    else:
        results_by_idx = {}
        for idx, fi in enumerate(file_list):
            results_by_idx[idx] = _download_and_read(fi["url"], parameter_id, CACHE_DIR)

    for idx in range(len(file_list)):
        local_path, data = results_by_idx[idx]
        # Extract metadata from the first CDF file.
        # Always read FILLVAL/VALIDMIN/VALIDMAX from CDF (ground truth)
        # since cached fill values may have different precision (float32 vs float64).
        if not frames:
            # Sync cached metadata with actual data CDF variables.
            # Runs once per dataset — cheap since file is already downloaded.
            _sync_metadata_with_data_cdf(dataset_id, local_path)

            try:
                cdf = cdflib.CDF(str(local_path))
                attrs = cdf.varattsget(parameter_id)
                if cdf_native:
                    units = attrs.get("UNITS", "") or ""
                    if isinstance(units, np.ndarray):
                        units = str(units)
                    description = (attrs.get("CATDESC", "")
                                   or attrs.get("FIELDNAM", "") or "")
                    if isinstance(description, np.ndarray):
                        description = str(description)
                fv = attrs.get("FILLVAL", None)
                if fv is not None:
                    try:
                        fill_value = float(fv)
                    except (ValueError, TypeError):
                        pass
                vmin = attrs.get("VALIDMIN", None)
                vmax = attrs.get("VALIDMAX", None)
                if vmin is not None:
                    try:
                        validmin = float(vmin)
                    except (ValueError, TypeError):
                        pass
                if vmax is not None:
                    try:
                        validmax = float(vmax)
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        frames.append(data)

    if not frames:
        raise ValueError(
            f"No data extracted for {dataset_id}/{parameter_id} "
            f"in range {time_min} to {time_max}"
        )

    # Branch: xarray DataArray (3D+) vs pandas DataFrame (1D/2D)
    is_xarray = isinstance(frames[0], xr.DataArray)

    if is_xarray:
        data = _postprocess_xarray(frames, parameter_id, time_min, time_max,
                                   fill_value, validmin, validmax)
    else:
        data = _postprocess_dataframe(frames, time_min, time_max,
                                      fill_value, validmin, validmax)

    n_time = data.sizes["time"] if is_xarray else len(data)
    shape_info = (f"{dict(data.sizes)}" if is_xarray
                  else f"{len(data)} rows, {len(data.columns)} columns")
    logger.debug(f"[CDF] {dataset_id}/{parameter_id}: {shape_info}")

    return {
        "data": data,
        "units": units,
        "description": description,
        "fill_value": fill_value,
    }


def _postprocess_dataframe(
    frames: list[pd.DataFrame],
    time_min: str,
    time_max: str,
    fill_value: float | None,
    validmin: float | None,
    validmax: float | None,
) -> pd.DataFrame:
    """Concatenate, clean, and trim DataFrame results from CDF files."""
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
            f"No data rows in range {time_min} to {time_max}"
        )

    # Ensure float64 dtype (CDF often stores float32)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    # Replace fill values with NaN.
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            for col in df.columns:
                mask = np.isclose(df[col].values, fill_f, rtol=1e-6,
                                  equal_nan=False)
                df.loc[mask, col] = np.nan
        except (ValueError, TypeError):
            pass

    # Replace out-of-range values with NaN using CDF VALIDMIN/VALIDMAX.
    if validmin is not None or validmax is not None:
        for col in df.columns:
            if validmin is not None:
                df.loc[df[col] < validmin, col] = np.nan
            if validmax is not None:
                df.loc[df[col] > validmax, col] = np.nan

    return df


def _postprocess_xarray(
    frames: list[xr.DataArray],
    parameter_id: str,
    time_min: str,
    time_max: str,
    fill_value: float | None,
    validmin: float | None,
    validmax: float | None,
) -> xr.DataArray:
    """Concatenate, clean, and trim xarray DataArray results from CDF files."""
    da = xr.concat(frames, dim="time")
    da = da.sortby("time")

    # Remove duplicate times
    _, unique_idx = np.unique(da.coords["time"].values, return_index=True)
    da = da.isel(time=unique_idx)

    # Trim to requested time range
    t_start = np.datetime64(time_min.rstrip("Z"))
    t_stop = np.datetime64(time_max.rstrip("Z"))
    da = da.sel(time=slice(t_start, t_stop))

    if da.sizes["time"] == 0:
        raise ValueError(
            f"No data rows for {parameter_id} in range {time_min} to {time_max}"
        )

    # Ensure float64 dtype
    da = da.astype(np.float64)

    # Replace fill values with NaN
    if fill_value is not None:
        try:
            fill_f = float(fill_value)
            da = da.where(~np.isclose(da.values, fill_f, rtol=1e-6, equal_nan=False))
        except (ValueError, TypeError):
            pass

    # Replace out-of-range values with NaN
    if validmin is not None:
        da = da.where(da >= validmin)
    if validmax is not None:
        da = da.where(da <= validmax)

    da.name = parameter_id
    return da


def _download_and_read(
    url: str, parameter_id: str, cache_dir: Path
) -> tuple[Path, pd.DataFrame | xr.DataArray]:
    """Download a CDF file and read one parameter. Thread-safe."""
    local_path = _download_cdf_file(url, cache_dir)
    data = _read_cdf_parameter(local_path, parameter_id)
    return local_path, data


def _sync_metadata_with_data_cdf(dataset_id: str, cdf_path: Path) -> None:
    """Compare data CDF variables against cached metadata and update if needed.

    Called once per fetch on the first data CDF file.  If the data CDF contains
    variables not in the cached metadata (or vice versa), updates the local
    metadata JSON with ``_validated`` annotations so discrepancies are visible
    to the LLM and future fetches.

    This is cheap — the CDF is already downloaded, we just read its variable list.
    """
    from knowledge.metadata_client import _find_local_cache
    import json as _json

    cache_path = _find_local_cache(dataset_id)
    if cache_path is None:
        logger.debug(f"[CDF] Metadata sync skipped for {dataset_id}: no local cache")
        return

    try:
        cached_info = _json.loads(cache_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.debug(f"[CDF] Metadata sync skipped for {dataset_id}: "
                     f"cache read failed: {exc}")
        return

    # Already validated — skip
    if cached_info.get("_validated"):
        logger.debug(f"[CDF] Metadata sync skipped for {dataset_id}: already validated")
        return

    logger.debug(f"[CDF] Metadata sync: comparing {dataset_id} against {cdf_path.name}")

    try:
        data_cdf = cdflib.CDF(str(cdf_path))
        data_info = data_cdf.cdf_info()
        data_vars = set(data_info.zVariables) | set(data_info.rVariables)
    except Exception as exc:
        logger.warning(f"[CDF] Metadata sync failed for {dataset_id}: "
                       f"could not read data CDF: {exc}")
        return

    cached_names = {
        p.get("name") for p in cached_info.get("parameters", [])
        if p.get("name", "").lower() != "time"
    }

    data_only = data_vars - cached_names
    master_only = cached_names - data_vars

    if not data_only and not master_only:
        logger.debug(f"[CDF] Metadata sync for {dataset_id}: perfect match "
                     f"({len(cached_names)} variables)")
        cached_info["_validated"] = True
        try:
            cache_path.write_text(
                _json.dumps(cached_info, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return

    # Log discrepancies at WARNING so they're visible in normal mode
    if master_only:
        logger.warning(
            f"[CDF] Metadata sync for {dataset_id}: {len(master_only)} vars in "
            f"master CDF but NOT in data CDF: {', '.join(sorted(master_only))}"
        )
    if data_only:
        logger.warning(
            f"[CDF] Metadata sync for {dataset_id}: {len(data_only)} vars in "
            f"data CDF but NOT in master CDF: {', '.join(sorted(data_only))}"
        )

    # Annotate existing parameters
    for param in cached_info.get("parameters", []):
        name = param.get("name", "")
        if name.lower() == "time":
            continue
        if name in master_only:
            param["_note"] = "in master CDF but not found in data CDF"
        elif name in data_vars:
            # Exists in both — confirmed
            pass

    # Add data-only variables to cached metadata
    added_count = 0
    for var_name in sorted(data_only):
        # Skip epoch/time and metadata types
        try:
            var_inq = data_cdf.varinq(var_name)
            if var_inq.Data_Type_Description in _SKIP_TYPES:
                continue
        except Exception:
            continue
        cached_info.setdefault("parameters", []).append({
            "name": var_name,
            "type": "",
            "units": "",
            "description": "",
            "_note": "found in data CDF but not in master CDF",
        })
        added_count += 1

    cached_info["_validated"] = True

    logger.info(
        f"[CDF] Metadata sync for {dataset_id}: updated cache — "
        f"{len(master_only)} master-only annotated, "
        f"{added_count} data-only added, "
        f"{len(cached_names & data_vars)} confirmed"
    )

    # Invalidate in-memory cache so next get_dataset_info() picks up changes
    from knowledge.metadata_client import _info_cache
    _info_cache.pop(dataset_id, None)

    try:
        cache_path.write_text(
            _json.dumps(cached_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.debug(f"[CDF] Metadata cache updated: {cache_path}")
    except OSError as exc:
        logger.warning(f"[CDF] Failed to write updated metadata cache "
                       f"for {dataset_id}: {exc}")


def _find_parameter_meta(info: dict, parameter_id: str) -> dict:
    """Find metadata for a specific parameter in metadata info."""
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


def _url_to_local_path(url: str, cache_base: Path) -> Path:
    """Resolve a CDAWeb CDF URL to its local cache path.

    Args:
        url: Full URL to the CDF file.
        cache_base: Local directory for cached files.

    Returns:
        Path where the file would be cached locally.
    """
    parsed = urlparse(url)
    path = parsed.path  # e.g., /sp_phys/data/ace/mag/.../file.cdf

    marker = "sp_phys/data/"
    idx = path.find(marker)
    if idx >= 0:
        rel_path = path[idx + len(marker):]
    else:
        rel_path = Path(parsed.path).name

    return cache_base / rel_path


def _check_download_size(
    file_list: list[dict], cache_dir: Path
) -> tuple[int, int, int, int]:
    """Calculate download size, excluding cached files.

    Args:
        file_list: List of dicts with 'url' and 'size' keys from _get_cdf_file_list().
        cache_dir: Local cache directory.

    Returns:
        Tuple of (download_bytes, total_bytes, n_cached, n_to_download).
    """
    download_bytes = 0
    total_bytes = 0
    n_cached = 0
    n_to_download = 0

    for fi in file_list:
        size = fi.get("size", 0)
        total_bytes += size
        local_path = _url_to_local_path(fi["url"], cache_dir)
        if local_path.exists() and local_path.stat().st_size > 0:
            n_cached += 1
        else:
            download_bytes += size
            n_to_download += 1

    return download_bytes, total_bytes, n_cached, n_to_download


def _download_cdf_file(url: str, cache_base: Path) -> Path:
    """Download a CDF file, using local cache if available.

    Preserves CDAWeb directory structure under cache_base.

    Args:
        url: Full URL to the CDF file.
        cache_base: Local directory for cached files.

    Returns:
        Path to the local CDF file.
    """
    local_path = _url_to_local_path(url, cache_base)

    # Skip download if cached
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.debug(f"[CDF] Cache hit: {local_path}")
        return local_path

    # Download
    logger.info(f"[CDF] Downloading: {local_path.name}")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    local_path.write_bytes(resp.content)
    size_mb = len(resp.content) / (1024 * 1024)
    logger.debug(f"[CDF] Downloaded {size_mb:.1f} MB -> {local_path}")

    return local_path


def _read_cdf_parameter(
    cdf_path: Path, parameter_id: str
) -> pd.DataFrame | xr.DataArray:
    """Extract one parameter from a CDF file.

    Args:
        cdf_path: Path to local CDF file.
        parameter_id: CDF variable name to read.

    Returns:
        DataFrame with DatetimeIndex for 1D/2D data, or xarray DataArray
        for 3D+ data.
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

    # Build DataFrame with integer column names (1D/2D) or DataArray (3D+)
    if param_data.ndim == 1:
        # Scalar parameter
        df = pd.DataFrame({1: param_data}, index=times)
        df.index.name = "time"
        return df
    elif param_data.ndim == 2:
        # Vector/multi-component parameter
        ncols = param_data.shape[1]
        columns = {i + 1: param_data[:, i] for i in range(ncols)}
        df = pd.DataFrame(columns, index=times)
        df.index.name = "time"
        return df
    else:
        # 3D+ variable — return xarray DataArray
        dims = ["time"] + [f"dim{i}" for i in range(1, param_data.ndim)]
        coords = {"time": times}
        da = xr.DataArray(param_data, dims=dims, coords=coords)
        da.name = parameter_id
        return da


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
