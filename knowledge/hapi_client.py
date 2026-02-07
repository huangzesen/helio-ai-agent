"""
HAPI client for CDAWeb parameter metadata discovery.

Fetches parameter info dynamically from the CDAWeb HAPI server and filters
to 1D plottable parameters (scalars and small vectors with size <= 3).

Supports local file cache: if a dataset's HAPI /info response is saved in
knowledge/missions/{mission}/hapi/{dataset_id}.json, it is loaded instantly
without a network request.
"""

import fnmatch
import json
import requests
from pathlib import Path
from typing import Optional

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi"

# Cache for HAPI responses to avoid repeated API calls
_info_cache: dict[str, dict] = {}

# Directory containing per-mission folders with HAPI cache files
_MISSIONS_DIR = Path(__file__).parent / "missions"


def _find_local_cache(dataset_id: str) -> Optional[Path]:
    """Scan mission subfolders for a locally cached HAPI /info file.

    Checks knowledge/missions/*/hapi/{dataset_id}.json across all mission
    directories. Only 8 dirs to scan — negligible cost.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "PSP_FLD_L2_MAG_RTN_1MIN")

    Returns:
        Path to local cache file, or None if not found.
    """
    for mission_dir in _MISSIONS_DIR.iterdir():
        if not mission_dir.is_dir():
            continue
        cache_file = mission_dir / "hapi" / f"{dataset_id}.json"
        if cache_file.exists():
            return cache_file
    return None


def get_dataset_info(dataset_id: str, use_cache: bool = True) -> dict:
    """Fetch parameter metadata from HAPI /info endpoint.

    Checks three sources in order:
    1. In-memory cache (fastest)
    2. Local file cache in knowledge/missions/*/hapi/ (instant, no network)
    3. Network request to CDAWeb HAPI server (fallback)

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "PSP_FLD_L2_MAG_RTN_1MIN")
        use_cache: Whether to use cached results (in-memory and local file)

    Returns:
        HAPI info response with startDate, stopDate, parameters, etc.

    Raises:
        requests.HTTPError: If the HAPI request fails and no cache is available.
    """
    # 1. In-memory cache
    if use_cache and dataset_id in _info_cache:
        return _info_cache[dataset_id]

    # 2. Local file cache
    if use_cache:
        local_path = _find_local_cache(dataset_id)
        if local_path is not None:
            info = json.loads(local_path.read_text(encoding="utf-8"))
            _info_cache[dataset_id] = info
            return info

    # 3. Network fallback
    resp = requests.get(
        f"{HAPI_BASE}/info",
        params={"id": dataset_id},
        timeout=30,
    )
    resp.raise_for_status()
    info = resp.json()

    if use_cache:
        _info_cache[dataset_id] = info

    return info


def list_parameters(dataset_id: str) -> list[dict]:
    """List plottable 1D parameters for a dataset.

    Fetches metadata from HAPI and filters to parameters that are:
    - Not the Time parameter
    - Numeric type (double or integer)
    - 1D with size <= 3 (scalars and small vectors)

    Args:
        dataset_id: CDAWeb dataset ID

    Returns:
        List of parameter dicts with name, description, units, size, dataset_id.
        Returns empty list if HAPI request fails.
    """
    try:
        info = get_dataset_info(dataset_id)
    except requests.RequestException as e:
        print(f"Warning: Could not fetch HAPI info for {dataset_id}: {e}")
        return []

    params = []
    for p in info.get("parameters", []):
        name = p.get("name", "")

        # Skip Time parameter
        if name.lower() == "time":
            continue

        # Normalize size to a list
        size = p.get("size")
        if size is None:
            size = [1]
        elif isinstance(size, int):
            size = [size]

        # Filter: 1D with size <= 3
        if len(size) == 1 and size[0] <= 3:
            ptype = p.get("type", "")
            if ptype in ("double", "integer"):
                params.append({
                    "name": name,
                    "description": p.get("description", ""),
                    "units": p.get("units", ""),
                    "size": size,
                    "dataset_id": dataset_id,
                })

    return params


def get_dataset_time_range(dataset_id: str) -> Optional[dict]:
    """Get the available time range for a dataset.

    Args:
        dataset_id: CDAWeb dataset ID

    Returns:
        Dict with 'start' and 'stop' ISO date strings, or None if unavailable.
    """
    try:
        info = get_dataset_info(dataset_id)
        return {
            "start": info.get("startDate"),
            "stop": info.get("stopDate"),
        }
    except requests.RequestException:
        return None


def list_cached_datasets(mission_id: str) -> Optional[dict]:
    """Load the _index.json summary for a mission's cached HAPI metadata.

    Args:
        mission_id: Mission identifier (e.g., "PSP", "psp"). Case-insensitive.

    Returns:
        Parsed _index.json dict with mission_id, dataset_count, datasets list,
        or None if no index file exists.
    """
    index_path = _MISSIONS_DIR / mission_id.lower() / "hapi" / "_index.json"
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


def _load_calibration_exclusions(mission_id: str) -> tuple[list[str], list[str]]:
    """Load calibration exclusion patterns and IDs for a mission.

    Reads from knowledge/missions/{mission}/hapi/_calibration_exclude.json.

    Args:
        mission_id: Mission identifier (case-insensitive).

    Returns:
        Tuple of (patterns, ids). Returns ([], []) if no exclusion file exists.
    """
    exclude_path = _MISSIONS_DIR / mission_id.lower() / "hapi" / "_calibration_exclude.json"
    if not exclude_path.exists():
        return [], []
    data = json.loads(exclude_path.read_text(encoding="utf-8"))
    return data.get("patterns", []), data.get("ids", [])


def browse_datasets(mission_id: str) -> Optional[list[dict]]:
    """Return non-calibration datasets from _index.json.

    Filters out datasets matching calibration exclusion patterns/IDs.
    Returns None if no _index.json exists for the mission.

    Args:
        mission_id: Mission identifier (e.g., 'PSP', 'ACE'). Case-insensitive.

    Returns:
        List of dataset summary dicts, or None if no index file exists.
    """
    index = list_cached_datasets(mission_id)
    if index is None:
        return None

    patterns, excluded_ids = _load_calibration_exclusions(mission_id)
    excluded_id_set = set(excluded_ids)

    result = []
    for ds in index.get("datasets", []):
        ds_id = ds.get("id", "")
        # Check exact ID exclusion
        if ds_id in excluded_id_set:
            continue
        # Check pattern exclusion
        if any(fnmatch.fnmatch(ds_id, pat) for pat in patterns):
            continue
        result.append(ds)

    return result


def clear_cache():
    """Clear the HAPI info cache."""
    global _info_cache
    _info_cache = {}
