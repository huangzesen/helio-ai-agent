"""
HAPI client for CDAWeb parameter metadata discovery.

Fetches parameter info dynamically from the CDAWeb HAPI server and filters
to 1D plottable parameters (scalars and small vectors with size <= 3).
"""

import requests
from typing import Optional

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi"

# Cache for HAPI responses to avoid repeated API calls
_info_cache: dict[str, dict] = {}


def get_dataset_info(dataset_id: str, use_cache: bool = True) -> dict:
    """Fetch parameter metadata from HAPI /info endpoint.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "PSP_FLD_L2_MAG_RTN_1MIN")
        use_cache: Whether to use cached results

    Returns:
        HAPI info response with startDate, stopDate, parameters, etc.

    Raises:
        requests.HTTPError: If the HAPI request fails.
    """
    if use_cache and dataset_id in _info_cache:
        return _info_cache[dataset_id]

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


def clear_cache():
    """Clear the HAPI info cache."""
    global _info_cache
    _info_cache = {}
