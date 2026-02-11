"""
Per-mission JSON loader with caching.

Loads mission data from knowledge/missions/*.json files.
Provides a clean API for accessing mission metadata, routing tables,
and dataset information without loading everything into memory upfront.
"""

import json
import logging
from pathlib import Path


logger = logging.getLogger("helio-agent")

# Directory containing per-mission JSON files
_MISSIONS_DIR = Path(__file__).parent / "missions"

# Module-level cache: mission_id (lowercase) -> parsed dict
_mission_cache: dict[str, dict] = {}

# Tracks which mission stems have already had their HAPI cache ensured
_hapi_cache_ensured: set[str] = set()


def load_mission(mission_id: str) -> dict:
    """Load a single mission's JSON data, with caching.

    Also ensures the HAPI metadata cache exists for this mission
    (downloads if missing). This is a fast no-op when the cache
    already exists.

    Args:
        mission_id: Mission identifier (e.g., "PSP", "ACE", "SolO").
                    Case-insensitive for file lookup; the JSON's "id" field
                    is the canonical casing.

    Returns:
        Parsed mission dict from the JSON file.

    Raises:
        FileNotFoundError: If no JSON file exists for this mission.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    cache_key = mission_id.lower().replace("-", "_")

    if cache_key not in _mission_cache:
        filepath = _MISSIONS_DIR / f"{cache_key}.json"
        if not filepath.exists():
            raise FileNotFoundError(
                f"No mission file found: {filepath}"
            )
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        _mission_cache[cache_key] = data

    _ensure_hapi_cache(cache_key)

    return _mission_cache[cache_key]


def _ensure_hapi_cache(mission_stem: str) -> None:
    """Ensure the HAPI metadata cache exists for a mission.

    Checks for _index.json and downloads if missing. Fast no-op
    when cache already exists or has already been checked this process.
    """
    if mission_stem in _hapi_cache_ensured:
        return
    _hapi_cache_ensured.add(mission_stem)

    from .hapi_client import list_cached_datasets
    if list_cached_datasets(mission_stem) is not None:
        return

    try:
        from .bootstrap import populate_mission_hapi_cache
        logger.info("[Bootstrap] Downloading metadata cache for %s...", mission_stem)
        populate_mission_hapi_cache(mission_stem)
    except Exception as e:
        logger.warning("Metadata cache download failed for %s: %s", mission_stem, e)


def load_all_missions() -> dict[str, dict]:
    """Load all mission JSON files, keyed by canonical mission ID.

    On first run (no JSON files exist), triggers auto-download from CDAWeb.

    Note: Does NOT trigger HAPI cache downloads — that only happens via
    load_mission() when a specific mission is accessed.

    Returns:
        Dict mapping mission ID (from JSON "id" field) to mission data.
        Example: {"PSP": {...}, "ACE": {...}, ...}
    """
    from .bootstrap import ensure_missions_populated
    ensure_missions_populated()

    result = {}
    for filepath in sorted(_MISSIONS_DIR.glob("*.json")):
        cache_key = filepath.stem  # e.g., "psp", "ace"
        if cache_key not in _mission_cache:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            _mission_cache[cache_key] = data
        mission = _mission_cache[cache_key]
        result[mission["id"]] = mission
    return result


def get_mission_ids() -> list[str]:
    """Get all known mission IDs from available JSON files.

    Returns:
        Sorted list of canonical mission IDs (e.g., ["ACE", "DSCOVR", ...]).
    """
    missions = load_all_missions()
    return sorted(missions.keys())


def get_routing_table() -> list[dict]:
    """Build a slim routing table for the main agent's system prompt.

    Each entry has the mission ID, name, and a list of capability keywords
    derived from instrument keywords (deduplicated).

    Returns:
        List of dicts: [{"id": "PSP", "name": "Parker Solar Probe",
                         "capabilities": ["magnetic field", "plasma"]}, ...]
    """
    missions = load_all_missions()
    table = []
    for mission_id, mission in missions.items():
        # Derive capabilities from instrument keyword unions
        capabilities = set()
        for inst in mission.get("instruments", {}).values():
            for kw in inst.get("keywords", []):
                # Group related keywords into higher-level capabilities
                if kw in ("magnetic", "field", "mag", "b-field", "bfield", "imf",
                           "mfi", "fgm", "impact", "magnetometer"):
                    capabilities.add("magnetic field")
                elif kw in ("plasma", "solar wind", "proton", "density", "velocity",
                            "temperature", "ion", "electron", "sweap", "swa",
                            "swe", "faraday", "plastic", "fpi"):
                    capabilities.add("plasma")
                elif kw in ("particle", "energetic", "cosmic ray"):
                    capabilities.add("energetic particles")
                elif kw in ("electric", "e-field"):
                    capabilities.add("electric field")
                elif kw in ("radio", "wave", "plasma wave"):
                    capabilities.add("radio/plasma waves")
                elif kw in ("index", "indices", "sym-h", "geomagnetic", "dst", "kp", "ae"):
                    capabilities.add("geomagnetic indices")
                elif kw in ("ephemeris", "orbit", "attitude", "position"):
                    capabilities.add("ephemeris")
                elif kw in ("composition", "charge state"):
                    capabilities.add("composition")
                elif kw in ("coronagraph", "heliograph"):
                    capabilities.add("coronagraph")
                elif kw in ("imaging", "remote sensing"):
                    capabilities.add("imaging")
        table.append({
            "id": mission_id,
            "name": mission["name"],
            "capabilities": sorted(capabilities),
        })
    return table


def get_mission_datasets(mission_id: str) -> list[str]:
    """Get all dataset IDs for a mission.

    Args:
        mission_id: Mission identifier (e.g., "PSP").

    Returns:
        List of dataset ID strings.
    """
    mission = load_mission(mission_id)
    dataset_ids = []
    for inst in mission.get("instruments", {}).values():
        for ds_id in inst.get("datasets", {}):
            dataset_ids.append(ds_id)
    return dataset_ids


def clear_cache():
    """Clear the mission cache. Useful for testing."""
    _mission_cache.clear()
    _hapi_cache_ensured.clear()
