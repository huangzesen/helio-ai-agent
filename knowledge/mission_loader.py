"""
Per-mission JSON loader with caching.

Loads mission data from knowledge/missions/*.json files.
Provides a clean API for accessing mission metadata, routing tables,
and dataset information without loading everything into memory upfront.

Supports override files in ``{data_dir}/mission_overrides/`` that are
deep-merged on top of the auto-generated base JSON at load time.
Override files are sparse patches — only fields that differ from the
auto-generated base need to be present.  Generic recursive deep-merge:
dicts merge recursively, everything else replaces.
"""

import json
import logging
from pathlib import Path


logger = logging.getLogger("helio-agent")

# Directory containing per-mission JSON files
_MISSIONS_DIR = Path(__file__).parent / "missions"

# Module-level cache: mission_id (lowercase) -> parsed dict
_mission_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Override support
# ---------------------------------------------------------------------------


def _get_overrides_dir() -> Path:
    """Return the directory for mission override files.

    Uses ``config.get_data_dir() / "mission_overrides"``.  Evaluated lazily
    because ``get_data_dir()`` depends on env/config not available at import
    time.
    """
    from config import get_data_dir
    return get_data_dir() / "mission_overrides"


def _load_override(cache_key: str) -> dict | None:
    """Load an override file for the given mission cache key.

    Returns:
        Parsed dict, or ``None`` if the file is missing, unreadable, or
        contains invalid JSON.
    """
    path = _get_overrides_dir() / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Override file %s is not a JSON object; ignoring", path)
            return None
        return data
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON in override file %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning("Cannot read override file %s: %s", path, exc)
        return None


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge *patch* into *base* (mutates *base* in place).

    - If both values are dicts, merge recursively.
    - Otherwise the patch value replaces the base value.

    Returns *base* for convenience.
    """
    for key, value in patch.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _save_override(cache_key: str, data: dict) -> None:
    """Write an override dict to disk.

    Creates the overrides directory if it doesn't exist.
    """
    overrides_dir = _get_overrides_dir()
    overrides_dir.mkdir(parents=True, exist_ok=True)
    path = overrides_dir / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.debug("Saved mission override: %s", path)


def update_mission_override(cache_key: str, patch: dict) -> dict:
    """Read-modify-write a mission override file.

    Loads the existing override (if any), deep-merges *patch* into it,
    writes the result back, and invalidates the mission cache so the
    next ``load_mission()`` call picks up the change.

    Args:
        cache_key: Mission cache key (e.g. ``"psp"``, ``"ace"``).
        patch: Sparse dict to merge into the override.

    Returns:
        The full override dict after merging.
    """
    existing = _load_override(cache_key) or {}
    _deep_merge(existing, patch)
    _save_override(cache_key, existing)
    # Invalidate cache so next load picks up the change
    _mission_cache.pop(cache_key, None)
    return existing


def load_mission(mission_id: str) -> dict:
    """Load a single mission's JSON data, with caching.

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
        # Apply user overrides before caching
        override = _load_override(cache_key)
        if override is not None:
            _deep_merge(data, override)
        _mission_cache[cache_key] = data

    return _mission_cache[cache_key]


def load_all_missions() -> dict[str, dict]:
    """Load all mission JSON files, keyed by canonical mission ID.

    On first run (no JSON files exist), triggers full auto-download
    from CDAWeb (catalog + parameter metadata via Master CDF).

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
            # Apply user overrides before caching
            override = _load_override(cache_key)
            if override is not None:
                _deep_merge(data, override)
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
