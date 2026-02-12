"""
CDAWeb parameter metadata client.

Fetches parameter info dynamically and filters to 1D plottable parameters
(scalars and small vectors with size <= 3).

Uses a three-layer resolution strategy:
1. In-memory cache (fastest)
2. Local file cache in knowledge/missions/*/metadata/ (instant, no network)
3. Master CDF skeleton file (network fallback)

Supports local file cache: if a dataset's info response is saved in
knowledge/missions/{mission}/metadata/{dataset_id}.json, it is loaded instantly
without a network request.
"""

import fnmatch
import json
import logging
import re
import requests
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

logger = logging.getLogger("helio-agent")

# Cache for metadata responses to avoid repeated API calls
_info_cache: dict[str, dict] = {}

# Cache for CDAWeb Notes HTML pages (keyed by base URL, e.g., NotesA.html)
_notes_cache: dict[str, str] = {}

# Directory containing per-mission folders with metadata cache files
_MISSIONS_DIR = Path(__file__).parent / "missions"


def _find_local_cache(dataset_id: str) -> Optional[Path]:
    """Scan mission subfolders for a locally cached metadata file.

    Checks knowledge/missions/*/metadata/{dataset_id}.json across all mission
    directories. Only 8 dirs to scan — negligible cost.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "PSP_FLD_L2_MAG_RTN_1MIN")

    Returns:
        Path to local cache file, or None if not found.
    """
    for mission_dir in _MISSIONS_DIR.iterdir():
        if not mission_dir.is_dir():
            continue
        cache_file = mission_dir / "metadata" / f"{dataset_id}.json"
        if cache_file.exists():
            return cache_file
    return None


def get_dataset_info(dataset_id: str, use_cache: bool = True) -> dict:
    """Fetch parameter metadata for a dataset.

    Checks three sources in order:
    1. In-memory cache (fastest)
    2. Local file cache in knowledge/missions/*/metadata/ (instant, no network)
    3. Master CDF skeleton file (network fallback)

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "PSP_FLD_L2_MAG_RTN_1MIN")
        use_cache: Whether to use cached results (in-memory and local file)

    Returns:
        Info dict with startDate, stopDate, parameters, etc.

    Raises:
        Exception: If all sources fail.
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

    # 3. Master CDF (network fallback)
    from .master_cdf import fetch_dataset_metadata_from_master
    info = fetch_dataset_metadata_from_master(dataset_id)
    if info is not None:
        logger.debug("Got metadata from Master CDF for %s", dataset_id)
        if use_cache:
            _info_cache[dataset_id] = info
            _save_to_local_cache(dataset_id, info)
        return info

    raise ValueError(
        f"No metadata available for dataset '{dataset_id}'. "
        f"Master CDF download failed."
    )


def _save_to_local_cache(dataset_id: str, info: dict) -> None:
    """Persist metadata to the local file cache.

    Finds the appropriate mission directory by scanning existing dirs.
    If no matching mission dir is found, skips silently.
    """
    for mission_dir in _MISSIONS_DIR.iterdir():
        if not mission_dir.is_dir():
            continue
        metadata_dir = mission_dir / "metadata"
        if metadata_dir.exists():
            # Check if this mission has any datasets with matching prefix
            # by looking at existing cache files
            existing = list(metadata_dir.glob("*.json"))
            if not existing:
                continue
            # Check prefix match (e.g., AC_ for ACE datasets)
            sample_name = existing[0].stem
            if sample_name.startswith("_"):
                if len(existing) > 1:
                    sample_name = existing[1].stem
                else:
                    continue
            # Simple heuristic: same mission prefix
            ds_prefix = dataset_id.split("_")[0] if "_" in dataset_id else ""
            sample_prefix = sample_name.split("_")[0] if "_" in sample_name else ""
            if ds_prefix and ds_prefix == sample_prefix:
                cache_file = metadata_dir / f"{dataset_id}.json"
                try:
                    cache_file.write_text(
                        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    logger.debug("Saved metadata to cache: %s", cache_file)
                except OSError:
                    pass
                return


def list_parameters(dataset_id: str) -> list[dict]:
    """List plottable 1D parameters for a dataset.

    Fetches metadata and filters to parameters that are:
    - Not the Time parameter
    - Numeric type (double or integer)
    - 1D with size <= 3 (scalars and small vectors)

    Args:
        dataset_id: CDAWeb dataset ID

    Returns:
        List of parameter dicts with name, description, units, size, dataset_id.
        Returns empty list if metadata fetch fails.
    """
    try:
        info = get_dataset_info(dataset_id)
    except (requests.RequestException, Exception) as e:
        logger.warning("Could not fetch info for %s: %s", dataset_id, e)
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
    except (requests.RequestException, ValueError):
        return None


def list_cached_datasets(mission_id: str) -> Optional[dict]:
    """Load the _index.json summary for a mission's cached metadata.

    Args:
        mission_id: Mission identifier (e.g., "PSP", "psp"). Case-insensitive.

    Returns:
        Parsed _index.json dict with mission_id, dataset_count, datasets list,
        or None if no index file exists.
    """
    index_path = _MISSIONS_DIR / mission_id.lower() / "metadata" / "_index.json"
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


def _load_calibration_exclusions(mission_id: str) -> tuple[list[str], list[str]]:
    """Load calibration exclusion patterns and IDs for a mission.

    Reads from knowledge/missions/{mission}/metadata/_calibration_exclude.json.

    Args:
        mission_id: Mission identifier (case-insensitive).

    Returns:
        Tuple of (patterns, ids). Returns ([], []) if no exclusion file exists.
    """
    exclude_path = _MISSIONS_DIR / mission_id.lower() / "metadata" / "_calibration_exclude.json"
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


def list_missions() -> list[dict]:
    """List all missions that have cached metadata on disk.

    Scans for _index.json files across all mission directories.
    Pure filesystem lookup — no network.

    Returns:
        List of dicts with mission_id and dataset_count.
    """
    results = []
    for mission_dir in sorted(_MISSIONS_DIR.iterdir()):
        if not mission_dir.is_dir():
            continue
        index_path = mission_dir / "metadata" / "_index.json"
        if not index_path.exists():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            results.append({
                "mission_id": index.get("mission_id", mission_dir.name.upper()),
                "dataset_count": index.get("dataset_count", len(index.get("datasets", []))),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return results


def validate_dataset_id(dataset_id: str) -> dict:
    """Check if a dataset ID exists in the local metadata cache.

    Uses _find_local_cache() to check all mission directories.
    No network call is made.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        Dict with valid (bool), mission_id (str|None), message (str).
    """
    cache_path = _find_local_cache(dataset_id)
    if cache_path is not None:
        # Extract mission_id from the path: .../missions/{mission}/metadata/{file}.json
        mission_id = cache_path.parent.parent.name.upper()
        return {
            "valid": True,
            "mission_id": mission_id,
            "message": f"Dataset '{dataset_id}' found in {mission_id} cache.",
        }
    return {
        "valid": False,
        "mission_id": None,
        "message": (
            f"Dataset '{dataset_id}' not found in local metadata cache. "
            f"Use browse_datasets(mission_id) to see available datasets."
        ),
    }


def validate_parameter_id(dataset_id: str, parameter_id: str) -> dict:
    """Check if a parameter ID exists in a cached dataset's metadata.

    Reads the cached dataset JSON directly — no network call.

    Args:
        dataset_id: CDAWeb dataset ID.
        parameter_id: Parameter name to validate.

    Returns:
        Dict with valid (bool), available_parameters (list[str]), message (str).
    """
    cache_path = _find_local_cache(dataset_id)
    if cache_path is None:
        return {
            "valid": False,
            "available_parameters": [],
            "message": (
                f"Cannot validate parameter — dataset '{dataset_id}' "
                f"not found in local metadata cache."
            ),
        }

    try:
        info = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "valid": False,
            "available_parameters": [],
            "message": f"Cannot read cache file for dataset '{dataset_id}'.",
        }

    # Collect all non-Time parameter names
    available = [
        p["name"]
        for p in info.get("parameters", [])
        if p.get("name", "").lower() != "time"
    ]

    if parameter_id in available:
        return {
            "valid": True,
            "available_parameters": available,
            "message": f"Parameter '{parameter_id}' is valid for dataset '{dataset_id}'.",
        }

    return {
        "valid": False,
        "available_parameters": available,
        "message": (
            f"Parameter '{parameter_id}' not found in dataset '{dataset_id}'. "
            f"Available parameters: {', '.join(available)}. "
            f"Use list_parameters('{dataset_id}') for details."
        ),
    }


class _HTMLToText(HTMLParser):
    """Minimal HTML-to-text converter using stdlib HTMLParser.

    Strips tags while preserving block structure (newlines for <br>, <p>, <hr>,
    headings, list items). Skips <script> and <style> content.
    """

    _BLOCK_TAGS = {"p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
    _SKIP_TAGS = {"script", "style"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in self._BLOCK_TAGS and not self._skip_depth:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str):
        if not self._skip_depth:
            self._parts.append(data)

    def get_text(self) -> str:
        """Return accumulated text, with collapsed whitespace."""
        raw = "".join(self._parts)
        # Collapse runs of blank lines to at most two newlines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _extract_dataset_section(html: str, dataset_id: str) -> Optional[str]:
    """Extract the section for a specific dataset from a CDAWeb Notes HTML page.

    CDAWeb Notes pages use anchors like <a name="DATASET_ID"> or
    <strong>DATASET_ID</strong> to mark section starts, with <hr> tags
    or "Back to top" links as section boundaries.

    Args:
        html: Full HTML text of the Notes page.
        dataset_id: CDAWeb dataset ID to look for (e.g., "AC_H2_MFI").

    Returns:
        HTML string for the dataset section, or None if not found.
    """
    # Try anchor patterns: name="ID", id="ID"
    patterns = [
        rf'(?:name|id)\s*=\s*"{re.escape(dataset_id)}"',
        rf'<strong>\s*{re.escape(dataset_id)}\s*</strong>',
    ]

    start_pos = None
    for pat in patterns:
        match = re.search(pat, html, re.IGNORECASE)
        if match:
            start_pos = match.start()
            break

    if start_pos is None:
        return None

    # Find section end: next <hr> or "Back to top" after the anchor
    remaining = html[start_pos:]
    end_patterns = [
        r'<hr\b[^>]*>',
        r'Back to top',
    ]
    end_pos = len(remaining)
    for ep in end_patterns:
        m = re.search(ep, remaining[100:], re.IGNORECASE)  # skip first 100 chars to avoid self-match
        if m:
            end_pos = min(end_pos, m.start() + 100)

    return remaining[:end_pos]


def _fetch_notes_section(resource_url: str, dataset_id: str) -> Optional[str]:
    """Fetch a CDAWeb Notes page (cached) and extract the dataset section as text.

    Args:
        resource_url: URL like "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI"
        dataset_id: CDAWeb dataset ID.

    Returns:
        Plain text of the dataset documentation section, or None on failure.
    """
    # Strip fragment to get base URL (the page to fetch)
    base_url = resource_url.split("#")[0]

    # Check cache
    if base_url not in _notes_cache:
        try:
            resp = requests.get(base_url, timeout=30)
            resp.raise_for_status()
            _notes_cache[base_url] = resp.text
        except Exception:
            return None

    html = _notes_cache[base_url]
    section_html = _extract_dataset_section(html, dataset_id)
    if section_html is None:
        return None

    parser = _HTMLToText()
    parser.feed(section_html)
    return parser.get_text()


def _fallback_resource_url(dataset_id: str) -> str:
    """Construct a plausible CDAWeb Notes URL from a dataset ID.

    CDAWeb Notes pages are organized by first letter: NotesA.html, NotesB.html, etc.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").

    Returns:
        URL string like "https://cdaweb.gsfc.nasa.gov/misc/NotesA.html#AC_H2_MFI"
    """
    first_letter = dataset_id[0].upper() if dataset_id else "A"
    return f"https://cdaweb.gsfc.nasa.gov/misc/Notes{first_letter}.html#{dataset_id}"


def get_dataset_docs(dataset_id: str, max_chars: int = 4000) -> dict:
    """Look up CDAWeb documentation for a dataset.

    Combines dataset metadata (contact, resourceURL) with the actual
    documentation text scraped from the CDAWeb Notes page.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").
        max_chars: Maximum characters for the documentation text.

    Returns:
        Dict with dataset_id, contact, resource_url, and documentation fields.
        documentation may be None if the Notes page could not be fetched or
        the dataset section was not found.
    """
    result = {"dataset_id": dataset_id, "contact": None, "resource_url": None, "documentation": None}

    # Try to get contact and resource URL from local cache or CDAWeb metadata
    try:
        info = get_dataset_info(dataset_id)
        result["contact"] = info.get("contact")
        result["resource_url"] = info.get("resourceURL")
    except Exception:
        pass

    # Enrich with CDAWeb REST API metadata (PI info, notes URL)
    if not result["contact"] or not result["resource_url"]:
        try:
            from .cdaweb_metadata import fetch_dataset_metadata
            cdaweb_meta = fetch_dataset_metadata()
            entry = cdaweb_meta.get(dataset_id)
            if entry:
                if not result["contact"] and entry.get("pi_name"):
                    result["contact"] = entry["pi_name"]
                    if entry.get("pi_affiliation"):
                        result["contact"] += f" @ {entry['pi_affiliation']}"
                if not result["resource_url"] and entry.get("notes_url"):
                    result["resource_url"] = entry["notes_url"]
        except Exception:
            pass

    # Determine the resource URL to fetch
    resource_url = result["resource_url"]
    if not resource_url:
        resource_url = _fallback_resource_url(dataset_id)
        result["resource_url"] = resource_url

    # Fetch and extract documentation
    doc_text = _fetch_notes_section(resource_url, dataset_id)
    if doc_text and len(doc_text) > max_chars:
        doc_text = doc_text[:max_chars] + "\n[truncated]"
    result["documentation"] = doc_text

    return result


def clear_cache():
    """Clear the metadata info cache and Notes page cache."""
    global _info_cache, _notes_cache
    _info_cache = {}
    _notes_cache = {}
