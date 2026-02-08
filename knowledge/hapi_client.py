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
import re
import requests
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi"

# Cache for HAPI responses to avoid repeated API calls
_info_cache: dict[str, dict] = {}

# Cache for CDAWeb Notes HTML pages (keyed by base URL, e.g., NotesA.html)
_notes_cache: dict[str, str] = {}

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
        except requests.RequestException:
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

    Combines HAPI /info metadata (contact, resourceURL) with the actual
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

    # Try to get contact and resource URL from HAPI /info
    try:
        info = get_dataset_info(dataset_id)
        result["contact"] = info.get("contact")
        result["resource_url"] = info.get("resourceURL")
    except requests.RequestException:
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
    """Clear the HAPI info cache and Notes page cache."""
    global _info_cache, _notes_cache
    _info_cache = {}
    _notes_cache = {}
