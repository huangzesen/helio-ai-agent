"""
Full CDAWeb HAPI catalog fetch, cache, and search.

Provides access to the complete CDAWeb catalog (2000+ datasets) via a
locally cached copy of the HAPI /catalog endpoint. The cache is refreshed
every 24 hours.

Used by the `search_full_catalog` tool to let users find datasets
across all CDAWeb missions, not just the curated ones.
"""

import json
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

HAPI_CATALOG_URL = "https://cdaweb.gsfc.nasa.gov/hapi/catalog"
CATALOG_CACHE = Path.home() / ".helio-agent" / "cdaweb_catalog.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def get_full_catalog() -> list[dict]:
    """Fetch and cache the full CDAWeb HAPI catalog.

    Returns a list of dicts with 'id' and 'title' keys.
    Uses a local file cache (refreshed every 24 hours).

    Returns:
        List of catalog entries, each with 'id' and 'title'.
    """
    # Check cache freshness
    if CATALOG_CACHE.exists():
        age = time.time() - CATALOG_CACHE.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            try:
                with open(CATALOG_CACHE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("catalog", [])
            except (json.JSONDecodeError, KeyError):
                pass  # Re-fetch on corrupt cache

    # Fetch from HAPI server
    if requests is None:
        return []

    try:
        resp = requests.get(HAPI_CATALOG_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        # If fetch fails and we have a stale cache, use it
        if CATALOG_CACHE.exists():
            try:
                with open(CATALOG_CACHE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("catalog", [])
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    # Save cache
    CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return data.get("catalog", [])


def search_catalog(query: str, max_results: int = 20) -> list[dict]:
    """Search the full CDAWeb catalog by keyword.

    Performs case-insensitive substring matching on dataset IDs and titles.
    Supports multi-word queries (all words must match).

    Args:
        query: Search terms (e.g., "cluster magnetic", "voyager 2").
        max_results: Maximum number of results to return.

    Returns:
        List of matching catalog entries with 'id' and 'title'.
    """
    catalog = get_full_catalog()
    if not catalog:
        return []

    words = query.lower().split()
    if not words:
        return []

    matches = []
    for entry in catalog:
        text = f"{entry.get('id', '')} {entry.get('title', '')}".lower()
        if all(w in text for w in words):
            matches.append({
                "id": entry.get("id", ""),
                "title": entry.get("title", ""),
            })
            if len(matches) >= max_results:
                break

    return matches


def get_catalog_stats() -> dict:
    """Return basic statistics about the cached catalog.

    Returns:
        Dict with 'total_datasets', 'cache_age_hours', 'cache_exists'.
    """
    catalog = get_full_catalog()
    stats = {
        "total_datasets": len(catalog),
        "cache_exists": CATALOG_CACHE.exists(),
        "cache_age_hours": None,
    }
    if CATALOG_CACHE.exists():
        age_hours = (time.time() - CATALOG_CACHE.stat().st_mtime) / 3600
        stats["cache_age_hours"] = round(age_hours, 1)
    return stats
