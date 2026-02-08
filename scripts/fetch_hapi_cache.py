#!/usr/bin/env python3
"""
Fetch and cache HAPI /info metadata for all datasets matching a mission.

Saves raw HAPI /info JSON responses to local files for instant offline lookup.
Also generates a lightweight _index.json summary per mission.

Usage:
    python scripts/fetch_hapi_cache.py --mission psp          # one mission
    python scripts/fetch_hapi_cache.py --all                  # all missions
    python scripts/fetch_hapi_cache.py --mission psp --force  # re-fetch all
    python scripts/fetch_hapi_cache.py --all --workers 20     # faster parallel
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for knowledge imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)

from knowledge.mission_prefixes import match_dataset_to_mission


# Constants
HAPI_SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
MISSIONS_DIR = Path(__file__).parent.parent / "knowledge" / "missions"
DEFAULT_WORKERS = 10


def fetch_hapi_catalog() -> list[dict]:
    """Fetch the full HAPI catalog from CDAWeb."""
    url = f"{HAPI_SERVER}/catalog"
    print(f"Fetching HAPI catalog from {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    catalog = data.get("catalog", [])
    print(f"  Found {len(catalog)} datasets in catalog")
    return catalog


def fetch_hapi_info(dataset_id: str) -> dict | None:
    """Fetch HAPI /info for a single dataset. Returns parsed JSON or None."""
    url = f"{HAPI_SERVER}/info"
    try:
        resp = requests.get(url, params={"id": dataset_id}, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def build_index_entry(dataset_id: str, info: dict, instrument_hint: str | None) -> dict:
    """Build a lightweight index entry from a HAPI /info response."""
    # Count non-Time parameters
    param_count = sum(
        1 for p in info.get("parameters", [])
        if p.get("name", "").lower() != "time"
    )

    start_date = info.get("startDate", "")
    stop_date = info.get("stopDate", "")
    # Truncate to date portion if present
    if start_date and "T" in start_date:
        start_date = start_date.split("T")[0]
    if stop_date and "T" in stop_date:
        stop_date = stop_date.split("T")[0]

    return {
        "id": dataset_id,
        "description": info.get("description", ""),
        "start_date": start_date,
        "stop_date": stop_date,
        "parameter_count": param_count,
        "instrument": instrument_hint or "",
    }


def ensure_calibration_exclude(mission_stem: str):
    """Create a basic _calibration_exclude.json if one doesn't exist."""
    hapi_dir = MISSIONS_DIR / mission_stem / "hapi"
    hapi_dir.mkdir(parents=True, exist_ok=True)
    exclude_file = hapi_dir / "_calibration_exclude.json"
    if not exclude_file.exists():
        exclude_data = {
            "description": "Auto-generated exclusion patterns for calibration/housekeeping data",
            "patterns": ["*_K0_*", "*_K1_*", "*_K2_*"],
            "ids": [],
        }
        with open(exclude_file, "w", encoding="utf-8") as f:
            json.dump(exclude_data, f, indent=2, ensure_ascii=False)
            f.write("\n")


def _fetch_and_save(ds_id: str, instrument_hint: str | None, cache_dir: Path) -> tuple[str, str, dict | None]:
    """Fetch a single dataset's HAPI info and save to cache. Thread-safe.

    Returns:
        (dataset_id, status, index_entry_or_None)
        status is one of: "fetched", "failed"
    """
    info = fetch_hapi_info(ds_id)
    if info is None:
        return ds_id, "failed", None

    cache_file = cache_dir / f"{ds_id}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    entry = build_index_entry(ds_id, info, instrument_hint)
    return ds_id, "fetched", entry


def cache_mission(
    mission_stem: str,
    hapi_catalog: list[dict],
    force: bool = False,
    verbose: bool = False,
    workers: int = DEFAULT_WORKERS,
):
    """Fetch and cache HAPI /info for all datasets matching a mission.

    Args:
        mission_stem: Lowercase mission file stem (e.g., "psp", "ace")
        hapi_catalog: Full HAPI catalog list
        force: Re-fetch even if local file exists
        verbose: Print detailed progress
        workers: Number of parallel fetch threads
    """
    # Verify mission JSON exists
    mission_json = MISSIONS_DIR / f"{mission_stem}.json"
    if not mission_json.exists():
        print(f"Error: Mission file not found: {mission_json}")
        return

    with open(mission_json, "r", encoding="utf-8") as f:
        mission_data = json.load(f)

    mission_id = mission_data.get("id", mission_stem.upper())

    # Create cache directory and ensure calibration exclude file exists
    cache_dir = MISSIONS_DIR / mission_stem / "hapi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_calibration_exclude(mission_stem)

    # Find matching datasets
    matched = []
    for entry in hapi_catalog:
        ds_id = entry.get("id", "")
        ds_mission, ds_instrument = match_dataset_to_mission(ds_id)
        if ds_mission == mission_stem:
            matched.append((ds_id, ds_instrument))

    print(f"\n{mission_id}: {len(matched)} datasets", end="", flush=True)

    index_entries = []
    fetched = 0
    skipped = 0
    errors = 0

    # Separate into cached (skip) and to-fetch lists
    to_fetch = []
    for ds_id, instrument_hint in matched:
        cache_file = cache_dir / f"{ds_id}.json"

        if cache_file.exists() and not force:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                index_entries.append(build_index_entry(ds_id, info, instrument_hint))
                skipped += 1
            except json.JSONDecodeError:
                to_fetch.append((ds_id, instrument_hint))
        else:
            to_fetch.append((ds_id, instrument_hint))

    if to_fetch:
        print(f" ({skipped} cached, {len(to_fetch)} to fetch)", flush=True)

        # Parallel fetch
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_and_save, ds_id, inst, cache_dir): (ds_id, inst)
                for ds_id, inst in to_fetch
            }
            for future in as_completed(futures):
                ds_id, status, entry = future.result()
                if status == "fetched" and entry:
                    index_entries.append(entry)
                    fetched += 1
                    if verbose:
                        print(f"  [ok] {ds_id} ({entry['parameter_count']} params)")
                else:
                    errors += 1
                    if verbose:
                        print(f"  [FAIL] {ds_id}")
    else:
        print(f" (all {skipped} cached)", flush=True)

    # Sort index by dataset ID
    index_entries.sort(key=lambda e: e["id"])

    # Write _index.json
    index_data = {
        "mission_id": mission_id,
        "dataset_count": len(index_entries),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": index_entries,
    }
    index_file = cache_dir / "_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  Done: {fetched} fetched, {skipped} cached, {errors} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and cache HAPI /info metadata for mission datasets"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mission",
        type=str,
        help="Cache one mission (e.g., psp, ace). Case-insensitive.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Cache all known missions",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if local cache exists",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel fetch threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Fetch HAPI catalog once
    hapi_catalog = fetch_hapi_catalog()

    if args.mission:
        mission_stem = args.mission.lower()
        cache_mission(
            mission_stem, hapi_catalog,
            force=args.force, verbose=args.verbose, workers=args.workers,
        )
    else:
        # All missions
        for filepath in sorted(MISSIONS_DIR.glob("*.json")):
            cache_mission(
                filepath.stem, hapi_catalog,
                force=args.force, verbose=args.verbose, workers=args.workers,
            )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
