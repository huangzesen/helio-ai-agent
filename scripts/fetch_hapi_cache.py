#!/usr/bin/env python3
"""
Fetch and cache HAPI /info metadata for all datasets matching a mission.

Saves raw HAPI /info JSON responses to local files for instant offline lookup.
Also generates a lightweight _index.json summary per mission.

Usage:
    python scripts/fetch_hapi_cache.py --mission psp          # one mission
    python scripts/fetch_hapi_cache.py --all                  # all 8 missions
    python scripts/fetch_hapi_cache.py --mission psp --force  # re-fetch all
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)


# Reuse constants from generate_mission_data.py
HAPI_SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
MISSIONS_DIR = Path(__file__).parent.parent / "knowledge" / "missions"
REQUEST_DELAY = 0.5

# Maps CDAWeb dataset ID prefixes to (mission_stem, instrument_hint)
# Order matters: longer prefixes checked first
MISSION_PREFIX_MAP = {
    "PSP_FLD": ("psp", "FIELDS/MAG"),
    "PSP_SWP_SPC": ("psp", "SWEAP"),
    "PSP_SWP_SPI": ("psp", "SWEAP/SPAN-I"),
    "PSP_SWP_SPA": ("psp", "SWEAP/SPAN-E"),
    "PSP_SWP_SPB": ("psp", "SWEAP/SPAN-E"),
    "PSP_SWP": ("psp", "SWEAP"),
    "PSP_ISOIS": ("psp", "ISOIS"),
    "PSP_": ("psp", None),
    "SOLO_L2_MAG": ("solo", "MAG"),
    "SOLO_L2_SWA": ("solo", "SWA-PAS"),
    "SOLO_": ("solo", None),
    "AC_H": ("ace", None),
    "AC_K": ("ace", None),
    "OMNI_HRO": ("omni", "Combined"),
    "OMNI_": ("omni", "Combined"),
    "WI_H": ("wind", None),
    "WI_": ("wind", None),
    "DSCOVR_H0_MAG": ("dscovr", "MAG"),
    "DSCOVR_H1_FC": ("dscovr", "FC"),
    "DSCOVR_": ("dscovr", None),
    "MMS1_FGM": ("mms", "FGM"),
    "MMS1_FPI": ("mms", "FPI-DIS"),
    "MMS1_": ("mms", None),
    "STA_L2_MAG": ("stereo_a", "MAG"),
    "STA_L2_PLA": ("stereo_a", "PLASTIC"),
    "STA_": ("stereo_a", None),
}


def match_dataset_to_mission(dataset_id: str) -> tuple[str | None, str | None]:
    """Map a CDAWeb dataset ID to a mission and optional instrument hint."""
    for prefix, (mission, instrument) in MISSION_PREFIX_MAP.items():
        if dataset_id.startswith(prefix):
            return mission, instrument
    return None, None


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
    except Exception as e:
        print(f"    Warning: Failed to fetch info for {dataset_id}: {e}")
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


def cache_mission(
    mission_stem: str,
    hapi_catalog: list[dict],
    force: bool = False,
    verbose: bool = False,
):
    """Fetch and cache HAPI /info for all datasets matching a mission.

    Args:
        mission_stem: Lowercase mission file stem (e.g., "psp", "ace")
        hapi_catalog: Full HAPI catalog list
        force: Re-fetch even if local file exists
        verbose: Print detailed progress
    """
    # Verify mission JSON exists
    mission_json = MISSIONS_DIR / f"{mission_stem}.json"
    if not mission_json.exists():
        print(f"Error: Mission file not found: {mission_json}")
        return

    with open(mission_json, "r", encoding="utf-8") as f:
        mission_data = json.load(f)

    mission_id = mission_data.get("id", mission_stem.upper())

    # Create cache directory
    cache_dir = MISSIONS_DIR / mission_stem / "hapi"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Find matching datasets
    matched = []
    for entry in hapi_catalog:
        ds_id = entry.get("id", "")
        ds_mission, ds_instrument = match_dataset_to_mission(ds_id)
        if ds_mission == mission_stem:
            matched.append((ds_id, ds_instrument))

    print(f"\n{mission_id}: {len(matched)} datasets found in HAPI catalog")

    index_entries = []
    fetched = 0
    skipped = 0
    errors = 0

    for ds_id, instrument_hint in matched:
        cache_file = cache_dir / f"{ds_id}.json"

        if cache_file.exists() and not force:
            # Load existing file for index
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                index_entries.append(build_index_entry(ds_id, info, instrument_hint))
                skipped += 1
                if verbose:
                    print(f"  [skip] {ds_id}")
            except json.JSONDecodeError:
                # Re-fetch if corrupted
                pass
            else:
                continue

        # Fetch from HAPI
        if verbose:
            print(f"  [fetch] {ds_id}...", end="", flush=True)

        info = fetch_hapi_info(ds_id)
        if info is None:
            errors += 1
            if verbose:
                print(" FAILED")
            continue

        # Save raw response
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            f.write("\n")

        index_entries.append(build_index_entry(ds_id, info, instrument_hint))
        fetched += 1

        if verbose:
            param_count = index_entries[-1]["parameter_count"]
            print(f" OK ({param_count} params)")

        time.sleep(REQUEST_DELAY)

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

    print(f"  Fetched: {fetched}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Index: {index_file} ({len(index_entries)} entries)")


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
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    args = parser.parse_args()

    # Fetch HAPI catalog once
    hapi_catalog = fetch_hapi_catalog()

    if args.mission:
        mission_stem = args.mission.lower()
        cache_mission(mission_stem, hapi_catalog, force=args.force, verbose=args.verbose)
    else:
        # All missions
        for filepath in sorted(MISSIONS_DIR.glob("*.json")):
            cache_mission(
                filepath.stem, hapi_catalog,
                force=args.force, verbose=args.verbose,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
