#!/usr/bin/env python3
"""
Auto-generate/update per-mission JSON files from CDAWeb HAPI metadata.

Queries the HAPI /catalog and /info endpoints to populate datasets and
parameters in knowledge/missions/*.json files.

Hand-curated fields (profile, keywords, tier) are preserved on merge.
New datasets default to tier="advanced".

Usage:
    python scripts/generate_mission_data.py              # Update all missions
    python scripts/generate_mission_data.py --mission PSP  # Update one mission
    python scripts/generate_mission_data.py --discover     # Show unknown datasets
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


# HAPI server base URL
HAPI_SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"

# Missions directory
MISSIONS_DIR = Path(__file__).parent.parent / "knowledge" / "missions"

# Maps CDAWeb dataset ID prefixes to mission IDs (lowercase JSON filenames)
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

# Rate limiting between HAPI /info requests (seconds)
REQUEST_DELAY = 0.5


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
    """Fetch HAPI /info for a dataset.

    Returns parsed JSON or None on error.
    """
    url = f"{HAPI_SERVER}/info?id={dataset_id}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        print(f"    Warning: Failed to fetch info for {dataset_id}: {e}")
        return None


def match_dataset_to_mission(dataset_id: str) -> tuple[str | None, str | None]:
    """Map a CDAWeb dataset ID to a mission and optional instrument.

    Returns:
        (mission_file_stem, instrument_id) or (None, None) if no match.
    """
    for prefix, (mission, instrument) in MISSION_PREFIX_MAP.items():
        if dataset_id.startswith(prefix):
            return mission, instrument
    return None, None


def load_mission_json(mission_stem: str) -> dict:
    """Load a mission JSON file."""
    filepath = MISSIONS_DIR / f"{mission_stem}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Mission file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_mission_json(mission_stem: str, data: dict):
    """Save a mission JSON file with sorted keys for clean diffs."""
    filepath = MISSIONS_DIR / f"{mission_stem}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")
    print(f"  Saved {filepath}")


def merge_dataset_info(
    existing_ds: dict | None,
    hapi_info: dict,
    dataset_id: str,
) -> dict:
    """Merge HAPI /info data into an existing dataset entry.

    Preserves: tier (hand-curated)
    Overwrites: description, start_date, stop_date, parameters, _meta
    """
    if existing_ds is None:
        existing_ds = {"tier": "advanced"}

    # Preserve tier
    tier = existing_ds.get("tier", "advanced")

    # Extract info from HAPI response
    description = hapi_info.get("description", existing_ds.get("description", ""))
    start_date = hapi_info.get("startDate", existing_ds.get("start_date", ""))
    stop_date = hapi_info.get("stopDate", existing_ds.get("stop_date", ""))

    # Build parameters array from HAPI response
    parameters = []
    for param in hapi_info.get("parameters", []):
        name = param.get("name", "")
        if name.lower() == "time":
            continue  # Skip the time parameter
        param_entry = {
            "name": name,
            "type": param.get("type", ""),
            "units": param.get("units", ""),
            "description": param.get("description", ""),
        }
        size = param.get("size")
        if size:
            param_entry["size"] = size
        parameters.append(param_entry)

    return {
        "tier": tier,
        "description": description,
        "start_date": start_date,
        "stop_date": stop_date,
        "parameters": parameters,
    }


def update_mission(mission_stem: str, hapi_catalog: list[dict], verbose: bool = False):
    """Update a single mission's JSON file with HAPI data.

    Args:
        mission_stem: Lowercase mission file stem (e.g., "psp", "ace")
        hapi_catalog: Full HAPI catalog list
        verbose: Print detailed progress
    """
    mission_data = load_mission_json(mission_stem)
    mission_id = mission_data["id"]
    print(f"\nUpdating {mission_id} ({mission_stem}.json)...")

    # Find all HAPI datasets that belong to this mission
    matched_datasets = []
    for entry in hapi_catalog:
        ds_id = entry.get("id", "")
        ds_mission, ds_instrument = match_dataset_to_mission(ds_id)
        if ds_mission == mission_stem:
            matched_datasets.append((ds_id, ds_instrument))

    print(f"  Found {len(matched_datasets)} HAPI datasets for {mission_id}")

    # Collect existing dataset IDs across all instruments
    existing_dataset_ids = set()
    for inst in mission_data.get("instruments", {}).values():
        existing_dataset_ids.update(inst.get("datasets", {}).keys())

    updated_count = 0
    new_count = 0

    for ds_id, suggested_instrument in matched_datasets:
        # Find which instrument this dataset belongs to
        target_instrument = None

        # First, check if dataset already exists in an instrument
        for inst_id, inst in mission_data.get("instruments", {}).items():
            if ds_id in inst.get("datasets", {}):
                target_instrument = inst_id
                break

        # If not found, use the suggested instrument from prefix mapping
        if target_instrument is None and suggested_instrument:
            if suggested_instrument in mission_data.get("instruments", {}):
                target_instrument = suggested_instrument

        # If still no instrument, skip (we can't auto-assign)
        if target_instrument is None:
            if verbose:
                print(f"    Skipping {ds_id}: no instrument mapping")
            continue

        # Fetch HAPI /info
        if verbose:
            print(f"    Fetching info for {ds_id}...")
        hapi_info = fetch_hapi_info(ds_id)
        if hapi_info is None:
            if verbose:
                print(f"    Warning: No info available for {ds_id}")
            continue

        time.sleep(REQUEST_DELAY)

        # Merge into the instrument's datasets
        inst = mission_data["instruments"][target_instrument]
        datasets = inst.setdefault("datasets", {})
        existing = datasets.get(ds_id)

        if existing:
            updated_count += 1
        else:
            new_count += 1

        datasets[ds_id] = merge_dataset_info(existing, hapi_info, ds_id)

        if verbose:
            status = "updated" if existing else "NEW"
            n_params = len(datasets[ds_id].get("parameters", []))
            print(f"    [{status}] {ds_id}: {n_params} parameters")

    # Update _meta
    mission_data["_meta"] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hapi_server": HAPI_SERVER,
    }

    save_mission_json(mission_stem, mission_data)
    print(f"  Summary: {updated_count} updated, {new_count} new datasets")


def discover_unmatched(hapi_catalog: list[dict]):
    """Show HAPI datasets that don't match any known mission prefix."""
    unmatched = []
    for entry in hapi_catalog:
        ds_id = entry.get("id", "")
        mission, _ = match_dataset_to_mission(ds_id)
        if mission is None:
            unmatched.append(ds_id)

    print(f"\n{len(unmatched)} datasets not matched to any mission:")
    # Group by prefix for readability
    prefixes = {}
    for ds_id in unmatched:
        prefix = ds_id.split("_")[0] if "_" in ds_id else ds_id[:5]
        prefixes.setdefault(prefix, []).append(ds_id)

    for prefix in sorted(prefixes.keys()):
        ids = prefixes[prefix]
        print(f"  {prefix}: {len(ids)} datasets")
        if len(ids) <= 3:
            for ds_id in ids:
                print(f"    {ds_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Update per-mission JSON files from CDAWeb HAPI metadata"
    )
    parser.add_argument(
        "--mission",
        type=str,
        help="Update only this mission (e.g., PSP, ACE). Case-insensitive.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Show HAPI datasets that don't match any known mission",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    args = parser.parse_args()

    # Fetch HAPI catalog
    hapi_catalog = fetch_hapi_catalog()

    if args.discover:
        discover_unmatched(hapi_catalog)
        return

    if args.mission:
        # Update a single mission
        mission_stem = args.mission.lower()
        try:
            update_mission(mission_stem, hapi_catalog, verbose=args.verbose)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Update all missions
        for filepath in sorted(MISSIONS_DIR.glob("*.json")):
            mission_stem = filepath.stem
            try:
                update_mission(mission_stem, hapi_catalog, verbose=args.verbose)
            except Exception as e:
                print(f"  Error updating {mission_stem}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
