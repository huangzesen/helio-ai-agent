"""
Auto-download mission data on first run.

When no mission JSON files exist (fresh clone), fetches the CDAWeb HAPI
catalog and populates all mission files + HAPI cache automatically.

This module is lazy-imported by mission_loader.load_all_missions() only
when no *.json files are found in knowledge/missions/.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

from .mission_prefixes import (
    match_dataset_to_mission,
    create_mission_skeleton,
    get_all_mission_stems,
    get_mission_name,
    PRIMARY_MISSIONS,
)


# Constants
HAPI_SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
MISSIONS_DIR = Path(__file__).parent / "missions"
DEFAULT_WORKERS = 10
MAX_RETRIES = 3

# Module-level flag: only check once per process
_bootstrap_checked = False


def ensure_missions_populated():
    """Check if primary mission data exists; download if missing.

    On first run, downloads only the 10 primary missions (~1-2 min).
    Users can download the full catalog (~50 missions) later with:
        python scripts/generate_mission_data.py --all

    Only runs once per process.
    """
    global _bootstrap_checked
    if _bootstrap_checked:
        return
    _bootstrap_checked = True

    primary_stems = set(PRIMARY_MISSIONS)
    existing_stems = {f.stem for f in MISSIONS_DIR.glob("*.json")}
    missing = primary_stems - existing_stems

    if not missing:
        return  # All primary missions present

    is_fresh = len(existing_stems) == 0

    print("\n" + "=" * 60)
    if is_fresh:
        print("  First run detected — downloading primary mission data.")
        print(f"  ({len(primary_stems)} missions, typically 1-2 minutes)")
    else:
        print(f"  {len(missing)} primary mission(s) missing.")
        print(f"  Downloading: {', '.join(sorted(missing))}")
    print()
    print("  To download ALL missions (~50), run:")
    print("    python scripts/generate_mission_data.py --all")
    print("=" * 60 + "\n")

    try:
        populate_missions(only_stems=missing)
    except Exception as e:
        print(f"\nWarning: Auto-download failed: {e}")
        print("The agent will start with a partial catalog.")
        print("You can retry by restarting, or run manually:")
        print("  python scripts/generate_mission_data.py --force\n")


def populate_missions(only_stems: set[str] | None = None):
    """Download and populate mission data from CDAWeb HAPI.

    Args:
        only_stems: If provided, only download these mission stems.
                    If None, download all missions found in the HAPI catalog.

    Steps:
      1. Fetch HAPI /catalog (all dataset IDs)
      2. Group datasets by mission via prefix matching
      3. Filter to only_stems if specified
      4. Create skeleton mission JSONs for missing ones
      5. Parallel-fetch HAPI /info for all datasets (with retries)
      6. Merge /info into mission JSONs
      7. Generate _index.json and _calibration_exclude.json per mission
    """
    if requests is None:
        raise RuntimeError(
            "'requests' package is required for auto-download. "
            "Install with: pip install requests"
        )

    start_time = time.time()

    # Step 1: Fetch HAPI catalog
    catalog = _fetch_catalog()

    # Step 1b: Fetch CDAWeb REST API metadata (InstrumentType per dataset)
    from .cdaweb_metadata import fetch_dataset_metadata
    print("Fetching CDAWeb dataset metadata (instrument types)...")
    cdaweb_meta = fetch_dataset_metadata()
    if cdaweb_meta:
        print(f"  Got metadata for {len(cdaweb_meta)} datasets")
    else:
        print("  Warning: CDAWeb metadata unavailable, falling back to prefix hints")

    # Step 2: Group datasets by mission
    mission_datasets = _group_by_mission(catalog)

    # Step 3: Filter to requested stems
    if only_stems:
        mission_datasets = {
            stem: ds for stem, ds in mission_datasets.items()
            if stem in only_stems
        }

    total_datasets = sum(len(ds) for ds in mission_datasets.values())
    print(f"Grouped {total_datasets} datasets into {len(mission_datasets)} missions\n")

    # Step 4: Create skeleton JSONs for missions that don't exist yet
    # Include all requested stems, even those with no HAPI datasets
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stems_to_skeleton = set(mission_datasets.keys())
    if only_stems:
        stems_to_skeleton |= only_stems
    for stem in sorted(stems_to_skeleton):
        filepath = MISSIONS_DIR / f"{stem}.json"
        if not filepath.exists():
            skeleton = create_mission_skeleton(stem)
            _save_json(filepath, skeleton)

    # Step 5: Parallel-fetch HAPI /info for all datasets
    all_fetch_items = []
    for stem, datasets in mission_datasets.items():
        cache_dir = MISSIONS_DIR / stem / "hapi"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for ds_id, instrument_hint in datasets:
            all_fetch_items.append((ds_id, instrument_hint, stem, cache_dir))

    print(f"Fetching HAPI /info for {len(all_fetch_items)} datasets...")
    results = _fetch_all_info(all_fetch_items)

    # Step 6: Merge results into mission JSONs
    _merge_into_missions(mission_datasets, results, cdaweb_meta)

    # Step 7: Generate per-mission index and calibration exclude files
    for stem in mission_datasets:
        _generate_index(stem)
        _ensure_calibration_exclude(stem)

    elapsed = time.time() - start_time
    n_success = sum(1 for r in results.values() if r is not None)
    n_failed = len(results) - n_success
    print(f"\nBootstrap complete in {elapsed:.0f}s: "
          f"{len(mission_datasets)} missions, "
          f"{n_success} datasets fetched"
          + (f", {n_failed} failed" if n_failed else ""))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_catalog() -> list[dict]:
    """Fetch the full HAPI catalog from CDAWeb."""
    url = f"{HAPI_SERVER}/catalog"
    print(f"Fetching HAPI catalog from {url}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    catalog = data.get("catalog", [])
    print(f"  Found {len(catalog)} datasets in catalog")
    return catalog


def _group_by_mission(catalog: list[dict]) -> dict[str, list[tuple[str, str | None]]]:
    """Group catalog entries by mission stem.

    Returns:
        Dict mapping mission_stem -> list of (dataset_id, instrument_hint).
    """
    groups: dict[str, list[tuple[str, str | None]]] = {}
    for entry in catalog:
        ds_id = entry.get("id", "")
        mission_stem, instrument_hint = match_dataset_to_mission(ds_id)
        if mission_stem:
            groups.setdefault(mission_stem, []).append((ds_id, instrument_hint))
    return groups


def _fetch_single_info(ds_id: str) -> dict | None:
    """Fetch HAPI /info for a single dataset. Returns parsed JSON or None."""
    url = f"{HAPI_SERVER}/info"
    try:
        resp = requests.get(url, params={"id": ds_id}, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _fetch_and_save(ds_id: str, cache_dir: Path) -> dict | None:
    """Fetch a single dataset's HAPI info and save to cache. Thread-safe.

    Returns parsed info dict or None on failure.
    """
    info = _fetch_single_info(ds_id)
    if info is None:
        return None

    cache_file = cache_dir / f"{ds_id}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError:
        pass  # Cache write failure is non-fatal

    return info


def _fetch_all_info(
    items: list[tuple[str, str | None, str, Path]],
) -> dict[str, dict | None]:
    """Parallel-fetch HAPI /info for all datasets, with retries.

    Args:
        items: List of (dataset_id, instrument_hint, mission_stem, cache_dir).

    Returns:
        Dict mapping dataset_id -> info dict (or None if all retries failed).
    """
    # Try tqdm for progress bar, fall back to simple counter
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    results: dict[str, dict | None] = {}
    pending = [(ds_id, cache_dir) for ds_id, _, _, cache_dir in items]

    for attempt in range(1, MAX_RETRIES + 1):
        if not pending:
            break

        if attempt > 1:
            print(f"\nRetry {attempt}/{MAX_RETRIES}: {len(pending)} datasets remaining...")

        failed = []

        if has_tqdm:
            pbar = tqdm(
                total=len(pending),
                desc=f"Downloading (attempt {attempt}/{MAX_RETRIES})"
                     if attempt > 1 else "Downloading",
                unit="ds",
                ncols=80,
            )
        else:
            counter = {"done": 0, "total": len(pending)}

        with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_and_save, ds_id, cache_dir): (ds_id, cache_dir)
                for ds_id, cache_dir in pending
            }

            for future in as_completed(futures):
                ds_id, cache_dir = futures[future]
                info = future.result()
                results[ds_id] = info

                if info is None:
                    failed.append((ds_id, cache_dir))

                if has_tqdm:
                    pbar.update(1)
                else:
                    counter["done"] += 1
                    if counter["done"] % 100 == 0 or counter["done"] == counter["total"]:
                        print(f"  {counter['done']}/{counter['total']} datasets processed")

        if has_tqdm:
            pbar.close()

        pending = failed

    if pending:
        print(f"\nWarning: {len(pending)} datasets failed after {MAX_RETRIES} attempts")

    return results


def _merge_dataset_info(hapi_info: dict, cdaweb_entry: dict | None = None) -> dict:
    """Extract dataset entry from HAPI /info + CDAWeb metadata.

    Only stores lightweight catalog info (description, dates, PI, DOI).
    Full parameter details stay in the per-dataset HAPI cache files at
    knowledge/missions/{mission}/hapi/{dataset_id}.json — loaded on demand
    by hapi_client.py when the agent needs them.

    Args:
        hapi_info: Response from HAPI /info endpoint.
        cdaweb_entry: Optional metadata from CDAWeb REST API for this dataset.
    """
    entry = {
        "description": "",
        "start_date": hapi_info.get("startDate", ""),
        "stop_date": hapi_info.get("stopDate", ""),
    }

    # Enrich with CDAWeb metadata
    if cdaweb_entry:
        entry["description"] = cdaweb_entry.get("label", "")
        if cdaweb_entry.get("pi_name"):
            entry["pi_name"] = cdaweb_entry["pi_name"]
        if cdaweb_entry.get("pi_affiliation"):
            entry["pi_affiliation"] = cdaweb_entry["pi_affiliation"]
        if cdaweb_entry.get("doi"):
            entry["doi"] = cdaweb_entry["doi"]
        if cdaweb_entry.get("notes_url"):
            entry["notes_url"] = cdaweb_entry["notes_url"]

    # Fall back to HAPI description if no CDAWeb label
    if not entry["description"]:
        entry["description"] = hapi_info.get("description", "")

    return entry


def _merge_into_missions(
    mission_datasets: dict[str, list[tuple[str, str | None]]],
    results: dict[str, dict | None],
    cdaweb_meta: dict[str, dict] | None = None,
):
    """Merge fetched HAPI /info results into mission JSON files.

    Uses CDAWeb InstrumentType metadata (when available) to group datasets
    into meaningful instrument categories instead of dumping into "General".

    Priority for instrument assignment:
      1. Dataset already exists in a named instrument → keep it
      2. Prefix hint from mission_prefixes → use it (preserves curated structure)
      3. CDAWeb InstrumentType → group by primary type
      4. Fallback → "General"

    After merging, backfills keywords for instruments that have keywords=[].
    """
    if cdaweb_meta is None:
        cdaweb_meta = {}

    from .cdaweb_metadata import pick_primary_type, get_type_info

    for stem, datasets in mission_datasets.items():
        filepath = MISSIONS_DIR / f"{stem}.json"
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            mission_data = json.load(f)

        updated = 0
        for ds_id, instrument_hint in datasets:
            info = results.get(ds_id)
            if info is None:
                continue

            # Priority 1: dataset already exists in a named instrument
            target_instrument = None
            for inst_id, inst in mission_data.get("instruments", {}).items():
                if ds_id in inst.get("datasets", {}):
                    target_instrument = inst_id
                    break

            # Priority 2: prefix hint from mission_prefixes
            if target_instrument is None and instrument_hint:
                if instrument_hint not in mission_data.get("instruments", {}):
                    mission_data.setdefault("instruments", {})[instrument_hint] = {
                        "name": instrument_hint,
                        "keywords": [],
                        "datasets": {},
                    }
                target_instrument = instrument_hint

            # Priority 3: CDAWeb InstrumentType grouping
            if target_instrument is None:
                meta = cdaweb_meta.get(ds_id)
                if meta and meta.get("instrument_types"):
                    primary_type = pick_primary_type(meta["instrument_types"])
                    if primary_type:
                        type_info = get_type_info(primary_type)
                        inst_key = type_info["id"]
                        if inst_key not in mission_data.get("instruments", {}):
                            mission_data.setdefault("instruments", {})[inst_key] = {
                                "name": type_info["name"],
                                "keywords": list(type_info["keywords"]),
                                "datasets": {},
                            }
                        target_instrument = inst_key

            # Priority 4: fallback to "General"
            if target_instrument is None:
                if "General" not in mission_data.get("instruments", {}):
                    mission_data.setdefault("instruments", {})["General"] = {
                        "name": "General",
                        "keywords": [],
                        "datasets": {},
                    }
                target_instrument = "General"

            inst = mission_data["instruments"][target_instrument]
            # Look up CDAWeb metadata; fall back to base ID without @N suffix
            cdaweb_entry = cdaweb_meta.get(ds_id)
            if cdaweb_entry is None and "@" in ds_id:
                cdaweb_entry = cdaweb_meta.get(ds_id.split("@")[0])
            inst.setdefault("datasets", {})[ds_id] = _merge_dataset_info(
                info, cdaweb_entry
            )
            updated += 1

            # Store observatory_group at mission level (from first dataset that has it)
            if cdaweb_entry and cdaweb_entry.get("observatory_group"):
                if "observatory_group" not in mission_data:
                    mission_data["observatory_group"] = cdaweb_entry["observatory_group"]

        # Backfill keywords for instruments that have keywords=[]
        _backfill_instrument_keywords(mission_data, cdaweb_meta)

        # Remove empty "General" if other instruments exist
        instruments = mission_data.get("instruments", {})
        if ("General" in instruments
                and not instruments["General"].get("datasets")
                and len(instruments) > 1):
            del instruments["General"]

        # Update _meta
        mission_data["_meta"] = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hapi_server": HAPI_SERVER,
        }

        _save_json(filepath, mission_data)

    print(f"Merged data into {len(mission_datasets)} mission JSON files")


def _backfill_instrument_keywords(
    mission_data: dict,
    cdaweb_meta: dict[str, dict],
):
    """Backfill keywords for instruments that have keywords=[].

    Looks up the InstrumentType of their datasets via CDAWeb metadata
    and sets keywords from the type info.
    """
    from .cdaweb_metadata import pick_primary_type, get_type_info

    for inst_id, inst in mission_data.get("instruments", {}).items():
        if inst.get("keywords"):
            continue  # Already has keywords

        # Collect InstrumentTypes from all datasets in this instrument
        all_types = set()
        for ds_id in inst.get("datasets", {}):
            meta = cdaweb_meta.get(ds_id)
            if meta and meta.get("instrument_types"):
                for t in meta["instrument_types"]:
                    all_types.add(t)

        if not all_types:
            continue

        # Pick primary type and use its keywords
        primary = pick_primary_type(list(all_types))
        if primary:
            type_info = get_type_info(primary)
            if type_info.get("keywords"):
                inst["keywords"] = list(type_info["keywords"])


def _generate_index(mission_stem: str):
    """Generate _index.json summary for a mission's HAPI cache."""
    cache_dir = MISSIONS_DIR / mission_stem / "hapi"
    if not cache_dir.exists():
        return

    index_entries = []
    for cache_file in sorted(cache_dir.glob("*.json")):
        if cache_file.name.startswith("_"):
            continue
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                info = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        ds_id = cache_file.stem
        param_count = sum(
            1 for p in info.get("parameters", [])
            if p.get("name", "").lower() != "time"
        )
        start_date = info.get("startDate", "")
        stop_date = info.get("stopDate", "")
        if start_date and "T" in start_date:
            start_date = start_date.split("T")[0]
        if stop_date and "T" in stop_date:
            stop_date = stop_date.split("T")[0]

        index_entries.append({
            "id": ds_id,
            "description": info.get("description", ""),
            "start_date": start_date,
            "stop_date": stop_date,
            "parameter_count": param_count,
            "instrument": "",
        })

    # Read mission ID from JSON
    mission_json = MISSIONS_DIR / f"{mission_stem}.json"
    mission_id = mission_stem.upper()
    if mission_json.exists():
        try:
            with open(mission_json, "r", encoding="utf-8") as f:
                mission_id = json.load(f).get("id", mission_id)
        except (json.JSONDecodeError, OSError):
            pass

    index_data = {
        "mission_id": mission_id,
        "dataset_count": len(index_entries),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": index_entries,
    }
    index_file = cache_dir / "_index.json"
    _save_json(index_file, index_data)


def _ensure_calibration_exclude(mission_stem: str):
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
        _save_json(exclude_file, exclude_data)


def clean_all_missions(only_stems: set[str] | None = None):
    """Delete mission JSONs and HAPI cache dirs for a fresh rebuild.

    Args:
        only_stems: If provided, only delete these mission stems.
                    If None, delete everything.

    Preserves the missions/ directory itself but removes generated content.
    Returns the count of (deleted_files, deleted_dirs) for logging.
    """
    global _bootstrap_checked
    _bootstrap_checked = False

    deleted_files = 0
    deleted_dirs = 0

    import shutil

    for filepath in MISSIONS_DIR.glob("*.json"):
        if only_stems and filepath.stem not in only_stems:
            continue
        filepath.unlink()
        deleted_files += 1

    for subdir in MISSIONS_DIR.iterdir():
        if subdir.is_dir():
            if only_stems and subdir.name not in only_stems:
                continue
            shutil.rmtree(subdir)
            deleted_dirs += 1

    return deleted_files, deleted_dirs


def _save_json(filepath: Path, data: dict):
    """Save a dict as JSON with consistent formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")
