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
    get_mission_name,
)


# Constants
HAPI_SERVER = "https://cdaweb.gsfc.nasa.gov/hapi"
MISSIONS_DIR = Path(__file__).parent / "missions"
DEFAULT_WORKERS = 10
MAX_RETRIES = 3


def ensure_missions_populated():
    """Check if mission data exists; if not, download it.

    Fast-path: if any *.json exists in missions/, returns immediately.
    Otherwise calls populate_missions() with full error handling.
    """
    if any(MISSIONS_DIR.glob("*.json")):
        return

    print("\n" + "=" * 60)
    print("  First run detected — no mission data found.")
    print("  Downloading CDAWeb HAPI metadata...")
    print("  (This is a one-time setup, typically 5-10 minutes)")
    print("=" * 60 + "\n")

    try:
        populate_missions()
    except Exception as e:
        print(f"\nWarning: Auto-download failed: {e}")
        print("The agent will start with an empty catalog.")
        print("You can retry by deleting knowledge/missions/*.json and restarting,")
        print("or run manually: python scripts/generate_mission_data.py --create-new\n")


def populate_missions():
    """Download and populate all mission data from CDAWeb HAPI.

    Steps:
      1. Fetch HAPI /catalog (all dataset IDs)
      2. Group datasets by mission via prefix matching
      3. Create skeleton mission JSONs
      4. Parallel-fetch HAPI /info for all datasets (with retries)
      5. Merge /info into mission JSONs
      6. Generate _index.json and _calibration_exclude.json per mission
    """
    if requests is None:
        raise RuntimeError(
            "'requests' package is required for auto-download. "
            "Install with: pip install requests"
        )

    start_time = time.time()

    # Step 1: Fetch HAPI catalog
    catalog = _fetch_catalog()

    # Step 2: Group datasets by mission
    mission_datasets = _group_by_mission(catalog)
    total_datasets = sum(len(ds) for ds in mission_datasets.values())
    print(f"Grouped {total_datasets} datasets into {len(mission_datasets)} missions\n")

    # Step 3: Create skeleton JSONs for all missions
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for stem in sorted(mission_datasets.keys()):
        filepath = MISSIONS_DIR / f"{stem}.json"
        if not filepath.exists():
            skeleton = create_mission_skeleton(stem)
            _save_json(filepath, skeleton)

    # Step 4: Parallel-fetch HAPI /info for all datasets
    all_fetch_items = []
    for stem, datasets in mission_datasets.items():
        cache_dir = MISSIONS_DIR / stem / "hapi"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for ds_id, instrument_hint in datasets:
            all_fetch_items.append((ds_id, instrument_hint, stem, cache_dir))

    print(f"Fetching HAPI /info for {len(all_fetch_items)} datasets...")
    results = _fetch_all_info(all_fetch_items)

    # Step 5: Merge results into mission JSONs
    _merge_into_missions(mission_datasets, results)

    # Step 6: Generate per-mission index and calibration exclude files
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


def _merge_dataset_info(hapi_info: dict) -> dict:
    """Extract dataset entry from HAPI /info response."""
    parameters = []
    for param in hapi_info.get("parameters", []):
        name = param.get("name", "")
        if name.lower() == "time":
            continue
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
        "description": hapi_info.get("description", ""),
        "start_date": hapi_info.get("startDate", ""),
        "stop_date": hapi_info.get("stopDate", ""),
        "parameters": parameters,
    }


def _merge_into_missions(
    mission_datasets: dict[str, list[tuple[str, str | None]]],
    results: dict[str, dict | None],
):
    """Merge fetched HAPI /info results into mission JSON files."""
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

            # Find target instrument
            target_instrument = None
            for inst_id, inst in mission_data.get("instruments", {}).items():
                if ds_id in inst.get("datasets", {}):
                    target_instrument = inst_id
                    break

            if target_instrument is None and instrument_hint:
                if instrument_hint in mission_data.get("instruments", {}):
                    target_instrument = instrument_hint

            if target_instrument is None:
                if "General" not in mission_data.get("instruments", {}):
                    mission_data.setdefault("instruments", {})["General"] = {
                        "name": "General",
                        "keywords": [],
                        "datasets": {},
                    }
                target_instrument = "General"

            inst = mission_data["instruments"][target_instrument]
            inst.setdefault("datasets", {})[ds_id] = _merge_dataset_info(info)
            updated += 1

        # Update _meta
        mission_data["_meta"] = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hapi_server": HAPI_SERVER,
        }

        _save_json(filepath, mission_data)

    print(f"Merged data into {len(mission_datasets)} mission JSON files")


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


def _save_json(filepath: Path, data: dict):
    """Save a dict as JSON with consistent formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")
