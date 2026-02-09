"""
Auto-download mission data on first run.

When no mission JSON files exist (fresh clone), fetches the CDAWeb HAPI
catalog and populates all mission files + HAPI cache automatically.

This module is lazy-imported by mission_loader.load_all_missions() only
when no *.json files are found in knowledge/missions/.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("helio-agent")

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

# Guard against concurrent HAPI cache downloads (e.g. Gradio threads)
_downloading: set[str] = set()


def ensure_missions_populated():
    """Check if any mission data exists; download lightweight catalog if missing.

    On first run, downloads all missions from HAPI catalog + CDAWeb REST API
    (just 2 HTTP calls, ~10-30s). Detailed per-dataset HAPI parameter cache
    is downloaded lazily when browse_datasets is first called for a mission.

    Only runs once per process.
    """
    global _bootstrap_checked
    if _bootstrap_checked:
        return
    _bootstrap_checked = True

    existing_stems = {f.stem for f in MISSIONS_DIR.glob("*.json")}

    if existing_stems:
        return  # Already bootstrapped

    logger.info("First run detected — downloading mission catalog "
                "(all missions, typically 10-30 seconds). "
                "Detailed parameter cache downloads on demand per mission. "
                "To pre-download all caches, use: --download-hapi-cache")

    try:
        populate_missions_lightweight()
    except Exception as e:
        logger.warning("Auto-download failed: %s. "
                       "The agent will start with a partial catalog. "
                       "Retry by restarting, or run: "
                       "python scripts/generate_mission_data.py --force", e)


def populate_missions_lightweight():
    """Fast bootstrap: create mission JSONs from HAPI catalog + CDAWeb metadata.

    Downloads only catalog-level metadata (2 HTTP calls, ~10-30s).
    Skips per-dataset HAPI /info calls — those are deferred to
    populate_mission_hapi_cache() which runs lazily on first browse.

    Steps:
      1. Fetch HAPI /catalog (all dataset IDs)
      2. Fetch CDAWeb REST API metadata (descriptions, dates, instrument types)
      3. Group datasets by mission via prefix matching
      4. Create skeleton mission JSONs
      5. Build synthetic info dicts from CDAWeb metadata (no HAPI /info)
      6. Merge into mission JSONs using existing _merge_into_missions()
    """
    if requests is None:
        raise RuntimeError(
            "'requests' package is required for auto-download. "
            "Install with: pip install requests"
        )

    start_time = time.time()

    # Step 1: Fetch HAPI catalog
    catalog = _fetch_catalog()

    # Step 2: Fetch CDAWeb REST API metadata
    from .cdaweb_metadata import fetch_dataset_metadata
    logger.info("Fetching CDAWeb dataset metadata (instrument types)...")
    cdaweb_meta = fetch_dataset_metadata()
    if cdaweb_meta:
        logger.info("Got metadata for %d datasets", len(cdaweb_meta))
    else:
        logger.warning("CDAWeb metadata unavailable, falling back to prefix hints")

    # Step 3: Group datasets by mission
    mission_datasets = _group_by_mission(catalog)
    total_datasets = sum(len(ds) for ds in mission_datasets.values())
    logger.info("Grouped %d datasets into %d missions",
                total_datasets, len(mission_datasets))

    # Step 4: Create skeleton JSONs
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for stem in sorted(mission_datasets.keys()):
        filepath = MISSIONS_DIR / f"{stem}.json"
        if not filepath.exists():
            skeleton = create_mission_skeleton(stem)
            _save_json(filepath, skeleton)

    # Step 5: Build synthetic info dicts from CDAWeb metadata
    # (no HAPI /info calls — just startDate/stopDate/description from CDAWeb)
    results: dict[str, dict | None] = {}
    for stem, datasets in mission_datasets.items():
        for ds_id, instrument_hint in datasets:
            cdaweb_entry = cdaweb_meta.get(ds_id)
            if cdaweb_entry:
                results[ds_id] = {
                    "startDate": cdaweb_entry.get("start_date", ""),
                    "stopDate": cdaweb_entry.get("stop_date", ""),
                    "description": cdaweb_entry.get("label", ""),
                }
            else:
                # No CDAWeb metadata — still register the dataset with empty info
                results[ds_id] = {
                    "startDate": "",
                    "stopDate": "",
                    "description": "",
                }

    # Step 6: Merge into mission JSONs (reuses all instrument-grouping logic)
    _merge_into_missions(mission_datasets, results, cdaweb_meta)

    # Skip _generate_index() and _ensure_calibration_exclude() — those need
    # actual HAPI cache files which we haven't downloaded yet.

    elapsed = time.time() - start_time
    logger.info("Lightweight bootstrap complete in %.0fs: %d missions, %d datasets",
                elapsed, len(mission_datasets), len(results))


def populate_mission_hapi_cache(mission_stem: str):
    """Download HAPI /info cache for a single mission's datasets.

    Called lazily when browse_datasets is first invoked for a mission
    that doesn't have an _index.json yet.

    Args:
        mission_stem: Lowercase mission stem (e.g., 'ace', 'psp').
    """
    if requests is None:
        raise RuntimeError("'requests' package required for HAPI cache download")

    # Guard against concurrent downloads of the same mission
    if mission_stem in _downloading:
        logger.debug("HAPI cache download already in progress for %s", mission_stem)
        return
    _downloading.add(mission_stem)

    try:
        mission_json = MISSIONS_DIR / f"{mission_stem}.json"
        if not mission_json.exists():
            logger.warning("No mission JSON for '%s', cannot download HAPI cache",
                           mission_stem)
            return

        with open(mission_json, "r", encoding="utf-8") as f:
            mission_data = json.load(f)

        # Collect all dataset IDs from instruments
        ds_ids = []
        for inst in mission_data.get("instruments", {}).values():
            for ds_id in inst.get("datasets", {}):
                ds_ids.append(ds_id)

        if not ds_ids:
            logger.warning("Mission '%s' has no datasets to cache", mission_stem)
            return

        cache_dir = MISSIONS_DIR / mission_stem / "hapi"
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading HAPI cache for %s (%d datasets)...",
                     mission_stem, len(ds_ids))

        items = [(ds_id, None, mission_stem, cache_dir) for ds_id in ds_ids]
        start_time = time.time()
        results = _fetch_all_info(items)

        # Generate index and calibration exclude
        _generate_index(mission_stem)
        _ensure_calibration_exclude(mission_stem)

        elapsed = time.time() - start_time
        n_success = sum(1 for r in results.values() if r is not None)
        logger.info("HAPI cache for %s complete in %.0fs: %d/%d datasets",
                     mission_stem, elapsed, n_success, len(ds_ids))
    finally:
        _downloading.discard(mission_stem)


def populate_all_hapi_caches(only_stems: set[str] | None = None):
    """Download HAPI cache for all missions that are missing _index.json.

    Args:
        only_stems: If provided, only download for these mission stems.
                    If None, download for all missions found in MISSIONS_DIR.
    """
    for filepath in sorted(MISSIONS_DIR.glob("*.json")):
        stem = filepath.stem
        if only_stems and stem not in only_stems:
            continue
        index_file = MISSIONS_DIR / stem / "hapi" / "_index.json"
        if index_file.exists():
            continue
        try:
            populate_mission_hapi_cache(stem)
        except Exception as e:
            logger.warning("HAPI cache download failed for %s: %s", stem, e)


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
    logger.info("Fetching CDAWeb dataset metadata (instrument types)...")
    cdaweb_meta = fetch_dataset_metadata()
    if cdaweb_meta:
        logger.info("Got metadata for %d datasets", len(cdaweb_meta))
    else:
        logger.warning("CDAWeb metadata unavailable, falling back to prefix hints")

    # Step 2: Group datasets by mission
    mission_datasets = _group_by_mission(catalog)

    # Step 3: Filter to requested stems
    if only_stems:
        mission_datasets = {
            stem: ds for stem, ds in mission_datasets.items()
            if stem in only_stems
        }

    total_datasets = sum(len(ds) for ds in mission_datasets.values())
    logger.info("Grouped %d datasets into %d missions",
                total_datasets, len(mission_datasets))

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

    logger.info("Fetching HAPI /info for %d datasets...", len(all_fetch_items))
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
    msg = (f"Bootstrap complete in {elapsed:.0f}s: "
           f"{len(mission_datasets)} missions, {n_success} datasets fetched")
    if n_failed:
        msg += f", {n_failed} failed"
    logger.info(msg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_catalog() -> list[dict]:
    """Fetch the full HAPI catalog from CDAWeb."""
    url = f"{HAPI_SERVER}/catalog"
    logger.info("Fetching HAPI catalog from %s...", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    catalog = data.get("catalog", [])
    logger.info("Found %d datasets in catalog", len(catalog))
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
    # Try tqdm for interactive terminal progress, fall back to logger.
    # The logger-based path also ensures Gradio's live log can display
    # progress (tqdm writes to stderr which bypasses the logging system).
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Detect if a Gradio _ListHandler is attached (i.e. we're inside the
    # web UI background thread).  In that case prefer logger-based progress
    # so the lines appear in the live log panel.
    _use_tqdm = has_tqdm and not any(
        type(h).__name__ == "_ListHandler"
        for h in logging.getLogger("helio-agent").handlers
    )

    results: dict[str, dict | None] = {}
    pending = [(ds_id, cache_dir) for ds_id, _, _, cache_dir in items]

    for attempt in range(1, MAX_RETRIES + 1):
        if not pending:
            break

        if attempt > 1:
            logger.info("Retry %d/%d: %d datasets remaining...",
                        attempt, MAX_RETRIES, len(pending))

        failed = []

        if _use_tqdm:
            pbar = tqdm(
                total=len(pending),
                desc=f"Downloading (attempt {attempt}/{MAX_RETRIES})"
                     if attempt > 1 else "Downloading",
                unit="ds",
                ncols=80,
            )
        else:
            counter = {"done": 0, "total": len(pending)}
            # Log every N items (more frequent for smaller batches)
            _log_every = max(1, len(pending) // 10)

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

                if _use_tqdm:
                    pbar.update(1)
                else:
                    counter["done"] += 1
                    if counter["done"] % _log_every == 0 or counter["done"] == counter["total"]:
                        pct = counter["done"] * 100 // counter["total"]
                        logger.info("Downloading: %d/%d datasets (%d%%)",
                                    counter["done"], counter["total"], pct)

        if _use_tqdm:
            pbar.close()

        pending = failed

    if pending:
        logger.warning("%d datasets failed after %d attempts",
                       len(pending), MAX_RETRIES)

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

    logger.info("Merged data into %d mission JSON files", len(mission_datasets))


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


def _patch_hapi_cache_dates(
    cache_dir: Path, ds_id: str, start_date: str, stop_date: str,
):
    """Patch startDate/stopDate in an individual HAPI cache file."""
    cache_file = cache_dir / f"{ds_id}.json"
    if not cache_file.exists():
        return
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        changed = False
        if start_date and info.get("startDate") != start_date:
            info["startDate"] = start_date
            changed = True
        if stop_date and info.get("stopDate") != stop_date:
            info["stopDate"] = stop_date
            changed = True
        if changed:
            _save_json(cache_file, info)
    except (json.JSONDecodeError, OSError):
        pass  # Non-fatal — mission JSON is the primary source of truth


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


def refresh_time_ranges(only_stems: set[str] | None = None) -> dict:
    """Lightweight refresh: update only start_date/stop_date in mission JSONs.

    Fetches the CDAWeb REST API catalog (single HTTP request, ~3s) to get
    fresh TimeInterval dates for all ~3000 datasets, then patches the
    existing mission JSONs.  No HAPI /info calls, no instrument regrouping.

    Args:
        only_stems: If provided, only refresh these mission stems.
                    If None, refresh every *.json in MISSIONS_DIR.

    Returns:
        Dict with keys: missions_updated, datasets_updated,
        datasets_failed, elapsed_seconds.
    """
    if requests is None:
        raise RuntimeError(
            "'requests' package is required for refresh. "
            "Install with: pip install requests"
        )

    start_time = time.time()

    # Step 1: Fetch all dataset dates from CDAWeb in one HTTP call
    from .cdaweb_metadata import fetch_dataset_metadata
    logger.info("Fetching CDAWeb catalog for time ranges...")
    cdaweb_meta = fetch_dataset_metadata()
    if not cdaweb_meta:
        logger.error("CDAWeb catalog unavailable — cannot refresh.")
        return {
            "missions_updated": 0,
            "datasets_updated": 0,
            "datasets_failed": 0,
            "elapsed_seconds": round(time.time() - start_time, 1),
        }
    logger.info("Got time ranges for %d datasets", len(cdaweb_meta))

    # Step 2: Walk existing mission JSONs and patch dates
    total_updated = 0
    total_failed = 0
    missions_updated = 0
    total_datasets = 0

    for filepath in sorted(MISSIONS_DIR.glob("*.json")):
        stem = filepath.stem
        if only_stems and stem not in only_stems:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                mission_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        cache_dir = MISSIONS_DIR / stem / "hapi"
        stem_updated = 0
        for inst in mission_data.get("instruments", {}).values():
            for ds_id, ds_entry in inst.get("datasets", {}).items():
                total_datasets += 1
                meta = cdaweb_meta.get(ds_id)
                if meta is None:
                    total_failed += 1
                    continue
                new_start = meta.get("start_date", "")
                new_stop = meta.get("stop_date", "")
                if new_start:
                    ds_entry["start_date"] = new_start
                if new_stop:
                    ds_entry["stop_date"] = new_stop
                stem_updated += 1

                # Also patch the individual HAPI cache file
                _patch_hapi_cache_dates(cache_dir, ds_id, new_start, new_stop)

        if stem_updated > 0:
            mission_data.setdefault("_meta", {})
            mission_data["_meta"]["generated_at"] = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            _save_json(filepath, mission_data)
            missions_updated += 1

        total_updated += stem_updated

        _generate_index(stem)

    elapsed = time.time() - start_time

    if total_failed:
        logger.warning("%d/%d dataset(s) not found in CDAWeb catalog "
                       "(dates left unchanged)", total_failed, total_datasets)

    msg = (f"Time-range refresh complete in {elapsed:.1f}s: "
           f"{missions_updated} missions, {total_updated} datasets updated")
    if total_failed:
        msg += f", {total_failed} failed"
    logger.info(msg)

    return {
        "missions_updated": missions_updated,
        "datasets_updated": total_updated,
        "datasets_failed": total_failed,
        "elapsed_seconds": round(elapsed, 1),
    }


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
