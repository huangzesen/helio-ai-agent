"""Test HAPI client for CDAWeb metadata discovery.

Validates:
  1. Can fetch the full HAPI catalog
  2. Can filter datasets by spacecraft pattern (PSP_*, SOLO_*)
  3. Can fetch parameter info for specific datasets
  4. Can filter to 1D parameters (scalars and small vectors)

Run: python spikes/dataset_knowledge/test_hapi.py
"""

import requests
import json
from pprint import pprint

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi"


def get_catalog():
    """Fetch full HAPI catalog."""
    print("Fetching HAPI catalog...")
    resp = requests.get(f"{HAPI_BASE}/catalog", timeout=30)
    resp.raise_for_status()
    return resp.json()["catalog"]


def filter_datasets_by_pattern(catalog: list, pattern: str) -> list:
    """Filter catalog to datasets matching a pattern (e.g., 'PSP_')."""
    prefix = pattern.rstrip("*")
    return [d for d in catalog if d["id"].startswith(prefix)]


def get_dataset_info(dataset_id: str) -> dict:
    """Fetch parameter metadata for a dataset."""
    print(f"Fetching info for {dataset_id}...")
    resp = requests.get(f"{HAPI_BASE}/info", params={"id": dataset_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def filter_1d_parameters(info: dict) -> list:
    """Filter to scalar/vector (size <= 3) numeric parameters.

    HAPI parameter 'size' field:
    - Omitted or [1] = scalar
    - [3] = 3-component vector
    - [N] where N > 3 = likely spectrogram or higher-dim data (exclude)
    """
    params = []
    for p in info.get("parameters", []):
        name = p.get("name", "")

        # Skip the Time parameter
        if name.lower() == "time":
            continue

        # Get size - default to scalar [1] if not specified
        size = p.get("size")
        if size is None:
            size = [1]
        elif isinstance(size, int):
            size = [size]

        # Filter: 1D means single dimension with size <= 3
        if len(size) == 1 and size[0] <= 3:
            ptype = p.get("type", "")
            if ptype in ("double", "integer", "isotime"):
                params.append(p)

    return params


def main():
    print("=" * 60)
    print("  HAPI Client Spike Test")
    print("=" * 60)
    print()

    # --- Test 1: Fetch catalog ---
    print("Test 1: Fetch HAPI catalog")
    catalog = get_catalog()
    print(f"  Total datasets in catalog: {len(catalog)}")
    print("  PASS")
    print()

    # --- Test 2: Filter by spacecraft ---
    print("Test 2: Filter datasets by spacecraft")

    psp_datasets = filter_datasets_by_pattern(catalog, "PSP_")
    print(f"  PSP datasets: {len(psp_datasets)}")
    if psp_datasets:
        print(f"    Examples: {[d['id'] for d in psp_datasets[:5]]}")

    solo_datasets = filter_datasets_by_pattern(catalog, "SOLO_")
    print(f"  Solar Orbiter datasets: {len(solo_datasets)}")
    if solo_datasets:
        print(f"    Examples: {[d['id'] for d in solo_datasets[:5]]}")

    if psp_datasets and solo_datasets:
        print("  PASS")
    else:
        print("  FAIL: No datasets found for PSP or SOLO")
        return
    print()

    # --- Test 3: Fetch specific dataset info ---
    print("Test 3: Fetch dataset info")

    # PSP magnetic field
    psp_mag_id = "PSP_FLD_L2_MAG_RTN_1MIN"
    try:
        psp_mag_info = get_dataset_info(psp_mag_id)
        print(f"  {psp_mag_id}:")
        print(f"    Start: {psp_mag_info.get('startDate', 'N/A')}")
        print(f"    Stop:  {psp_mag_info.get('stopDate', 'N/A')}")
        print(f"    Parameters: {len(psp_mag_info.get('parameters', []))}")
    except Exception as e:
        print(f"  FAIL fetching {psp_mag_id}: {e}")
        # Try alternate dataset
        psp_mag_id = "PSP_FLD_L2_MAG_RTN"
        psp_mag_info = get_dataset_info(psp_mag_id)
        print(f"  Fallback to {psp_mag_id}")

    # Solar Orbiter magnetic field
    solo_mag_id = "SOLO_L2_MAG-RTN-NORMAL-1-MINUTE"
    try:
        solo_mag_info = get_dataset_info(solo_mag_id)
        print(f"  {solo_mag_id}:")
        print(f"    Start: {solo_mag_info.get('startDate', 'N/A')}")
        print(f"    Stop:  {solo_mag_info.get('stopDate', 'N/A')}")
        print(f"    Parameters: {len(solo_mag_info.get('parameters', []))}")
    except Exception as e:
        print(f"  FAIL fetching {solo_mag_id}: {e}")
        solo_mag_info = None

    print("  PASS")
    print()

    # --- Test 4: Filter to 1D parameters ---
    print("Test 4: Filter to 1D parameters")

    psp_1d_params = filter_1d_parameters(psp_mag_info)
    print(f"  {psp_mag_id} - 1D parameters ({len(psp_1d_params)}):")
    for p in psp_1d_params[:5]:
        size = p.get("size", [1])
        units = p.get("units", "")
        desc = p.get("description", "")[:50]
        print(f"    - {p['name']}: size={size}, units={units}")
        if desc:
            print(f"      {desc}...")

    if solo_mag_info:
        solo_1d_params = filter_1d_parameters(solo_mag_info)
        print(f"  {solo_mag_id} - 1D parameters ({len(solo_1d_params)}):")
        for p in solo_1d_params[:5]:
            size = p.get("size", [1])
            units = p.get("units", "")
            print(f"    - {p['name']}: size={size}, units={units}")

    if psp_1d_params:
        print("  PASS")
    else:
        print("  FAIL: No 1D parameters found")
        return
    print()

    # --- Summary ---
    print("=" * 60)
    print("ALL TESTS PASS")
    print()
    print("Key findings:")
    print(f"  - HAPI catalog has {len(catalog)} total datasets")
    print(f"  - PSP has {len(psp_datasets)} datasets")
    print(f"  - Solar Orbiter has {len(solo_datasets)} datasets")
    print(f"  - PSP MAG dataset has {len(psp_1d_params)} 1D plottable parameters")


if __name__ == "__main__":
    main()
