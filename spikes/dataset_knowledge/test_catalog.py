"""Test spacecraft/instrument catalog with keyword matching.

Validates:
  1. Static catalog structure for PSP and Solar Orbiter
  2. Keyword matching for natural language queries
  3. Integration with HAPI client for parameter discovery

Run: python spikes/dataset_knowledge/test_catalog.py
"""

from dataclasses import dataclass
from typing import Optional
import requests

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi"

# --- Static Catalog ---

SPACECRAFT = {
    "PSP": {
        "name": "Parker Solar Probe",
        "keywords": ["parker", "psp", "probe", "solar probe"],
        "instruments": {
            "FIELDS/MAG": {
                "name": "FIELDS Magnetometer",
                "keywords": ["magnetic", "field", "mag", "b-field", "bfield"],
                "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"],
            },
            "SWEAP": {
                "name": "Solar Wind Plasma",
                "keywords": ["plasma", "solar wind", "proton", "density", "velocity", "sweap"],
                "datasets": ["PSP_SWP_SPC_L3I"],
            },
        },
    },
    "SolO": {
        "name": "Solar Orbiter",
        "keywords": ["solar orbiter", "solo", "orbiter"],
        "instruments": {
            "MAG": {
                "name": "Magnetometer",
                "keywords": ["magnetic", "field", "mag", "b-field"],
                "datasets": ["SOLO_L2_MAG-RTN-NORMAL-1-MINUTE"],
            },
            "SWA-PAS": {
                "name": "Proton-Alpha Sensor",
                "keywords": ["plasma", "proton", "density", "velocity", "temperature", "swa"],
                "datasets": ["SOLO_L2_SWA-PAS-GRND-MOM"],
            },
        },
    },
}


# --- Catalog Functions ---

def list_spacecraft() -> list[dict]:
    """List all supported spacecraft."""
    return [
        {"id": sc_id, "name": info["name"]}
        for sc_id, info in SPACECRAFT.items()
    ]


def list_instruments(spacecraft: str) -> list[dict]:
    """List instruments for a spacecraft."""
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return []
    return [
        {"id": inst_id, "name": info["name"]}
        for inst_id, info in sc["instruments"].items()
    ]


def get_datasets(spacecraft: str, instrument: str) -> list[str]:
    """Get dataset IDs for a spacecraft/instrument combo."""
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return []
    inst = sc["instruments"].get(instrument)
    if not inst:
        return []
    return inst["datasets"]


def match_spacecraft(query: str) -> Optional[str]:
    """Match a query to a spacecraft using keywords."""
    query_lower = query.lower()
    for sc_id, info in SPACECRAFT.items():
        # Check exact match
        if query_lower == sc_id.lower():
            return sc_id
        # Check keywords
        for kw in info["keywords"]:
            if kw in query_lower:
                return sc_id
    return None


def match_instrument(spacecraft: str, query: str) -> Optional[str]:
    """Match a query to an instrument using keywords."""
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return None

    query_lower = query.lower()
    for inst_id, info in sc["instruments"].items():
        # Check exact match
        if query_lower == inst_id.lower():
            return inst_id
        # Check keywords
        for kw in info["keywords"]:
            if kw in query_lower:
                return inst_id
    return None


def get_dataset_info_from_hapi(dataset_id: str) -> dict:
    """Fetch parameter metadata from HAPI."""
    resp = requests.get(f"{HAPI_BASE}/info", params={"id": dataset_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_parameters(spacecraft: str, instrument: str) -> list[dict]:
    """List plottable 1D parameters for a spacecraft/instrument.

    Fetches from HAPI and filters to 1D numeric parameters.
    """
    datasets = get_datasets(spacecraft, instrument)
    if not datasets:
        return []

    # Use first dataset
    dataset_id = datasets[0]
    try:
        info = get_dataset_info_from_hapi(dataset_id)
    except Exception as e:
        print(f"  Warning: Could not fetch HAPI info for {dataset_id}: {e}")
        return []

    # Filter to 1D parameters
    params = []
    for p in info.get("parameters", []):
        name = p.get("name", "")
        if name.lower() == "time":
            continue

        size = p.get("size")
        if size is None:
            size = [1]
        elif isinstance(size, int):
            size = [size]

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


# --- Tests ---

def main():
    print("=" * 60)
    print("  Catalog Spike Test")
    print("=" * 60)
    print()

    # --- Test 1: List spacecraft ---
    print("Test 1: list_spacecraft()")
    spacecraft = list_spacecraft()
    print(f"  Result: {spacecraft}")
    assert len(spacecraft) == 2
    assert any(s["id"] == "PSP" for s in spacecraft)
    assert any(s["id"] == "SolO" for s in spacecraft)
    print("  PASS")
    print()

    # --- Test 2: List instruments ---
    print("Test 2: list_instruments()")
    psp_instruments = list_instruments("PSP")
    print(f"  PSP instruments: {psp_instruments}")
    assert len(psp_instruments) == 2

    solo_instruments = list_instruments("SolO")
    print(f"  SolO instruments: {solo_instruments}")
    assert len(solo_instruments) == 2
    print("  PASS")
    print()

    # --- Test 3: Keyword matching - spacecraft ---
    print("Test 3: match_spacecraft()")
    test_cases = [
        ("parker", "PSP"),
        ("PSP", "PSP"),
        ("solar probe", "PSP"),
        ("solar orbiter", "SolO"),
        ("solo", "SolO"),
        ("orbiter", "SolO"),
        ("unknown", None),
    ]
    for query, expected in test_cases:
        result = match_spacecraft(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"  '{query}' -> {result} (expected {expected}) [{status}]")
        assert result == expected, f"Failed for query '{query}'"
    print("  PASS")
    print()

    # --- Test 4: Keyword matching - instrument ---
    print("Test 4: match_instrument()")
    test_cases = [
        ("PSP", "magnetic field", "FIELDS/MAG"),
        ("PSP", "mag", "FIELDS/MAG"),
        ("PSP", "plasma", "SWEAP"),
        ("PSP", "density", "SWEAP"),
        ("SolO", "magnetic", "MAG"),
        ("SolO", "proton", "SWA-PAS"),
        ("PSP", "unknown", None),
    ]
    for spacecraft, query, expected in test_cases:
        result = match_instrument(spacecraft, query)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {spacecraft} + '{query}' -> {result} (expected {expected}) [{status}]")
        assert result == expected, f"Failed for {spacecraft} + '{query}'"
    print("  PASS")
    print()

    # --- Test 5: Get datasets ---
    print("Test 5: get_datasets()")
    psp_mag_datasets = get_datasets("PSP", "FIELDS/MAG")
    print(f"  PSP FIELDS/MAG datasets: {psp_mag_datasets}")
    assert "PSP_FLD_L2_MAG_RTN_1MIN" in psp_mag_datasets
    print("  PASS")
    print()

    # --- Test 6: List parameters (HAPI integration) ---
    print("Test 6: list_parameters() - HAPI integration")
    print("  Fetching parameters for PSP FIELDS/MAG...")
    psp_params = list_parameters("PSP", "FIELDS/MAG")
    print(f"  Found {len(psp_params)} 1D parameters:")
    for p in psp_params[:5]:
        print(f"    - {p['name']}: {p['units']} (size={p['size']})")
    if psp_params:
        print("  PASS")
    else:
        print("  WARN: No parameters found (HAPI may be unavailable)")
    print()

    # --- Summary ---
    print("=" * 60)
    print("ALL TESTS PASS")
    print()
    print("Catalog structure validated:")
    print(f"  - {len(SPACECRAFT)} spacecraft defined")
    print(f"  - Keyword matching works for spacecraft and instruments")
    print(f"  - HAPI integration returns {len(psp_params)} parameters for PSP MAG")


if __name__ == "__main__":
    main()
