#!/usr/bin/env python3
"""
Test 4: Knowledge Modules on macOS

Tests knowledge base functionality:
- Static catalog search (keyword matching)
- HAPI client (API calls to CDAWeb)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_catalog_import():
    """Test importing catalog module."""
    print("\n[1/6] Testing catalog import...")

    try:
        from knowledge import catalog
        print(f"  ✓ Catalog module imported")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import catalog: {e}")
        return False


def test_catalog_search():
    """Test keyword-based dataset search."""
    print("\n[2/6] Testing catalog search...")

    try:
        from knowledge.catalog import search_by_keywords

        # Test 1: Parker Solar Probe magnetic field
        result = search_by_keywords("parker magnetic field")
        if result and result.get("spacecraft") == "PSP":
            print(f"  ✓ Found Parker: {result['spacecraft_name']}")
            print(f"    Instrument: {result.get('instrument_name')}")
            print(f"    Datasets: {result.get('datasets')}")
        else:
            print(f"  ✗ Parker search failed: {result}")
            return False

        # Test 2: ACE
        result = search_by_keywords("ace plasma")
        if result and result.get("spacecraft") == "ACE":
            print(f"  ✓ Found ACE: {result['spacecraft_name']}")
        else:
            print(f"  ✗ ACE search failed")
            return False

        # Test 3: Solar Orbiter
        result = search_by_keywords("solar orbiter mag")
        if result and result.get("spacecraft") == "SolO":
            print(f"  ✓ Found Solar Orbiter: {result['spacecraft_name']}")
        else:
            print(f"  ✗ Solar Orbiter search failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Catalog search exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hapi_import():
    """Test importing HAPI client."""
    print("\n[3/6] Testing HAPI client import...")

    try:
        from knowledge import metadata_client
        print(f"  ✓ HAPI client imported")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import HAPI client: {e}")
        return False


def test_hapi_connection():
    """Test connecting to CDAWeb HAPI server."""
    print("\n[4/6] Testing HAPI server connection...")

    try:
        from knowledge.metadata_client import get_dataset_info

        # Request info for ACE magnetic field dataset
        info = get_dataset_info("AC_H2_MFI")

        if "parameters" in info:
            print(f"  ✓ HAPI response received")
            print(f"    Parameters: {len(info['parameters'])}")
            print(f"    Time range: {info.get('startDate')} to {info.get('stopDate')}")
            return True
        else:
            print(f"  ✗ Unexpected HAPI response: {info}")
            return False

    except Exception as e:
        print(f"  ✗ HAPI connection failed: {e}")
        print(f"    This might be a network issue or CDAWeb server problem")
        return False


def test_list_parameters():
    """Test listing plottable parameters."""
    print("\n[5/6] Testing parameter listing...")

    try:
        from knowledge.metadata_client import list_parameters

        # Get parameters for Parker Solar Probe magnetic field
        params = list_parameters("PSP_FLD_L2_MAG_RTN_1MIN")

        if params:
            print(f"  ✓ Found {len(params)} parameters")
            for p in params[:3]:  # Show first 3
                print(f"    - {p['name']}: {p.get('description', 'N/A')}")
            return True
        else:
            print(f"  ✗ No parameters found")
            return False

    except Exception as e:
        print(f"  ✗ Parameter listing failed: {e}")
        return False


def test_cache():
    """Test HAPI caching."""
    print("\n[6/6] Testing HAPI cache...")

    try:
        from knowledge.metadata_client import get_dataset_info, _info_cache

        # Clear cache
        _info_cache.clear()

        # First call - should hit API
        info1 = get_dataset_info("AC_H2_MFI", use_cache=True)

        # Second call - should use cache
        info2 = get_dataset_info("AC_H2_MFI", use_cache=True)

        if info1 == info2 and "AC_H2_MFI" in _info_cache:
            print(f"  ✓ Cache working correctly")
            return True
        else:
            print(f"  ✗ Cache not working as expected")
            return False

    except Exception as e:
        print(f"  ✗ Cache test failed: {e}")
        return False


def main():
    """Run all knowledge module tests."""
    print("=" * 60)
    print("Knowledge Modules Test (macOS)")
    print("=" * 60)

    results = [
        test_catalog_import(),
        test_catalog_search(),
        test_hapi_import(),
        test_hapi_connection(),
        test_list_parameters(),
        test_cache(),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\n✓ All knowledge modules work on macOS!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
