#!/usr/bin/env python3
"""
Test 3: Autoplot Bridge Commands on macOS

Tests that Autoplot bridge operations work correctly:
- Plot CDAWeb data
- Change time range
- Export PNG
- State tracking
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_connection():
    """Test basic Autoplot connection."""
    print("\n[1/5] Testing Autoplot connection...")

    try:
        from autoplot_bridge.connection import get_script_context

        ctx = get_script_context()
        print(f"  ✓ Connection established: {type(ctx)}")
        return True

    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plot_cdaweb():
    """Test plotting CDAWeb data."""
    print("\n[2/5] Testing CDAWeb plot command...")

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()

        # Plot ACE magnetic field data (1 day)
        result = cmd.plot_cdaweb(
            dataset_id="AC_H2_MFI",
            parameter_id="Magnitude",
            time_range="2024-01-01 to 2024-01-02"
        )

        if result.get("status") == "success":
            print(f"  ✓ Plot succeeded")
            print(f"    URI: {result['uri']}")
            return True
        else:
            print(f"  ✗ Plot failed: {result}")
            return False

    except Exception as e:
        print(f"  ✗ Exception during plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_range_change():
    """Test changing time range of existing plot."""
    print("\n[3/5] Testing time range change...")

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()

        # Change to a different time range
        result = cmd.set_time_range("2024-01-02 to 2024-01-03")

        if result.get("status") == "success":
            print(f"  ✓ Time range changed to: {result['time_range']}")
            return True
        else:
            print(f"  ✗ Time range change failed: {result}")
            return False

    except Exception as e:
        print(f"  ✗ Exception during time range change: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export_png():
    """Test PNG export."""
    print("\n[4/5] Testing PNG export...")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"test_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()

        result = cmd.export_png(str(output_file))

        if result.get("status") == "success":
            print(f"  ✓ PNG exported: {result['filepath']}")
            print(f"    Size: {result.get('size_bytes', 0)} bytes")

            # Verify file exists and has content
            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"  ✓ File verified on disk")
                return True
            else:
                print(f"  ✗ File missing or empty")
                return False
        else:
            print(f"  ✗ Export failed: {result}")
            return False

    except Exception as e:
        print(f"  ✗ Exception during export: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_tracking():
    """Test plot state tracking."""
    print("\n[5/5] Testing state tracking...")

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()

        state = cmd.get_current_state()

        print(f"  Current URI: {state.get('uri')}")
        print(f"  Current time range: {state.get('time_range')}")

        if state.get('uri'):
            print(f"  ✓ State tracking works")
            return True
        else:
            print(f"  ✗ No state found (might need to plot first)")
            return False

    except Exception as e:
        print(f"  ✗ Exception getting state: {e}")
        return False


def main():
    """Run all Autoplot bridge tests."""
    print("=" * 60)
    print("Autoplot Bridge Commands Test (macOS)")
    print("=" * 60)
    print("\nNOTE: This will open an Autoplot window.")
    print("      The plot should appear on your screen.")

    results = [
        test_connection(),
        test_plot_cdaweb(),
        test_time_range_change(),
        test_export_png(),
        test_state_tracking(),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\n✓ All Autoplot bridge commands work on macOS!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
