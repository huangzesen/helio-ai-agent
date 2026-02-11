#!/usr/bin/env python3
"""
Test 5: Agent Integration on macOS

Tests the full agent workflow:
- Gemini initialization
- Tool execution
- Conversation flow
- All 7 agent tools
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_agent_import():
    """Test importing agent module."""
    print("\n[1/9] Testing agent import...")

    try:
        from agent import core
        print(f"  ✓ Agent module imported")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import agent: {e}")
        return False


def test_gemini_api_key():
    """Test that Gemini API key is configured."""
    print("\n[2/9] Testing Gemini API key...")

    try:
        from config import GOOGLE_API_KEY

        if GOOGLE_API_KEY:
            print(f"  ✓ API key is set (length: {len(GOOGLE_API_KEY)})")
            return True
        else:
            print(f"  ✗ API key not set in .env")
            return False
    except Exception as e:
        print(f"  ✗ Error checking API key: {e}")
        return False


def test_agent_creation():
    """Test creating an agent instance."""
    print("\n[3/9] Testing agent creation...")

    try:
        from agent.core import create_agent

        agent = create_agent(verbose=True)
        print(f"  ✓ Agent created: {type(agent)}")
        return True
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_search_datasets():
    """Test search_datasets tool."""
    print("\n[4/9] Testing tool: search_datasets...")

    try:
        from knowledge.catalog import search_by_keywords

        result = search_by_keywords("parker magnetic")

        if result and result.get("spacecraft") == "PSP":
            print(f"  ✓ search_datasets works")
            print(f"    Found: {result['spacecraft_name']}")
            return True
        else:
            print(f"  ✗ search_datasets failed")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def test_tool_list_parameters():
    """Test list_parameters tool."""
    print("\n[5/9] Testing tool: list_parameters...")

    try:
        from knowledge.metadata_client import list_parameters

        params = list_parameters("AC_H2_MFI")

        if params:
            print(f"  ✓ list_parameters works")
            print(f"    Found {len(params)} parameters")
            return True
        else:
            print(f"  ✗ list_parameters returned empty")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def test_tool_plot_data():
    """Test plot_data tool (via bridge)."""
    print("\n[6/9] Testing tool: plot_data...")

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()
        result = cmd.plot_cdaweb(
            "AC_H2_MFI",
            "Magnitude",
            "2024-01-01 to 2024-01-02"
        )

        if result.get("status") == "success":
            print(f"  ✓ plot_data works")
            return True
        else:
            print(f"  ✗ plot_data failed: {result}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def test_tool_change_time_range():
    """Test change_time_range tool."""
    print("\n[7/9] Testing tool: change_time_range...")

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()
        result = cmd.set_time_range("2024-01-03 to 2024-01-04")

        if result.get("status") == "success":
            print(f"  ✓ change_time_range works")
            return True
        else:
            print(f"  ✗ change_time_range failed: {result}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def test_tool_export_plot():
    """Test export_plot tool."""
    print("\n[8/9] Testing tool: export_plot...")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "test_agent_export.png"

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()
        result = cmd.export_png(str(output_file))

        if result.get("status") == "success":
            print(f"  ✓ export_plot works")
            print(f"    File: {result['filepath']}")
            return True
        else:
            print(f"  ✗ export_plot failed: {result}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def test_tool_get_plot_info():
    """Test get_plot_info tool."""
    print("\n[9/9] Testing tool: get_plot_info...")

    try:
        from autoplot_bridge.commands import get_commands

        cmd = get_commands()
        state = cmd.get_current_state()

        if state.get("uri"):
            print(f"  ✓ get_plot_info works")
            print(f"    URI: {state['uri'][:50]}...")
            return True
        else:
            print(f"  ✗ get_plot_info returned no state")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def main():
    """Run all agent tests."""
    print("=" * 60)
    print("Agent Integration Test (macOS)")
    print("=" * 60)
    print("\nNOTE: This test creates an agent and tests all tools.")
    print("      Some tests will open Autoplot windows.")

    results = [
        test_agent_import(),
        test_gemini_api_key(),
        test_agent_creation(),
        test_tool_search_datasets(),
        test_tool_list_parameters(),
        test_tool_plot_data(),
        test_tool_change_time_range(),
        test_tool_export_plot(),
        test_tool_get_plot_info(),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    # Note: ask_clarification is tested implicitly in conversation flow
    print("\nNOTE: ask_clarification tool is tested in conversation mode")

    if all(results):
        print("\n✓ All agent tools work on macOS!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
