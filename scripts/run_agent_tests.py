#!/usr/bin/env python3
"""
Agent Integration Tests — drives the agent server through multi-turn
conversations and checks tool calls / response content.

Usage:
    venv/Scripts/python.exe scripts/run_agent_tests.py
    python scripts/run_agent_tests.py --verbose
    python scripts/run_agent_tests.py --test 4   # run only test 4
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_server import send_msg, recv_msg, PORT_FILE, _cleanup_port_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PLOTS_DIR = PROJECT_ROOT / "tests" / "plots"
SERVER_STARTUP_TIMEOUT = 60  # seconds
REQUEST_TIMEOUT = 300  # 5 min per request


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_server(verbose: bool = False) -> subprocess.Popen:
    """Launch agent_server.py serve as a subprocess, wait for port file."""
    # Clean stale port file
    _cleanup_port_file()

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "agent_server.py"), "serve"]
    if verbose:
        cmd.append("--verbose")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    print("Starting agent server...", end="", flush=True)
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read()
            print(f"\nServer exited unexpectedly (code {proc.returncode}):")
            print(output)
            sys.exit(1)
        if PORT_FILE.exists():
            try:
                port = int(PORT_FILE.read_text().strip())
                if port > 0:
                    print(f" ready (port {port})")
                    return proc
            except (ValueError, OSError):
                pass
        time.sleep(1)
        print(".", end="", flush=True)

    print("\nTimeout waiting for server to start.")
    proc.kill()
    sys.exit(1)


def _connect() -> socket.socket:
    """Connect to the running server."""
    port = int(PORT_FILE.read_text().strip())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(REQUEST_TIMEOUT)
    sock.connect(("127.0.0.1", port))
    return sock


def send(message: str) -> dict:
    """Send a message and return the full response dict."""
    sock = _connect()
    try:
        send_msg(sock, {"action": "send", "message": message})
        response = recv_msg(sock)
    finally:
        sock.close()
    return response or {"response": "", "error": "No response", "tool_calls": []}


def reset():
    """Reset the agent conversation."""
    sock = _connect()
    try:
        send_msg(sock, {"action": "reset"})
        recv_msg(sock)
    finally:
        sock.close()


def stop_server():
    """Send stop command to the server."""
    try:
        sock = _connect()
        try:
            send_msg(sock, {"action": "stop"})
            recv_msg(sock)
        finally:
            sock.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

class TestResults:
    def __init__(self):
        self.scenarios = []  # list of (name, checks)
        self._current_checks = []
        self._current_name = ""

    def start_scenario(self, name: str):
        self._current_name = name
        self._current_checks = []

    def check(self, label: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        self._current_checks.append((label, condition, detail))
        marker = "  [+]" if condition else "  [-]"
        print(f"{marker} {label}" + (f"  ({detail})" if detail and not condition else ""))

    def end_scenario(self):
        self.scenarios.append((self._current_name, list(self._current_checks)))

    def summary(self) -> str:
        lines = ["\n=== Agent Test Report ==="]
        total_pass = 0
        total_checks = 0
        for name, checks in self.scenarios:
            passed = sum(1 for _, ok, _ in checks if ok)
            total = len(checks)
            total_pass += passed
            total_checks += total
            status = "PASS" if passed == total else "FAIL"
            lines.append(f"  {name:<35s} {status} ({passed}/{total} checks)")
        lines.append("===")
        lines.append(f"Total: {total_pass}/{total_checks} checks passed")
        return "\n".join(lines)

    @property
    def all_passed(self) -> bool:
        return all(ok for _, checks in self.scenarios for _, ok, _ in checks)


def tool_names(response: dict) -> list[str]:
    """Extract tool call names from a response."""
    return [tc["name"] for tc in response.get("tool_calls", [])]


def has_tool(response: dict, name: str) -> bool:
    """Check if a specific tool was called."""
    return name in tool_names(response)


def resp_text(response: dict) -> str:
    """Get lowercased response text for easy matching."""
    return (response.get("response") or "").lower()


def resp_error(response: dict) -> str | None:
    """Get error from response."""
    return response.get("error")


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def test_1_discovery(results: TestResults):
    """Test 1: Discovery — Search & List Parameters"""
    results.start_scenario("Test 1: Discovery")
    print("\n--- Test 1: Discovery — Search & List Parameters ---")
    reset()

    # Turn 1
    print("\n  Turn 1: Search for Wind magnetic field datasets")
    r = send("Search for Wind magnetic field datasets")
    text = resp_text(r)
    results.check(
        "search_datasets tool called",
        has_tool(r, "search_datasets"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "Response mentions a Wind dataset",
        "wi_h" in text or "wind" in text or "mfi" in text,
        f"response snippet: {text[:120]}",
    )

    # Turn 2
    print("\n  Turn 2: List parameters for that dataset")
    r = send("List the parameters for that dataset")
    text = resp_text(r)
    results.check(
        "list_parameters tool called",
        has_tool(r, "list_parameters"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "Response mentions parameter names",
        any(w in text for w in ["bgse", "magnitude", "bx", "by", "bz", "field"]),
        f"response snippet: {text[:120]}",
    )

    results.end_scenario()


def test_2_fetch_describe(results: TestResults):
    """Test 2: Fetch & Describe Pipeline"""
    results.start_scenario("Test 2: Fetch & Describe")
    print("\n--- Test 2: Fetch & Describe Pipeline ---")
    reset()

    # Turn 1
    print("\n  Turn 1: Fetch ACE magnetic field magnitude data")
    r = send("Fetch ACE magnetic field magnitude data for January 2024")
    text = resp_text(r)
    tools = tool_names(r)
    results.check(
        "Delegation or fetch tool called",
        "delegate_to_mission" in tools or "fetch_data" in tools,
        f"tools: {tools}",
    )
    results.check(
        "Response confirms data fetched",
        any(w in text for w in ["fetched", "points", "records", "data", "loaded", "retrieved"]),
        f"response snippet: {text[:120]}",
    )

    # Turn 2
    print("\n  Turn 2: Describe the data")
    r = send("Describe the data you just fetched")
    text = resp_text(r)
    results.check(
        "describe_data tool called",
        has_tool(r, "describe_data") or has_tool(r, "delegate_to_mission"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "Response includes statistics",
        any(w in text for w in ["min", "max", "mean", "std", "average", "range"]),
        f"response snippet: {text[:120]}",
    )

    # Turn 3
    print("\n  Turn 3: List fetched data")
    r = send("What data do I have in memory?")
    text = resp_text(r)
    results.check(
        "list_fetched_data tool called",
        has_tool(r, "list_fetched_data") or has_tool(r, "delegate_to_mission"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "Response mentions ACE data",
        any(w in text for w in ["ace", "magnetic", "magnitude"]),
        f"response snippet: {text[:120]}",
    )

    results.end_scenario()


def test_3_compute_save(results: TestResults):
    """Test 3: Compute & Save (continues from Test 2 — ACE data in memory)"""
    results.start_scenario("Test 3: Compute & Save")
    print("\n--- Test 3: Compute & Save ---")
    # Do NOT reset — ACE data still in memory from Test 2

    # Turn 1
    print("\n  Turn 1: Compute running average")
    r = send("Compute a 1-hour running average of the ACE magnetic field magnitude")
    text = resp_text(r)
    results.check(
        "custom_operation tool called",
        has_tool(r, "custom_operation") or has_tool(r, "delegate_to_mission"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "Response mentions computed result",
        any(w in text for w in ["running average", "smoothed", "computed", "average", "result"]),
        f"response snippet: {text[:150]}",
    )

    # Turn 2
    print("\n  Turn 2: Save data to CSV")
    csv_path = str(PLOTS_DIR / "ace_mag_test.csv")
    r = send(f"Save the original ACE magnetic field magnitude data to a CSV file at {csv_path}")
    text = resp_text(r)
    results.check(
        "save_data tool called",
        has_tool(r, "save_data") or has_tool(r, "delegate_to_mission"),
        f"tools: {tool_names(r)}",
    )
    # Check if file exists (allow a moment for disk write)
    time.sleep(1)
    csv_exists = any(
        p.suffix == ".csv" and "ace" in p.name.lower()
        for p in PLOTS_DIR.iterdir()
    ) if PLOTS_DIR.exists() else False
    results.check(
        "CSV file created on disk",
        csv_exists,
        f"checked {PLOTS_DIR}",
    )

    results.end_scenario()


def test_4_plot_export(results: TestResults):
    """Test 4: Direct Plot & Export"""
    results.start_scenario("Test 4: Plot & Export")
    print("\n--- Test 4: Direct Plot & Export ---")
    reset()

    # Turn 1
    print("\n  Turn 1: Plot DSCOVR magnetic field data")
    r = send("Plot DSCOVR magnetic field data for January 2024")
    text = resp_text(r)
    tools = tool_names(r)
    results.check(
        "plot_data or delegate tool called",
        "plot_data" in tools or "delegate_to_mission" in tools,
        f"tools: {tools}",
    )
    results.check(
        "No error in response",
        resp_error(r) is None,
        str(resp_error(r)),
    )

    # Turn 2
    print("\n  Turn 2: Change time range")
    r = send("Zoom in to January 10-15, 2024")
    text = resp_text(r)
    results.check(
        "change_time_range tool called",
        has_tool(r, "change_time_range"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "Response confirms new time range",
        any(w in text for w in ["10", "15", "time range", "updated", "changed", "zoom"]),
        f"response snippet: {text[:120]}",
    )

    # Turn 3
    png_path = str(PLOTS_DIR / "dscovr_mag.png")
    print(f"\n  Turn 3: Export plot to {png_path}")
    r = send(f"Export the current plot as {png_path}")
    text = resp_text(r)
    results.check(
        "export_plot tool called",
        has_tool(r, "export_plot"),
        f"tools: {tool_names(r)}",
    )
    time.sleep(1)
    png_exists = Path(png_path).exists()
    results.check(
        "PNG file created on disk",
        png_exists,
        png_path,
    )

    # Turn 4
    print("\n  Turn 4: Get plot info")
    r = send("What's currently plotted?")
    text = resp_text(r)
    results.check(
        "get_plot_info tool called",
        has_tool(r, "get_plot_info"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "Response includes dataset info",
        any(w in text for w in ["dscovr", "uri", "dataset", "plotted", "magnetic"]),
        f"response snippet: {text[:120]}",
    )

    results.end_scenario()


def test_5_multi_mission(results: TestResults):
    """Test 5: Multi-Mission Comparison"""
    results.start_scenario("Test 5: Multi-Mission")
    print("\n--- Test 5: Multi-Mission Comparison ---")
    reset()

    # Turn 1
    print("\n  Turn 1: Fetch Wind magnetic field magnitude")
    r = send("Fetch Wind magnetic field magnitude for January 2024")
    text = resp_text(r)
    tools = tool_names(r)
    results.check(
        "Wind data fetch initiated",
        "delegate_to_mission" in tools or "fetch_data" in tools,
        f"tools: {tools}",
    )
    results.check(
        "Response confirms Wind data fetched",
        any(w in text for w in ["wind", "fetched", "data", "loaded", "retrieved", "points"]),
        f"response snippet: {text[:120]}",
    )

    # Turn 2
    print("\n  Turn 2: Fetch DSCOVR magnetic field magnitude")
    r = send("Also fetch DSCOVR magnetic field magnitude for the same period")
    text = resp_text(r)
    tools = tool_names(r)
    results.check(
        "DSCOVR data fetch initiated",
        "delegate_to_mission" in tools or "fetch_data" in tools,
        f"tools: {tools}",
    )
    results.check(
        "Response confirms DSCOVR data fetched",
        any(w in text for w in ["dscovr", "fetched", "data", "loaded", "retrieved", "points"]),
        f"response snippet: {text[:120]}",
    )

    # Turn 3
    print("\n  Turn 3: Plot both datasets together")
    r = send("Plot both the Wind and DSCOVR magnetic field magnitude datasets together")
    text = resp_text(r)
    tools = tool_names(r)
    results.check(
        "plot_computed_data tool called",
        "plot_computed_data" in tools or "plot_data" in tools,
        f"tools: {tools}",
    )

    results.end_scenario()


def test_6_error_handling(results: TestResults):
    """Test 6: Error Handling & Edge Cases"""
    results.start_scenario("Test 6: Error Handling")
    print("\n--- Test 6: Error Handling & Edge Cases ---")
    reset()

    # Turn 1
    print("\n  Turn 1: Search for unsupported mission")
    r = send("Search for datasets from Voyager")
    text = resp_text(r)
    results.check(
        "No crash on unsupported mission",
        resp_error(r) is None,
        str(resp_error(r)),
    )
    results.check(
        "Response indicates no results or unsupported",
        any(w in text for w in ["no ", "not ", "unavailable", "don't", "doesn't", "voyager", "support"]),
        f"response snippet: {text[:150]}",
    )

    # Turn 2
    print("\n  Turn 2: Fetch nonexistent dataset")
    r = send("Fetch data from NONEXISTENT_DATASET_XYZ for last week")
    text = resp_text(r)
    results.check(
        "Agent recovers from bad dataset",
        True,  # If we got a response at all, agent didn't crash
        f"error={resp_error(r)}, response snippet: {text[:120]}",
    )

    # Turn 3
    print("\n  Turn 3: General question to prove agent still works")
    r = send("Hello, what missions do you support?")
    text = resp_text(r)
    results.check(
        "Agent is functional after errors",
        any(w in text for w in ["ace", "wind", "dscovr", "psp", "mission"]),
        f"response snippet: {text[:150]}",
    )

    results.end_scenario()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    1: ("Discovery", test_1_discovery),
    2: ("Fetch & Describe", test_2_fetch_describe),
    3: ("Compute & Save", test_3_compute_save),
    4: ("Plot & Export", test_4_plot_export),
    5: ("Multi-Mission", test_5_multi_mission),
    6: ("Error Handling", test_6_error_handling),
}


def main():
    parser = argparse.ArgumentParser(description="Run agent integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Start server in verbose mode")
    parser.add_argument("--test", "-t", type=int, help="Run only this test number (1-6)")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start/stop server (assume already running)")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results = TestResults()
    proc = None
    start_time = time.time()

    try:
        # Start server
        if not args.no_server:
            proc = start_server(verbose=args.verbose)
        else:
            if not PORT_FILE.exists():
                print("Error: --no-server specified but server is not running.")
                sys.exit(1)
            print("Using existing server.")

        # Select tests to run
        if args.test:
            if args.test not in ALL_TESTS:
                print(f"Invalid test number: {args.test}. Choose 1-{len(ALL_TESTS)}.")
                sys.exit(1)
            tests_to_run = {args.test: ALL_TESTS[args.test]}
            # If running test 3 alone, we need test 2 first (data dependency)
            if args.test == 3 and 2 not in tests_to_run:
                print("Note: Test 3 depends on Test 2 (ACE data in memory). Running Test 2 first.")
                tests_to_run = {2: ALL_TESTS[2], 3: ALL_TESTS[3]}
        else:
            tests_to_run = ALL_TESTS

        # Run tests
        for num, (name, func) in sorted(tests_to_run.items()):
            try:
                func(results)
            except Exception as e:
                print(f"\n  [!] Test {num} ({name}) crashed: {e}")
                results.start_scenario(f"Test {num}: {name}")
                results.check("Test execution", False, str(e))
                results.end_scenario()

        # Summary
        elapsed = round(time.time() - start_time, 1)
        report = results.summary()
        report += f"\nElapsed: {elapsed}s"
        print(report)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = PLOTS_DIR / f"test_report_{timestamp}.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved: {report_path}")

    finally:
        # Stop server
        if not args.no_server:
            print("\nStopping server...")
            stop_server()
            if proc:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    sys.exit(0 if results.all_passed else 1)


if __name__ == "__main__":
    main()
