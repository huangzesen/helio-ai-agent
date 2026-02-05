"""
Autoplot command wrappers for plot operations, time range control, and export.

Usage:
    from autoplot_bridge.commands import get_commands
    cmd = get_commands()
    cmd.plot_cdaweb("AC_H2_MFI", "Magnitude", "2024-01-01 to 2024-01-02")
    cmd.export_png("output.png")
"""

import sys
import threading
import time
from pathlib import Path

import jpype

from autoplot_bridge.connection import get_script_context


class AutoplotCommands:
    """Wrapper around Autoplot ScriptContext with state tracking."""

    def __init__(self, verbose: bool = False):
        self._ctx = None
        self._current_uri = None
        self._current_time_range = None
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Autoplot] {msg}")
            sys.stdout.flush()

    def _run_with_elapsed(self, label: str, func, *args, **kwargs):
        """Run a function while printing elapsed time if verbose."""
        if not self.verbose:
            return func(*args, **kwargs)

        stop_event = threading.Event()
        start = time.time()

        def print_elapsed():
            while not stop_event.wait(5.0):
                elapsed = time.time() - start
                print(f"  [Autoplot] ... {label} ({elapsed:.0f}s elapsed)")
                sys.stdout.flush()

        timer_thread = threading.Thread(target=print_elapsed, daemon=True)
        timer_thread.start()
        try:
            result = func(*args, **kwargs)
        finally:
            stop_event.set()
            timer_thread.join()
        elapsed = time.time() - start
        self._log(f"{label} done ({elapsed:.1f}s)")
        return result

    @property
    def ctx(self):
        """Lazy-initialize ScriptContext on first access."""
        if self._ctx is None:
            self._log("Initializing ScriptContext...")
            self._ctx = get_script_context(verbose=self.verbose)
        return self._ctx

    def plot_cdaweb(self, dataset_id: str, parameter_id: str, time_range: str) -> dict:
        """
        Plot CDAWeb data.

        Args:
            dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI")
            parameter_id: Parameter within the dataset (e.g., "Magnitude")
            time_range: Time range string (e.g., "2024-01-01 to 2024-01-02")

        Returns:
            dict with status and URI
        """
        # Build CDAWeb URI with URL-encoded time range
        uri = f"vap+cdaweb:ds={dataset_id}&id={parameter_id}&timerange={time_range.replace(' ', '+')}"

        # Ensure ScriptContext is initialized before logging the plot step
        _ = self.ctx

        self._log(f"Plotting URI: {uri}")
        self._log("Downloading data and rendering plot...")
        self._run_with_elapsed("Plotting", self.ctx.plot, uri)
        self._log("Waiting for render to complete...")
        self._run_with_elapsed("Rendering", self.ctx.waitUntilIdle)

        self._current_uri = uri
        self._current_time_range = time_range

        return {
            "status": "success",
            "uri": uri,
            "dataset_id": dataset_id,
            "parameter_id": parameter_id,
            "time_range": time_range,
        }

    def set_time_range(self, time_range: str) -> dict:
        """
        Set the time range for the current plot.

        Args:
            time_range: Time range string (e.g., "2024-01-01 to 2024-01-02")

        Returns:
            dict with status and new time range
        """
        self._log(f"Setting time range: {time_range}")
        DatumRangeUtil = jpype.JClass("org.das2.datum.DatumRangeUtil")
        tr = DatumRangeUtil.parseTimeRange(time_range)
        dom = self.ctx.getDocumentModel()
        dom.setTimeRange(tr)
        self._log("Time range updated.")

        self._current_time_range = time_range

        return {
            "status": "success",
            "time_range": time_range,
        }

    def export_png(self, filepath: str) -> dict:
        """
        Export current plot to PNG file.

        Args:
            filepath: Output file path (will be created/overwritten)

        Returns:
            dict with status and file path
        """
        # writeToPng needs forward slashes on Windows
        filepath_normalized = filepath.replace("\\", "/")

        # Ensure parent directory exists
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self._log(f"Exporting PNG to {filepath_normalized}...")
        self.ctx.waitUntilIdle()
        self._run_with_elapsed("Exporting PNG", self.ctx.writeToPng, filepath_normalized)

        # Verify file was created
        path_obj = Path(filepath)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            return {
                "status": "success",
                "filepath": str(path_obj.resolve()),
                "size_bytes": path_obj.stat().st_size,
            }
        else:
            return {
                "status": "error",
                "message": f"PNG file not created or is empty: {filepath}",
            }

    def get_current_state(self) -> dict:
        """
        Get the current plot state.

        Returns:
            dict with current URI and time range (or None if not set)
        """
        return {
            "uri": self._current_uri,
            "time_range": self._current_time_range,
        }


# Singleton instance
_commands = None


def get_commands(verbose: bool = False) -> AutoplotCommands:
    """Get the singleton AutoplotCommands instance."""
    global _commands
    if _commands is None:
        _commands = AutoplotCommands(verbose=verbose)
    return _commands
