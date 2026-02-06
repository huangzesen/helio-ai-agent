"""
Autoplot command wrappers for plot operations, time range control, and export.

Usage:
    from autoplot_bridge.commands import get_commands
    cmd = get_commands()
    cmd.plot_cdaweb("AC_H2_MFI", "Magnitude", "2024-01-01 to 2024-01-02")
    cmd.export_png("output.png")
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import jpype

from agent.time_utils import TimeRange
from autoplot_bridge.connection import get_script_context

if TYPE_CHECKING:
    from data_ops.store import DataEntry


class AutoplotCommands:
    """Wrapper around Autoplot ScriptContext with state tracking."""

    def __init__(self, verbose: bool = False):
        self._ctx = None
        self._current_uri = None
        self._current_time_range = None
        self.verbose = verbose
        self._label_colors: dict = {}  # label → java.awt.Color

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

    def plot_cdaweb(self, dataset_id: str, parameter_id: str, time_range: TimeRange) -> dict:
        """
        Plot CDAWeb data.

        Args:
            dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI")
            parameter_id: Parameter within the dataset (e.g., "Magnitude")
            time_range: TimeRange object with UTC start/end datetimes.

        Returns:
            dict with status and URI
        """
        tr_str = time_range.to_autoplot_string()
        # Build CDAWeb URI with URL-encoded time range
        uri = f"vap+cdaweb:ds={dataset_id}&id={parameter_id}&timerange={tr_str.replace(' ', '+')}"

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
            "time_range": tr_str,
        }

    def set_time_range(self, time_range: TimeRange) -> dict:
        """
        Set the time range for the current plot.

        Args:
            time_range: TimeRange object with UTC start/end datetimes.

        Returns:
            dict with status and new time range
        """
        tr_str = time_range.to_autoplot_string()
        self._log(f"Setting time range: {tr_str}")
        DatumRangeUtil = jpype.JClass("org.das2.datum.DatumRangeUtil")
        tr = DatumRangeUtil.parseTimeRange(tr_str)
        dom = self.ctx.getDocumentModel()
        dom.setTimeRange(tr)
        self._log("Time range updated.")

        self._current_time_range = time_range

        return {
            "status": "success",
            "time_range": tr_str,
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

    def _numpy_to_qdataset(self, time_arr: np.ndarray, values: np.ndarray,
                            units: str = "", label: str = "") -> "jpype.JObject":
        """Convert numpy time + values arrays to an Autoplot QDataSet.

        Args:
            time_arr: 1D datetime64[ns] timestamps.
            values: 1D float64 data array (scalar only — vectors must be
                    decomposed before calling this method).
            units: Physical units string (e.g., "nT").
            label: Series label for the legend.

        Returns:
            A rank-1 QDataSet with DEPEND_0 (time), UNITS, LABEL, and TITLE.
        """
        DDataSet = jpype.JClass("org.das2.qds.DDataSet")
        QDataSet = jpype.JClass("org.das2.qds.QDataSet")
        Units = jpype.JClass("org.das2.datum.Units")

        n = len(time_arr)

        # --- Build time dataset (Units.t2000 = seconds since 2000-01-01) ---
        epoch_2000 = np.datetime64("2000-01-01T00:00:00", "ns")
        time_seconds = (time_arr.astype("datetime64[ns]") - epoch_2000).astype(np.float64) / 1e9

        time_ds = DDataSet.createRank1(n)
        for i in range(n):
            time_ds.putValue(i, float(time_seconds[i]))
        time_ds.putProperty(QDataSet.UNITS, Units.t2000)

        # --- Build values dataset ---
        vals = values.ravel() if values.ndim > 1 else values
        val_ds = DDataSet.createRank1(n)
        for i in range(n):
            val_ds.putValue(i, float(vals[i]))

        # Attach time as DEPEND_0
        val_ds.putProperty(QDataSet.DEPEND_0, time_ds)

        # Units (best-effort lookup; skip on failure)
        if units:
            try:
                val_ds.putProperty(QDataSet.UNITS, Units.lookupUnits(units))
            except Exception:
                pass

        if label:
            val_ds.putProperty(QDataSet.LABEL, label)
            val_ds.putProperty(QDataSet.TITLE, label)

        return val_ds

    def plot_dataset(self, entries: list[DataEntry],
                     title: str = "", filename: str = "") -> dict:
        """Plot one or more DataEntry timeseries in the Autoplot canvas.

        Vector entries (n, 3) are decomposed into scalar x/y/z components.
        Multiple series are overlaid using setLayoutOverplot.

        Args:
            entries: DataEntry objects to plot.
            title: Optional plot title.
            filename: If provided, export to PNG after plotting.

        Returns:
            dict with status, labels, num_series, and optional filepath.
        """
        if not entries:
            return {"status": "error", "message": "No entries to plot"}

        # Check for empty arrays
        for entry in entries:
            if len(entry.time) == 0:
                return {"status": "error",
                        "message": f"Entry '{entry.label}' has no data points"}

        # Decompose vectors into scalar components
        datasets = []  # list of (label, QDataSet)
        for entry in entries:
            if entry.values.ndim == 2 and entry.values.shape[1] > 1:
                comp_names = ["x", "y", "z"]
                for col in range(entry.values.shape[1]):
                    comp = comp_names[col] if col < 3 else str(col)
                    comp_label = f"{entry.label}.{comp}"
                    ds = self._numpy_to_qdataset(
                        entry.time, entry.values[:, col],
                        units=entry.units, label=comp_label,
                    )
                    datasets.append((comp_label, ds))
            else:
                ds = self._numpy_to_qdataset(
                    entry.time, entry.values,
                    units=entry.units, label=entry.label,
                )
                datasets.append((entry.label, ds))

        # Ensure ScriptContext is initialized
        _ = self.ctx

        n_series = len(datasets)

        if n_series == 1:
            label, ds = datasets[0]
            self._log(f"Plotting '{label}' in Autoplot...")
            self._run_with_elapsed("Plotting", self.ctx.plot, ds)
        else:
            self._log(f"Overplotting {n_series} series in Autoplot...")
            self.ctx.setLayoutOverplot(n_series)
            for idx, (label, ds) in enumerate(datasets):
                self._log(f"  Series {idx}: '{label}'")
                self._run_with_elapsed(f"Plotting series {idx}", self.ctx.plot, idx, ds)

        self._log("Waiting for render to complete...")
        self._run_with_elapsed("Rendering", self.ctx.waitUntilIdle)

        # Color handling for overplotted series:
        # - First plot (nothing cached): generate distinct colors via HSB
        # - Later plots: reuse cached colors, new additions get black
        if n_series > 1:
            try:
                Color = jpype.JClass("java.awt.Color")
                dom = self.ctx.getDocumentModel()
                has_prior_colors = any(lbl in self._label_colors for lbl, _ in datasets)

                hue = 0.0
                for idx in range(n_series):
                    label = datasets[idx][0]
                    if label in self._label_colors:
                        color = self._label_colors[label]
                    elif has_prior_colors:
                        # New addition to an existing set → black
                        color = Color.BLACK
                    else:
                        # First-time plot — generate distinct colors
                        color = Color.getHSBColor(hue, 0.75, 0.80)
                        hue += 0.618033  # golden ratio spacing
                        if hue >= 1.0:
                            hue -= 1.0
                    self._label_colors[label] = color
                    dom.getPlotElements(idx).getStyle().setColor(color)

                self._log(f"Assigned {n_series} colors ({len(self._label_colors)} cached)")
            except Exception as e:
                self._log(f"Could not set series colors: {e}")

        # Set title via DOM if provided
        if title:
            try:
                dom = self.ctx.getDocumentModel()
                dom.getPlots(0).setTitle(title)
            except Exception:
                self._log("Could not set plot title via DOM")

        result = {
            "status": "success",
            "labels": [lbl for lbl, _ in datasets],
            "num_series": n_series,
        }

        # Export to PNG if requested
        if filename:
            if not filename.endswith(".png"):
                filename += ".png"
            export_result = self.export_png(filename)
            if export_result.get("status") == "success":
                result["filepath"] = export_result["filepath"]
            else:
                result["export_warning"] = export_result.get("message", "Export failed")

        return result

    def get_current_state(self) -> dict:
        """
        Get the current plot state.

        Returns:
            dict with current URI and time range (or None if not set)
        """
        tr_str = self._current_time_range.to_autoplot_string() if self._current_time_range else None
        return {
            "uri": self._current_uri,
            "time_range": tr_str,
        }


# Singleton instance
_commands = None


def get_commands(verbose: bool = False) -> AutoplotCommands:
    """Get the singleton AutoplotCommands instance."""
    global _commands
    if _commands is None:
        _commands = AutoplotCommands(verbose=verbose)
    return _commands
