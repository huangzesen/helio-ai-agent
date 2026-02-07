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

    def __init__(self, verbose: bool = False, gui_mode: bool = False):
        self._ctx = None
        self._current_uri = None
        self._current_time_range = None
        self.verbose = verbose
        self.gui_mode = gui_mode
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
            self._ctx = get_script_context(
                verbose=self.verbose,
                headless=not self.gui_mode,
            )
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
            "display": "gui_window" if self.gui_mode else "headless",
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
        try:
            self.ctx.waitUntilIdle()
        except Exception as e:
            # NullPointerException when no plot has been rendered yet
            return {
                "status": "error",
                "message": "No plot to export. Plot data first before exporting.",
            }
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

        # Ensure JVM is running before converting to QDataSet
        _ = self.ctx

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
            "display": "gui_window" if self.gui_mode else "headless",
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

    # --- GUI-mode interactive methods ---

    def reset(self) -> dict:
        """Reset the Autoplot canvas, clearing all plots and state.

        Returns:
            dict with status.
        """
        self._log("Resetting canvas...")
        self.ctx.reset()
        self.ctx.waitUntilIdle()
        self._current_uri = None
        self._current_time_range = None
        self._label_colors.clear()
        return {"status": "success", "message": "Canvas reset."}

    def set_plot_title(self, title: str) -> dict:
        """Set or change the title of the current plot.

        Args:
            title: The title text.

        Returns:
            dict with status.
        """
        self._log(f"Setting plot title: {title}")
        dom = self.ctx.getDocumentModel()
        dom.getPlots(0).setTitle(title)
        self.ctx.waitUntilIdle()
        return {"status": "success", "title": title}

    def set_axis_label(self, axis: str, label: str) -> dict:
        """Set a label on an axis of the current plot.

        Args:
            axis: Which axis — 'y' or 'z'.
            label: The text label.

        Returns:
            dict with status.
        """
        self._log(f"Setting {axis}-axis label: {label}")
        dom = self.ctx.getDocumentModel()
        plot = dom.getPlots(0)
        if axis.lower() == "y":
            plot.getYaxis().setLabel(label)
        elif axis.lower() == "z":
            plot.getZaxis().setLabel(label)
        else:
            return {"status": "error", "message": f"Unsupported axis '{axis}'. Use 'y' or 'z'."}
        self.ctx.waitUntilIdle()
        return {"status": "success", "axis": axis, "label": label}

    def toggle_log_scale(self, axis: str, enabled: bool) -> dict:
        """Enable or disable logarithmic scale on an axis.

        Args:
            axis: Which axis — 'y' or 'z'.
            enabled: True for log scale, False for linear.

        Returns:
            dict with status.
        """
        self._log(f"Setting {axis}-axis log scale: {enabled}")
        dom = self.ctx.getDocumentModel()
        plot = dom.getPlots(0)
        if axis.lower() == "y":
            plot.getYaxis().setLog(enabled)
        elif axis.lower() == "z":
            plot.getZaxis().setLog(enabled)
        else:
            return {"status": "error", "message": f"Unsupported axis '{axis}'. Use 'y' or 'z'."}
        self.ctx.waitUntilIdle()
        return {"status": "success", "axis": axis, "log_scale": enabled}

    def set_axis_range(self, axis: str, min_val: float, max_val: float) -> dict:
        """Manually set the range of a plot axis.

        Args:
            axis: Which axis — 'y' or 'z'.
            min_val: Minimum value.
            max_val: Maximum value.

        Returns:
            dict with status.
        """
        self._log(f"Setting {axis}-axis range: {min_val} to {max_val}")
        dom = self.ctx.getDocumentModel()
        plot = dom.getPlots(0)
        DatumRange = jpype.JClass("org.das2.datum.DatumRange")
        Units = jpype.JClass("org.das2.datum.Units")
        dr = DatumRange(min_val, max_val, Units.dimensionless)
        if axis.lower() == "y":
            plot.getYaxis().setRange(dr)
        elif axis.lower() == "z":
            plot.getZaxis().setRange(dr)
        else:
            return {"status": "error", "message": f"Unsupported axis '{axis}'. Use 'y' or 'z'."}
        self.ctx.waitUntilIdle()
        return {"status": "success", "axis": axis, "min": min_val, "max": max_val}

    def save_session(self, filepath: str) -> dict:
        """Save the current Autoplot session to a .vap file.

        Args:
            filepath: Output file path (.vap extension).

        Returns:
            dict with status and filepath.
        """
        if not filepath.endswith(".vap"):
            filepath += ".vap"
        filepath_normalized = filepath.replace("\\", "/")
        self._log(f"Saving session to {filepath_normalized}...")
        self.ctx.save(filepath_normalized)
        return {"status": "success", "filepath": filepath}

    def load_session(self, filepath: str) -> dict:
        """Load a previously saved Autoplot session from a .vap file.

        Args:
            filepath: Path to the .vap file.

        Returns:
            dict with status and filepath.
        """
        filepath_normalized = filepath.replace("\\", "/")
        self._log(f"Loading session from {filepath_normalized}...")
        self.ctx.load(filepath_normalized)
        self.ctx.waitUntilIdle()
        # Clear tracked state since the loaded session has its own
        self._current_uri = None
        self._current_time_range = None
        self._label_colors.clear()
        return {"status": "success", "filepath": filepath}

    def set_render_type(self, render_type: str, index: int = 0) -> dict:
        """Change how data is rendered in the plot.

        Args:
            render_type: Render type name (e.g., 'series', 'scatter', 'spectrogram').
            index: Plot element index (0-based).

        Returns:
            dict with status.
        """
        self._log(f"Setting render type to '{render_type}' for plot element {index}")
        try:
            dom = self.ctx.getDocumentModel()
            pele = dom.getPlotElements(index)
            RenderType = jpype.JClass("org.autoplot.RenderType")
            rt = RenderType.valueOf(render_type)
            pele.setRenderType(rt)
            self.ctx.waitUntilIdle()
            return {"status": "success", "render_type": render_type, "index": index}
        except Exception as e:
            return {"status": "error", "message": f"Failed to set render type: {e}"}

    def set_color_table(self, name: str) -> dict:
        """Set the color table (colormap) for the current plot.

        Args:
            name: Color table name (e.g., 'matlab_jet', 'grayscale').

        Returns:
            dict with status.
        """
        self._log(f"Setting color table to '{name}'")
        try:
            dom = self.ctx.getDocumentModel()
            DasColorBar = jpype.JClass("org.das2.graph.DasColorBar")
            Type = DasColorBar.Type
            ct = Type.parse(name)
            dom.getPlots(0).getZaxis().setColortable(ct)
            self.ctx.waitUntilIdle()
            return {"status": "success", "color_table": name}
        except Exception as e:
            return {"status": "error", "message": f"Failed to set color table: {e}"}

    def set_canvas_size(self, width: int, height: int) -> dict:
        """Set the canvas (image) size in pixels.

        Args:
            width: Width in pixels.
            height: Height in pixels.

        Returns:
            dict with status.
        """
        self._log(f"Setting canvas size to {width}x{height}")
        try:
            self.ctx.setCanvasSize(width, height)
            self.ctx.waitUntilIdle()
            return {"status": "success", "width": width, "height": height}
        except Exception as e:
            return {"status": "error", "message": f"Failed to set canvas size: {e}"}

    def export_pdf(self, filepath: str) -> dict:
        """Export current plot to a PDF file.

        Args:
            filepath: Output file path (.pdf extension added if missing)

        Returns:
            dict with status and file path
        """
        if not filepath.endswith(".pdf"):
            filepath += ".pdf"
        filepath_normalized = filepath.replace("\\", "/")

        # Ensure parent directory exists
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self._log(f"Exporting PDF to {filepath_normalized}...")
        try:
            self.ctx.waitUntilIdle()
        except Exception:
            return {
                "status": "error",
                "message": "No plot to export. Plot data first before exporting.",
            }
        self._run_with_elapsed("Exporting PDF", self.ctx.writeToPdf, filepath_normalized)

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
                "message": f"PDF file not created or is empty: {filepath}",
            }

    def execute_script(self, code: str) -> dict:
        """Execute a validated Autoplot script in a restricted namespace.

        Args:
            code: Python code using sc, dom, Color, etc.

        Returns:
            dict with status, output, and optional result.
        """
        _ = self.ctx  # ensure JVM initialized
        from autoplot_bridge.script_runner import run_autoplot_script
        return run_autoplot_script(code, self.ctx)

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


def get_commands(verbose: bool = False, gui_mode: bool = False) -> AutoplotCommands:
    """Get the singleton AutoplotCommands instance.

    Args:
        verbose: If True, print debug info.
        gui_mode: If True, launch Autoplot with visible GUI window.

    Raises:
        RuntimeError: If called with a different gui_mode than the existing instance.
    """
    global _commands
    if _commands is None:
        _commands = AutoplotCommands(verbose=verbose, gui_mode=gui_mode)
    elif _commands.gui_mode != gui_mode:
        raise RuntimeError(
            f"AutoplotCommands already created with gui_mode={_commands.gui_mode}, "
            f"cannot change to gui_mode={gui_mode} (JVM mode is set at startup)."
        )
    return _commands
