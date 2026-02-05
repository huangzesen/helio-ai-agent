"""
Autoplot command wrappers for plot operations, time range control, and export.

Usage:
    from autoplot_bridge.commands import get_commands
    cmd = get_commands()
    cmd.plot_cdaweb("AC_H2_MFI", "Magnitude", "2024-01-01 to 2024-01-02")
    cmd.export_png("output.png")
"""

from pathlib import Path

import jpype

from autoplot_bridge.connection import get_script_context


class AutoplotCommands:
    """Wrapper around Autoplot ScriptContext with state tracking."""

    def __init__(self):
        self._ctx = None
        self._current_uri = None
        self._current_time_range = None

    @property
    def ctx(self):
        """Lazy-initialize ScriptContext on first access."""
        if self._ctx is None:
            self._ctx = get_script_context()
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

        self.ctx.plot(uri)

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
        DatumRangeUtil = jpype.JClass("org.das2.datum.DatumRangeUtil")
        tr = DatumRangeUtil.parseTimeRange(time_range)
        dom = self.ctx.getDocumentModel()
        dom.setTimeRange(tr)

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

        self.ctx.writeToPng(filepath_normalized)

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


def get_commands() -> AutoplotCommands:
    """Get the singleton AutoplotCommands instance."""
    global _commands
    if _commands is None:
        _commands = AutoplotCommands()
    return _commands
