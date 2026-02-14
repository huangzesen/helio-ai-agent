"""
Operations log for pipeline reproducibility.

Records every data-producing operation as a JSON-serializable dict so the
full pipeline can eventually be replayed.  This phase implements recording
only — replay comes later.

Storage: ``~/.helio-agent/sessions/{session_id}/operations.json``
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class OperationsLog:
    """Ordered, thread-safe log of data-producing operations."""

    def __init__(self):
        self._records: list[dict] = []
        self._counter: int = 0
        self._lock = threading.Lock()

    def record(
        self,
        tool: str,
        args: dict[str, Any],
        outputs: list[str],
        inputs: Optional[list[str]] = None,
        status: str = "success",
        error: Optional[str] = None,
    ) -> dict:
        """Append an operation record and return it.

        Args:
            tool: Tool name (e.g. "fetch_data", "custom_operation").
            args: Tool-specific arguments dict.
            outputs: Labels produced by this operation.
            inputs: Labels consumed by this operation.
            status: "success" or "error".
            error: Error message if status is "error".

        Returns:
            The recorded operation dict.
        """
        with self._lock:
            self._counter += 1
            record = {
                "id": f"op_{self._counter:03d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "status": status,
                "inputs": inputs or [],
                "outputs": outputs,
                "args": args,
                "error": error,
            }
            self._records.append(record)
            return record

    def get_records(self) -> list[dict]:
        """Return a copy of all records."""
        with self._lock:
            return list(self._records)

    def save_to_file(self, path: Path) -> None:
        """Write records to a JSON file."""
        with self._lock:
            data = list(self._records)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_records(self, records: list[dict]) -> int:
        """Restore from an in-memory list and resume the counter.

        Returns:
            Number of records loaded.
        """
        with self._lock:
            self._records = list(records)
            self._counter = self._max_counter_from_records(self._records)
            return len(self._records)

    def load_from_file(self, path: Path) -> int:
        """Load records from a JSON file and resume the counter.

        Returns:
            Number of records loaded.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.load_from_records(data)

    @staticmethod
    def _max_counter_from_records(records: list[dict]) -> int:
        """Extract the highest numeric ID from op_NNN-style IDs."""
        max_id = 0
        for rec in records:
            op_id = rec.get("id", "")
            if op_id.startswith("op_"):
                try:
                    max_id = max(max_id, int(op_id[3:]))
                except ValueError:
                    pass
        return max_id

    def clear(self) -> None:
        """Reset records and counter."""
        with self._lock:
            self._records.clear()
            self._counter = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# Module-level singleton
_log: Optional[OperationsLog] = None


def get_operations_log() -> OperationsLog:
    """Return the global OperationsLog singleton."""
    global _log
    if _log is None:
        _log = OperationsLog()
    return _log


def reset_operations_log() -> None:
    """Reset the global OperationsLog (mainly for testing)."""
    global _log
    _log = None
