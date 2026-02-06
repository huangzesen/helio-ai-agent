"""
In-memory timeseries store.

DataEntry holds a single timeseries (time + values + metadata).
DataStore is a singleton dict-like container keyed by label strings.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DataEntry:
    """A single timeseries stored in memory.

    Attributes:
        label: Unique identifier (e.g., "AC_H2_MFI.BGSEc" or "Bmag").
        time: 1D array of datetime64[ns] timestamps.
        values: Data array — (n,) for scalars, (n,3) for vectors.
        units: Physical units string (e.g., "nT").
        description: Human-readable description.
        source: Origin — "hapi" for fetched data, "computed" for derived data.
    """

    label: str
    time: np.ndarray
    values: np.ndarray
    units: str = ""
    description: str = ""
    source: str = "computed"

    def summary(self) -> dict:
        """Return a compact summary dict suitable for LLM responses."""
        shape_desc = "scalar" if self.values.ndim == 1 else f"vector[{self.values.shape[1]}]"
        return {
            "label": self.label,
            "num_points": len(self.time),
            "shape": shape_desc,
            "units": self.units,
            "time_min": str(self.time[0]) if len(self.time) > 0 else None,
            "time_max": str(self.time[-1]) if len(self.time) > 0 else None,
            "description": self.description,
            "source": self.source,
        }


class DataStore:
    """Singleton in-memory store mapping labels to DataEntry objects."""

    def __init__(self):
        self._entries: dict[str, DataEntry] = {}

    def put(self, entry: DataEntry) -> None:
        """Store a DataEntry, overwriting any existing entry with the same label."""
        self._entries[entry.label] = entry

    def get(self, label: str) -> Optional[DataEntry]:
        """Retrieve a DataEntry by label, or None if not found."""
        return self._entries.get(label)

    def has(self, label: str) -> bool:
        """Check if a label exists in the store."""
        return label in self._entries

    def remove(self, label: str) -> bool:
        """Remove an entry by label. Returns True if it existed."""
        if label in self._entries:
            del self._entries[label]
            return True
        return False

    def list_entries(self) -> list[dict]:
        """Return summary dicts for all stored entries."""
        return [entry.summary() for entry in self._entries.values()]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# Module-level singleton
_store: Optional[DataStore] = None


def get_store() -> DataStore:
    """Return the global DataStore singleton."""
    global _store
    if _store is None:
        _store = DataStore()
    return _store


def reset_store() -> None:
    """Reset the global DataStore (mainly for testing)."""
    global _store
    _store = None
