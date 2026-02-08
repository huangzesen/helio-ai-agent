"""
In-memory timeseries store.

DataEntry holds a single timeseries as a pandas DataFrame (DatetimeIndex + value columns).
DataStore is a singleton dict-like container keyed by label strings.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DataEntry:
    """A single timeseries stored in memory.

    Attributes:
        label: Unique identifier (e.g., "AC_H2_MFI.BGSEc" or "Bmag").
        data: DataFrame with DatetimeIndex and one or more float64 columns.
        units: Physical units string (e.g., "nT").
        description: Human-readable description.
        source: Origin — "hapi" for fetched data, "computed" for derived data.
    """

    label: str
    data: pd.DataFrame
    units: str = ""
    description: str = ""
    source: str = "computed"

    @property
    def time(self) -> np.ndarray:
        """Backward compat: numpy datetime64[ns] array."""
        return self.data.index.values

    @property
    def values(self) -> np.ndarray:
        """Backward compat: numpy float64 array — (n,) for scalar, (n,k) for vector."""
        v = self.data.values
        if v.shape[1] == 1:
            return v.squeeze(axis=1)
        return v

    def summary(self) -> dict:
        """Return a compact summary dict suitable for LLM responses."""
        n = len(self.data)
        ncols = len(self.data.columns)
        shape_desc = "scalar" if ncols == 1 else f"vector[{ncols}]"
        return {
            "label": self.label,
            "num_points": n,
            "shape": shape_desc,
            "units": self.units,
            "time_min": str(self.data.index[0]) if n > 0 else None,
            "time_max": str(self.data.index[-1]) if n > 0 else None,
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

    def memory_usage_bytes(self) -> int:
        """Return approximate total memory usage of all stored DataFrames."""
        return sum(
            entry.data.memory_usage(deep=True).sum()
            for entry in self._entries.values()
        )

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
