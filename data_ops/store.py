"""
In-memory timeseries store.

DataEntry holds a single timeseries as a pandas DataFrame (DatetimeIndex + value columns).
DataStore is a singleton dict-like container keyed by label strings.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Characters unsafe for filenames on Windows
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')


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
    metadata: dict | None = None

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
        if self.metadata and self.metadata.get("type") == "spectrogram":
            shape_desc = f"spectrogram[{ncols} bins]"
        else:
            shape_desc = "scalar" if ncols == 1 else f"vector[{ncols}]"
        result = {
            "label": self.label,
            "columns": list(self.data.columns),
            "num_points": n,
            "shape": shape_desc,
            "units": self.units,
            "time_min": str(self.data.index[0]) if n > 0 else None,
            "time_max": str(self.data.index[-1]) if n > 0 else None,
            "description": self.description,
            "source": self.source,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


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

    def save_to_directory(self, dir_path: Path) -> None:
        """Persist all DataEntries to a directory as pickled DataFrames.

        Writes each DataFrame as ``{safe_label}.pkl`` and an ``_index.json``
        mapping original labels to filenames and metadata.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        index = {}
        for label, entry in self._entries.items():
            safe = _UNSAFE_CHARS.sub("_", label)
            pkl_name = f"{safe}.pkl"
            entry.data.to_pickle(dir_path / pkl_name)
            entry_meta = {
                "filename": pkl_name,
                "units": entry.units,
                "description": entry.description,
                "source": entry.source,
            }
            if entry.metadata is not None:
                entry_meta["metadata"] = entry.metadata
            index[label] = entry_meta

        with open(dir_path / "_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def load_from_directory(self, dir_path: Path) -> int:
        """Restore DataEntries from a directory written by ``save_to_directory``.

        Returns:
            Number of entries loaded.
        """
        dir_path = Path(dir_path)
        index_path = dir_path / "_index.json"
        if not index_path.exists():
            return 0

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        count = 0
        for label, info in index.items():
            pkl_path = dir_path / info["filename"]
            if not pkl_path.exists():
                continue
            df = pd.read_pickle(pkl_path)
            entry = DataEntry(
                label=label,
                data=df,
                units=info.get("units", ""),
                description=info.get("description", ""),
                source=info.get("source", "computed"),
                metadata=info.get("metadata"),
            )
            self.put(entry)
            count += 1

        return count

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


def build_source_map(
    store: DataStore, labels: list[str]
) -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    """Build a mapping of sandbox variable names to DataFrames from store labels.

    Each label becomes ``df_<SUFFIX>`` where SUFFIX is the part after the last
    '.' in the label.  If the label has no '.', the full label is used as suffix.

    Args:
        store: DataStore to look up entries.
        labels: List of store labels.

    Returns:
        Tuple of (source_map, error_string).  On success error_string is None.
        On failure source_map is None.
    """
    source_map: dict[str, pd.DataFrame] = {}
    for label in labels:
        entry = store.get(label)
        if entry is None:
            return None, f"Label '{label}' not found in store"
        suffix = label.rsplit(".", 1)[-1]
        var_name = f"df_{suffix}"
        if var_name in source_map:
            return None, (
                f"Duplicate sandbox variable '{var_name}' — labels "
                f"'{label}' and another share suffix '{suffix}'. "
                f"Use labels with distinct suffixes."
            )
        source_map[var_name] = entry.data
    return source_map, None


def describe_sources(store: DataStore, labels: list[str]) -> dict:
    """Return lightweight summaries for a list of store labels.

    For each label, computes: columns, point count, cadence, NaN%, and time range.
    Cheaper than full ``describe_data`` — just what the LLM needs for correct code.

    Args:
        store: DataStore to look up entries.
        labels: List of store labels.

    Returns:
        Dict keyed by sandbox variable name (``df_SUFFIX``), each containing
        label, columns, points, cadence, nan_pct, and time_range.
    """
    result = {}
    for label in labels:
        entry = store.get(label)
        if entry is None:
            continue
        df = entry.data
        suffix = label.rsplit(".", 1)[-1]
        var_name = f"df_{suffix}"

        # Cadence: median time delta
        cadence_str = ""
        if len(df) > 1:
            dt = pd.Series(df.index).diff().dropna().median()
            cadence_str = str(dt)

        # NaN percentage
        total_cells = df.size
        nan_pct = round(df.isna().sum().sum() / total_cells * 100, 1) if total_cells > 0 else 0.0

        # Time range
        time_range = []
        if len(df) > 0:
            time_range = [str(df.index[0].date()), str(df.index[-1].date())]

        result[var_name] = {
            "label": label,
            "columns": list(df.columns),
            "points": len(df),
            "cadence": cadence_str,
            "nan_pct": nan_pct,
            "time_range": time_range,
        }
    return result
