"""
Long-term memory system for cross-session user preferences and session summaries.

Stores memories in ~/.helio-agent/memory.json. Memories are automatically
extracted at session boundaries and injected into future conversations.

Memory types:
    - "preference": User habits, plot styles, spacecraft of interest
    - "summary": Brief summaries of past analysis sessions
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .logging import get_logger

logger = get_logger()

# Maximum memories injected per type (budget: ~10k tokens total)
MAX_PREFERENCES = 15
MAX_SUMMARIES = 10
MAX_PITFALLS = 20


@dataclass
class Memory:
    """A single memory entry."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: str = "preference"  # "preference" or "summary"
    content: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_session: str = ""
    enabled: bool = True


class MemoryStore:
    """Manages long-term memories persisted as JSON."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path.home() / ".helio-agent" / "memory.json"
        self.path = path
        self.cold_path = path.with_name("memory_cold.json")
        self._global_enabled: bool = True
        self._memories: list[Memory] = []
        self.load()

    # ---- Persistence ----

    def load(self) -> None:
        """Load memories from disk."""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._global_enabled = data.get("global_enabled", True)
            self._memories = [
                Memory(**m) for m in data.get("memories", [])
            ]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"[Memory] Could not load {self.path}: {e}")
            self._memories = []

    def save(self) -> None:
        """Atomically save memories to disk (write temp + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {
            "global_enabled": self._global_enabled,
            "memories": [asdict(m) for m in self._memories],
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.path)

    # ---- CRUD ----

    def add(self, memory: Memory) -> None:
        """Add a memory and save."""
        self._memories.append(memory)
        self.save()

    def remove(self, memory_id: str) -> bool:
        """Remove a memory by ID. Returns True if found."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.id != memory_id]
        if len(self._memories) < before:
            self.save()
            return True
        return False

    def toggle(self, memory_id: str, enabled: bool) -> bool:
        """Toggle a memory's enabled state. Returns True if found."""
        for m in self._memories:
            if m.id == memory_id:
                m.enabled = enabled
                self.save()
                return True
        return False

    def replace_all(self, memories: list[Memory]) -> None:
        """Replace all memories with a new list and save."""
        self._memories = memories
        self.save()

    def archive_to_cold(self, memories: list[Memory]) -> None:
        """Append memories to cold storage file for archival."""
        if not memories:
            return
        # Load existing cold storage
        existing = []
        if self.cold_path.exists():
            try:
                with open(self.cold_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.extend(asdict(m) for m in memories)
        # Write atomically
        self.cold_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cold_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        tmp.replace(self.cold_path)
        logger.debug(f"[Memory] Archived {len(memories)} memories to cold storage")

    def read_cold(self) -> list[dict]:
        """Load all cold-storage memories as dicts."""
        if not self.cold_path.exists():
            return []
        try:
            with open(self.cold_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[Memory] Could not load cold storage {self.cold_path}: {e}")
            return []

    def search_cold(self, query: str, mem_type: str | None = None, limit: int = 20) -> list[dict]:
        """Search cold memories by substring match on content. Optional type filter."""
        entries = self.read_cold()
        query_lower = query.lower()
        results = []
        for entry in entries:
            if mem_type and entry.get("type") != mem_type:
                continue
            if query_lower in entry.get("content", "").lower():
                results.append(entry)
        return results[-limit:]

    def clear_all(self) -> int:
        """Remove all memories. Returns count removed."""
        count = len(self._memories)
        self._memories = []
        self.save()
        return count

    # ---- Global toggle ----

    def toggle_global(self, enabled: bool) -> None:
        """Enable or disable the entire memory system."""
        self._global_enabled = enabled
        self.save()

    def is_global_enabled(self) -> bool:
        """Check if the memory system is globally enabled."""
        return self._global_enabled

    # ---- Queries ----

    def get_all(self) -> list[Memory]:
        """Return all memories."""
        return list(self._memories)

    def get_enabled(self) -> list[Memory]:
        """Return only enabled memories."""
        return [m for m in self._memories if m.enabled]

    # ---- Prompt building ----

    def build_prompt_section(self) -> str:
        """Build a markdown section for injection into user messages.

        Returns empty string if disabled or no enabled memories.
        Caps at MAX_PREFERENCES + MAX_SUMMARIES.
        """
        if not self._global_enabled:
            return ""

        enabled = self.get_enabled()
        if not enabled:
            return ""

        preferences = [m for m in enabled if m.type == "preference"][:MAX_PREFERENCES]
        summaries = [m for m in enabled if m.type == "summary"][:MAX_SUMMARIES]
        pitfalls = [m for m in enabled if m.type == "pitfall"][:MAX_PITFALLS]

        if not preferences and not summaries and not pitfalls:
            return ""

        parts = ["## Your Memory of This User"]

        if preferences:
            parts.append("")
            parts.append("### Preferences")
            for m in preferences:
                parts.append(f"- {m.content}")

        if summaries:
            parts.append("")
            parts.append("### Past Sessions")
            for m in summaries:
                date_str = m.created_at[:10] if m.created_at else ""
                if date_str:
                    parts.append(f"- ({date_str}) {m.content}")
                else:
                    parts.append(f"- {m.content}")

        if pitfalls:
            parts.append("")
            parts.append("## Operational Knowledge")
            parts.append("Follow these lessons learned from past sessions:")
            for m in pitfalls:
                parts.append(f"- {m.content}")

        return "\n".join(parts)
