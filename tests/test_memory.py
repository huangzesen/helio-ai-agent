"""
Tests for agent.memory — MemoryStore and Memory dataclass.

Run with: python -m pytest tests/test_memory.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from agent.memory import Memory, MemoryStore, MAX_PREFERENCES, MAX_SUMMARIES


@pytest.fixture
def tmp_path_file():
    """Provide a temporary file path for memory storage."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "memory.json"


@pytest.fixture
def store(tmp_path_file):
    """Provide a MemoryStore backed by a temp file."""
    return MemoryStore(path=tmp_path_file)


# ---- Basic CRUD ----

class TestMemoryCRUD:
    def test_add_and_get_all(self, store):
        m = Memory(type="preference", content="Prefers dark theme")
        store.add(m)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].content == "Prefers dark theme"

    def test_remove(self, store):
        m = Memory(id="abc123", type="preference", content="Test")
        store.add(m)
        assert store.remove("abc123")
        assert len(store.get_all()) == 0

    def test_remove_nonexistent(self, store):
        assert not store.remove("nonexistent")

    def test_toggle(self, store):
        m = Memory(id="t1", type="preference", content="Test", enabled=True)
        store.add(m)
        store.toggle("t1", False)
        assert not store.get_all()[0].enabled

    def test_toggle_nonexistent(self, store):
        assert not store.toggle("nonexistent", True)

    def test_clear_all(self, store):
        store.add(Memory(type="preference", content="A"))
        store.add(Memory(type="summary", content="B"))
        count = store.clear_all()
        assert count == 2
        assert len(store.get_all()) == 0

    def test_clear_empty(self, store):
        assert store.clear_all() == 0


# ---- Persistence ----

class TestPersistence:
    def test_save_and_reload(self, tmp_path_file):
        store1 = MemoryStore(path=tmp_path_file)
        store1.add(Memory(id="p1", type="preference", content="Dark theme"))
        store1.add(Memory(id="s1", type="summary", content="Analyzed ACE data"))

        # Create new store from same file
        store2 = MemoryStore(path=tmp_path_file)
        assert len(store2.get_all()) == 2
        assert store2.get_all()[0].id == "p1"
        assert store2.get_all()[1].id == "s1"

    def test_global_enabled_persists(self, tmp_path_file):
        store1 = MemoryStore(path=tmp_path_file)
        store1.toggle_global(False)

        store2 = MemoryStore(path=tmp_path_file)
        assert not store2.is_global_enabled()

    def test_corrupt_file_handled(self, tmp_path_file):
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_file.write_text("not valid json")
        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 0

    def test_missing_file_ok(self, tmp_path_file):
        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 0
        assert store.is_global_enabled()

    def test_atomic_write(self, tmp_path_file):
        store = MemoryStore(path=tmp_path_file)
        store.add(Memory(type="preference", content="Test"))
        # Check that temp file doesn't linger
        assert not tmp_path_file.with_suffix(".tmp").exists()
        # Check that actual file exists
        assert tmp_path_file.exists()
        data = json.loads(tmp_path_file.read_text())
        assert len(data["memories"]) == 1


# ---- Global toggle ----

class TestGlobalToggle:
    def test_default_enabled(self, store):
        assert store.is_global_enabled()

    def test_toggle_global(self, store):
        store.toggle_global(False)
        assert not store.is_global_enabled()
        store.toggle_global(True)
        assert store.is_global_enabled()


# ---- Enabled filter ----

class TestEnabledFilter:
    def test_get_enabled(self, store):
        store.add(Memory(id="a", type="preference", content="A", enabled=True))
        store.add(Memory(id="b", type="preference", content="B", enabled=False))
        store.add(Memory(id="c", type="summary", content="C", enabled=True))
        enabled = store.get_enabled()
        assert len(enabled) == 2
        assert all(m.enabled for m in enabled)


# ---- Prompt building ----

class TestBuildPromptSection:
    def test_empty_returns_empty(self, store):
        assert store.build_prompt_section() == ""

    def test_disabled_returns_empty(self, store):
        store.add(Memory(type="preference", content="X"))
        store.toggle_global(False)
        assert store.build_prompt_section() == ""

    def test_preferences_only(self, store):
        store.add(Memory(type="preference", content="Prefers dark theme"))
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "- Prefers dark theme" in section
        assert "### Past Sessions" not in section

    def test_summaries_only(self, store):
        store.add(Memory(
            type="summary",
            content="Analyzed PSP data",
            created_at="2026-02-08T10:00:00",
        ))
        section = store.build_prompt_section()
        assert "### Past Sessions" in section
        assert "(2026-02-08)" in section
        assert "Analyzed PSP data" in section
        assert "### Preferences" not in section

    def test_mixed(self, store):
        store.add(Memory(type="preference", content="Prefers dark theme"))
        store.add(Memory(
            type="summary",
            content="Analyzed PSP data",
            created_at="2026-02-08T10:00:00",
        ))
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "### Past Sessions" in section

    def test_disabled_memories_excluded(self, store):
        store.add(Memory(
            id="a", type="preference", content="Visible", enabled=True,
        ))
        store.add(Memory(
            id="b", type="preference", content="Hidden", enabled=False,
        ))
        section = store.build_prompt_section()
        assert "Visible" in section
        assert "Hidden" not in section

    def test_cap_preferences(self, store):
        for i in range(MAX_PREFERENCES + 5):
            store.add(Memory(type="preference", content=f"Pref {i}"))
        section = store.build_prompt_section()
        # Should have exactly MAX_PREFERENCES preference lines
        pref_lines = [l for l in section.split("\n") if l.startswith("- Pref ")]
        assert len(pref_lines) == MAX_PREFERENCES

    def test_cap_summaries(self, store):
        for i in range(MAX_SUMMARIES + 5):
            store.add(Memory(
                type="summary",
                content=f"Session {i}",
                created_at="2026-01-01T00:00:00",
            ))
        section = store.build_prompt_section()
        sum_lines = [l for l in section.split("\n") if "Session " in l]
        assert len(sum_lines) == MAX_SUMMARIES

    def test_header_present(self, store):
        store.add(Memory(type="preference", content="Test"))
        section = store.build_prompt_section()
        assert section.startswith("## Your Memory of This User")


# ---- Memory dataclass defaults ----

class TestMemoryDefaults:
    def test_default_id(self):
        m = Memory()
        assert m.id  # non-empty
        assert len(m.id) == 12

    def test_default_type(self):
        m = Memory()
        assert m.type == "preference"

    def test_default_enabled(self):
        m = Memory()
        assert m.enabled is True

    def test_created_at_set(self):
        m = Memory()
        assert m.created_at  # non-empty ISO string
