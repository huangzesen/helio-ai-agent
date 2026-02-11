"""
Tests for agent.memory — MemoryStore and Memory dataclass.

Run with: python -m pytest tests/test_memory.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from agent.memory import Memory, MemoryStore, MAX_PREFERENCES, MAX_SUMMARIES, MAX_PITFALLS


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

    def test_replace_all(self, store):
        store.add(Memory(id="old1", type="preference", content="Old A"))
        store.add(Memory(id="old2", type="summary", content="Old B"))
        new_memories = [
            Memory(id="new1", type="pitfall", content="New X"),
        ]
        store.replace_all(new_memories)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].id == "new1"
        assert store.get_all()[0].content == "New X"

    def test_replace_all_persists(self, tmp_path_file):
        store1 = MemoryStore(path=tmp_path_file)
        store1.add(Memory(type="preference", content="Original"))
        store1.replace_all([Memory(id="r1", type="pitfall", content="Replaced")])
        store2 = MemoryStore(path=tmp_path_file)
        assert len(store2.get_all()) == 1
        assert store2.get_all()[0].id == "r1"

    def test_replace_all_empty(self, store):
        store.add(Memory(type="preference", content="A"))
        store.replace_all([])
        assert len(store.get_all()) == 0

    def test_archive_to_cold(self, store):
        memories = [
            Memory(id="c1", type="preference", content="Cold A"),
            Memory(id="c2", type="pitfall", content="Cold B"),
        ]
        store.archive_to_cold(memories)
        assert store.cold_path.exists()
        data = json.loads(store.cold_path.read_text())
        assert len(data) == 2
        assert data[0]["id"] == "c1"
        assert data[1]["content"] == "Cold B"

    def test_archive_to_cold_appends(self, store):
        store.archive_to_cold([Memory(id="a1", type="preference", content="First")])
        store.archive_to_cold([Memory(id="a2", type="pitfall", content="Second")])
        data = json.loads(store.cold_path.read_text())
        assert len(data) == 2
        assert data[0]["id"] == "a1"
        assert data[1]["id"] == "a2"

    def test_archive_to_cold_empty_noop(self, store):
        store.archive_to_cold([])
        assert not store.cold_path.exists()

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


# ---- Pitfall prompt rendering ----

class TestPitfallPrompt:
    def test_pitfalls_only(self, store):
        store.add(Memory(type="pitfall", content="OMNI data may have empty CSV strings"))
        section = store.build_prompt_section()
        assert "## Operational Knowledge" in section
        assert "Follow these lessons learned" in section
        assert "- OMNI data may have empty CSV strings" in section
        assert "### Preferences" not in section

    def test_pitfalls_with_preferences(self, store):
        store.add(Memory(type="preference", content="Prefers dark theme"))
        store.add(Memory(type="pitfall", content="MMS dataset IDs require @0 suffix"))
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "## Operational Knowledge" in section
        assert "- MMS dataset IDs require @0 suffix" in section

    def test_cap_pitfalls(self, store):
        for i in range(MAX_PITFALLS + 5):
            store.add(Memory(type="pitfall", content=f"Pitfall {i}"))
        section = store.build_prompt_section()
        pitfall_lines = [l for l in section.split("\n") if l.startswith("- Pitfall ")]
        assert len(pitfall_lines) == MAX_PITFALLS

    def test_disabled_pitfalls_excluded(self, store):
        store.add(Memory(
            id="p1", type="pitfall", content="Visible pitfall", enabled=True,
        ))
        store.add(Memory(
            id="p2", type="pitfall", content="Hidden pitfall", enabled=False,
        ))
        section = store.build_prompt_section()
        assert "Visible pitfall" in section
        assert "Hidden pitfall" not in section

    def test_all_three_types(self, store):
        store.add(Memory(type="preference", content="Prefers log scale"))
        store.add(Memory(
            type="summary", content="Analyzed ACE data",
            created_at="2026-02-09T10:00:00",
        ))
        store.add(Memory(type="pitfall", content="Rolling windows need DatetimeIndex"))
        section = store.build_prompt_section()
        assert "### Preferences" in section
        assert "### Past Sessions" in section
        assert "## Operational Knowledge" in section


# ---- Cold storage reads ----

class TestColdStorageRead:
    def test_read_cold_empty(self, store):
        assert store.read_cold() == []

    def test_read_cold_after_archive(self, store):
        store.archive_to_cold([
            Memory(id="c1", type="preference", content="Dark theme"),
            Memory(id="c2", type="summary", content="Analyzed ACE mag data"),
        ])
        cold = store.read_cold()
        assert len(cold) == 2
        assert cold[0]["id"] == "c1"
        assert cold[1]["content"] == "Analyzed ACE mag data"

    def test_read_cold_corrupt_file(self, store):
        store.cold_path.parent.mkdir(parents=True, exist_ok=True)
        store.cold_path.write_text("not json")
        assert store.read_cold() == []

    def test_search_cold_by_keyword(self, store):
        store.archive_to_cold([
            Memory(id="s1", type="summary", content="Analyzed ACE magnetic field"),
            Memory(id="s2", type="summary", content="Plotted PSP solar wind speed"),
            Memory(id="s3", type="pitfall", content="ACE data has gaps in January"),
        ])
        results = store.search_cold("ACE")
        assert len(results) == 2
        assert results[0]["id"] == "s1"
        assert results[1]["id"] == "s3"

    def test_search_cold_case_insensitive(self, store):
        store.archive_to_cold([
            Memory(id="s1", type="summary", content="Magnetic field analysis"),
        ])
        results = store.search_cold("magnetic")
        assert len(results) == 1
        results = store.search_cold("MAGNETIC")
        assert len(results) == 1

    def test_search_cold_with_type_filter(self, store):
        store.archive_to_cold([
            Memory(id="s1", type="summary", content="ACE analysis session"),
            Memory(id="s2", type="pitfall", content="ACE data gaps"),
            Memory(id="s3", type="preference", content="ACE preferred mission"),
        ])
        results = store.search_cold("ACE", mem_type="pitfall")
        assert len(results) == 1
        assert results[0]["id"] == "s2"

    def test_search_cold_with_limit(self, store):
        store.archive_to_cold([
            Memory(id=f"m{i}", type="summary", content=f"Session {i} analysis")
            for i in range(10)
        ])
        results = store.search_cold("analysis", limit=3)
        assert len(results) == 3
        # Should return the last 3 matches
        assert results[0]["id"] == "m7"
        assert results[2]["id"] == "m9"

    def test_search_cold_no_match(self, store):
        store.archive_to_cold([
            Memory(id="s1", type="summary", content="Analyzed ACE data"),
        ])
        results = store.search_cold("nonexistent")
        assert results == []

    def test_search_cold_empty_store(self, store):
        results = store.search_cold("anything")
        assert results == []


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

    def test_default_scope(self):
        m = Memory()
        assert m.scope == "generic"

    def test_created_at_set(self):
        m = Memory()
        assert m.created_at  # non-empty ISO string


# ---- Scoped pitfalls ----

class TestScopedPitfalls:
    def test_scope_default_generic(self, store):
        """Pitfalls without explicit scope default to 'generic'."""
        m = Memory(type="pitfall", content="Some lesson")
        store.add(m)
        assert store.get_all()[0].scope == "generic"

    def test_get_pitfalls_by_scope(self, store):
        """get_pitfalls_by_scope returns only matching enabled pitfalls."""
        store.add(Memory(type="pitfall", content="PSP fill values", scope="mission:PSP"))
        store.add(Memory(type="pitfall", content="ACE data gaps", scope="mission:ACE"))
        store.add(Memory(type="pitfall", content="Generic lesson", scope="generic"))
        store.add(Memory(type="pitfall", content="Plotly y_range", scope="visualization"))
        store.add(Memory(
            id="dis", type="pitfall", content="Disabled PSP",
            scope="mission:PSP", enabled=False,
        ))

        psp = store.get_pitfalls_by_scope("mission:PSP")
        assert len(psp) == 1
        assert psp[0].content == "PSP fill values"

        ace = store.get_pitfalls_by_scope("mission:ACE")
        assert len(ace) == 1
        assert ace[0].content == "ACE data gaps"

        viz = store.get_pitfalls_by_scope("visualization")
        assert len(viz) == 1
        assert viz[0].content == "Plotly y_range"

        generic = store.get_pitfalls_by_scope("generic")
        assert len(generic) == 1

    def test_build_prompt_only_generic(self, store):
        """build_prompt_section only includes generic-scoped pitfalls."""
        store.add(Memory(type="pitfall", content="Generic operational advice", scope="generic"))
        store.add(Memory(type="pitfall", content="PSP SPC fill values", scope="mission:PSP"))
        store.add(Memory(type="pitfall", content="Plotly rendering tip", scope="visualization"))

        section = store.build_prompt_section()
        assert "Generic operational advice" in section
        assert "PSP SPC fill values" not in section
        assert "Plotly rendering tip" not in section

    def test_get_scoped_pitfall_texts(self, store):
        """get_scoped_pitfall_texts returns content strings capped at MAX_PITFALLS."""
        store.add(Memory(type="pitfall", content="PSP lesson 1", scope="mission:PSP"))
        store.add(Memory(type="pitfall", content="PSP lesson 2", scope="mission:PSP"))
        store.add(Memory(type="pitfall", content="ACE lesson", scope="mission:ACE"))

        texts = store.get_scoped_pitfall_texts("mission:PSP")
        assert texts == ["PSP lesson 1", "PSP lesson 2"]

        texts = store.get_scoped_pitfall_texts("mission:ACE")
        assert texts == ["ACE lesson"]

        texts = store.get_scoped_pitfall_texts("mission:WIND")
        assert texts == []

    def test_get_scoped_pitfall_texts_capped(self, store):
        """Texts are capped at MAX_PITFALLS."""
        for i in range(MAX_PITFALLS + 5):
            store.add(Memory(
                type="pitfall", content=f"Viz lesson {i}", scope="visualization",
            ))
        texts = store.get_scoped_pitfall_texts("visualization")
        assert len(texts) == MAX_PITFALLS

    def test_backward_compat_no_scope(self, tmp_path_file):
        """Old JSON without scope field loads with scope='generic'."""
        tmp_path_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global_enabled": True,
            "memories": [
                {
                    "id": "old1",
                    "type": "pitfall",
                    "content": "Old pitfall without scope",
                    "created_at": "2026-01-01T00:00:00",
                    "source_session": "",
                    "enabled": True,
                }
            ],
        }
        tmp_path_file.write_text(json.dumps(data))
        store = MemoryStore(path=tmp_path_file)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].scope == "generic"
        assert store.get_all()[0].content == "Old pitfall without scope"
