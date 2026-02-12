"""
Tests for agent.memory_agent — MemoryAgent passive analysis.

Run with: python -m pytest tests/test_memory_agent.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.llm import LLMResponse
from agent.memory import Memory, MemoryStore
from agent.memory_agent import (
    MemoryAgent,
    AnalysisResult,
    LOG_GROWTH_THRESHOLD,
    ERROR_COUNT_THRESHOLD,
    STATE_FILE,
    REPORTS_DIR,
)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for state/memory/reports."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def memory_store(tmp_dir):
    """Provide a MemoryStore backed by a temp file."""
    return MemoryStore(path=tmp_dir / "memory.json")


@pytest.fixture
def agent(memory_store):
    """Provide a MemoryAgent with mock adapter."""
    adapter = MagicMock()
    return MemoryAgent(
        adapter=adapter,
        model_name="test-model",
        memory_store=memory_store,
        verbose=False,
    )


# ---- _should_analyze() ----

class TestShouldAnalyze:
    def test_triggers_on_log_growth(self, agent, tmp_dir):
        """Should trigger when log grows by threshold bytes."""
        state = {
            "last_log_file": "agent_20260209.log",
            "last_log_byte_offset": 1000,
        }
        with patch.object(agent, "_load_state", return_value=state):
            log_path = tmp_dir / "agent_20260209.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=1000 + LOG_GROWTH_THRESHOLD):
                    assert agent._should_analyze()

    def test_no_trigger_below_log_growth(self, agent, tmp_dir):
        """Should NOT trigger when log growth is below threshold."""
        state = {
            "last_log_file": "agent_20260209.log",
            "last_log_byte_offset": 1000,
        }
        with patch.object(agent, "_load_state", return_value=state):
            log_path = tmp_dir / "agent_20260209.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=1000 + LOG_GROWTH_THRESHOLD - 1):
                    with patch.object(agent, "_count_new_errors", return_value=0):
                        assert not agent._should_analyze()

    def test_triggers_on_log_rotation(self, agent, tmp_dir):
        """Should trigger when log file name changes (new day) and size exceeds threshold."""
        state = {
            "last_log_file": "agent_20260208.log",
            "last_log_byte_offset": 50000,
        }
        with patch.object(agent, "_load_state", return_value=state):
            log_path = tmp_dir / "agent_20260209.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=LOG_GROWTH_THRESHOLD):
                    assert agent._should_analyze()

    def test_triggers_on_error_count(self, agent, tmp_dir):
        """Should trigger when error count meets threshold."""
        state = {"last_log_file": "agent_20260209.log", "last_log_byte_offset": 1000}
        with patch.object(agent, "_load_state", return_value=state):
            log_path = tmp_dir / "agent_20260209.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=1000):
                    with patch.object(agent, "_count_new_errors", return_value=ERROR_COUNT_THRESHOLD):
                        assert agent._should_analyze()

    def test_no_trigger_small_log_no_errors(self, agent, tmp_dir):
        """Should NOT trigger with small log and no errors."""
        with patch.object(agent, "_load_state", return_value={}):
            log_path = tmp_dir / "fake.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=0):
                    with patch.object(agent, "_count_new_errors", return_value=0):
                        assert not agent._should_analyze()


# ---- State persistence ----

class TestStatePersistence:
    def test_save_and_load_roundtrip(self, agent, tmp_dir):
        """State should round-trip through save/load."""
        state = {
            "last_log_file": "agent_20260209.log",
            "last_log_byte_offset": 12345,
            "last_analysis_timestamp": "2026-02-09T14:30:00",
        }
        state_file = tmp_dir / "state.json"
        with patch("agent.memory_agent.STATE_FILE", state_file):
            with patch("agent.memory_agent.STATE_DIR", tmp_dir):
                agent._save_state(state)
                loaded = agent._load_state()
        assert loaded == state

    def test_load_missing_file(self, agent, tmp_dir):
        """Missing state file returns empty dict."""
        with patch("agent.memory_agent.STATE_FILE", tmp_dir / "missing.json"):
            assert agent._load_state() == {}

    def test_load_corrupt_file(self, agent, tmp_dir):
        """Corrupt state file returns empty dict."""
        state_file = tmp_dir / "state.json"
        state_file.write_text("not valid json")
        with patch("agent.memory_agent.STATE_FILE", state_file):
            assert agent._load_state() == {}


# ---- _count_new_errors() ----

class TestCountNewErrors:
    def test_counts_errors_and_warnings(self, agent, tmp_dir):
        """Should count both ERROR and WARNING lines."""
        log_content = (
            "2026-02-09 14:00:00 | DEBUG    | helio-agent | Normal debug\n"
            "2026-02-09 14:00:01 | ERROR    | helio-agent | Something failed\n"
            "2026-02-09 14:00:02 | WARNING  | helio-agent | Something wrong\n"
            "2026-02-09 14:00:03 | INFO     | helio-agent | Normal info\n"
            "2026-02-09 14:00:04 | ERROR    | helio-agent | Another error\n"
        )
        with patch.object(agent, "_read_new_log_content", return_value=log_content):
            assert agent._count_new_errors({}) == 3

    def test_empty_log(self, agent):
        """Empty log content should return 0."""
        with patch.object(agent, "_read_new_log_content", return_value=""):
            assert agent._count_new_errors({}) == 0


# ---- _parse_analysis_response() ----

class TestParseAnalysisResponse:
    def test_valid_json(self, agent):
        """Should parse valid JSON."""
        text = json.dumps({
            "preferences": ["Dark theme"],
            "summary": "Analyzed ACE data",
            "pitfalls": ["OMNI may have empty strings"],
            "error_patterns": [],
        })
        result = agent._parse_analysis_response(text)
        assert result is not None
        assert result["preferences"] == ["Dark theme"]
        assert result["pitfalls"] == ["OMNI may have empty strings"]

    def test_json_with_markdown_fencing(self, agent):
        """Should strip markdown fencing."""
        text = "```json\n" + json.dumps({"preferences": [], "summary": ""}) + "\n```"
        result = agent._parse_analysis_response(text)
        assert result is not None
        assert result["preferences"] == []

    def test_invalid_json(self, agent):
        """Should return None for invalid JSON."""
        assert agent._parse_analysis_response("not json") is None

    def test_empty_string(self, agent):
        """Should return None for empty string."""
        assert agent._parse_analysis_response("") is None

    def test_none_text(self, agent):
        """Should handle None-ish input."""
        assert agent._parse_analysis_response("") is None


# ---- Report generation ----

class TestReportGeneration:
    def test_report_created(self, agent, tmp_dir):
        """Should create a markdown report file."""
        result = AnalysisResult(
            pitfalls=["Always validate time ranges"],
            error_patterns=[{
                "component": "data_ops/fetch.py",
                "occurrences": 2,
                "pattern": "CSV parse failure",
                "suggested_fix": "Handle empty strings as NaN",
            }],
        )
        with patch("agent.memory_agent.REPORTS_DIR", tmp_dir / "reports"):
            path = agent._save_report(
                result, "session_123", "agent_20260209.log",
                0, 50000,
            )
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "# Analysis Report" in content
        assert "session_123" in content
        assert "CSV parse failure" in content
        assert "Always validate time ranges" in content

    def test_report_without_errors(self, agent, tmp_dir):
        """Report with only pitfalls should still work."""
        result = AnalysisResult(
            pitfalls=["Check for NaN values"],
        )
        with patch("agent.memory_agent.REPORTS_DIR", tmp_dir / "reports"):
            path = agent._save_report(result, "", "log.log", 0, 100)
        content = Path(path).read_text()
        assert "## Error Patterns" not in content
        assert "Check for NaN values" in content


# ---- Pitfall dedup ----

class TestPitfallDedup:
    def test_dedup_exact_substring(self, agent, memory_store):
        """Pitfall that is substring of existing should be skipped."""
        existing = ["OMNI data for recent dates may have empty CSV strings"]
        count = agent._save_pitfalls(
            ["OMNI data for recent dates may have empty CSV strings"],
            "session_1",
            existing,
        )
        assert count == 0

    def test_dedup_case_insensitive(self, agent, memory_store):
        """Dedup should be case-insensitive."""
        existing = ["omni data may fail"]
        count = agent._save_pitfalls(
            ["OMNI data may fail"],
            "session_1",
            existing,
        )
        assert count == 0

    def test_new_pitfall_added(self, agent, memory_store):
        """New pitfall should be added."""
        count = agent._save_pitfalls(
            ["MMS dataset IDs require @0 suffix"],
            "session_1",
            [],
        )
        assert count == 1
        pitfalls = [m for m in memory_store.get_all() if m.type == "pitfall"]
        assert len(pitfalls) == 1
        assert pitfalls[0].content == "MMS dataset IDs require @0 suffix"

    def test_batch_dedup(self, agent, memory_store):
        """Multiple same pitfalls in one batch — only first should be added."""
        count = agent._save_pitfalls(
            ["Lesson A", "Lesson A", "Lesson B"],
            "session_1",
            [],
        )
        assert count == 2


# ---- Prompt building ----

class TestBuildAnalysisPrompt:
    def test_includes_log_content(self, agent):
        prompt = agent._build_analysis_prompt(
            "2026-02-09 | ERROR | fetch failed",
            [], [], [],
        )
        assert "fetch failed" in prompt
        assert "error_patterns" in prompt

    def test_includes_existing_memories(self, agent):
        prompt = agent._build_analysis_prompt(
            "",
            ["Prefers dark theme"],
            ["OMNI may fail"],
            ["Analyzed ACE"],
        )
        assert "Prefers dark theme" in prompt
        assert "OMNI may fail" in prompt
        assert "Analyzed ACE" in prompt
        assert "do NOT duplicate" in prompt


# ---- consolidate() ----

class TestConsolidate:
    def test_skips_when_under_limit(self, agent, memory_store):
        """Should not call LLM when memory count is at or below max_total."""
        memory_store.add(Memory(type="preference", content="A"))
        memory_store.add(Memory(type="pitfall", content="B"))
        removed = agent.consolidate(max_total=5)
        assert removed == 0
        # Client should NOT have been called
        agent.adapter.generate.assert_not_called()

    def test_consolidates_when_over_limit(self, agent, memory_store):
        """Should call LLM and reduce memory count."""
        for i in range(15):
            memory_store.add(Memory(
                id=f"m{i}", type="preference", content=f"Pref {i}",
            ))
        # Mock LLM response with 5 merged entries
        consolidated = json.dumps([
            {"id": "m0", "type": "preference", "content": "Merged pref A"},
            {"id": "m3", "type": "preference", "content": "Merged pref B"},
            {"id": "new1", "type": "pitfall", "content": "Merged pitfall"},
            {"id": "m7", "type": "summary", "content": "Merged summary"},
            {"id": "m10", "type": "preference", "content": "Merged pref C"},
        ])
        mock_response = MagicMock()
        mock_response.text = consolidated
        agent.adapter.generate.return_value = mock_response

        removed = agent.consolidate(max_total=10)
        assert removed == 10  # 15 - 5
        assert len(memory_store.get_all()) == 5
        contents = [m.content for m in memory_store.get_all()]
        assert "Merged pref A" in contents
        assert "Merged pitfall" in contents

    def test_preserves_disabled_memories(self, agent, memory_store):
        """Disabled memories should be kept even if not in LLM output."""
        # Add some enabled
        for i in range(12):
            memory_store.add(Memory(
                id=f"e{i}", type="preference", content=f"Enabled {i}",
            ))
        # Add a disabled memory
        memory_store.add(Memory(
            id="dis1", type="preference", content="Disabled one", enabled=False,
        ))

        consolidated = json.dumps([
            {"id": "e0", "type": "preference", "content": "Kept pref"},
        ])
        mock_response = MagicMock()
        mock_response.text = consolidated
        agent.adapter.generate.return_value = mock_response

        agent.consolidate(max_total=10)
        all_memories = memory_store.get_all()
        assert len(all_memories) == 2  # 1 disabled + 1 consolidated
        disabled = [m for m in all_memories if not m.enabled]
        assert len(disabled) == 1
        assert disabled[0].id == "dis1"

    def test_handles_llm_failure(self, agent, memory_store):
        """Should return 0 and keep memories intact on LLM failure."""
        for i in range(12):
            memory_store.add(Memory(
                id=f"f{i}", type="preference", content=f"Pref {i}",
            ))
        agent.adapter.generate.side_effect = Exception("API error")

        removed = agent.consolidate(max_total=10)
        assert removed == 0
        assert len(memory_store.get_all()) == 12

    def test_handles_invalid_json_response(self, agent, memory_store):
        """Should return 0 on unparseable LLM response."""
        for i in range(12):
            memory_store.add(Memory(
                id=f"j{i}", type="preference", content=f"Pref {i}",
            ))
        mock_response = MagicMock()
        mock_response.text = "not valid json at all"
        agent.adapter.generate.return_value = mock_response

        removed = agent.consolidate(max_total=10)
        assert removed == 0
        assert len(memory_store.get_all()) == 12

    def test_handles_empty_response(self, agent, memory_store):
        """Should return 0 if LLM returns empty list."""
        for i in range(12):
            memory_store.add(Memory(
                id=f"z{i}", type="preference", content=f"Pref {i}",
            ))
        mock_response = MagicMock()
        mock_response.text = "[]"
        agent.adapter.generate.return_value = mock_response

        removed = agent.consolidate(max_total=10)
        assert removed == 0
        assert len(memory_store.get_all()) == 12

    def test_strips_markdown_fencing(self, agent, memory_store):
        """Should handle LLM response wrapped in markdown fencing."""
        for i in range(12):
            memory_store.add(Memory(
                id=f"md{i}", type="pitfall", content=f"Pitfall {i}",
            ))
        consolidated = json.dumps([
            {"id": "md0", "type": "pitfall", "content": "Merged pitfall A"},
            {"id": "md1", "type": "pitfall", "content": "Merged pitfall B"},
        ])
        mock_response = MagicMock()
        mock_response.text = f"```json\n{consolidated}\n```"
        agent.adapter.generate.return_value = mock_response

        removed = agent.consolidate(max_total=10)
        assert removed == 10  # 12 - 2
        assert len(memory_store.get_enabled()) == 2

    def test_evicted_memories_archived_to_cold(self, agent, memory_store):
        """Evicted memories should be written to cold storage, not deleted."""
        for i in range(12):
            memory_store.add(Memory(
                id=f"arc{i}", type="pitfall", content=f"Pitfall {i}",
            ))
        # LLM keeps only 2
        consolidated = json.dumps([
            {"id": "arc0", "type": "pitfall", "content": "Pitfall 0"},
            {"id": "arc5", "type": "pitfall", "content": "Pitfall 5"},
        ])
        mock_response = MagicMock()
        mock_response.text = consolidated
        agent.adapter.generate.return_value = mock_response

        agent.consolidate(max_total=10)
        # Cold storage should have the 10 evicted memories
        assert memory_store.cold_path.exists()
        cold_data = json.loads(memory_store.cold_path.read_text())
        assert len(cold_data) == 10
        cold_ids = {m["id"] for m in cold_data}
        assert "arc0" not in cold_ids  # kept
        assert "arc5" not in cold_ids  # kept
        assert "arc1" in cold_ids  # evicted
        assert "arc11" in cold_ids  # evicted
