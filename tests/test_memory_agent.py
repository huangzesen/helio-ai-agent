"""
Tests for agent.memory_agent — MemoryAgent passive analysis.

Run with: python -m pytest tests/test_memory_agent.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.memory import Memory, MemoryStore
from agent.memory_agent import (
    MemoryAgent,
    AnalysisResult,
    LOG_GROWTH_THRESHOLD,
    ERROR_COUNT_THRESHOLD,
    TURN_COUNT_THRESHOLD,
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
    """Provide a MemoryAgent with mock client."""
    client = MagicMock()
    return MemoryAgent(
        client=client,
        model_name="test-model",
        memory_store=memory_store,
        verbose=False,
    )


# ---- should_analyze() ----

class TestShouldAnalyze:
    def test_triggers_on_turn_count(self, agent, tmp_dir):
        """Should trigger when turn_count exceeds threshold."""
        state = {"last_turn_count": 0}
        with patch.object(agent, "_load_state", return_value=state):
            with patch("agent.memory_agent.get_current_log_path", return_value=tmp_dir / "fake.log"):
                with patch("agent.memory_agent.get_log_size", return_value=0):
                    assert agent.should_analyze(TURN_COUNT_THRESHOLD)

    def test_no_trigger_below_turn_threshold(self, agent, tmp_dir):
        """Should NOT trigger when turn_count is below threshold."""
        state = {"last_turn_count": 0}
        with patch.object(agent, "_load_state", return_value=state):
            with patch("agent.memory_agent.get_current_log_path", return_value=tmp_dir / "fake.log"):
                with patch("agent.memory_agent.get_log_size", return_value=0):
                    assert not agent.should_analyze(TURN_COUNT_THRESHOLD - 1)

    def test_triggers_on_log_growth(self, agent, tmp_dir):
        """Should trigger when log grows by threshold bytes."""
        state = {
            "last_turn_count": 100,
            "last_log_file": "agent_20260209.log",
            "last_log_byte_offset": 1000,
        }
        with patch.object(agent, "_load_state", return_value=state):
            log_path = tmp_dir / "agent_20260209.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=1000 + LOG_GROWTH_THRESHOLD):
                    assert agent.should_analyze(100)  # same turn count

    def test_no_trigger_below_log_growth(self, agent, tmp_dir):
        """Should NOT trigger when log growth is below threshold."""
        state = {
            "last_turn_count": 100,
            "last_log_file": "agent_20260209.log",
            "last_log_byte_offset": 1000,
        }
        with patch.object(agent, "_load_state", return_value=state):
            log_path = tmp_dir / "agent_20260209.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=1000 + LOG_GROWTH_THRESHOLD - 1):
                    assert not agent.should_analyze(100)

    def test_triggers_on_log_rotation(self, agent, tmp_dir):
        """Should trigger when log file name changes (new day) and size exceeds threshold."""
        state = {
            "last_turn_count": 100,
            "last_log_file": "agent_20260208.log",
            "last_log_byte_offset": 50000,
        }
        with patch.object(agent, "_load_state", return_value=state):
            log_path = tmp_dir / "agent_20260209.log"
            with patch("agent.memory_agent.get_current_log_path", return_value=log_path):
                with patch("agent.memory_agent.get_log_size", return_value=LOG_GROWTH_THRESHOLD):
                    assert agent.should_analyze(100)

    def test_triggers_on_error_count(self, agent, tmp_dir):
        """Should trigger when error count meets threshold."""
        state = {"last_turn_count": 100}
        with patch.object(agent, "_load_state", return_value=state):
            with patch("agent.memory_agent.get_current_log_path", return_value=tmp_dir / "fake.log"):
                with patch("agent.memory_agent.get_log_size", return_value=0):
                    with patch.object(agent, "_count_new_errors", return_value=ERROR_COUNT_THRESHOLD):
                        assert agent.should_analyze(100)

    def test_empty_state_no_trigger_low_turns(self, agent, tmp_dir):
        """Empty state file with low turn count should not trigger."""
        with patch.object(agent, "_load_state", return_value={}):
            with patch("agent.memory_agent.get_current_log_path", return_value=tmp_dir / "fake.log"):
                with patch("agent.memory_agent.get_log_size", return_value=0):
                    assert not agent.should_analyze(TURN_COUNT_THRESHOLD - 1)


# ---- State persistence ----

class TestStatePersistence:
    def test_save_and_load_roundtrip(self, agent, tmp_dir):
        """State should round-trip through save/load."""
        state = {
            "last_log_file": "agent_20260209.log",
            "last_log_byte_offset": 12345,
            "last_analysis_timestamp": "2026-02-09T14:30:00",
            "last_turn_count": 25,
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
                0, 50000, 15,
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
            path = agent._save_report(result, "", "log.log", 0, 100, 5)
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
    def test_includes_conversation_and_log(self, agent):
        prompt = agent._build_analysis_prompt(
            "User: Show ACE data\nAgent: Here is the data",
            "2026-02-09 | ERROR | fetch failed",
            [], [], [],
        )
        assert "Show ACE data" in prompt
        assert "fetch failed" in prompt
        assert "error_patterns" in prompt

    def test_includes_existing_memories(self, agent):
        prompt = agent._build_analysis_prompt(
            "User: Hello",
            "",
            ["Prefers dark theme"],
            ["OMNI may fail"],
            ["Analyzed ACE"],
        )
        assert "Prefers dark theme" in prompt
        assert "OMNI may fail" in prompt
        assert "Analyzed ACE" in prompt
        assert "do NOT duplicate" in prompt
