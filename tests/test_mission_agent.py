"""
Tests for agent.mission_agent — MissionAgent class.

Tests the mission-specific sub-agent without requiring a Gemini API key.
Verifies prompt construction and structural behavior.

Run with: python -m pytest tests/test_mission_agent.py
"""

import pytest
from knowledge.catalog import SPACECRAFT
from knowledge.prompt_builder import build_mission_prompt


class TestBuildMissionPromptForAgent:
    """Verify that build_mission_prompt produces usable prompts for MissionAgent."""

    def test_all_missions_produce_prompt(self):
        for sc_id in SPACECRAFT:
            prompt = build_mission_prompt(sc_id)
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_psp_prompt_is_focused(self):
        prompt = build_mission_prompt("PSP")
        assert "Parker Solar Probe" in prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in prompt
        # Should NOT mention other missions
        assert "AC_H2_MFI" not in prompt
        assert "OMNI_HRO_1MIN" not in prompt

    def test_ace_prompt_is_focused(self):
        prompt = build_mission_prompt("ACE")
        assert "Advanced Composition Explorer" in prompt
        assert "AC_H2_MFI" in prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" not in prompt

    def test_prompt_contains_specialist_identity(self):
        prompt = build_mission_prompt("PSP")
        assert "specialist agent" in prompt.lower()

    def test_prompt_directs_to_list_parameters(self):
        prompt = build_mission_prompt("PSP")
        assert "list_parameters" in prompt

    def test_invalid_mission_raises(self):
        with pytest.raises(KeyError):
            build_mission_prompt("NONEXISTENT")


class TestMissionAgentImport:
    """Verify MissionAgent can be imported and has expected interface."""

    def test_import(self):
        from agent.mission_agent import MissionAgent
        assert MissionAgent is not None

    def test_class_has_execute_task(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "execute_task")

    def test_class_has_get_token_usage(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "get_token_usage")
