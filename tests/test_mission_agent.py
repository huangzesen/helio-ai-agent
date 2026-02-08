"""
Tests for agent.mission_agent — MissionAgent class.

Tests the mission-specific sub-agent without requiring a Gemini API key.
Verifies prompt construction, structural behavior, and tool filtering.

Run with: python -m pytest tests/test_mission_agent.py
"""

import pytest
from knowledge.catalog import SPACECRAFT
from knowledge.prompt_builder import build_mission_prompt
from agent.tools import get_tool_schemas
from agent.mission_agent import MISSION_TOOL_CATEGORIES, MISSION_EXTRA_TOOLS


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

    def test_prompt_contains_data_specialist_identity(self):
        prompt = build_mission_prompt("PSP")
        assert "data specialist agent" in prompt.lower()

    def test_prompt_directs_to_list_parameters(self):
        prompt = build_mission_prompt("PSP")
        assert "list_parameters" in prompt

    def test_invalid_mission_raises(self):
        with pytest.raises(KeyError):
            build_mission_prompt("NONEXISTENT")


class TestMissionAgentToolFiltering:
    """Verify mission sub-agents do NOT have plotting tools."""

    PLOTTING_TOOLS = {"plot_data", "change_time_range", "export_plot", "get_plot_info", "plot_computed_data"}

    def test_mission_tools_exclude_plotting(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        assert names.isdisjoint(self.PLOTTING_TOOLS), (
            f"Mission sub-agents should not have plotting tools, found: {names & self.PLOTTING_TOOLS}"
        )

    def test_mission_tools_include_fetch(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        assert "fetch_data" in names
        assert "list_fetched_data" in names

    def test_mission_tools_exclude_compute(self):
        """MissionAgent no longer has compute tools — those moved to DataOpsAgent."""
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        assert "custom_operation" not in names
        assert "describe_data" not in names
        assert "save_data" not in names

    def test_mission_tools_include_discovery(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        assert "search_datasets" in names
        assert "list_parameters" in names
        assert "get_data_availability" in names
        assert "get_dataset_docs" in names

    def test_mission_tools_include_conversation(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        assert "ask_clarification" in names


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

    def test_class_has_process_request(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "process_request")
