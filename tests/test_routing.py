"""
Tests for the always-delegate routing logic in core.py and tool filtering.

Tests _is_general_request() heuristics, mission detection routing,
and tool category filtering without requiring a Gemini API key.

Run with: python -m pytest tests/test_routing.py -v
"""

import pytest
from agent.core import AutoplotAgent
from agent.tools import get_tool_schemas


class TestIsGeneralRequest:
    """Test the _is_general_request() static method."""

    # Meta questions
    @pytest.mark.parametrize("text", [
        "help",
        "what can you do",
        "what can you do?",
        "What missions do you support?",
        "what missions are available",
        "how do I use this?",
        "how does the time range work?",
        "hello",
        "hi",
        "Hi there!",
        "thanks",
        "thank you",
        "Thanks for the help!",
    ])
    def test_meta_questions_are_general(self, text):
        assert AutoplotAgent._is_general_request(text) is True

    # Plot follow-ups
    @pytest.mark.parametrize("text", [
        "zoom in to last 3 days",
        "zoom into January",
        "export this as png",
        "export the plot",
        "save the plot as output.png",
        "change the time range to last month",
        "change time to 2024-01-15",
        "what is currently plotted?",
        "what's showing?",
        "get plot info",
    ])
    def test_plot_followups_are_general(self, text):
        assert AutoplotAgent._is_general_request(text) is True

    # Mission-specific requests should NOT be general
    @pytest.mark.parametrize("text", [
        "show me PSP magnetic field data",
        "plot ACE solar wind for last week",
        "fetch OMNI data",
        "parker probe magnetic field",
        "compare PSP and ACE",
        "what data does Solar Orbiter have?",
        "show wind plasma data",
        "MMS magnetopause crossing",
        "show me the magnetic field magnitude",
        "fetch and plot density data",
    ])
    def test_mission_requests_are_not_general(self, text):
        assert AutoplotAgent._is_general_request(text) is False


class TestToolCategoryFiltering:
    """Test get_tool_schemas() category filtering."""

    PLOTTING_TOOLS = {"plot_data", "change_time_range", "export_plot", "get_plot_info", "plot_computed_data"}

    def test_no_filter_returns_all_tools(self):
        all_tools = get_tool_schemas()
        assert len(all_tools) == 14
        names = {t["name"] for t in all_tools}
        assert "plot_data" in names
        assert "fetch_data" in names

    def test_mission_categories_exclude_plotting(self):
        mission_tools = get_tool_schemas(categories=["discovery", "data_ops", "conversation"])
        names = {t["name"] for t in mission_tools}
        # Should not include any plotting tools
        assert names.isdisjoint(self.PLOTTING_TOOLS), f"Unexpected plotting tools: {names & self.PLOTTING_TOOLS}"
        # Should include data tools
        assert "fetch_data" in names
        assert "search_datasets" in names
        assert "custom_operation" in names
        assert "ask_clarification" in names

    def test_plotting_category_only(self):
        plot_tools = get_tool_schemas(categories=["plotting"])
        names = {t["name"] for t in plot_tools}
        assert names == self.PLOTTING_TOOLS

    def test_every_tool_has_category(self):
        for tool in get_tool_schemas():
            assert "category" in tool, f"Tool '{tool['name']}' missing category field"

    def test_empty_categories_returns_nothing(self):
        assert get_tool_schemas(categories=[]) == []


class TestMissionAgentImportAndInterface:
    """Verify the new process_request method exists on MissionAgent."""

    def test_process_request_method_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "process_request")
        assert callable(getattr(MissionAgent, "process_request"))

    def test_execute_task_still_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "execute_task")
