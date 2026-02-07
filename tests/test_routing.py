"""
Tests for routing, tool filtering, and the delegate_to_mission tool.

Tests tool category filtering and LLM-driven routing architecture
without requiring a Gemini API key.

Run with: python -m pytest tests/test_routing.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.mission_agent import MISSION_TOOL_CATEGORIES


class TestToolCategoryFiltering:
    """Test get_tool_schemas() category filtering."""

    PLOTTING_TOOLS = {
        "plot_data", "change_time_range", "export_plot", "get_plot_info", "plot_computed_data",
        "reset_plot", "set_plot_title", "set_axis_label", "toggle_log_scale",
        "set_axis_range", "save_session", "load_session",
    }

    def test_no_filter_returns_all_tools(self):
        all_tools = get_tool_schemas()
        assert len(all_tools) == 22
        names = {t["name"] for t in all_tools}
        assert "plot_data" in names
        assert "fetch_data" in names
        assert "delegate_to_mission" in names

    def test_mission_categories_exclude_plotting_and_routing(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES)
        names = {t["name"] for t in mission_tools}
        # Should not include any plotting tools
        assert names.isdisjoint(self.PLOTTING_TOOLS), f"Unexpected plotting tools: {names & self.PLOTTING_TOOLS}"
        # Should not include routing tools (no recursive delegation)
        assert "delegate_to_mission" not in names
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


class TestDelegateToMissionTool:
    """Test that the delegate_to_mission tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_mission" in names

    def test_tool_has_routing_category(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_mission")
        assert tool["category"] == "routing"

    def test_tool_not_in_mission_agent_tools(self):
        """Sub-agents should NOT have delegate_to_mission (no recursive delegation)."""
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES)
        names = {t["name"] for t in mission_tools}
        assert "delegate_to_mission" not in names

    def test_tool_requires_mission_id_and_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_mission")
        assert "mission_id" in tool["parameters"]["properties"]
        assert "request" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["mission_id", "request"]


class TestMissionAgentImportAndInterface:
    """Verify MissionAgent interface."""

    def test_process_request_method_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "process_request")
        assert callable(getattr(MissionAgent, "process_request"))

    def test_execute_task_still_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "execute_task")
