"""
Tests for routing, tool filtering, and the delegate tools.

Tests tool category filtering and LLM-driven routing architecture
without requiring a Gemini API key.

Run with: python -m pytest tests/test_routing.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.mission_agent import MISSION_TOOL_CATEGORIES
from agent.autoplot_agent import AUTOPLOT_TOOL_CATEGORIES, AUTOPLOT_EXTRA_TOOLS
from agent.core import ORCHESTRATOR_CATEGORIES


class TestToolCategoryFiltering:
    """Test get_tool_schemas() category filtering."""

    def test_no_filter_returns_all_tools(self):
        all_tools = get_tool_schemas()
        assert len(all_tools) == 14
        names = {t["name"] for t in all_tools}
        assert "execute_autoplot" in names
        assert "fetch_data" in names
        assert "delegate_to_mission" in names
        assert "delegate_to_autoplot" in names

    def test_mission_categories_exclude_autoplot_and_routing(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES)
        names = {t["name"] for t in mission_tools}
        # Should not include autoplot tools
        assert "execute_autoplot" not in names
        # Should not include routing tools (no recursive delegation)
        assert "delegate_to_mission" not in names
        assert "delegate_to_autoplot" not in names
        # Should include data tools
        assert "fetch_data" in names
        assert "search_datasets" in names
        assert "custom_operation" in names
        assert "ask_clarification" in names

    def test_autoplot_category_only(self):
        autoplot_tools = get_tool_schemas(categories=AUTOPLOT_TOOL_CATEGORIES)
        names = {t["name"] for t in autoplot_tools}
        assert names == {"execute_autoplot", "autoplot_script"}

    def test_autoplot_with_extras(self):
        tools = get_tool_schemas(
            categories=AUTOPLOT_TOOL_CATEGORIES,
            extra_names=AUTOPLOT_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert names == {"execute_autoplot", "autoplot_script", "list_fetched_data"}

    def test_orchestrator_categories(self):
        orch_tools = get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES)
        names = {t["name"] for t in orch_tools}
        # Should include routing
        assert "delegate_to_mission" in names
        assert "delegate_to_autoplot" in names
        # Should include discovery and data ops
        assert "search_datasets" in names
        assert "fetch_data" in names
        # Should NOT include autoplot
        assert "execute_autoplot" not in names

    def test_every_tool_has_category(self):
        for tool in get_tool_schemas():
            assert "category" in tool, f"Tool '{tool['name']}' missing category field"

    def test_empty_categories_returns_nothing(self):
        assert get_tool_schemas(categories=[]) == []

    def test_extra_names_without_categories(self):
        tools = get_tool_schemas(categories=[], extra_names=["fetch_data"])
        assert len(tools) == 1
        assert tools[0]["name"] == "fetch_data"


class TestDelegateToMissionTool:
    """Test that the delegate_to_mission tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_mission" in names

    def test_tool_has_routing_category(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_mission")
        assert tool["category"] == "routing"

    def test_tool_not_in_mission_agent_tools(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES)
        names = {t["name"] for t in mission_tools}
        assert "delegate_to_mission" not in names

    def test_tool_requires_mission_id_and_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_mission")
        assert "mission_id" in tool["parameters"]["properties"]
        assert "request" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["mission_id", "request"]


class TestDelegateToAutoplotTool:
    """Test that the delegate_to_autoplot tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_autoplot" in names

    def test_tool_has_routing_category(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_autoplot")
        assert tool["category"] == "routing"

    def test_tool_requires_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_autoplot")
        assert "request" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["request"]

    def test_tool_not_in_autoplot_agent_tools(self):
        tools = get_tool_schemas(
            categories=AUTOPLOT_TOOL_CATEGORIES,
            extra_names=AUTOPLOT_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "delegate_to_autoplot" not in names


class TestMissionAgentImportAndInterface:
    """Verify MissionAgent interface."""

    def test_process_request_method_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "process_request")
        assert callable(getattr(MissionAgent, "process_request"))

    def test_execute_task_still_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "execute_task")
