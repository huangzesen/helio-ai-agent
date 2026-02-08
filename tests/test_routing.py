"""
Tests for routing, tool filtering, and the delegate tools.

Tests tool category filtering and LLM-driven routing architecture
without requiring a Gemini API key.

Run with: python -m pytest tests/test_routing.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.mission_agent import MISSION_TOOL_CATEGORIES
from agent.visualization_agent import VIZ_TOOL_CATEGORIES, VIZ_EXTRA_TOOLS
from agent.core import ORCHESTRATOR_CATEGORIES, ORCHESTRATOR_EXTRA_TOOLS


class TestToolCategoryFiltering:
    """Test get_tool_schemas() category filtering."""

    def test_no_filter_returns_all_tools(self):
        all_tools = get_tool_schemas()
        assert len(all_tools) == 13
        names = {t["name"] for t in all_tools}
        assert "execute_visualization" in names
        assert "fetch_data" in names
        assert "delegate_to_mission" in names
        assert "delegate_to_visualization" in names

    def test_mission_categories_exclude_visualization_and_routing(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES)
        names = {t["name"] for t in mission_tools}
        # Should not include visualization tools
        assert "execute_visualization" not in names
        # Should not include routing tools (no recursive delegation)
        assert "delegate_to_mission" not in names
        assert "delegate_to_visualization" not in names
        # Should include data tools
        assert "fetch_data" in names
        assert "search_datasets" in names
        assert "custom_operation" in names
        assert "ask_clarification" in names

    def test_visualization_category_only(self):
        viz_tools = get_tool_schemas(categories=VIZ_TOOL_CATEGORIES)
        names = {t["name"] for t in viz_tools}
        assert names == {"execute_visualization"}

    def test_visualization_with_extras(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert names == {"execute_visualization", "list_fetched_data"}

    def test_orchestrator_categories(self):
        orch_tools = get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS)
        names = {t["name"] for t in orch_tools}
        # Should include routing
        assert "delegate_to_mission" in names
        assert "delegate_to_visualization" in names
        # Should include discovery
        assert "search_datasets" in names
        # Should include list_fetched_data (extra tool)
        assert "list_fetched_data" in names
        # Should NOT include data_ops (delegated to mission agents)
        assert "fetch_data" not in names
        assert "custom_operation" not in names
        # Should NOT include visualization
        assert "execute_visualization" not in names

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


class TestDelegateToVisualizationTool:
    """Test that the delegate_to_visualization tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_visualization" in names

    def test_tool_has_routing_category(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_visualization")
        assert tool["category"] == "routing"

    def test_tool_requires_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_visualization")
        assert "request" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["request"]

    def test_tool_not_in_viz_agent_tools(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "delegate_to_visualization" not in names


class TestMissionAgentImportAndInterface:
    """Verify MissionAgent interface."""

    def test_process_request_method_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "process_request")
        assert callable(getattr(MissionAgent, "process_request"))

    def test_execute_task_still_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "execute_task")
