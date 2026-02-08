"""
Tests for the VisualizationAgent sub-agent.

Run with: python -m pytest tests/test_visualization_agent.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.visualization_agent import VisualizationAgent, VIZ_TOOL_CATEGORIES, VIZ_EXTRA_TOOLS


class TestVizAgentToolFiltering:
    """Test that VisualizationAgent gets the right tools."""

    def test_viz_tools_include_execute_visualization(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "execute_visualization" in names

    def test_viz_tools_include_custom_visualization(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "custom_visualization" in names

    def test_viz_tools_include_list_fetched_data(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "list_fetched_data" in names

    def test_viz_tools_count_is_3(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        assert len(tools) == 3  # execute_visualization + custom_visualization + list_fetched_data

    def test_viz_tools_exclude_routing(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "delegate_to_mission" not in names
        assert "delegate_to_visualization" not in names

    def test_viz_tools_exclude_discovery(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "search_datasets" not in names
        assert "list_parameters" not in names


class TestVizAgentInterface:
    """Verify VisualizationAgent interface matches MissionAgent pattern."""

    def test_has_process_request(self):
        assert hasattr(VisualizationAgent, "process_request")
        assert callable(getattr(VisualizationAgent, "process_request"))

    def test_has_execute_task(self):
        assert hasattr(VisualizationAgent, "execute_task")
        assert callable(getattr(VisualizationAgent, "execute_task"))

    def test_has_get_token_usage(self):
        assert hasattr(VisualizationAgent, "get_token_usage")
        assert callable(getattr(VisualizationAgent, "get_token_usage"))
