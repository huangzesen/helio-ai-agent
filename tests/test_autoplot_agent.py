"""
Tests for the AutoplotAgent sub-agent.

Run with: python -m pytest tests/test_autoplot_agent.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.autoplot_agent import AutoplotAgent, AUTOPLOT_TOOL_CATEGORIES, AUTOPLOT_EXTRA_TOOLS


class TestAutoplotAgentToolFiltering:
    """Test that AutoplotAgent gets the right tools."""

    def test_autoplot_tools_include_execute_autoplot(self):
        tools = get_tool_schemas(
            categories=AUTOPLOT_TOOL_CATEGORIES,
            extra_names=AUTOPLOT_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "execute_autoplot" in names

    def test_autoplot_tools_include_list_fetched_data(self):
        tools = get_tool_schemas(
            categories=AUTOPLOT_TOOL_CATEGORIES,
            extra_names=AUTOPLOT_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "list_fetched_data" in names

    def test_autoplot_tools_count_is_3(self):
        tools = get_tool_schemas(
            categories=AUTOPLOT_TOOL_CATEGORIES,
            extra_names=AUTOPLOT_EXTRA_TOOLS,
        )
        assert len(tools) == 3

    def test_autoplot_tools_exclude_routing(self):
        tools = get_tool_schemas(
            categories=AUTOPLOT_TOOL_CATEGORIES,
            extra_names=AUTOPLOT_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "delegate_to_mission" not in names
        assert "delegate_to_autoplot" not in names

    def test_autoplot_tools_exclude_discovery(self):
        tools = get_tool_schemas(
            categories=AUTOPLOT_TOOL_CATEGORIES,
            extra_names=AUTOPLOT_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "search_datasets" not in names
        assert "list_parameters" not in names


class TestAutoplotAgentInterface:
    """Verify AutoplotAgent interface matches MissionAgent pattern."""

    def test_has_process_request(self):
        assert hasattr(AutoplotAgent, "process_request")
        assert callable(getattr(AutoplotAgent, "process_request"))

    def test_has_execute_task(self):
        assert hasattr(AutoplotAgent, "execute_task")
        assert callable(getattr(AutoplotAgent, "execute_task"))

    def test_has_get_token_usage(self):
        assert hasattr(AutoplotAgent, "get_token_usage")
        assert callable(getattr(AutoplotAgent, "get_token_usage"))
