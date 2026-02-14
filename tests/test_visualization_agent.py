"""
Tests for the VisualizationAgent sub-agent.

Run with: python -m pytest tests/test_visualization_agent.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.base_agent import BaseSubAgent
from agent.visualization_agent import (
    VisualizationAgent, VIZ_TOOL_CATEGORIES, VIZ_EXTRA_TOOLS,
    _extract_labels_from_instruction,
)


class TestVizAgentToolFiltering:
    """Test that VisualizationAgent gets the right tools."""

    def test_viz_tools_include_render_plotly_json(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "render_plotly_json" in names

    def test_viz_tools_include_manage_plot(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "manage_plot" in names

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
        assert len(tools) == 3  # render_plotly_json + manage_plot + list_fetched_data

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


class TestExtractLabels:
    """Test _extract_labels_from_instruction helper."""

    def test_extracts_labels_from_store_format(self):
        instruction = (
            "Plot ACE and Wind magnetic field\n\n"
            "Data currently in memory:\n"
            "  - AC_H0_MFI.Magnitude (37800 pts)\n"
            "  - WI_H0_MFI@0.BF1 (10080 pts)"
        )
        labels = _extract_labels_from_instruction(instruction)
        assert labels == ["AC_H0_MFI.Magnitude", "WI_H0_MFI@0.BF1"]

    def test_extracts_multiple_labels(self):
        instruction = (
            "Task instruction\n\n"
            "Data currently in memory:\n"
            "  - A (100 pts)\n"
            "  - B (200 pts)\n"
            "  - C (300 pts)\n"
            "  - D (400 pts)"
        )
        labels = _extract_labels_from_instruction(instruction)
        assert labels == ["A", "B", "C", "D"]

    def test_returns_empty_for_no_labels(self):
        instruction = "Just plot something"
        labels = _extract_labels_from_instruction(instruction)
        assert labels == []

    def test_handles_dots_and_at_signs(self):
        instruction = "  - WI_H0_MFI@0.BGSE (10080 pts)"
        labels = _extract_labels_from_instruction(instruction)
        assert labels == ["WI_H0_MFI@0.BGSE"]


class TestForceToolCallAttribute:
    """Verify _force_tool_call_in_tasks is disabled for VisualizationAgent."""

    def test_viz_agent_disables_forced_tool_call(self):
        """VisualizationAgent must not force tool calls — render_plotly_json
        requires complex JSON that the LLM emits as empty {} under forced mode."""
        assert VisualizationAgent._force_tool_call_in_tasks is False

    def test_base_agent_default_is_true(self):
        """Other sub-agents should still default to forced tool calling."""
        assert BaseSubAgent._force_tool_call_in_tasks is True


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


class TestExecuteTaskOverride:
    """Verify VisualizationAgent overrides execute_task with think→execute."""

    def test_execute_task_override_exists(self):
        """VisualizationAgent must define its own execute_task, not just inherit."""
        assert "execute_task" in VisualizationAgent.__dict__

    def test_needs_think_for_plot_task(self):
        """Plot-creation instructions should trigger the think phase."""
        agent_cls = VisualizationAgent
        # _needs_think_phase is an instance method but only uses self for nothing
        # — test via the unbound check on a known plot instruction
        assert agent_cls._needs_think_phase(None, "Plot ACE magnetic field data")
        assert agent_cls._needs_think_phase(None, "Show me the solar wind speed")
        assert agent_cls._needs_think_phase(None, "Create a spectrogram of PSP data")
        assert agent_cls._needs_think_phase(None, "Display OMNI proton density")
        assert agent_cls._needs_think_phase(None, "Compare Wind and ACE Bz")

    def test_skips_think_for_style_task(self):
        """Style/manage instructions should skip the think phase."""
        agent_cls = VisualizationAgent
        assert not agent_cls._needs_think_phase(None, "Change the title to 'Solar Wind'")
        assert not agent_cls._needs_think_phase(None, "Zoom in to January 10-15")
        assert not agent_cls._needs_think_phase(None, "Export as PNG")
        assert not agent_cls._needs_think_phase(None, "Switch to log scale")
        assert not agent_cls._needs_think_phase(None, "Add a legend")
