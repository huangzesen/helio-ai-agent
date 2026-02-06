"""
Tests for the always-delegate routing logic in core.py.

Tests _is_general_request() heuristics and mission detection routing
without requiring a Gemini API key.

Run with: python -m pytest tests/test_routing.py -v
"""

import pytest
from agent.core import AutoplotAgent


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


class TestMissionAgentImportAndInterface:
    """Verify the new process_request method exists on MissionAgent."""

    def test_process_request_method_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "process_request")
        assert callable(getattr(MissionAgent, "process_request"))

    def test_execute_task_still_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "execute_task")
