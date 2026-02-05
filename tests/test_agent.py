"""
Tests for the agent core logic.

Run with: python -m pytest tests/test_agent.py

Note: Most tests mock external dependencies to avoid network calls and JVM startup.
"""

import pytest
from unittest.mock import patch, MagicMock
from agent.prompts import parse_relative_time, format_tool_result
from datetime import datetime, timedelta


class TestParseRelativeTime:
    def test_last_week(self):
        result = parse_relative_time("last week")
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        assert result == f"{week_ago} to {today}"

    def test_last_month(self):
        result = parse_relative_time("last month")
        today = datetime.now().date()
        month_ago = today - timedelta(days=30)
        assert result == f"{month_ago} to {today}"

    def test_last_n_days(self):
        result = parse_relative_time("last 3 days")
        today = datetime.now().date()
        three_days_ago = today - timedelta(days=3)
        assert result == f"{three_days_ago} to {today}"

    def test_month_year(self):
        result = parse_relative_time("January 2024")
        assert result == "2024-01-01 to 2024-02-01"

    def test_december_year(self):
        result = parse_relative_time("December 2024")
        assert result == "2024-12-01 to 2025-01-01"

    def test_single_date(self):
        result = parse_relative_time("2024-06-15")
        assert result == "2024-06-15 to 2024-06-16"

    def test_already_formatted(self):
        result = parse_relative_time("2024-01-01 to 2024-01-31")
        assert result == "2024-01-01 to 2024-01-31"

    def test_abbreviated_month(self):
        result = parse_relative_time("Jan 2024")
        assert result == "2024-01-01 to 2024-02-01"


class TestFormatToolResult:
    def test_format_error(self):
        result = format_tool_result("any_tool", {"status": "error", "message": "Something failed"})
        assert "Error" in result
        assert "Something failed" in result

    def test_format_search_result(self):
        result = format_tool_result("search_datasets", {
            "status": "success",
            "spacecraft": "PSP",
            "spacecraft_name": "Parker Solar Probe",
            "instrument": "FIELDS/MAG",
            "instrument_name": "FIELDS Magnetometer",
            "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"],
        })
        assert "Parker Solar Probe" in result
        assert "FIELDS/MAG" in result

    def test_format_parameters_result(self):
        result = format_tool_result("list_parameters", {
            "status": "success",
            "parameters": [
                {"name": "Magnitude", "units": "nT", "size": [1], "description": "Magnetic field magnitude"},
                {"name": "Vector", "units": "nT", "size": [3], "description": "B field vector"},
            ]
        })
        assert "Magnitude" in result
        assert "nT" in result
        assert "vector:3" in result

    def test_format_plot_result(self):
        result = format_tool_result("plot_data", {
            "status": "success",
            "dataset_id": "AC_H2_MFI",
            "parameter_id": "Magnitude",
            "time_range": "2024-01-01 to 2024-01-07",
        })
        assert "AC_H2_MFI" in result
        assert "Magnitude" in result

    def test_format_time_range_change(self):
        result = format_tool_result("change_time_range", {
            "status": "success",
            "time_range": "2024-01-15 to 2024-01-20",
        })
        assert "2024-01-15 to 2024-01-20" in result

    def test_format_export(self):
        result = format_tool_result("export_plot", {
            "status": "success",
            "filepath": "/path/to/output.png",
        })
        assert "/path/to/output.png" in result

    def test_format_plot_info_no_plot(self):
        result = format_tool_result("get_plot_info", {
            "uri": None,
            "time_range": None,
        })
        assert "No plot" in result

    def test_format_plot_info_with_plot(self):
        result = format_tool_result("get_plot_info", {
            "uri": "vap+cdaweb:ds=AC_H2_MFI&id=Magnitude",
            "time_range": "2024-01-01 to 2024-01-07",
        })
        assert "AC_H2_MFI" in result
        assert "2024-01-01 to 2024-01-07" in result


class TestAgentToolExecution:
    """Tests for agent tool execution logic (mocked)."""

    @pytest.fixture
    def mock_genai(self):
        """Mock google.generativeai module."""
        with patch("agent.core.genai") as mock:
            yield mock

    @pytest.fixture
    def mock_autoplot(self):
        """Mock autoplot commands."""
        with patch("agent.core.get_commands") as mock:
            mock_commands = MagicMock()
            mock.return_value = mock_commands
            yield mock_commands

    def test_search_datasets_tool(self):
        """Test that search_datasets calls catalog.search_by_keywords."""
        from agent.core import AutoplotAgent

        with patch("agent.core.genai"):
            with patch("agent.core.search_by_keywords") as mock_search:
                mock_search.return_value = {
                    "spacecraft": "PSP",
                    "spacecraft_name": "Parker Solar Probe",
                    "instrument": "FIELDS/MAG",
                    "instrument_name": "FIELDS Magnetometer",
                    "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"],
                }

                agent = AutoplotAgent.__new__(AutoplotAgent)
                agent.verbose = False
                agent._autoplot = None

                result = agent._execute_tool("search_datasets", {"query": "parker magnetic"})

                mock_search.assert_called_once_with("parker magnetic")
                assert result["spacecraft"] == "PSP"

    def test_list_parameters_tool(self):
        """Test that list_parameters calls HAPI client."""
        from agent.core import AutoplotAgent

        with patch("agent.core.genai"):
            with patch("agent.core.hapi_list_parameters") as mock_list:
                mock_list.return_value = [
                    {"name": "Magnitude", "units": "nT", "size": [1], "description": "", "dataset_id": "AC_H2_MFI"},
                ]

                agent = AutoplotAgent.__new__(AutoplotAgent)
                agent.verbose = False
                agent._autoplot = None

                result = agent._execute_tool("list_parameters", {"dataset_id": "AC_H2_MFI"})

                mock_list.assert_called_once_with("AC_H2_MFI")
                assert len(result["parameters"]) == 1

    def test_ask_clarification_tool(self):
        """Test that ask_clarification returns question data."""
        from agent.core import AutoplotAgent

        with patch("agent.core.genai"):
            agent = AutoplotAgent.__new__(AutoplotAgent)
            agent.verbose = False
            agent._autoplot = None

            result = agent._execute_tool("ask_clarification", {
                "question": "Which parameter?",
                "options": ["Magnitude", "Vector"],
                "context": "Multiple parameters available",
            })

            assert result["status"] == "clarification_needed"
            assert result["question"] == "Which parameter?"
            assert result["options"] == ["Magnitude", "Vector"]
