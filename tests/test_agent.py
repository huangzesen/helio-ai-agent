"""
Tests for the agent core logic.

Run with: python -m pytest tests/test_agent.py

Note: Most tests mock external dependencies to avoid network calls and JVM startup.
"""

import pytest
from unittest.mock import patch, MagicMock
from agent.time_utils import parse_time_range, TimeRange, TimeRangeError
from agent.prompts import format_tool_result
from datetime import datetime, timedelta, timezone


class TestParseTimeRange:
    def test_last_week(self):
        result = parse_time_range("last week")
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        assert result.end == today
        assert result.start == today - timedelta(days=7)

    def test_last_month(self):
        result = parse_time_range("last month")
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        assert result.end == today
        assert result.start == today - timedelta(days=30)

    def test_last_n_days(self):
        result = parse_time_range("last 3 days")
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        assert result.end == today
        assert result.start == today - timedelta(days=3)

    def test_month_year(self):
        result = parse_time_range("January 2024")
        assert result.to_time_range_string() == "2024-01-01 to 2024-02-01"

    def test_december_year(self):
        result = parse_time_range("December 2024")
        assert result.to_time_range_string() == "2024-12-01 to 2025-01-01"

    def test_single_date(self):
        result = parse_time_range("2024-06-15")
        assert result.to_time_range_string() == "2024-06-15 to 2024-06-16"

    def test_already_formatted(self):
        result = parse_time_range("2024-01-01 to 2024-01-31")
        assert result.to_time_range_string() == "2024-01-01 to 2024-01-31"

    def test_abbreviated_month(self):
        result = parse_time_range("Jan 2024")
        assert result.to_time_range_string() == "2024-01-01 to 2024-02-01"

    def test_datetime_range(self):
        result = parse_time_range("2024-01-15T06:00 to 2024-01-15T18:00")
        assert result.start == datetime(2024, 1, 15, 6, 0, tzinfo=timezone.utc)
        assert result.end == datetime(2024, 1, 15, 18, 0, tzinfo=timezone.utc)
        assert "T" in result.to_time_range_string()

    def test_datetime_range_with_seconds(self):
        result = parse_time_range("2024-01-15T06:00:30 to 2024-01-15T18:30:45")
        assert result.start == datetime(2024, 1, 15, 6, 0, 30, tzinfo=timezone.utc)
        assert result.end == datetime(2024, 1, 15, 18, 30, 45, tzinfo=timezone.utc)

    def test_unparseable_raises_error(self):
        with pytest.raises(TimeRangeError, match="Could not parse"):
            parse_time_range("gobbledygook")

    def test_bad_date_format_raises_error(self):
        with pytest.raises(TimeRangeError, match="Could not parse"):
            parse_time_range("15/01/2024")

    def test_start_after_end_raises_error(self):
        with pytest.raises(ValueError, match="must be before"):
            parse_time_range("2024-01-20 to 2024-01-15")

    def test_day_precision_omits_time(self):
        result = parse_time_range("2024-01-15 to 2024-01-20")
        s = result.to_time_range_string()
        assert "T" not in s
        assert s == "2024-01-15 to 2024-01-20"

    def test_sub_day_includes_time(self):
        result = parse_time_range("2024-01-15T06:00 to 2024-01-15T18:00")
        s = result.to_time_range_string()
        assert "T" in s
        assert s == "2024-01-15T06:00:00 to 2024-01-15T18:00:00"


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
    def mock_renderer(self):
        """Mock renderer commands."""
        with patch("agent.core.get_commands") as mock:
            mock_commands = MagicMock()
            mock.return_value = mock_commands
            yield mock_commands

    def test_search_datasets_tool(self):
        """Test that search_datasets calls catalog.search_by_keywords."""
        from agent.core import OrchestratorAgent

        with patch("agent.core.genai"):
            with patch("agent.core.search_by_keywords") as mock_search:
                mock_search.return_value = {
                    "spacecraft": "PSP",
                    "spacecraft_name": "Parker Solar Probe",
                    "instrument": "FIELDS/MAG",
                    "instrument_name": "FIELDS Magnetometer",
                    "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"],
                }

                agent = OrchestratorAgent.__new__(OrchestratorAgent)
                agent.verbose = False
                agent._renderer = None

                result = agent._execute_tool("search_datasets", {"query": "parker magnetic"})

                mock_search.assert_called_once_with("parker magnetic")
                assert result["spacecraft"] == "PSP"

    def test_list_parameters_tool(self):
        """Test that list_parameters calls HAPI client."""
        from agent.core import OrchestratorAgent

        with patch("agent.core.genai"):
            with patch("agent.core.hapi_list_parameters") as mock_list:
                mock_list.return_value = [
                    {"name": "Magnitude", "units": "nT", "size": [1], "description": "", "dataset_id": "AC_H2_MFI"},
                ]

                agent = OrchestratorAgent.__new__(OrchestratorAgent)
                agent.verbose = False
                agent._renderer = None

                result = agent._execute_tool("list_parameters", {"dataset_id": "AC_H2_MFI"})

                mock_list.assert_called_once_with("AC_H2_MFI")
                assert len(result["parameters"]) == 1

    def test_ask_clarification_tool(self):
        """Test that ask_clarification returns question data."""
        from agent.core import OrchestratorAgent

        with patch("agent.core.genai"):
            agent = OrchestratorAgent.__new__(OrchestratorAgent)
            agent.verbose = False
            agent._renderer = None

            result = agent._execute_tool("ask_clarification", {
                "question": "Which parameter?",
                "options": ["Magnitude", "Vector"],
                "context": "Multiple parameters available",
            })

            assert result["status"] == "clarification_needed"
            assert result["question"] == "Which parameter?"
            assert result["options"] == ["Magnitude", "Vector"]


class TestValidateTimeRange:
    """Test _validate_time_range auto-clamping logic."""

    @pytest.fixture
    def agent(self):
        """Create a minimal OrchestratorAgent instance for testing."""
        from agent.core import OrchestratorAgent
        with patch("agent.core.genai"):
            a = OrchestratorAgent.__new__(OrchestratorAgent)
            a.verbose = False
            a._renderer = None
            return a

    def _dt(self, s):
        """Shorthand for UTC datetime."""
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

    def test_fully_within_range_returns_none(self, agent):
        with patch("agent.core.get_dataset_time_range") as mock:
            mock.return_value = {"start": "2020-01-01", "stop": "2025-01-01"}
            result = agent._validate_time_range(
                "TEST", self._dt("2024-01-01"), self._dt("2024-02-01")
            )
            assert result is None

    def test_hapi_failure_returns_none(self, agent):
        with patch("agent.core.get_dataset_time_range") as mock:
            mock.return_value = None
            result = agent._validate_time_range(
                "TEST", self._dt("2024-01-01"), self._dt("2024-02-01")
            )
            assert result is None

    def test_request_after_stop_shifts_window(self, agent):
        """'Last week' when dataset ends months ago → shifts to last available week."""
        with patch("agent.core.get_dataset_time_range") as mock:
            mock.return_value = {"start": "2020-01-01", "stop": "2025-06-15"}
            result = agent._validate_time_range(
                "TEST", self._dt("2026-01-01"), self._dt("2026-01-08")
            )
            assert result is not None
            assert result["end"] == self._dt("2025-06-15")
            # Duration preserved (7 days)
            assert result["start"] == self._dt("2025-06-08")
            assert "after" in result["note"]
            assert "Auto-adjusted" in result["note"]

    def test_request_before_start_shifts_window(self, agent):
        """Request for 1990 when dataset starts in 2018 → shifts to first available period."""
        with patch("agent.core.get_dataset_time_range") as mock:
            mock.return_value = {"start": "2018-08-12", "stop": "2025-06-15"}
            result = agent._validate_time_range(
                "TEST", self._dt("1990-01-01"), self._dt("1990-02-01")
            )
            assert result is not None
            assert result["start"] == self._dt("2018-08-12")
            # Duration preserved (31 days)
            expected_end = self._dt("2018-08-12") + timedelta(days=31)
            assert result["end"] == expected_end
            assert "before" in result["note"]

    def test_partial_overlap_clamps_start(self, agent):
        """Request starts before available → clamps start."""
        with patch("agent.core.get_dataset_time_range") as mock:
            mock.return_value = {"start": "2020-01-01", "stop": "2025-06-15"}
            result = agent._validate_time_range(
                "TEST", self._dt("2019-06-01"), self._dt("2020-06-01")
            )
            assert result is not None
            assert result["start"] == self._dt("2020-01-01")
            assert result["end"] == self._dt("2020-06-01")
            assert "Clamped" in result["note"]

    def test_partial_overlap_clamps_end(self, agent):
        """Request ends after available → clamps end."""
        with patch("agent.core.get_dataset_time_range") as mock:
            mock.return_value = {"start": "2020-01-01", "stop": "2025-06-15"}
            result = agent._validate_time_range(
                "TEST", self._dt("2025-05-01"), self._dt("2026-01-01")
            )
            assert result is not None
            assert result["start"] == self._dt("2025-05-01")
            assert result["end"] == self._dt("2025-06-15")
            assert "Clamped" in result["note"]

    def test_shift_preserves_duration_capped_at_available(self, agent):
        """If requested duration exceeds available range, cap to full available range."""
        with patch("agent.core.get_dataset_time_range") as mock:
            # Dataset covers only 6 months, but requesting 2 years after stop
            mock.return_value = {"start": "2025-01-01", "stop": "2025-06-15"}
            result = agent._validate_time_range(
                "TEST", self._dt("2026-01-01"), self._dt("2028-01-01")
            )
            assert result is not None
            # Should cap start at avail_start
            assert result["start"] == self._dt("2025-01-01")
            assert result["end"] == self._dt("2025-06-15")
