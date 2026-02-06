"""
Tests for agent.planner — complexity detection and plan formatting.

Run with: python -m pytest tests/test_planner.py
"""

import pytest

from agent.planner import is_complex_request, format_plan_for_display
from agent.tasks import Task, TaskPlan, TaskStatus, PlanStatus, create_task, create_plan
from datetime import datetime


class TestIsComplexRequest:
    """Test the complexity detection heuristics."""

    # --- Requests that SHOULD be detected as complex ---

    def test_multiple_and_conjunctions(self):
        assert is_complex_request("Plot PSP and ACE data and compute averages")
        assert is_complex_request("Fetch data and compute magnitude and show the result")

    def test_then_sequential(self):
        assert is_complex_request("Fetch the data, then plot it")
        assert is_complex_request("Then show me the results")

    def test_after_sequential(self):
        assert is_complex_request("After fetching, compute the magnitude")

    def test_first_then_sequential(self):
        assert is_complex_request("First get PSP data, then compare with ACE")

    def test_finally_sequential(self):
        assert is_complex_request("Fetch, compute, and finally plot the data")

    def test_compare_keyword(self):
        assert is_complex_request("Compare PSP and ACE magnetic fields")
        assert is_complex_request("I want to compare the data from two sources")

    def test_difference_between(self):
        assert is_complex_request("Show the difference between PSP and ACE data")

    def test_vs_keyword(self):
        assert is_complex_request("PSP vs ACE magnetic field")
        assert is_complex_request("PSP vs. Solar Orbiter")

    def test_multiple_plot_operations(self):
        assert is_complex_request("Plot PSP and also plot ACE data")
        assert is_complex_request("Show PSP data and show ACE data")

    def test_fetch_and_compute(self):
        assert is_complex_request("Fetch the magnetic field and compute its magnitude")
        assert is_complex_request("Get the data and calculate the average")

    def test_compute_and_plot(self):
        assert is_complex_request("Compute the magnitude and plot it")
        assert is_complex_request("Calculate the derivative and show the result")

    def test_multiple_spacecraft(self):
        assert is_complex_request("Show Parker and ACE magnetic field data")
        assert is_complex_request("Get PSP and OMNI solar wind")
        assert is_complex_request("Compare ACE and Solo magnetic field")

    def test_smooth_and_plot(self):
        assert is_complex_request("Smooth the data and plot it")

    def test_average_and_compare(self):
        assert is_complex_request("Average the data and compare with OMNI")
        assert is_complex_request("Calculate running average and plot both")

    def test_magnitude_and_something(self):
        assert is_complex_request("Compute magnitude and then plot")
        assert is_complex_request("Get the magnitude and compare")

    # --- Requests that should NOT be detected as complex ---

    def test_simple_show_data(self):
        assert not is_complex_request("Show me ACE data")
        assert not is_complex_request("Show Parker magnetic field")

    def test_simple_plot(self):
        assert not is_complex_request("Plot ACE magnetic field for last week")
        assert not is_complex_request("Plot the solar wind data")

    def test_simple_query(self):
        assert not is_complex_request("What data is available for PSP?")
        assert not is_complex_request("List parameters for AC_H2_MFI")

    def test_simple_time_change(self):
        assert not is_complex_request("Zoom in to last 3 days")
        assert not is_complex_request("Change time range to January 2024")

    def test_simple_export(self):
        assert not is_complex_request("Export this as my_plot.png")
        assert not is_complex_request("Save the plot")

    def test_single_spacecraft_data(self):
        assert not is_complex_request("Show me PSP magnetic field for last month")
        assert not is_complex_request("Get ACE solar wind velocity")

    def test_help_questions(self):
        assert not is_complex_request("How do I plot data?")
        assert not is_complex_request("What spacecraft do you support?")


class TestFormatPlanForDisplay:
    """Test the plan formatting function."""

    def test_format_pending_plan(self):
        tasks = [
            create_task("Fetch PSP data", "Use fetch_data..."),
            create_task("Compute magnitude", "Use compute_magnitude..."),
            create_task("Plot result", "Use plot_computed_data..."),
        ]
        plan = create_plan("Test request", tasks)

        output = format_plan_for_display(plan)
        assert "Plan: 3 steps" in output
        assert "Fetch PSP data" in output
        assert "Compute magnitude" in output
        assert "Plot result" in output
        assert "0/3 completed" in output
        # All tasks pending, should show ASCII 'o' (Windows-compatible)
        assert "[o]" in output

    def test_format_in_progress_plan(self):
        tasks = [
            Task(id="1", description="Step 1", instruction="I1", status=TaskStatus.COMPLETED),
            Task(id="2", description="Step 2", instruction="I2", status=TaskStatus.IN_PROGRESS),
            Task(id="3", description="Step 3", instruction="I3", status=TaskStatus.PENDING),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
            status=PlanStatus.EXECUTING,
        )

        output = format_plan_for_display(plan)
        assert "[+]" in output  # Completed
        assert "[*]" in output  # In progress
        assert "[o]" in output  # Pending
        assert "1/3 completed" in output

    def test_format_failed_task(self):
        tasks = [
            Task(
                id="1",
                description="Fetch data",
                instruction="I1",
                status=TaskStatus.FAILED,
                error="Network timeout"
            ),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "[x]" in output  # Failed
        assert "Network timeout" in output
        assert "1 failed" in output

    def test_format_completed_plan(self):
        tasks = [
            Task(id="1", description="Step 1", instruction="I1", status=TaskStatus.COMPLETED),
            Task(id="2", description="Step 2", instruction="I2", status=TaskStatus.COMPLETED),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
            status=PlanStatus.COMPLETED,
        )

        output = format_plan_for_display(plan)
        assert "2/2 completed" in output
        assert "failed" not in output

    def test_format_skipped_task(self):
        tasks = [
            Task(id="1", description="Step 1", instruction="I1", status=TaskStatus.COMPLETED),
            Task(id="2", description="Step 2", instruction="I2", status=TaskStatus.SKIPPED),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "[-]" in output  # Skipped
