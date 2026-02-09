"""
Tests for agent.planner — complexity detection, PlannerAgent, and plan formatting.

Run with: python -m pytest tests/test_planner.py
"""

import pytest

from agent.planner import (
    is_complex_request,
    format_plan_for_display,
    PlannerAgent,
    PLANNER_RESPONSE_SCHEMA,
    MAX_ROUNDS,
)
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


class TestPlannerAgentInterface:
    """Test PlannerAgent class structure and interface."""

    def test_has_required_methods(self):
        """PlannerAgent should have the expected public methods."""
        assert hasattr(PlannerAgent, "start_planning")
        assert hasattr(PlannerAgent, "continue_planning")
        assert hasattr(PlannerAgent, "get_token_usage")
        assert hasattr(PlannerAgent, "reset")

    def test_init_no_chat(self):
        """PlannerAgent starts with no active chat."""
        # Use None client — we won't make API calls
        agent = PlannerAgent(client=None, model_name="test-model")
        assert agent._chat is None
        assert agent.model_name == "test-model"
        assert agent.verbose is False

    def test_get_token_usage_initial(self):
        """Token usage starts at zero."""
        agent = PlannerAgent(client=None, model_name="test-model")
        usage = agent.get_token_usage()
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

    def test_reset_clears_chat(self):
        """Reset should clear the chat session."""
        agent = PlannerAgent(client=None, model_name="test-model")
        agent._chat = "something"
        agent.reset()
        assert agent._chat is None

    def test_continue_planning_without_chat_returns_none(self):
        """continue_planning should return None if no chat session exists."""
        agent = PlannerAgent(client=None, model_name="test-model")
        result = agent.continue_planning([{"description": "test", "status": "completed"}])
        assert result is None

    def test_max_rounds_constant(self):
        """MAX_ROUNDS should be 5."""
        assert MAX_ROUNDS == 5


class TestPlannerResponseSchema:
    """Test the PLANNER_RESPONSE_SCHEMA structure."""

    def test_schema_is_object(self):
        assert PLANNER_RESPONSE_SCHEMA["type"] == "object"

    def test_required_fields(self):
        required = PLANNER_RESPONSE_SCHEMA["required"]
        assert "status" in required
        assert "reasoning" in required
        assert "tasks" in required

    def test_status_enum(self):
        status_schema = PLANNER_RESPONSE_SCHEMA["properties"]["status"]
        assert status_schema["type"] == "string"
        assert set(status_schema["enum"]) == {"continue", "done"}

    def test_tasks_is_array(self):
        tasks_schema = PLANNER_RESPONSE_SCHEMA["properties"]["tasks"]
        assert tasks_schema["type"] == "array"

    def test_task_item_required_fields(self):
        task_schema = PLANNER_RESPONSE_SCHEMA["properties"]["tasks"]["items"]
        assert "description" in task_schema["required"]
        assert "instruction" in task_schema["required"]

    def test_task_item_has_mission(self):
        task_schema = PLANNER_RESPONSE_SCHEMA["properties"]["tasks"]["items"]
        assert "mission" in task_schema["properties"]

    def test_summary_field_exists(self):
        assert "summary" in PLANNER_RESPONSE_SCHEMA["properties"]
        assert PLANNER_RESPONSE_SCHEMA["properties"]["summary"]["type"] == "string"


class TestTaskRoundField:
    """Test the round field on Task dataclass."""

    def test_default_round_is_zero(self):
        task = create_task("Test", "instruction")
        assert task.round == 0

    def test_round_in_to_dict(self):
        task = create_task("Test", "instruction")
        task.round = 2
        d = task.to_dict()
        assert d["round"] == 2

    def test_round_from_dict(self):
        d = {
            "id": "test-id",
            "description": "Test",
            "instruction": "Do something",
            "status": "pending",
            "round": 3,
        }
        task = Task.from_dict(d)
        assert task.round == 3

    def test_round_from_dict_missing(self):
        """Missing round field should default to 0."""
        d = {
            "id": "test-id",
            "description": "Test",
            "instruction": "Do something",
            "status": "pending",
        }
        task = Task.from_dict(d)
        assert task.round == 0


class TestTaskPlanAddTasks:
    """Test the add_tasks() method on TaskPlan."""

    def test_add_tasks_appends(self):
        plan = create_plan("Test", [])
        assert len(plan.tasks) == 0

        tasks = [create_task("A", "a"), create_task("B", "b")]
        plan.add_tasks(tasks)
        assert len(plan.tasks) == 2

    def test_add_tasks_incremental(self):
        t1 = create_task("A", "a")
        plan = create_plan("Test", [t1])
        assert len(plan.tasks) == 1

        t2 = create_task("B", "b")
        t3 = create_task("C", "c")
        plan.add_tasks([t2, t3])
        assert len(plan.tasks) == 3
        assert plan.tasks[0].description == "A"
        assert plan.tasks[1].description == "B"
        assert plan.tasks[2].description == "C"


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

    def test_format_mission_tagged_task(self):
        tasks = [
            Task(id="1", description="Fetch PSP data", instruction="I1",
                 mission="PSP", status=TaskStatus.PENDING),
            Task(id="2", description="Fetch ACE data", instruction="I2",
                 mission="ACE", status=TaskStatus.COMPLETED),
            Task(id="3", description="Compare", instruction="I3",
                 status=TaskStatus.PENDING),  # No mission
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Compare PSP and ACE",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "[PSP]" in output
        assert "[ACE]" in output
        # Task 3 has no mission tag
        lines = output.split("\n")
        compare_line = [l for l in lines if "Compare" in l][0]
        assert "[PSP]" not in compare_line
        assert "[ACE]" not in compare_line

    def test_format_with_rounds(self):
        """Tasks with non-zero rounds should be grouped by round."""
        tasks = [
            Task(id="1", description="Fetch ACE", instruction="I1",
                 mission="ACE", status=TaskStatus.COMPLETED, round=1),
            Task(id="2", description="Fetch Wind", instruction="I2",
                 mission="WIND", status=TaskStatus.COMPLETED, round=1),
            Task(id="3", description="Compute magnitude", instruction="I3",
                 mission="__data_ops__", status=TaskStatus.COMPLETED, round=2),
            Task(id="4", description="Plot comparison", instruction="I4",
                 mission="__visualization__", status=TaskStatus.PENDING, round=3),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Compare ACE and Wind mag",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "Round 1:" in output
        assert "Round 2:" in output
        assert "Round 3:" in output
        assert "Plan: 4 steps" in output
        assert "Fetch ACE" in output
        assert "Plot comparison" in output

    def test_format_without_rounds(self):
        """Tasks with round=0 should display without round headers."""
        tasks = [
            Task(id="1", description="Step A", instruction="I1", status=TaskStatus.PENDING),
            Task(id="2", description="Step B", instruction="I2", status=TaskStatus.PENDING),
        ]
        plan = TaskPlan(
            id="plan",
            user_request="Test",
            tasks=tasks,
            created_at=datetime.now(),
        )

        output = format_plan_for_display(plan)
        assert "Round" not in output
        assert "Step A" in output
        assert "Step B" in output
