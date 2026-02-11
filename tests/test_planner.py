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
    PLANNER_TOOL_CATEGORIES,
    PLANNER_EXTRA_TOOLS,
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
        assert agent.tool_executor is None
        assert agent._function_declarations == []

    def test_get_token_usage_initial(self):
        """Token usage starts at zero."""
        agent = PlannerAgent(client=None, model_name="test-model")
        usage = agent.get_token_usage()
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["thinking_tokens"] == 0

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


class TestPlannerAgentWithTools:
    """Test PlannerAgent tool integration (no API calls needed)."""

    def _dummy_executor(self, tool_name, tool_args):
        return {"status": "success", "message": "mock"}

    def test_init_with_tool_executor_has_declarations(self):
        """When tool_executor is provided, function declarations should be built."""
        agent = PlannerAgent(
            client=None,
            model_name="test-model",
            tool_executor=self._dummy_executor,
        )
        assert agent.tool_executor is not None
        assert len(agent._function_declarations) > 0

    def test_init_without_tool_executor_no_declarations(self):
        """When tool_executor is None, no function declarations."""
        agent = PlannerAgent(client=None, model_name="test-model")
        assert agent.tool_executor is None
        assert agent._function_declarations == []

    def test_tool_names_include_discovery(self):
        """Function declarations should include discovery tools."""
        agent = PlannerAgent(
            client=None,
            model_name="test-model",
            tool_executor=self._dummy_executor,
        )
        tool_names = {fd.name for fd in agent._function_declarations}
        # Discovery tools
        assert "search_datasets" in tool_names
        assert "list_parameters" in tool_names
        assert "get_data_availability" in tool_names
        assert "browse_datasets" in tool_names
        assert "get_dataset_docs" in tool_names
        assert "search_full_catalog" in tool_names
        # Extra tool
        assert "list_fetched_data" in tool_names

    def test_tool_names_exclude_routing_and_visualization(self):
        """Function declarations should NOT include routing or visualization tools."""
        agent = PlannerAgent(
            client=None,
            model_name="test-model",
            tool_executor=self._dummy_executor,
        )
        tool_names = {fd.name for fd in agent._function_declarations}
        # Routing tools should NOT be present
        assert "delegate_to_mission" not in tool_names
        assert "delegate_to_visualization" not in tool_names
        assert "delegate_to_data_ops" not in tool_names
        assert "request_planning" not in tool_names
        # Visualization tools should NOT be present
        assert "plot_data" not in tool_names
        assert "style_plot" not in tool_names
        assert "manage_plot" not in tool_names
        # Fetch / compute tools should NOT be present
        assert "fetch_data" not in tool_names
        assert "custom_operation" not in tool_names

    def test_planner_tool_categories_constant(self):
        """PLANNER_TOOL_CATEGORIES should be discovery only."""
        assert PLANNER_TOOL_CATEGORIES == ["discovery"]

    def test_planner_extra_tools_constant(self):
        """PLANNER_EXTRA_TOOLS should include list_fetched_data."""
        assert "list_fetched_data" in PLANNER_EXTRA_TOOLS

    def test_has_run_discovery_method(self):
        """PlannerAgent should have a _run_discovery method for two-phase planning."""
        assert hasattr(PlannerAgent, "_run_discovery")
        agent = PlannerAgent(
            client=None,
            model_name="test-model",
            tool_executor=self._dummy_executor,
        )
        assert callable(agent._run_discovery)


class TestBuildParameterReference:
    """Test the structured parameter reference builder."""

    def test_empty_when_no_list_parameters(self):
        """Returns empty string when no list_parameters results."""
        assert PlannerAgent._build_parameter_reference({}) == ""
        assert PlannerAgent._build_parameter_reference({"search_datasets": []}) == ""

    def test_basic_parameter_reference(self):
        """Builds a reference from list_parameters results."""
        tool_results = {
            "list_parameters": [
                {
                    "args": {"dataset_id": "AC_H2_MFI"},
                    "result": {
                        "status": "success",
                        "parameters": [
                            {"name": "Time", "type": "isotime", "units": "UTC"},
                            {"name": "BGSEc", "type": "double", "units": "nT", "size": [3]},
                            {"name": "Magnitude", "type": "double", "units": "nT"},
                        ],
                    },
                },
            ],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "## DATASET REFERENCE" in ref
        assert "AC_H2_MFI" in ref
        assert "BGSEc" in ref
        assert "Magnitude" in ref
        # Time should be excluded
        assert "\n  - Time" not in ref

    def test_includes_availability(self):
        """Includes data availability when get_data_availability results exist."""
        tool_results = {
            "list_parameters": [
                {
                    "args": {"dataset_id": "AC_H2_MFI"},
                    "result": {
                        "status": "success",
                        "parameters": [
                            {"name": "BGSEc", "type": "double", "units": "nT"},
                        ],
                    },
                },
            ],
            "get_data_availability": [
                {
                    "args": {"dataset_id": "AC_H2_MFI"},
                    "result": {
                        "status": "success",
                        "start_date": "1998-01-01",
                        "end_date": "2025-12-31",
                    },
                },
            ],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "1998-01-01 to 2025-12-31" in ref

    def test_skips_datasets_with_no_parameters(self):
        """Datasets with 0 parameters are marked as unavailable."""
        tool_results = {
            "list_parameters": [
                {
                    "args": {"dataset_id": "VG1_PWS_LR"},
                    "result": {
                        "status": "success",
                        "parameters": [],
                    },
                },
            ],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "NO PARAMETERS AVAILABLE" in ref
        assert "VG1_PWS_LR" in ref

    def test_multiple_datasets(self):
        """Handles multiple datasets in one reference."""
        tool_results = {
            "list_parameters": [
                {
                    "args": {"dataset_id": "DS_A"},
                    "result": {
                        "status": "success",
                        "parameters": [
                            {"name": "ParamA", "type": "double", "units": "nT"},
                        ],
                    },
                },
                {
                    "args": {"dataset_id": "DS_B"},
                    "result": {
                        "status": "success",
                        "parameters": [
                            {"name": "ParamB", "type": "double", "units": "cm/s"},
                        ],
                    },
                },
            ],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "DS_A" in ref
        assert "ParamA" in ref
        assert "DS_B" in ref
        assert "ParamB" in ref

    def test_only_instruction_present(self):
        """The reference includes the instruction to use only listed dataset IDs."""
        tool_results = {
            "list_parameters": [
                {
                    "args": {"dataset_id": "TEST"},
                    "result": {
                        "status": "success",
                        "parameters": [{"name": "X", "type": "double"}],
                    },
                },
            ],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "Use ONLY" in ref
        assert "candidate_datasets" in ref
