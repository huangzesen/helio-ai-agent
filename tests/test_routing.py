"""
Tests for routing, tool filtering, and the delegate tools.

Tests tool category filtering and LLM-driven routing architecture
without requiring a Gemini API key.

Run with: python -m pytest tests/test_routing.py -v
"""

import pytest
from agent.tools import get_tool_schemas
from agent.mission_agent import MISSION_TOOL_CATEGORIES, MISSION_EXTRA_TOOLS
from agent.data_ops_agent import DATAOPS_TOOL_CATEGORIES, DATAOPS_EXTRA_TOOLS
from agent.data_extraction_agent import EXTRACTION_CATEGORIES, EXTRACTION_EXTRA_TOOLS
from agent.visualization_agent import VIZ_TOOL_CATEGORIES, VIZ_EXTRA_TOOLS
from agent.core import ORCHESTRATOR_CATEGORIES, ORCHESTRATOR_EXTRA_TOOLS


class TestToolCategoryFiltering:
    """Test get_tool_schemas() category filtering."""

    def test_no_filter_returns_all_tools(self):
        all_tools = get_tool_schemas()
        assert len(all_tools) == 23  # 22 + request_planning
        names = {t["name"] for t in all_tools}
        assert "execute_visualization" in names
        assert "custom_visualization" in names
        assert "fetch_data" in names
        assert "delegate_to_mission" in names
        assert "delegate_to_visualization" in names
        assert "delegate_to_data_ops" in names
        assert "delegate_to_data_extraction" in names
        assert "get_dataset_docs" in names
        assert "read_document" in names

    def test_mission_categories_exclude_visualization_and_routing(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        # Should not include visualization tools
        assert "execute_visualization" not in names
        # Should not include routing tools (no recursive delegation)
        assert "delegate_to_mission" not in names
        assert "delegate_to_visualization" not in names
        assert "delegate_to_data_ops" not in names
        # Should include fetch + discovery tools
        assert "fetch_data" in names
        assert "search_datasets" in names
        assert "list_fetched_data" in names
        assert "ask_clarification" in names
        # Should NOT include compute tools (moved to DataOpsAgent)
        assert "custom_operation" not in names
        assert "describe_data" not in names
        assert "save_data" not in names
        # Should NOT include document tools
        assert "read_document" not in names

    def test_visualization_category_only(self):
        viz_tools = get_tool_schemas(categories=VIZ_TOOL_CATEGORIES)
        names = {t["name"] for t in viz_tools}
        assert names == {"execute_visualization", "custom_visualization"}

    def test_visualization_with_extras(self):
        tools = get_tool_schemas(
            categories=VIZ_TOOL_CATEGORIES,
            extra_names=VIZ_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert names == {"execute_visualization", "custom_visualization", "list_fetched_data"}

    def test_orchestrator_categories(self):
        orch_tools = get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS)
        names = {t["name"] for t in orch_tools}
        # Should include routing
        assert "delegate_to_mission" in names
        assert "delegate_to_visualization" in names
        assert "delegate_to_data_ops" in names
        assert "delegate_to_data_extraction" in names
        assert "request_planning" in names
        # Should include discovery
        assert "search_datasets" in names
        # Should include list_fetched_data (extra tool)
        assert "list_fetched_data" in names
        # Should include document tools
        assert "read_document" in names
        # Should NOT include data_ops (delegated to sub-agents)
        assert "fetch_data" not in names
        assert "custom_operation" not in names
        # Should NOT include visualization
        assert "execute_visualization" not in names

    def test_dataops_categories(self):
        dataops_tools = get_tool_schemas(
            categories=DATAOPS_TOOL_CATEGORIES,
            extra_names=DATAOPS_EXTRA_TOOLS,
        )
        names = {t["name"] for t in dataops_tools}
        # Should include compute tools
        assert "custom_operation" in names
        assert "describe_data" in names
        assert "save_data" in names
        # Should include list_fetched_data (extra)
        assert "list_fetched_data" in names
        # Should include conversation
        assert "ask_clarification" in names
        # Should NOT include fetch (mission-specific)
        assert "fetch_data" not in names
        # Should NOT include store_dataframe (moved to data_extraction)
        assert "store_dataframe" not in names
        # Should NOT include routing or visualization
        assert "delegate_to_mission" not in names
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


class TestDelegateToDataOpsTool:
    """Test that the delegate_to_data_ops tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_data_ops" in names

    def test_tool_has_routing_category(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_data_ops")
        assert tool["category"] == "routing"

    def test_tool_requires_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_data_ops")
        assert "request" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["request"]

    def test_tool_not_in_mission_agent_tools(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        assert "delegate_to_data_ops" not in names

    def test_tool_not_in_dataops_agent_tools(self):
        dataops_tools = get_tool_schemas(
            categories=DATAOPS_TOOL_CATEGORIES,
            extra_names=DATAOPS_EXTRA_TOOLS,
        )
        names = {t["name"] for t in dataops_tools}
        assert "delegate_to_data_ops" not in names


class TestDataExtractionCategories:
    """Test DataExtractionAgent tool filtering."""

    def test_extraction_agent_gets_store_dataframe(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "store_dataframe" in names

    def test_extraction_agent_gets_read_document(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "read_document" in names

    def test_extraction_agent_gets_list_fetched_data(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "list_fetched_data" in names

    def test_extraction_agent_gets_ask_clarification(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "ask_clarification" in names

    def test_extraction_agent_excludes_fetch(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "fetch_data" not in names

    def test_extraction_agent_excludes_compute(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "custom_operation" not in names
        assert "describe_data" not in names
        assert "save_data" not in names

    def test_extraction_agent_excludes_routing(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "delegate_to_mission" not in names
        assert "delegate_to_visualization" not in names

    def test_extraction_agent_excludes_visualization(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "execute_visualization" not in names
        assert "custom_visualization" not in names


class TestDelegateToDataExtractionTool:
    """Test that the delegate_to_data_extraction tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "delegate_to_data_extraction" in names

    def test_tool_has_routing_category(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_data_extraction")
        assert tool["category"] == "routing"

    def test_tool_requires_request(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_data_extraction")
        assert "request" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["request"]

    def test_tool_has_optional_context(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "delegate_to_data_extraction")
        assert "context" in tool["parameters"]["properties"]

    def test_tool_not_in_extraction_agent_tools(self):
        tools = get_tool_schemas(
            categories=EXTRACTION_CATEGORIES,
            extra_names=EXTRACTION_EXTRA_TOOLS,
        )
        names = {t["name"] for t in tools}
        assert "delegate_to_data_extraction" not in names


class TestDataExtractionAgentImportAndInterface:
    """Verify DataExtractionAgent interface."""

    def test_import(self):
        from agent.data_extraction_agent import DataExtractionAgent
        assert DataExtractionAgent is not None

    def test_process_request_method_exists(self):
        from agent.data_extraction_agent import DataExtractionAgent
        assert hasattr(DataExtractionAgent, "process_request")
        assert callable(getattr(DataExtractionAgent, "process_request"))

    def test_execute_task_method_exists(self):
        from agent.data_extraction_agent import DataExtractionAgent
        assert hasattr(DataExtractionAgent, "execute_task")
        assert callable(getattr(DataExtractionAgent, "execute_task"))

    def test_get_token_usage_method_exists(self):
        from agent.data_extraction_agent import DataExtractionAgent
        assert hasattr(DataExtractionAgent, "get_token_usage")
        assert callable(getattr(DataExtractionAgent, "get_token_usage"))


class TestMissionAgentImportAndInterface:
    """Verify MissionAgent interface."""

    def test_process_request_method_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "process_request")
        assert callable(getattr(MissionAgent, "process_request"))

    def test_execute_task_still_exists(self):
        from agent.mission_agent import MissionAgent
        assert hasattr(MissionAgent, "execute_task")


class TestDataOpsAgentImportAndInterface:
    """Verify DataOpsAgent interface."""

    def test_import(self):
        from agent.data_ops_agent import DataOpsAgent
        assert DataOpsAgent is not None

    def test_process_request_method_exists(self):
        from agent.data_ops_agent import DataOpsAgent
        assert hasattr(DataOpsAgent, "process_request")
        assert callable(getattr(DataOpsAgent, "process_request"))

    def test_execute_task_method_exists(self):
        from agent.data_ops_agent import DataOpsAgent
        assert hasattr(DataOpsAgent, "execute_task")
        assert callable(getattr(DataOpsAgent, "execute_task"))

    def test_get_token_usage_method_exists(self):
        from agent.data_ops_agent import DataOpsAgent
        assert hasattr(DataOpsAgent, "get_token_usage")
        assert callable(getattr(DataOpsAgent, "get_token_usage"))


class TestRequestPlanningTool:
    """Test that the request_planning tool is properly configured."""

    def test_tool_exists(self):
        names = {t["name"] for t in get_tool_schemas()}
        assert "request_planning" in names

    def test_tool_has_routing_category(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "request_planning")
        assert tool["category"] == "routing"

    def test_tool_requires_request_and_reasoning(self):
        tool = next(t for t in get_tool_schemas() if t["name"] == "request_planning")
        assert "request" in tool["parameters"]["properties"]
        assert "reasoning" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["request", "reasoning"]

    def test_tool_in_orchestrator_tools(self):
        orch_tools = get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS)
        names = {t["name"] for t in orch_tools}
        assert "request_planning" in names

    def test_tool_not_in_mission_agent_tools(self):
        mission_tools = get_tool_schemas(categories=MISSION_TOOL_CATEGORIES, extra_names=MISSION_EXTRA_TOOLS)
        names = {t["name"] for t in mission_tools}
        assert "request_planning" not in names

    def test_tool_not_in_viz_agent_tools(self):
        viz_tools = get_tool_schemas(categories=VIZ_TOOL_CATEGORIES, extra_names=VIZ_EXTRA_TOOLS)
        names = {t["name"] for t in viz_tools}
        assert "request_planning" not in names

    def test_tool_not_in_dataops_agent_tools(self):
        dataops_tools = get_tool_schemas(categories=DATAOPS_TOOL_CATEGORIES, extra_names=DATAOPS_EXTRA_TOOLS)
        names = {t["name"] for t in dataops_tools}
        assert "request_planning" not in names

    def test_tool_not_in_extraction_agent_tools(self):
        extraction_tools = get_tool_schemas(categories=EXTRACTION_CATEGORIES, extra_names=EXTRACTION_EXTRA_TOOLS)
        names = {t["name"] for t in extraction_tools}
        assert "request_planning" not in names
