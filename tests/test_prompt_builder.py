"""
Tests for dynamic prompt generation from catalog.

Verifies that prompt_builder.py produces prompts containing all spacecraft,
datasets, and parameters from the catalog — the single source of truth.

Run with: python -m pytest tests/test_prompt_builder.py
"""

import pytest
from knowledge.catalog import SPACECRAFT
from knowledge.prompt_builder import (
    generate_spacecraft_overview,
    generate_dataset_quick_reference,
    generate_planner_dataset_reference,
    generate_mission_profiles,
    build_mission_prompt,
    build_system_prompt,
    build_planning_prompt,
)


class TestGenerateSpacecraftOverview:
    def test_all_spacecraft_present(self):
        table = generate_spacecraft_overview()
        for sc_id, sc in SPACECRAFT.items():
            assert sc["name"] in table, f"{sc['name']} missing from overview"

    def test_is_markdown_table(self):
        table = generate_spacecraft_overview()
        lines = table.strip().split("\n")
        assert lines[0].startswith("|")
        assert "---" in lines[1]


class TestGenerateDatasetQuickReference:
    def test_all_datasets_present(self):
        table = generate_dataset_quick_reference()
        for sc_id, sc in SPACECRAFT.items():
            for inst_id, inst in sc["instruments"].items():
                for ds in inst["datasets"]:
                    assert ds in table, f"Dataset {ds} missing from quick reference"

    def test_directs_to_list_parameters(self):
        table = generate_dataset_quick_reference()
        # Parameter details come from HAPI, not hardcoded
        assert "list_parameters" in table

    def test_is_markdown_table(self):
        table = generate_dataset_quick_reference()
        lines = table.strip().split("\n")
        assert lines[0].startswith("|")


class TestGeneratePlannerDatasetReference:
    def test_all_spacecraft_present(self):
        ref = generate_planner_dataset_reference()
        for sc_id, sc in SPACECRAFT.items():
            assert sc["name"] in ref, f"{sc['name']} missing from planner reference"

    def test_all_datasets_present(self):
        ref = generate_planner_dataset_reference()
        for sc_id, sc in SPACECRAFT.items():
            for inst_id, inst in sc["instruments"].items():
                for ds in inst["datasets"]:
                    assert ds in ref, f"Dataset {ds} missing from planner reference"


class TestGenerateMissionProfiles:
    def test_all_profiled_missions_present(self):
        profiles = generate_mission_profiles()
        for sc_id, sc in SPACECRAFT.items():
            if sc.get("profile"):
                assert sc["name"] in profiles, f"{sc['name']} missing from profiles"

    def test_analysis_patterns_included(self):
        profiles = generate_mission_profiles()
        # PSP should have switchback analysis tip
        assert "Switchback" in profiles or "switchback" in profiles


class TestBuildMissionPrompt:
    def test_psp_prompt_contains_mission_info(self):
        prompt = build_mission_prompt("PSP")
        assert "Parker Solar Probe" in prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in prompt
        assert "FIELDS" in prompt

    def test_ace_prompt_contains_mission_info(self):
        prompt = build_mission_prompt("ACE")
        assert "Advanced Composition Explorer" in prompt
        assert "AC_H2_MFI" in prompt

    def test_prompt_does_not_contain_other_missions(self):
        prompt = build_mission_prompt("PSP")
        assert "AC_H2_MFI" not in prompt
        assert "Advanced Composition Explorer" not in prompt

    def test_invalid_mission_raises(self):
        with pytest.raises(KeyError):
            build_mission_prompt("NONEXISTENT")

    def test_all_missions_can_build(self):
        for sc_id in SPACECRAFT:
            prompt = build_mission_prompt(sc_id)
            assert len(prompt) > 50

    def test_directs_to_list_parameters(self):
        prompt = build_mission_prompt("PSP")
        assert "list_parameters" in prompt

    def test_mission_prompt_has_data_ops_docs(self):
        prompt = build_mission_prompt("PSP")
        assert "## Data Operations Workflow" in prompt
        assert "custom_operation" in prompt
        assert "Magnitude" in prompt

    def test_mission_prompt_has_tiered_datasets(self):
        prompt = build_mission_prompt("PSP")
        assert "## Primary Datasets" in prompt

    def test_mission_prompt_has_analysis_patterns(self):
        prompt = build_mission_prompt("PSP")
        assert "## Analysis Patterns" in prompt
        assert "Switchback" in prompt


class TestBuildSystemPrompt:
    def test_contains_today_placeholder(self):
        prompt = build_system_prompt()
        assert "{today}" in prompt

    def test_contains_all_spacecraft(self):
        prompt = build_system_prompt()
        for sc_id, sc in SPACECRAFT.items():
            assert sc["name"] in prompt, f"{sc['name']} missing from system prompt"

    def test_contains_workflow_sections(self):
        prompt = build_system_prompt()
        assert "## Workflow" in prompt
        assert "## Time Range Handling" in prompt

    def test_contains_routing_table(self):
        prompt = build_system_prompt()
        assert "## Supported Missions" in prompt
        assert "Capabilities" in prompt

    def test_slim_prompt_has_no_dataset_ids(self):
        prompt = build_system_prompt()
        # Dataset IDs should NOT be in the main agent prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" not in prompt
        assert "AC_H2_MFI" not in prompt

    def test_slim_prompt_has_no_mission_profiles(self):
        prompt = build_system_prompt()
        # Analysis tips and mission-specific knowledge moved to sub-agents
        assert "## Mission-Specific Knowledge" not in prompt
        assert "Switchback detection" not in prompt


class TestBuildPlanningPrompt:
    def test_contains_user_request_placeholder(self):
        prompt = build_planning_prompt()
        assert "{user_request}" in prompt

    def test_contains_all_datasets(self):
        prompt = build_planning_prompt()
        for sc_id, sc in SPACECRAFT.items():
            for inst_id, inst in sc["instruments"].items():
                for ds in inst["datasets"]:
                    assert ds in prompt, f"Dataset {ds} missing from planning prompt"

    def test_contains_tool_docs(self):
        prompt = build_planning_prompt()
        assert "fetch_data" in prompt
        assert "custom_operation" in prompt
        assert "plot_computed_data" in prompt

    def test_contains_planning_guidelines(self):
        prompt = build_planning_prompt()
        assert "Planning Guidelines" in prompt

    def test_contains_mission_tagging_instructions(self):
        prompt = build_planning_prompt()
        assert "Mission Tagging" in prompt
        assert "depends_on" in prompt

    def test_contains_spacecraft_ids_for_tagging(self):
        prompt = build_planning_prompt()
        assert "PSP" in prompt
        assert "ACE" in prompt
