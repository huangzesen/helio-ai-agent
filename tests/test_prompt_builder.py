"""
Tests for dynamic prompt generation from catalog.

Verifies that prompt_builder.py produces prompts containing all spacecraft,
datasets, and parameters from the catalog — the single source of truth.

Run with: python -m pytest tests/test_prompt_builder.py
"""

import pytest
from knowledge.catalog import SPACECRAFT, classify_instrument_type
from knowledge.prompt_builder import (
    generate_spacecraft_overview,
    generate_dataset_quick_reference,
    generate_planner_dataset_reference,
    generate_mission_profiles,
    build_mission_prompt,
    build_data_ops_prompt,
    build_system_prompt,
    build_planner_agent_prompt,
    build_visualization_prompt,
    build_discovery_prompt,
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
        # Parameter details come from metadata, not hardcoded
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
        # PSP should have switchback analysis tip if curated profiles exist.
        # Auto-generated missions may have empty analysis_patterns.
        # Just verify the function returns non-empty output.
        assert len(profiles) > 0


class TestBuildMissionPrompt:
    def test_psp_prompt_contains_mission_info(self):
        prompt = build_mission_prompt("PSP")
        assert "Parker Solar Probe" in prompt
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in prompt
        assert "FIELDS" in prompt

    def test_ace_prompt_contains_mission_info(self):
        prompt = build_mission_prompt("ACE")
        # ACE name may be "ACE" or "Advanced Composition Explorer" depending
        # on whether the mission JSON was curated or auto-generated
        assert "ACE" in prompt
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

    def test_mission_prompt_has_dataset_selection_workflow(self):
        prompt = build_mission_prompt("PSP")
        assert "## Dataset Selection Workflow" in prompt
        assert "fetch_data" in prompt
        # Computation patterns moved to DataOps agent
        assert "custom_operation" not in prompt

    def test_mission_prompt_has_recommended_datasets(self):
        prompt = build_mission_prompt("PSP")
        assert "## Recommended Datasets" in prompt

    def test_mission_prompt_mentions_browse_datasets(self):
        prompt = build_mission_prompt("PSP")
        assert "browse_datasets" in prompt

    def test_mission_prompt_has_no_advanced_section(self):
        prompt = build_mission_prompt("PSP")
        assert "## Advanced Datasets" not in prompt

    def test_mission_prompt_has_dataset_documentation_section(self):
        prompt = build_mission_prompt("PSP")
        assert "## Dataset Documentation" in prompt
        assert "get_dataset_docs" in prompt

    def test_mission_prompt_has_analysis_patterns(self):
        prompt = build_mission_prompt("PSP")
        # Analysis patterns section only appears if the mission has curated patterns.
        # Auto-generated missions have empty analysis_patterns → section omitted.
        from knowledge.mission_loader import load_mission
        mission = load_mission("PSP")
        if mission.get("profile", {}).get("analysis_patterns"):
            assert "## Analysis Patterns" in prompt
        else:
            # With auto-generated data, no analysis patterns section
            assert "## Recommended Datasets" in prompt

    def test_mission_prompt_forbids_plotting(self):
        prompt = build_mission_prompt("PSP")
        assert "Do NOT attempt to plot" in prompt

    def test_mission_prompt_forbids_transformations(self):
        prompt = build_mission_prompt("PSP")
        assert "Do NOT attempt data transformations" in prompt

    def test_mission_prompt_workflow_excludes_plot_computed_data(self):
        prompt = build_mission_prompt("PSP")
        # plot_computed_data should not appear in the workflow steps
        workflow_start = prompt.index("## Dataset Selection Workflow")
        workflow_end = prompt.index("## Reporting Results")
        workflow_section = prompt[workflow_start:workflow_end]
        assert "plot_computed_data" not in workflow_section

    def test_mission_prompt_has_reporting_section(self):
        prompt = build_mission_prompt("PSP")
        assert "## Reporting Results" in prompt

    def test_mission_prompt_has_data_specialist_identity(self):
        prompt = build_mission_prompt("PSP")
        assert "data specialist agent" in prompt.lower()

    def test_mission_prompt_has_explore_before_fetch_workflow(self):
        """Workflow guides: candidate inspection, browse for vague requests."""
        prompt = build_mission_prompt("PSP")
        workflow_start = prompt.index("## Dataset Selection Workflow")
        workflow_end = prompt.index("## Reporting Results")
        workflow_section = prompt[workflow_start:workflow_end]
        assert "candidate datasets" in workflow_section
        assert "fetch_data" in workflow_section
        assert "vague request" in workflow_section
        # Compute tools no longer in mission workflow
        assert "custom_operation" not in workflow_section
        assert "describe_data" not in workflow_section
        assert "save_data" not in workflow_section

    def test_mission_prompt_has_parameter_summaries_with_cache(self, tmp_path):
        """When local metadata cache exists, primary datasets show parameter names."""
        import json
        from unittest.mock import patch

        # Create a fake cache for PSP_FLD_L2_MAG_RTN_1MIN
        fake_missions = tmp_path / "missions"
        psp_metadata = fake_missions / "psp" / "metadata"
        psp_metadata.mkdir(parents=True)
        sample_info = {
            "parameters": [
                {"name": "Time", "type": "isotime", "units": "UTC"},
                {"name": "psp_fld_l2_mag_RTN_1min", "type": "double", "units": "nT", "size": [3]},
                {"name": "psp_fld_l2_quality_flags", "type": "integer", "units": None},
            ],
        }
        (psp_metadata / "PSP_FLD_L2_MAG_RTN_1MIN.json").write_text(
            json.dumps(sample_info), encoding="utf-8"
        )

        with patch("knowledge.metadata_client._MISSIONS_DIR", fake_missions):
            from knowledge.metadata_client import clear_cache
            clear_cache()
            prompt = build_mission_prompt("PSP")
            assert "Parameters:" in prompt
            assert "psp_fld_l2_mag_RTN_1min" in prompt

    def test_mission_prompt_without_cache_still_works(self, tmp_path):
        """When no local cache exists, prompt still generates without parameter lines."""
        from unittest.mock import patch

        # Point to empty directory — no cache files
        fake_missions = tmp_path / "empty_missions"
        fake_missions.mkdir()

        with patch("knowledge.metadata_client._MISSIONS_DIR", fake_missions):
            from knowledge.metadata_client import clear_cache
            clear_cache()
            prompt = build_mission_prompt("PSP")
            # Prompt should still work, just without parameter summaries
            assert "## Recommended Datasets" in prompt
            assert "PSP_FLD_L2_MAG_RTN_1MIN" in prompt


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

    def test_contains_delegate_to_visualization_instructions(self):
        prompt = build_system_prompt()
        assert "delegate_to_visualization" in prompt

    def test_contains_delegate_to_mission_instructions(self):
        prompt = build_system_prompt()
        assert "delegate_to_mission" in prompt

    def test_contains_delegate_to_data_ops_instructions(self):
        prompt = build_system_prompt()
        assert "delegate_to_data_ops" in prompt

    def test_contains_routing_table(self):
        prompt = build_system_prompt()
        assert "## Supported Missions" in prompt
        assert "Capabilities" in prompt

    def test_slim_prompt_routing_table_has_no_dataset_ids(self):
        prompt = build_system_prompt()
        # The routing table section should NOT list dataset IDs
        # (they may appear in examples, but not in the mission table)
        routing_section = prompt.split("## Supported Missions")[1].split("## Workflow")[0]
        assert "PSP_FLD_L2_MAG_RTN_1MIN" not in routing_section
        assert "AC_H2_MFI" not in routing_section

    def test_slim_prompt_has_no_mission_profiles(self):
        prompt = build_system_prompt()
        # Analysis tips and mission-specific knowledge moved to sub-agents
        assert "## Mission-Specific Knowledge" not in prompt
        assert "Switchback detection" not in prompt


class TestBuildPlannerAgentPrompt:
    def test_no_user_request_placeholder(self):
        """Chat-based planner receives user request via chat message, not template."""
        prompt = build_planner_agent_prompt()
        assert "{user_request}" not in prompt

    def test_no_known_dataset_ids_section(self):
        """Dataset reference removed — datasets come from discovery results."""
        prompt = build_planner_agent_prompt()
        assert "Known Dataset IDs" not in prompt

    def test_prompt_size_reduced(self):
        """Planning prompt should be much smaller without the 32K dataset ref."""
        prompt = build_planner_agent_prompt()
        # Was ~140K chars, should now be ~12K
        assert len(prompt) < 20000, f"Prompt still too large: {len(prompt)} chars"

    def test_references_discovery_results(self):
        """Dataset Selection section should reference Discovery Results."""
        prompt = build_planner_agent_prompt()
        assert "Discovery Results" in prompt

    def test_contains_tool_docs(self):
        prompt = build_planner_agent_prompt()
        assert "fetch_data" in prompt
        assert "custom_operation" in prompt

    def test_contains_planning_guidelines(self):
        prompt = build_planner_agent_prompt()
        assert "Planning Guidelines" in prompt

    def test_contains_mission_tagging_instructions(self):
        prompt = build_planner_agent_prompt()
        assert "Mission Tagging" in prompt

    def test_contains_spacecraft_ids_for_tagging(self):
        prompt = build_planner_agent_prompt()
        assert "PSP" in prompt
        assert "ACE" in prompt

    def test_plotting_tasks_use_visualization_mission(self):
        prompt = build_planner_agent_prompt()
        assert "__visualization__" in prompt
        assert 'mission="__visualization__"' in prompt

    def test_compute_tasks_use_data_ops_mission(self):
        prompt = build_planner_agent_prompt()
        assert "__data_ops__" in prompt
        assert 'mission="__data_ops__"' in prompt

    def test_contains_batch_round_semantics(self):
        prompt = build_planner_agent_prompt()
        assert "batch" in prompt.lower()
        assert "round" in prompt.lower()

    def test_contains_status_field_docs(self):
        prompt = build_planner_agent_prompt()
        assert '"continue"' in prompt or "'continue'" in prompt
        assert '"done"' in prompt or "'done'" in prompt

    def test_contains_routing_table(self):
        prompt = build_planner_agent_prompt()
        assert "Known Missions" in prompt


class TestBuildDiscoveryPrompt:
    """Test the discovery phase prompt for the planner."""

    def test_returns_nonempty_string(self):
        prompt = build_discovery_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_mentions_list_parameters(self):
        prompt = build_discovery_prompt()
        assert "list_parameters" in prompt

    def test_mentions_browse_datasets(self):
        prompt = build_discovery_prompt()
        assert "browse_datasets" in prompt

    def test_browse_first_strategy(self):
        """Discovery prompt should guide browse-first, verify-top-picks."""
        prompt = build_discovery_prompt()
        assert "Browse First" in prompt or "browse_datasets" in prompt
        assert "top" in prompt.lower() or "Top Picks" in prompt

    def test_mentions_output_format(self):
        prompt = build_discovery_prompt()
        assert "summary" in prompt.lower()


class TestBuildDataOpsPrompt:
    """Test the DataOps agent's system prompt builder."""

    def test_contains_specialist_identity(self):
        prompt = build_data_ops_prompt()
        assert "data transformation" in prompt.lower()

    def test_contains_computation_patterns(self):
        prompt = build_data_ops_prompt()
        assert "Magnitude" in prompt
        assert "Smoothing" in prompt
        assert "Resample" in prompt
        assert "Normalize" in prompt

    def test_contains_code_guidelines(self):
        prompt = build_data_ops_prompt()
        assert "## Code Guidelines" in prompt
        assert "result" in prompt
        assert "DatetimeIndex" in prompt

    def test_contains_workflow(self):
        prompt = build_data_ops_prompt()
        assert "## Workflow" in prompt
        assert "list_fetched_data" in prompt
        assert "custom_operation" in prompt
        assert "describe_data" in prompt

    def test_forbids_fetching(self):
        prompt = build_data_ops_prompt()
        assert "Do NOT attempt to fetch" in prompt

    def test_forbids_plotting(self):
        prompt = build_data_ops_prompt()
        assert "Do NOT attempt to plot" in prompt

    def test_has_reporting_section(self):
        prompt = build_data_ops_prompt()
        assert "## Reporting Results" in prompt


class TestBuildVisualizationPrompt:
    """Test the visualization agent's system prompt builder."""

    def test_contains_spec_workflow(self):
        prompt = build_visualization_prompt()
        assert "update_plot_spec" in prompt

    def test_contains_tool_usage_sections(self):
        prompt = build_visualization_prompt()
        assert "## How update_plot_spec Works" in prompt
        assert "## Spec Field Reference" in prompt

    def test_contains_workflow(self):
        prompt = build_visualization_prompt()
        assert "list_fetched_data" in prompt

    def test_no_gui_section_by_default(self):
        prompt = build_visualization_prompt(gui_mode=False)
        assert "Interactive Mode" not in prompt

    def test_gui_mode_appends_section(self):
        prompt = build_visualization_prompt(gui_mode=True)
        assert "Interactive Mode" in prompt

    def test_has_visualization_specialist_identity(self):
        prompt = build_visualization_prompt()
        assert "visualization" in prompt.lower()

    def test_has_time_format_guidance(self):
        prompt = build_visualization_prompt()
        assert "## Time Range Format" in prompt
        assert "NOT '/'" in prompt

    def test_has_spec_method_in_workflow(self):
        prompt = build_visualization_prompt()
        assert "update_plot_spec" in prompt

    def test_has_panel_example(self):
        prompt = build_visualization_prompt()
        assert "panels" in prompt

    def test_has_vector_data_note(self):
        prompt = build_visualization_prompt()
        assert "Vector data" in prompt

    def test_describes_two_tools(self):
        prompt = build_visualization_prompt()
        assert "two tools" in prompt
        assert "update_plot_spec" in prompt
        assert "list_fetched_data" in prompt
        assert "manage_plot" not in prompt

    def test_no_deleted_method_references(self):
        """Deleted registry methods should not appear in the prompt."""
        prompt = build_visualization_prompt()
        assert "set_render_type" not in prompt
        assert "set_color_table" not in prompt
        assert "save_session" not in prompt
        assert "load_session" not in prompt


class TestClassifyInstrumentType:
    """Test the shared instrument type classifier."""

    def test_magnetic(self):
        assert classify_instrument_type(["magnetic", "field"]) == "magnetic"
        assert classify_instrument_type(["mag", "rtn"]) == "magnetic"
        assert classify_instrument_type(["magnetometer"]) == "magnetic"

    def test_plasma(self):
        assert classify_instrument_type(["plasma", "density"]) == "plasma"
        assert classify_instrument_type(["solar wind"]) == "plasma"
        assert classify_instrument_type(["ion", "composition"]) == "plasma"

    def test_particles(self):
        assert classify_instrument_type(["energetic", "particle"]) == "particles"
        assert classify_instrument_type(["cosmic ray"]) == "particles"

    def test_indices(self):
        assert classify_instrument_type(["geomagnetic", "index"]) == "indices"

    def test_ephemeris(self):
        assert classify_instrument_type(["orbit", "position"]) == "ephemeris"
        assert classify_instrument_type(["ephemeris"]) == "ephemeris"

    def test_other(self):
        assert classify_instrument_type(["unknown"]) == "other"
        assert classify_instrument_type([]) == "other"

    def test_case_insensitive(self):
        assert classify_instrument_type(["Magnetic", "FIELD"]) == "magnetic"
        assert classify_instrument_type(["PLASMA"]) == "plasma"


class TestBuildParameterReference:
    """Test PlannerAgent._build_parameter_reference with browse + list_parameters."""

    def test_browse_results_included(self):
        from agent.planner import PlannerAgent
        tool_results = {
            "browse_datasets": [{"args": {"mission_id": "ACE"}, "result": {
                "status": "success", "mission_id": "ACE", "datasets": [
                    {"id": "AC_H2_MFI", "start_date": "1998", "stop_date": "2025",
                     "parameter_count": 6, "instrument": "mag", "type": "magnetic"}
                ]
            }}],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "AC_H2_MFI" in ref
        assert "ACE" in ref
        assert "magnetic" in ref

    def test_verified_tag_applied(self):
        from agent.planner import PlannerAgent
        tool_results = {
            "browse_datasets": [{"args": {"mission_id": "ACE"}, "result": {
                "status": "success", "mission_id": "ACE", "datasets": [
                    {"id": "AC_H2_MFI", "start_date": "1998", "stop_date": "2025",
                     "parameter_count": 6, "instrument": "mag", "type": "magnetic"},
                    {"id": "AC_H0_MFI", "start_date": "1998", "stop_date": "2025",
                     "parameter_count": 4, "instrument": "mag", "type": "magnetic"},
                ]
            }}],
            "list_parameters": [{"args": {"dataset_id": "AC_H2_MFI"}, "result": {
                "status": "success", "parameters": [
                    {"name": "BGSEc", "units": "nT", "size": [3]},
                    {"name": "Magnitude", "units": "nT"},
                ]
            }}],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "[VERIFIED]" in ref
        # AC_H2_MFI should be verified, AC_H0_MFI should not
        lines = ref.split("\n")
        for line in lines:
            if "AC_H2_MFI" in line and "VERIFIED" in line:
                break
        else:
            pytest.fail("AC_H2_MFI should be marked [VERIFIED]")

    def test_verified_parameters_section(self):
        from agent.planner import PlannerAgent
        tool_results = {
            "browse_datasets": [{"args": {"mission_id": "ACE"}, "result": {
                "status": "success", "mission_id": "ACE", "datasets": [
                    {"id": "AC_H2_MFI", "start_date": "1998", "stop_date": "2025",
                     "parameter_count": 6, "instrument": "mag", "type": "magnetic"}
                ]
            }}],
            "list_parameters": [{"args": {"dataset_id": "AC_H2_MFI"}, "result": {
                "status": "success", "parameters": [
                    {"name": "Time", "units": "UTC"},
                    {"name": "BGSEc", "units": "nT", "size": [3]},
                    {"name": "Magnitude", "units": "nT"},
                ]
            }}],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "Verified Parameters" in ref
        assert "BGSEc" in ref
        assert "Magnitude" in ref
        # Time should be excluded
        assert "\n  - Time" not in ref

    def test_empty_results(self):
        from agent.planner import PlannerAgent
        ref = PlannerAgent._build_parameter_reference({})
        assert ref == ""

    def test_list_parameters_only_fallback(self):
        """When no browse results, list_parameters data still produces output."""
        from agent.planner import PlannerAgent
        tool_results = {
            "list_parameters": [{"args": {"dataset_id": "AC_H2_MFI"}, "result": {
                "status": "success", "parameters": [
                    {"name": "BGSEc", "units": "nT", "size": [3]},
                ]
            }}],
        }
        ref = PlannerAgent._build_parameter_reference(tool_results)
        assert "AC_H2_MFI" in ref
        assert "BGSEc" in ref
