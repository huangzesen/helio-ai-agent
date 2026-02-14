"""
Tests for the visualization tool registry.

Run with: python -m pytest tests/test_registry.py -v
"""

import pytest
from rendering.registry import (
    TOOLS, get_method, validate_args, render_method_catalog,
)


class TestRegistryStructure:
    """Verify that all registry entries have required fields."""

    def test_all_tools_have_name(self):
        for t in TOOLS:
            assert "name" in t, f"Tool missing 'name': {t}"

    def test_all_tools_have_description(self):
        for t in TOOLS:
            assert "description" in t, f"Tool '{t.get('name', '?')}' missing 'description'"

    def test_all_tools_have_parameters(self):
        for t in TOOLS:
            assert "parameters" in t, f"Tool '{t['name']}' missing 'parameters'"
            assert isinstance(t["parameters"], list)

    def test_no_duplicate_names(self):
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

    def test_tool_count(self):
        assert len(TOOLS) == 2  # render_plotly_json, manage_plot

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"render_plotly_json", "manage_plot"}

    def test_parameters_have_required_fields(self):
        for t in TOOLS:
            for p in t["parameters"]:
                assert "name" in p, f"{t['name']}: param missing 'name'"
                assert "type" in p, f"{t['name']}.{p.get('name', '?')}: missing 'type'"
                assert "required" in p, f"{t['name']}.{p['name']}: missing 'required'"
                assert "description" in p, f"{t['name']}.{p['name']}: missing 'description'"

    def test_render_plotly_json_has_figure_json_param(self):
        t = get_method("render_plotly_json")
        assert t is not None
        param_names = [p["name"] for p in t["parameters"]]
        assert "figure_json" in param_names
        param = next(p for p in t["parameters"] if p["name"] == "figure_json")
        assert param["required"] is True

    def test_manage_plot_has_action_param(self):
        t = get_method("manage_plot")
        assert t is not None
        param_names = [p["name"] for p in t["parameters"]]
        assert "action" in param_names
        action_param = next(p for p in t["parameters"] if p["name"] == "action")
        assert action_param["required"] is True
        assert "enum" in action_param

    def test_manage_plot_has_format_param(self):
        t = get_method("manage_plot")
        assert t is not None
        param_names = [p["name"] for p in t["parameters"]]
        assert "format" in param_names
        fmt_param = next(p for p in t["parameters"] if p["name"] == "format")
        assert fmt_param["required"] is False
        assert fmt_param["default"] == "png"
        assert set(fmt_param["enum"]) == {"png", "pdf"}


class TestGetMethod:
    def test_known_tool(self):
        t = get_method("render_plotly_json")
        assert t is not None
        assert t["name"] == "render_plotly_json"

    def test_unknown_tool(self):
        assert get_method("nonexistent") is None

    def test_removed_tools_not_found(self):
        """Removed legacy tools should not exist in the registry."""
        assert get_method("plot_data") is None
        assert get_method("style_plot") is None

    def test_all_tools_retrievable(self):
        for t in TOOLS:
            assert get_method(t["name"]) is t

    def test_old_methods_not_found(self):
        """Old method names should no longer exist in the registry."""
        for name in ("plot_stored_data", "set_time_range", "export",
                      "get_plot_state", "plot_spectrogram", "reset",
                      "execute_visualization", "custom_visualization"):
            assert get_method(name) is None, f"{name} should have been removed"


class TestValidateArgs:
    def test_missing_required_param(self):
        errors = validate_args("render_plotly_json", {})
        assert any("figure_json" in e for e in errors)

    def test_valid_args(self):
        errors = validate_args("render_plotly_json", {
            "figure_json": {"data": [{"data_label": "X"}], "layout": {}}
        })
        assert errors == []

    def test_unknown_tool(self):
        errors = validate_args("nonexistent", {})
        assert any("Unknown tool" in e for e in errors)

    def test_manage_plot_action_enum_validation(self):
        errors = validate_args("manage_plot", {"action": "invalid"})
        assert any("Invalid value" in e for e in errors)

    def test_manage_plot_action_enum_valid(self):
        errors = validate_args("manage_plot", {"action": "reset"})
        assert errors == []

    def test_manage_plot_format_enum_valid(self):
        errors = validate_args("manage_plot", {"action": "export", "format": "pdf"})
        assert errors == []

    def test_manage_plot_format_enum_invalid(self):
        errors = validate_args("manage_plot", {"action": "export", "format": "invalid"})
        assert any("Invalid value" in e for e in errors)

    def test_removed_tools_return_unknown(self):
        errors = validate_args("plot_data", {})
        assert any("Unknown tool" in e for e in errors)
        errors = validate_args("style_plot", {})
        assert any("Unknown tool" in e for e in errors)


class TestRenderMethodCatalog:
    def test_returns_string(self):
        result = render_method_catalog()
        assert isinstance(result, str)

    def test_has_header(self):
        result = render_method_catalog()
        assert "## Visualization Tools" in result

    def test_all_tools_listed(self):
        result = render_method_catalog()
        for t in TOOLS:
            assert f"**{t['name']}**" in result

    def test_enum_values_shown(self):
        result = render_method_catalog()
        assert "`png`" in result
        assert "`pdf`" in result

    def test_descriptions_included(self):
        result = render_method_catalog()
        assert "data_label" in result
        assert "Imperative operations" in result

    def test_examples_section(self):
        result = render_method_catalog()
        assert "## Examples" in result
        assert "render_plotly_json" in result
        assert "manage_plot" in result
