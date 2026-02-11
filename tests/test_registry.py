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

    def test_tool_count_is_3(self):
        assert len(TOOLS) == 3

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"plot_data", "style_plot", "manage_plot"}

    def test_parameters_have_required_fields(self):
        for t in TOOLS:
            for p in t["parameters"]:
                assert "name" in p, f"{t['name']}: param missing 'name'"
                assert "type" in p, f"{t['name']}.{p.get('name', '?')}: missing 'type'"
                assert "required" in p, f"{t['name']}.{p['name']}: missing 'required'"
                assert "description" in p, f"{t['name']}.{p['name']}: missing 'description'"

    def test_plot_data_has_labels_param(self):
        t = get_method("plot_data")
        assert t is not None
        param_names = [p["name"] for p in t["parameters"]]
        assert "labels" in param_names
        labels_param = next(p for p in t["parameters"] if p["name"] == "labels")
        assert labels_param["required"] is True

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

    def test_style_plot_all_optional(self):
        t = get_method("style_plot")
        assert t is not None
        for p in t["parameters"]:
            assert p["required"] is False, f"style_plot param '{p['name']}' should be optional"


class TestGetMethod:
    def test_known_tool(self):
        t = get_method("plot_data")
        assert t is not None
        assert t["name"] == "plot_data"

    def test_unknown_tool(self):
        assert get_method("nonexistent") is None

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
        errors = validate_args("plot_data", {})
        assert any("labels" in e for e in errors)

    def test_valid_args(self):
        errors = validate_args("plot_data", {"labels": "ACE_Bmag"})
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

    def test_style_plot_no_required_params(self):
        errors = validate_args("style_plot", {})
        assert errors == []

    def test_style_plot_log_scale_accepts_strings(self):
        # log_scale is "string or object" — no enum validation (accepts dicts too)
        errors = validate_args("style_plot", {"log_scale": "y"})
        assert errors == []
        errors = validate_args("style_plot", {"log_scale": "linear"})
        assert errors == []


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
        assert "`line`" in result
        assert "`spectrogram`" in result

    def test_descriptions_included(self):
        result = render_method_catalog()
        assert "fresh plot" in result
        assert "aesthetic" in result.lower()
        assert "Structural operations" in result

    def test_examples_section(self):
        result = render_method_catalog()
        assert "## Examples" in result
        assert "plot_data" in result
        assert "style_plot" in result
        assert "manage_plot" in result
