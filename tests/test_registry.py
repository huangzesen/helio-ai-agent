"""
Tests for the visualization method registry.

Run with: python -m pytest tests/test_registry.py -v
"""

import pytest
from rendering.registry import (
    METHODS, get_method, validate_args, render_method_catalog,
)


class TestRegistryStructure:
    """Verify that all registry entries have required fields."""

    def test_all_methods_have_name(self):
        for m in METHODS:
            assert "name" in m, f"Method missing 'name': {m}"

    def test_all_methods_have_description(self):
        for m in METHODS:
            assert "description" in m, f"Method '{m.get('name', '?')}' missing 'description'"

    def test_all_methods_have_parameters(self):
        for m in METHODS:
            assert "parameters" in m, f"Method '{m['name']}' missing 'parameters'"
            assert isinstance(m["parameters"], list)

    def test_no_duplicate_names(self):
        names = [m["name"] for m in METHODS]
        assert len(names) == len(set(names)), f"Duplicate method names: {[n for n in names if names.count(n) > 1]}"

    def test_method_count_is_6(self):
        assert len(METHODS) == 6

    def test_parameters_have_required_fields(self):
        for m in METHODS:
            for p in m["parameters"]:
                assert "name" in p, f"{m['name']}: param missing 'name'"
                assert "type" in p, f"{m['name']}.{p.get('name', '?')}: missing 'type'"
                assert "required" in p, f"{m['name']}.{p['name']}: missing 'required'"
                assert "description" in p, f"{m['name']}.{p['name']}: missing 'description'"

    def test_plot_stored_data_has_index_param(self):
        m = get_method("plot_stored_data")
        assert m is not None
        param_names = [p["name"] for p in m["parameters"]]
        assert "index" in param_names
        idx_param = next(p for p in m["parameters"] if p["name"] == "index")
        assert idx_param["required"] is False
        assert idx_param["default"] == -1
        assert idx_param["type"] == "integer"

    def test_export_has_format_param(self):
        m = get_method("export")
        assert m is not None
        param_names = [p["name"] for p in m["parameters"]]
        assert "format" in param_names
        fmt_param = next(p for p in m["parameters"] if p["name"] == "format")
        assert fmt_param["required"] is False
        assert fmt_param["default"] == "png"
        assert set(fmt_param["enum"]) == {"png", "pdf"}


class TestGetMethod:
    def test_known_method(self):
        m = get_method("plot_stored_data")
        assert m is not None
        assert m["name"] == "plot_stored_data"

    def test_unknown_method(self):
        assert get_method("nonexistent") is None

    def test_all_methods_retrievable(self):
        for m in METHODS:
            assert get_method(m["name"]) is m

    def test_deleted_methods_not_found(self):
        """Thin wrapper methods should no longer exist in the registry."""
        for name in ("set_title", "set_axis_label", "toggle_log_scale",
                      "set_axis_range", "set_canvas_size", "set_render_type",
                      "set_color_table", "save_session", "load_session",
                      "export_png", "export_pdf"):
            assert get_method(name) is None, f"{name} should have been removed"


class TestValidateArgs:
    def test_missing_required_param(self):
        errors = validate_args("plot_stored_data", {})
        assert any("labels" in e for e in errors)

    def test_valid_args(self):
        errors = validate_args("plot_stored_data", {
            "labels": "ACE_Bmag",
        })
        assert errors == []

    def test_unknown_method(self):
        errors = validate_args("nonexistent", {})
        assert any("Unknown method" in e for e in errors)

    def test_export_format_enum_validation(self):
        errors = validate_args("export", {"filename": "test", "format": "invalid"})
        assert any("Invalid value" in e for e in errors)

    def test_export_format_enum_valid(self):
        errors = validate_args("export", {"filename": "test", "format": "pdf"})
        assert errors == []

    def test_no_params_method(self):
        errors = validate_args("reset", {})
        assert errors == []

    def test_plot_stored_data_with_index(self):
        errors = validate_args("plot_stored_data", {
            "labels": "ACE_Bmag",
            "index": 1,
        })
        assert errors == []

    def test_plot_stored_data_without_index(self):
        errors = validate_args("plot_stored_data", {"labels": "ACE_Bmag"})
        assert errors == []


class TestRenderMethodCatalog:
    def test_returns_string(self):
        result = render_method_catalog()
        assert isinstance(result, str)

    def test_has_header(self):
        result = render_method_catalog()
        assert "## Available Methods" in result

    def test_all_methods_listed(self):
        result = render_method_catalog()
        for m in METHODS:
            assert f"**{m['name']}**" in result

    def test_enum_values_shown(self):
        result = render_method_catalog()
        assert "`png`" in result
        assert "`pdf`" in result

    def test_descriptions_included(self):
        result = render_method_catalog()
        assert "Plot one or more in-memory" in result
        assert "Reset the plot canvas" in result
