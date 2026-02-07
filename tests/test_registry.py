"""
Tests for the Autoplot method registry.

Run with: python -m pytest tests/test_registry.py -v
"""

import pytest
from autoplot_bridge.registry import (
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

    def test_method_count_is_16(self):
        assert len(METHODS) == 16

    def test_parameters_have_required_fields(self):
        for m in METHODS:
            for p in m["parameters"]:
                assert "name" in p, f"{m['name']}: param missing 'name'"
                assert "type" in p, f"{m['name']}.{p.get('name', '?')}: missing 'type'"
                assert "required" in p, f"{m['name']}.{p['name']}: missing 'required'"
                assert "description" in p, f"{m['name']}.{p['name']}: missing 'description'"


class TestGetMethod:
    def test_known_method(self):
        m = get_method("plot_cdaweb")
        assert m is not None
        assert m["name"] == "plot_cdaweb"

    def test_unknown_method(self):
        assert get_method("nonexistent") is None

    def test_all_methods_retrievable(self):
        for m in METHODS:
            assert get_method(m["name"]) is m


class TestValidateArgs:
    def test_missing_required_param(self):
        errors = validate_args("plot_cdaweb", {"dataset_id": "X"})
        assert any("parameter_id" in e for e in errors)
        assert any("time_range" in e for e in errors)

    def test_valid_args(self):
        errors = validate_args("plot_cdaweb", {
            "dataset_id": "AC_H2_MFI",
            "parameter_id": "Magnitude",
            "time_range": "last week",
        })
        assert errors == []

    def test_unknown_method(self):
        errors = validate_args("nonexistent", {})
        assert any("Unknown method" in e for e in errors)

    def test_enum_validation(self):
        errors = validate_args("set_render_type", {"render_type": "invalid_type"})
        assert any("Invalid value" in e for e in errors)

    def test_enum_valid_value(self):
        errors = validate_args("set_render_type", {"render_type": "scatter"})
        assert errors == []

    def test_no_params_method(self):
        errors = validate_args("reset", {})
        assert errors == []

    def test_optional_params_not_required(self):
        errors = validate_args("set_render_type", {"render_type": "scatter"})
        assert errors == []  # index is optional

    def test_axis_enum_validation(self):
        errors = validate_args("set_axis_label", {"axis": "x", "label": "test"})
        assert any("Invalid value" in e for e in errors)

    def test_axis_enum_valid(self):
        errors = validate_args("set_axis_label", {"axis": "y", "label": "B (nT)"})
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
        assert "`scatter`" in result
        assert "`spectrogram`" in result

    def test_descriptions_included(self):
        result = render_method_catalog()
        assert "Plot CDAWeb data" in result
        assert "Reset the Autoplot canvas" in result
