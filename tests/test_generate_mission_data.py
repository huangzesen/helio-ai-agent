"""
Tests for the HAPI auto-generation script's merge logic.

Tests the merge_dataset_info function to ensure it preserves
hand-curated fields (tier) while overwriting HAPI-derived fields.

Run with: python -m pytest tests/test_generate_mission_data.py -v
"""

import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_mission_data import (
    merge_dataset_info,
    match_dataset_to_mission,
)


class TestMergeDatasetInfo:
    """Test the merge logic for combining HAPI data with existing JSON."""

    def test_new_dataset_defaults_to_advanced(self):
        """A new dataset (no existing entry) should get tier='advanced'."""
        hapi_info = {
            "description": "New dataset from HAPI",
            "startDate": "2020-01-01",
            "stopDate": "2025-01-01",
            "parameters": [
                {"name": "Time", "type": "isotime"},
                {"name": "Bx", "type": "double", "units": "nT", "description": "X component"},
            ],
        }
        result = merge_dataset_info(None, hapi_info, "NEW_DATASET_ID")
        assert result["tier"] == "advanced"
        assert result["description"] == "New dataset from HAPI"
        assert result["start_date"] == "2020-01-01"
        assert result["stop_date"] == "2025-01-01"
        # Time parameter should be filtered out
        assert len(result["parameters"]) == 1
        assert result["parameters"][0]["name"] == "Bx"

    def test_preserves_existing_tier_primary(self):
        """If existing dataset has tier='primary', preserve it."""
        existing = {
            "tier": "primary",
            "description": "Old description",
            "parameters": [],
        }
        hapi_info = {
            "description": "Updated description from HAPI",
            "startDate": "2018-10-06",
            "stopDate": "2025-12-31",
            "parameters": [
                {"name": "Time", "type": "isotime"},
                {"name": "Bmag", "type": "double", "units": "nT"},
            ],
        }
        result = merge_dataset_info(existing, hapi_info, "TEST_DS")
        assert result["tier"] == "primary"  # Preserved!
        assert result["description"] == "Updated description from HAPI"  # Overwritten
        assert len(result["parameters"]) == 1  # Time filtered

    def test_preserves_existing_tier_advanced(self):
        existing = {"tier": "advanced", "parameters": []}
        hapi_info = {"description": "test", "parameters": []}
        result = merge_dataset_info(existing, hapi_info, "TEST_DS")
        assert result["tier"] == "advanced"

    def test_overwrites_dates(self):
        existing = {
            "tier": "primary",
            "start_date": "2018-01-01",
            "stop_date": "2020-01-01",
            "parameters": [],
        }
        hapi_info = {
            "startDate": "2018-10-06",
            "stopDate": "2025-11-30",
            "parameters": [],
        }
        result = merge_dataset_info(existing, hapi_info, "TEST_DS")
        assert result["start_date"] == "2018-10-06"
        assert result["stop_date"] == "2025-11-30"

    def test_overwrites_parameters(self):
        existing = {
            "tier": "primary",
            "parameters": [{"name": "old_param"}],
        }
        hapi_info = {
            "parameters": [
                {"name": "Time", "type": "isotime"},
                {"name": "new_param", "type": "double", "units": "nT", "description": "New"},
            ],
        }
        result = merge_dataset_info(existing, hapi_info, "TEST_DS")
        assert len(result["parameters"]) == 1
        assert result["parameters"][0]["name"] == "new_param"

    def test_parameter_size_preserved(self):
        hapi_info = {
            "parameters": [
                {"name": "Bvec", "type": "double", "units": "nT", "size": [3], "description": "B vector"},
            ],
        }
        result = merge_dataset_info(None, hapi_info, "TEST_DS")
        assert result["parameters"][0]["size"] == [3]

    def test_empty_hapi_info(self):
        result = merge_dataset_info(None, {"parameters": []}, "TEST_DS")
        assert result["tier"] == "advanced"
        assert result["parameters"] == []
        assert result["description"] == ""


class TestMatchDatasetToMission:
    """Test the prefix-based mission matching."""

    @pytest.mark.parametrize("dataset_id,expected_mission", [
        ("PSP_FLD_L2_MAG_RTN_1MIN", "psp"),
        ("PSP_SWP_SPC_L3I", "psp"),
        ("PSP_SWP_SPI_SF00_L3_MOM", "psp"),
        ("PSP_SWP_SPA_SF0_L3_PAD", "psp"),
        ("PSP_SWP_SPB_SF0_L3_PAD", "psp"),
        ("PSP_ISOIS-EPIHI_L2-HET-RATES60", "psp"),
        ("AC_H2_MFI", "ace"),
        ("SOLO_L2_MAG-RTN-NORMAL-1-MINUTE", "solo"),
        ("OMNI_HRO_1MIN", "omni"),
        ("WI_H2_MFI", "wind"),
        ("DSCOVR_H0_MAG", "dscovr"),
        ("MMS1_FGM_SRVY_L2", "mms"),
        ("STA_L2_MAG_RTN", "stereo_a"),
        ("UNKNOWN_DATASET", None),
    ])
    def test_dataset_prefix_matching(self, dataset_id, expected_mission):
        mission, _ = match_dataset_to_mission(dataset_id)
        assert mission == expected_mission

    @pytest.mark.parametrize("dataset_id,expected_instrument", [
        ("PSP_FLD_L2_MAG_RTN_1MIN", "FIELDS/MAG"),
        ("PSP_SWP_SPC_L3I", "SWEAP"),
        ("PSP_SWP_SPI_SF00_L3_MOM", "SWEAP/SPAN-I"),
        ("PSP_SWP_SPA_SF0_L3_PAD", "SWEAP/SPAN-E"),
        ("PSP_SWP_SPB_SF0_L3_PAD", "SWEAP/SPAN-E"),
        ("PSP_ISOIS-EPIHI_L2-HET-RATES60", "ISOIS"),
    ])
    def test_instrument_suggestion(self, dataset_id, expected_instrument):
        _, instrument = match_dataset_to_mission(dataset_id)
        assert instrument == expected_instrument
