"""
Tests for the per-mission JSON loader.

Run with: python -m pytest tests/test_mission_loader.py -v
"""

import json
import pytest
from pathlib import Path

from knowledge.mission_loader import (
    load_mission,
    load_all_missions,
    get_routing_table,
    get_mission_datasets,
    get_mission_ids,
    clear_cache,
    _MISSIONS_DIR,
)


@pytest.fixture(autouse=True)
def fresh_cache():
    """Clear the module cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestLoadMission:
    def test_load_psp(self):
        mission = load_mission("PSP")
        assert mission["id"] == "PSP"
        assert mission["name"] == "Parker Solar Probe"
        assert "FIELDS/MAG" in mission["instruments"]

    def test_load_ace(self):
        mission = load_mission("ACE")
        assert mission["id"] == "ACE"
        assert "MAG" in mission["instruments"]

    def test_load_is_case_insensitive(self):
        mission = load_mission("psp")
        assert mission["id"] == "PSP"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_mission("NONEXISTENT")

    def test_caching(self):
        m1 = load_mission("PSP")
        m2 = load_mission("PSP")
        assert m1 is m2  # Same object = cached

    def test_mission_has_profile(self):
        mission = load_mission("PSP")
        profile = mission["profile"]
        assert "description" in profile
        assert "coordinate_systems" in profile
        assert "analysis_patterns" in profile

    def test_datasets_are_dicts(self):
        """In JSON, datasets are dicts keyed by dataset ID, not lists."""
        mission = load_mission("PSP")
        for inst in mission["instruments"].values():
            datasets = inst["datasets"]
            assert isinstance(datasets, dict)
            for ds_id, ds_info in datasets.items():
                assert isinstance(ds_id, str)
                assert "tier" in ds_info


class TestLoadAllMissions:
    def test_loads_all_8_missions(self):
        missions = load_all_missions()
        assert len(missions) == 8
        expected_ids = {"PSP", "SolO", "ACE", "OMNI", "WIND", "DSCOVR", "MMS", "STEREO_A"}
        assert set(missions.keys()) == expected_ids

    def test_keyed_by_canonical_id(self):
        missions = load_all_missions()
        for mission_id, mission in missions.items():
            assert mission["id"] == mission_id


class TestGetMissionIds:
    def test_returns_sorted_ids(self):
        ids = get_mission_ids()
        assert len(ids) == 8
        assert ids == sorted(ids)
        assert "PSP" in ids
        assert "ACE" in ids


class TestGetRoutingTable:
    def test_returns_all_missions(self):
        table = get_routing_table()
        assert len(table) == 8

    def test_entry_structure(self):
        table = get_routing_table()
        for entry in table:
            assert "id" in entry
            assert "name" in entry
            assert "capabilities" in entry
            assert isinstance(entry["capabilities"], list)

    def test_psp_capabilities(self):
        table = get_routing_table()
        psp = next(e for e in table if e["id"] == "PSP")
        assert "magnetic field" in psp["capabilities"]
        assert "plasma" in psp["capabilities"]

    def test_omni_has_geomagnetic(self):
        table = get_routing_table()
        omni = next(e for e in table if e["id"] == "OMNI")
        assert "geomagnetic indices" in omni["capabilities"]


class TestGetMissionDatasets:
    def test_psp_all_datasets(self):
        datasets = get_mission_datasets("PSP")
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in datasets
        assert "PSP_SWP_SPC_L3I" in datasets

    def test_psp_primary_only(self):
        datasets = get_mission_datasets("PSP", tier="primary")
        assert len(datasets) >= 2
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in datasets

    def test_psp_advanced_empty_initially(self):
        datasets = get_mission_datasets("PSP", tier="advanced")
        assert len(datasets) == 0  # No advanced datasets in initial JSON

    def test_nonexistent_mission_raises(self):
        with pytest.raises(FileNotFoundError):
            get_mission_datasets("NONEXISTENT")


class TestJsonFileIntegrity:
    """Verify all mission JSON files are well-formed."""

    def test_all_json_files_parse(self):
        for filepath in _MISSIONS_DIR.glob("*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "id" in data
            assert "name" in data
            assert "keywords" in data
            assert "instruments" in data

    def test_all_json_files_have_meta(self):
        for filepath in _MISSIONS_DIR.glob("*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "_meta" in data
            assert "hapi_server" in data["_meta"]

    def test_all_datasets_have_tier(self):
        for filepath in _MISSIONS_DIR.glob("*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for inst in data["instruments"].values():
                for ds_id, ds_info in inst["datasets"].items():
                    assert "tier" in ds_info, f"{data['id']}/{ds_id} missing tier"
                    assert ds_info["tier"] in ("primary", "advanced"), \
                        f"{data['id']}/{ds_id} has invalid tier: {ds_info['tier']}"
