"""
Tests for local HAPI metadata cache functionality.

Tests the local file cache in hapi_client.py: _find_local_cache(),
get_dataset_info() with local files, list_parameters from cache,
and list_cached_datasets().

Run with: python -m pytest tests/test_hapi_cache.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from knowledge.hapi_client import (
    _find_local_cache,
    get_dataset_info,
    list_parameters,
    list_cached_datasets,
    clear_cache,
)


# Sample HAPI /info response for testing
SAMPLE_HAPI_INFO = {
    "HAPI": "3.0",
    "startDate": "2018-10-06T00:00:00.000Z",
    "stopDate": "2025-12-31T00:00:00.000Z",
    "parameters": [
        {"name": "Time", "type": "isotime", "units": "UTC", "length": 24},
        {
            "name": "psp_fld_l2_mag_RTN_1min",
            "type": "double",
            "units": "nT",
            "size": [3],
            "description": "Magnetic field in RTN coordinates",
        },
        {
            "name": "psp_fld_l2_quality_flags",
            "type": "integer",
            "units": None,
            "size": [1],
            "description": "Quality flags",
        },
    ],
}

SAMPLE_INDEX = {
    "mission_id": "PSP",
    "dataset_count": 2,
    "generated_at": "2026-02-07T00:00:00Z",
    "datasets": [
        {
            "id": "PSP_FLD_L2_MAG_RTN_1MIN",
            "description": "PSP FIELDS Magnetometer 1-min RTN",
            "start_date": "2018-10-06",
            "stop_date": "2025-12-31",
            "parameter_count": 2,
            "instrument": "FIELDS/MAG",
        },
        {
            "id": "PSP_SWP_SPC_L3I",
            "description": "PSP SWEAP SPC Level 3i",
            "start_date": "2018-10-06",
            "stop_date": "2025-12-31",
            "parameter_count": 5,
            "instrument": "SWEAP",
        },
    ],
}


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear in-memory cache before each test."""
    clear_cache()


@pytest.fixture
def fake_missions_dir(tmp_path):
    """Create a temporary missions directory with cache files."""
    missions_dir = tmp_path / "missions"

    # Create PSP cache
    psp_hapi = missions_dir / "psp" / "hapi"
    psp_hapi.mkdir(parents=True)

    # Write sample HAPI /info file
    cache_file = psp_hapi / "PSP_FLD_L2_MAG_RTN_1MIN.json"
    cache_file.write_text(json.dumps(SAMPLE_HAPI_INFO), encoding="utf-8")

    # Write _index.json
    index_file = psp_hapi / "_index.json"
    index_file.write_text(json.dumps(SAMPLE_INDEX), encoding="utf-8")

    # Also create a non-directory file (the psp.json mission file)
    mission_file = missions_dir / "psp.json"
    mission_file.write_text("{}", encoding="utf-8")

    return missions_dir


class TestFindLocalCache:
    def test_returns_path_when_exists(self, fake_missions_dir):
        """_find_local_cache returns the path when a cache file exists."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            result = _find_local_cache("PSP_FLD_L2_MAG_RTN_1MIN")
            assert result is not None
            assert result.name == "PSP_FLD_L2_MAG_RTN_1MIN.json"
            assert result.exists()

    def test_returns_none_when_missing(self, fake_missions_dir):
        """_find_local_cache returns None for a dataset not in cache."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            result = _find_local_cache("NONEXISTENT_DATASET")
            assert result is None

    def test_skips_non_directory_entries(self, fake_missions_dir):
        """_find_local_cache skips files like psp.json (not directories)."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            # Should not crash even though psp.json exists at missions_dir level
            result = _find_local_cache("PSP_FLD_L2_MAG_RTN_1MIN")
            assert result is not None


class TestGetDatasetInfoLocalCache:
    def test_uses_local_cache(self, fake_missions_dir):
        """get_dataset_info loads from local file, no network needed."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            info = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
            assert info is not None
            assert info["startDate"] == "2018-10-06T00:00:00.000Z"
            assert len(info["parameters"]) == 3

    def test_local_cache_populates_memory_cache(self, fake_missions_dir):
        """Reading from local file also populates the in-memory cache."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            info1 = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
            # Delete the file to prove second call uses memory
            cache_file = fake_missions_dir / "psp" / "hapi" / "PSP_FLD_L2_MAG_RTN_1MIN.json"
            cache_file.unlink()
            info2 = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
            assert info1 == info2

    def test_use_cache_false_skips_local(self, fake_missions_dir):
        """use_cache=False skips both memory and local file, hits network."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir), \
             patch("knowledge.hapi_client.requests.get") as mock_get:
            # Simulate network call
            mock_get.return_value.status_code = 200
            mock_get.return_value.raise_for_status = lambda: None
            mock_get.return_value.json.return_value = {"parameters": []}
            get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN", use_cache=False)
            # Network SHOULD have been called because use_cache=False
            mock_get.assert_called_once()


class TestListParametersFromCache:
    def test_lists_parameters_from_local_cache(self, fake_missions_dir):
        """list_parameters works with locally cached HAPI /info."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            params = list_parameters("PSP_FLD_L2_MAG_RTN_1MIN")
            assert len(params) == 2
            names = [p["name"] for p in params]
            assert "psp_fld_l2_mag_RTN_1min" in names
            assert "psp_fld_l2_quality_flags" in names
            # Time should be excluded
            assert "Time" not in names

    def test_parameter_structure(self, fake_missions_dir):
        """Parameters from cache have correct structure."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            params = list_parameters("PSP_FLD_L2_MAG_RTN_1MIN")
            mag_param = next(p for p in params if p["name"] == "psp_fld_l2_mag_RTN_1min")
            assert mag_param["units"] == "nT"
            assert mag_param["size"] == [3]
            assert mag_param["dataset_id"] == "PSP_FLD_L2_MAG_RTN_1MIN"

    def test_returns_empty_for_uncached_invalid_dataset(self, fake_missions_dir):
        """list_parameters returns empty list for dataset not in cache or network."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            params = list_parameters("NONEXISTENT_DATASET_XYZ")
            assert params == []


class TestListCachedDatasets:
    def test_loads_index(self, fake_missions_dir):
        """list_cached_datasets loads the _index.json correctly."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            index = list_cached_datasets("PSP")
            assert index is not None
            assert index["mission_id"] == "PSP"
            assert index["dataset_count"] == 2
            assert len(index["datasets"]) == 2

    def test_case_insensitive(self, fake_missions_dir):
        """Mission ID lookup is case-insensitive."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            index = list_cached_datasets("psp")
            assert index is not None
            assert index["mission_id"] == "PSP"

    def test_returns_none_when_no_index(self, fake_missions_dir):
        """Returns None when no _index.json exists for the mission."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            result = list_cached_datasets("NONEXISTENT")
            assert result is None

    def test_index_dataset_entries(self, fake_missions_dir):
        """Index entries have expected fields."""
        with patch("knowledge.hapi_client._MISSIONS_DIR", fake_missions_dir):
            index = list_cached_datasets("PSP")
            ds = index["datasets"][0]
            assert "id" in ds
            assert "description" in ds
            assert "start_date" in ds
            assert "stop_date" in ds
            assert "parameter_count" in ds
            assert "instrument" in ds
