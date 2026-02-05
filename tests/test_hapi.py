"""
Tests for the HAPI client.

Run with: python -m pytest tests/test_hapi.py

Note: These tests require network access to CDAWeb HAPI server.
Some tests are marked slow and can be skipped with: pytest -m "not slow"
"""

import pytest
from knowledge.hapi_client import (
    get_dataset_info,
    list_parameters,
    get_dataset_time_range,
    clear_cache,
)


@pytest.fixture(autouse=True)
def clear_hapi_cache():
    """Clear cache before each test."""
    clear_cache()


class TestGetDatasetInfo:
    @pytest.mark.slow
    def test_fetch_psp_mag_info(self):
        """Test fetching PSP MAG dataset info from HAPI."""
        info = get_dataset_info("PSP_FLD_L2_MAG_RTN_1MIN")
        assert info is not None
        assert "startDate" in info
        assert "stopDate" in info
        assert "parameters" in info
        assert len(info["parameters"]) > 0

    @pytest.mark.slow
    def test_fetch_ace_mag_info(self):
        """Test fetching ACE MAG dataset info from HAPI."""
        info = get_dataset_info("AC_H2_MFI")
        assert info is not None
        assert "parameters" in info

    def test_caching_works(self):
        """Test that caching prevents duplicate requests."""
        # First call populates cache
        info1 = get_dataset_info("AC_H2_MFI")
        # Second call should use cache
        info2 = get_dataset_info("AC_H2_MFI", use_cache=True)
        assert info1 == info2

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset ID raises an error."""
        with pytest.raises(Exception):
            get_dataset_info("INVALID_DATASET_XYZ_123")


class TestListParameters:
    @pytest.mark.slow
    def test_list_psp_mag_parameters(self):
        """Test listing parameters for PSP MAG."""
        params = list_parameters("PSP_FLD_L2_MAG_RTN_1MIN")
        assert len(params) > 0

        # Check parameter structure
        for p in params:
            assert "name" in p
            assert "dataset_id" in p
            assert p["dataset_id"] == "PSP_FLD_L2_MAG_RTN_1MIN"

    @pytest.mark.slow
    def test_parameters_are_1d(self):
        """Test that returned parameters are 1D with size <= 3."""
        params = list_parameters("AC_H2_MFI")
        for p in params:
            assert len(p["size"]) == 1
            assert p["size"][0] <= 3

    @pytest.mark.slow
    def test_excludes_time_parameter(self):
        """Test that Time parameter is excluded."""
        params = list_parameters("AC_H2_MFI")
        names = [p["name"].lower() for p in params]
        assert "time" not in names

    def test_invalid_dataset_returns_empty(self):
        """Test that invalid dataset returns empty list."""
        params = list_parameters("INVALID_DATASET_XYZ")
        assert params == []


class TestGetDatasetTimeRange:
    @pytest.mark.slow
    def test_get_time_range(self):
        """Test getting dataset time range."""
        time_range = get_dataset_time_range("AC_H2_MFI")
        assert time_range is not None
        assert "start" in time_range
        assert "stop" in time_range
        assert time_range["start"] is not None

    def test_invalid_dataset_returns_none(self):
        """Test that invalid dataset returns None."""
        time_range = get_dataset_time_range("INVALID_XYZ")
        assert time_range is None
