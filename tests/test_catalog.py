"""
Tests for the spacecraft catalog and keyword matching.

Run with: python -m pytest tests/test_catalog.py
"""

import pytest
from knowledge.catalog import (
    SPACECRAFT,
    list_spacecraft,
    list_instruments,
    get_datasets,
    match_spacecraft,
    match_instrument,
    search_by_keywords,
)


class TestListSpacecraft:
    def test_returns_all_spacecraft(self):
        spacecraft = list_spacecraft()
        assert len(spacecraft) == 8
        ids = [s["id"] for s in spacecraft]
        assert "PSP" in ids
        assert "SolO" in ids
        assert "ACE" in ids
        assert "OMNI" in ids
        assert "WIND" in ids
        assert "DSCOVR" in ids
        assert "MMS" in ids
        assert "STEREO_A" in ids

    def test_returns_dicts_with_id_and_name(self):
        spacecraft = list_spacecraft()
        for s in spacecraft:
            assert "id" in s
            assert "name" in s


class TestListInstruments:
    def test_psp_instruments(self):
        instruments = list_instruments("PSP")
        assert len(instruments) == 5
        ids = [i["id"] for i in instruments]
        assert "FIELDS/MAG" in ids
        assert "SWEAP" in ids
        assert "SWEAP/SPAN-I" in ids
        assert "SWEAP/SPAN-E" in ids
        assert "ISOIS" in ids

    def test_ace_instruments(self):
        instruments = list_instruments("ACE")
        assert len(instruments) == 2
        ids = [i["id"] for i in instruments]
        assert "MAG" in ids
        assert "SWEPAM" in ids

    def test_invalid_spacecraft_returns_empty(self):
        instruments = list_instruments("INVALID")
        assert instruments == []


class TestGetDatasets:
    def test_psp_mag_datasets(self):
        datasets = get_datasets("PSP", "FIELDS/MAG")
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in datasets

    def test_ace_mag_datasets(self):
        datasets = get_datasets("ACE", "MAG")
        assert "AC_H2_MFI" in datasets

    def test_invalid_returns_empty(self):
        assert get_datasets("INVALID", "MAG") == []
        assert get_datasets("PSP", "INVALID") == []


class TestMatchSpacecraft:
    @pytest.mark.parametrize("query,expected", [
        ("parker", "PSP"),
        ("PARKER", "PSP"),
        ("psp", "PSP"),
        ("PSP", "PSP"),
        ("solar probe", "PSP"),
        ("solar orbiter", "SolO"),
        ("solo", "SolO"),
        ("orbiter", "SolO"),
        ("ace", "ACE"),
        ("ACE", "ACE"),
        ("omni", "OMNI"),
        ("wind", "WIND"),
        ("WIND", "WIND"),
        ("dscovr", "DSCOVR"),
        ("mms", "MMS"),
        ("MMS", "MMS"),
        ("magnetospheric multiscale", "MMS"),
        ("stereo", "STEREO_A"),
        ("stereo-a", "STEREO_A"),
        ("stereo a", "STEREO_A"),
        ("unknown", None),
        ("xyz123", None),
    ])
    def test_match_spacecraft(self, query, expected):
        assert match_spacecraft(query) == expected


class TestMatchInstrument:
    @pytest.mark.parametrize("spacecraft,query,expected", [
        ("PSP", "magnetic field", "FIELDS/MAG"),
        ("PSP", "mag", "FIELDS/MAG"),
        ("PSP", "plasma", "SWEAP"),
        ("PSP", "density", "SWEAP"),
        ("PSP", "velocity", "SWEAP"),
        ("PSP", "span", "SWEAP/SPAN-I"),
        ("PSP", "span-i", "SWEAP/SPAN-I"),
        ("PSP", "ion spectrometer", "SWEAP/SPAN-I"),
        ("PSP", "span-e", "SWEAP/SPAN-E"),
        ("PSP", "electron", "SWEAP/SPAN-E"),
        ("PSP", "isois", "ISOIS"),
        ("PSP", "energetic particle", "ISOIS"),
        ("PSP", "epi-hi", "ISOIS"),
        ("SolO", "magnetic", "MAG"),
        ("SolO", "proton", "SWA-PAS"),
        ("ACE", "magnetic", "MAG"),
        ("ACE", "imf", "MAG"),
        ("ACE", "solar wind", "SWEPAM"),
        ("PSP", "unknown", None),
    ])
    def test_match_instrument(self, spacecraft, query, expected):
        assert match_instrument(spacecraft, query) == expected

    def test_invalid_spacecraft_returns_none(self):
        assert match_instrument("INVALID", "magnetic") is None


class TestSearchByKeywords:
    def test_parker_magnetic(self):
        result = search_by_keywords("parker magnetic field")
        assert result is not None
        assert result["spacecraft"] == "PSP"
        assert result["instrument"] == "FIELDS/MAG"
        assert "PSP_FLD_L2_MAG_RTN_1MIN" in result["datasets"]

    def test_ace_solar_wind(self):
        result = search_by_keywords("ace solar wind")
        assert result is not None
        assert result["spacecraft"] == "ACE"
        assert result["instrument"] == "SWEPAM"
        assert "AC_H0_SWE" in result["datasets"]

    def test_spacecraft_only_no_instrument(self):
        result = search_by_keywords("parker")
        assert result is not None
        assert result["spacecraft"] == "PSP"
        # No instrument keyword, so no instrument match
        assert result["instrument"] is None
        assert result["datasets"] == []
        assert "available_instruments" in result

    def test_no_match(self):
        result = search_by_keywords("nonexistent mission xyz123")
        assert result is None

    def test_omni_combined(self):
        result = search_by_keywords("omni solar wind")
        assert result is not None
        assert result["spacecraft"] == "OMNI"
        assert result["instrument"] == "Combined"
        assert "OMNI_HRO_1MIN" in result["datasets"]

    def test_wind_magnetic(self):
        result = search_by_keywords("wind magnetic")
        assert result is not None
        assert result["spacecraft"] == "WIND"
        assert "WI_H2_MFI" in result["datasets"]

    def test_dscovr_plasma(self):
        result = search_by_keywords("dscovr plasma")
        assert result is not None
        assert result["spacecraft"] == "DSCOVR"
        assert result["instrument"] == "FC"

    def test_mms_magnetic(self):
        result = search_by_keywords("mms magnetic")
        assert result is not None
        assert result["spacecraft"] == "MMS"
        assert "MMS1_FGM_SRVY_L2@0" in result["datasets"]

    def test_stereo_magnetic(self):
        result = search_by_keywords("stereo magnetic")
        assert result is not None
        assert result["spacecraft"] == "STEREO_A"
        assert "STA_L1_MAG_RTN" in result["datasets"]
