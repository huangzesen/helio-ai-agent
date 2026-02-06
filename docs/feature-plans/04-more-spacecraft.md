# Feature 04: More Spacecraft in Catalog

## Summary

Add WIND, DSCOVR, MMS, and STEREO-A to the spacecraft catalog. These are popular missions with well-documented CDAWeb dataset IDs.

## Motivation

Going from 4 to 8 supported spacecraft instantly makes the agent more useful and demo-worthy. These missions are frequently used alongside the existing ones (ACE, PSP, Solar Orbiter, OMNI) for multi-spacecraft studies.

## Files to Modify

### 1. `knowledge/catalog.py` — Add spacecraft entries

Add after the OMNI entry in the `SPACECRAFT` dict:

```python
"WIND": {
    "name": "Wind",
    "keywords": ["wind"],
    "instruments": {
        "MFI": {
            "name": "Magnetic Fields Investigation",
            "keywords": ["magnetic", "field", "mag", "b-field", "mfi", "imf"],
            "datasets": ["WI_H2_MFI"],
        },
        "SWE": {
            "name": "Solar Wind Experiment",
            "keywords": ["plasma", "solar wind", "proton", "density", "velocity", "electron", "swe"],
            "datasets": ["WI_H1_SWE"],
        },
    },
},
"DSCOVR": {
    "name": "Deep Space Climate Observatory",
    "keywords": ["dscovr", "deep space", "climate observatory"],
    "instruments": {
        "MAG": {
            "name": "Fluxgate Magnetometer",
            "keywords": ["magnetic", "field", "mag", "b-field", "imf"],
            "datasets": ["DSCOVR_H0_MAG"],
        },
        "FC": {
            "name": "Faraday Cup",
            "keywords": ["plasma", "solar wind", "proton", "density", "velocity", "faraday"],
            "datasets": ["DSCOVR_H1_FC"],
        },
    },
},
"MMS": {
    "name": "Magnetospheric Multiscale",
    "keywords": ["mms", "magnetospheric multiscale"],
    "instruments": {
        "FGM": {
            "name": "Fluxgate Magnetometer",
            "keywords": ["magnetic", "field", "mag", "b-field", "fgm"],
            "datasets": ["MMS1_FGM_SRVY_L2"],
        },
        "FPI-DIS": {
            "name": "Fast Plasma (Ion)",
            "keywords": ["plasma", "ion", "density", "velocity", "fpi"],
            "datasets": ["MMS1_FPI_FAST_L2_DIS-MOMS"],
        },
    },
},
"STEREO_A": {
    "name": "STEREO-A",
    "keywords": ["stereo", "stereo-a", "stereo a", "ahead"],
    "instruments": {
        "MAG": {
            "name": "IMPACT Magnetometer",
            "keywords": ["magnetic", "field", "mag", "b-field", "impact"],
            "datasets": ["STA_L2_MAG_RTN"],
        },
        "PLASTIC": {
            "name": "Plasma and Suprathermal",
            "keywords": ["plasma", "solar wind", "proton", "density", "velocity", "plastic"],
            "datasets": ["STA_L2_PLA_1DMAX_1MIN"],
        },
    },
},
```

### 2. `agent/prompts.py` — Update the spacecraft table

Replace the current spacecraft table in `SYSTEM_PROMPT` with:

```
| Spacecraft | Instruments | Example Data |
|------------|-------------|--------------|
| Parker Solar Probe (PSP) | FIELDS/MAG, SWEAP | Magnetic field, solar wind plasma |
| Solar Orbiter (SolO) | MAG, SWA-PAS | Magnetic field, proton moments |
| ACE | MAG, SWEPAM | IMF, solar wind |
| OMNI | Combined | Multi-spacecraft propagated data |
| Wind | MFI, SWE | Magnetic field, solar wind |
| DSCOVR | MAG, FC | Real-time L1 solar wind |
| MMS | FGM, FPI | High-res magnetospheric data |
| STEREO-A | MAG, PLASTIC | Off-Sun-Earth-line observations |
```

### 3. `docs/capability-summary.md` — Update spacecraft table

Add corresponding rows to the "Supported Spacecraft" table.

## Verification Before Implementation

For each new dataset ID, verify it exists on CDAWeb HAPI by running:

```python
import requests
r = requests.get("https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H2_MFI")
print(r.status_code, r.json().get("parameters", [])[:3])
```

Do this for all 8 new dataset IDs. If any are wrong, find the correct ID from:
https://cdaweb.gsfc.nasa.gov/hapi/catalog

## Testing

Add to `tests/test_catalog.py`:

```python
def test_search_wind():
    result = search_by_keywords("wind magnetic")
    assert result is not None
    assert result["spacecraft"] == "WIND"
    assert "WI_H2_MFI" in result["datasets"]

def test_search_dscovr():
    result = search_by_keywords("dscovr solar wind")
    assert result is not None
    assert result["spacecraft"] == "DSCOVR"

def test_search_mms():
    result = search_by_keywords("mms magnetic")
    assert result is not None
    assert result["spacecraft"] == "MMS"

def test_search_stereo():
    result = search_by_keywords("stereo magnetic")
    assert result is not None
    assert result["spacecraft"] == "STEREO_A"

def test_list_spacecraft_count():
    """Verify all 8 spacecraft are listed."""
    sc = list_spacecraft()
    assert len(sc) == 8
```

## Notes

- **WIND keyword collision**: "wind" is a common English word but unlikely to appear in scientific queries without spacecraft context. The keyword match is whole-word within the query, so "wind speed" would match — this is acceptable since WIND data includes solar wind speed. Monitor for false positives.
- **MMS multi-spacecraft**: MMS has 4 spacecraft (MMS1-4). We start with MMS1 only. Could add a follow-up feature for spacecraft selection.
- **STEREO-B**: STEREO-B lost contact in 2014. Only STEREO-A (ahead) is included.
- **Dataset ID verification**: Always verify against the live HAPI catalog before committing, as CDAWeb dataset IDs occasionally change.
