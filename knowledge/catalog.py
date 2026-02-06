"""
Static spacecraft/instrument catalog with keyword matching.

Provides fast local lookup to map natural language queries to CDAWeb dataset IDs.
HAPI API is used separately to fetch parameter metadata.
"""

from typing import Optional


SPACECRAFT = {
    "PSP": {
        "name": "Parker Solar Probe",
        "keywords": ["parker", "psp", "probe", "solar probe"],
        "profile": {
            "description": "Inner heliosphere probe studying the solar corona and young solar wind",
            "coordinate_systems": ["RTN"],
            "typical_cadence": "1-minute",
            "data_caveats": ["RTN frame rotates with spacecraft orbital position"],
            "analysis_patterns": [
                "Switchback detection: compute radial component sign changes in Br",
                "Parker spiral angle: atan2(Bt, Br) compared to expected spiral",
                "Perihelion passes: check distance and compare field magnitude",
            ],
        },
        "instruments": {
            "FIELDS/MAG": {
                "name": "FIELDS Magnetometer",
                "keywords": ["magnetic", "field", "mag", "b-field", "bfield"],
                "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"],
            },
            "SWEAP": {
                "name": "Solar Wind Plasma",
                "keywords": ["plasma", "solar wind", "proton", "density", "velocity", "sweap"],
                "datasets": ["PSP_SWP_SPC_L3I"],
            },
        },
    },
    "SolO": {
        "name": "Solar Orbiter",
        "keywords": ["solar orbiter", "solo", "orbiter"],
        "profile": {
            "description": "ESA/NASA mission studying the Sun from close range with in-situ and remote sensing",
            "coordinate_systems": ["RTN"],
            "typical_cadence": "1-minute",
            "data_caveats": ["RTN frame; some data gaps during commissioning periods"],
            "analysis_patterns": [
                "Compare with PSP for radial evolution of solar wind",
                "Check B magnitude at varying heliocentric distances",
            ],
        },
        "instruments": {
            "MAG": {
                "name": "Magnetometer",
                "keywords": ["magnetic", "field", "mag", "b-field"],
                "datasets": ["SOLO_L2_MAG-RTN-NORMAL-1-MINUTE"],
            },
            "SWA-PAS": {
                "name": "Proton-Alpha Sensor",
                "keywords": ["plasma", "proton", "density", "velocity", "temperature", "swa"],
                "datasets": ["SOLO_L2_SWA-PAS-GRND-MOM"],
            },
        },
    },
    "ACE": {
        "name": "Advanced Composition Explorer",
        "keywords": ["ace", "advanced composition"],
        "profile": {
            "description": "L1 monitor for solar wind and interplanetary magnetic field since 1997",
            "coordinate_systems": ["GSE"],
            "typical_cadence": "1-minute (16-second available)",
            "data_caveats": ["GSE coordinates; long baseline ideal for solar cycle studies"],
            "analysis_patterns": [
                "IMF sector structure: check Bx sign for toward/away",
                "Solar wind speed categorization: slow (<400) vs fast (>600) km/s",
                "Upstream monitor: compare with OMNI propagated data",
            ],
        },
        "instruments": {
            "MAG": {
                "name": "Magnetometer",
                "keywords": ["magnetic", "field", "mag", "b-field", "imf"],
                "datasets": ["AC_H2_MFI"],
            },
            "SWEPAM": {
                "name": "Solar Wind Plasma",
                "keywords": ["plasma", "solar wind", "proton", "density", "velocity", "temperature"],
                "datasets": ["AC_H0_SWE"],
            },
        },
    },
    "OMNI": {
        "name": "OMNI Combined Data",
        "keywords": ["omni", "combined", "propagated", "bow shock"],
        "profile": {
            "description": "Multi-spacecraft time-shifted solar wind data propagated to the bow shock nose",
            "coordinate_systems": ["GSE", "GSM"],
            "typical_cadence": "1-minute (5-minute also available)",
            "data_caveats": [
                "Combined from multiple L1 monitors; source spacecraft varies",
                "Some empty-string fill values in CSV; use coerce parsing",
            ],
            "analysis_patterns": [
                "Geomagnetic coupling: correlate Bz with SYM-H",
                "Solar wind driver identification: speed + density + B for CME/CIR",
                "Reference dataset for multi-spacecraft comparisons",
            ],
        },
        "instruments": {
            "Combined": {
                "name": "Multi-spacecraft Combined",
                "keywords": ["solar wind", "imf", "magnetic", "density", "velocity", "sym-h", "geomagnetic"],
                "datasets": ["OMNI_HRO_1MIN"],
            },
        },
    },
    "WIND": {
        "name": "Wind",
        "keywords": ["wind"],
        "profile": {
            "description": "L1 solar wind monitor operating since 1994; complements ACE",
            "coordinate_systems": ["GSE"],
            "typical_cadence": "1-minute (3-second burst available)",
            "data_caveats": ["GSE coordinates; longest continuous L1 dataset"],
            "analysis_patterns": [
                "Cross-calibrate with ACE MAG for same events",
                "Long-term solar cycle trends",
            ],
        },
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
        "profile": {
            "description": "NOAA L1 real-time solar wind monitor (successor to ACE for space weather)",
            "coordinate_systems": ["GSE"],
            "typical_cadence": "1-minute",
            "data_caveats": ["Real-time data may have quality flags; check for fill values"],
            "analysis_patterns": [
                "Real-time space weather monitoring",
                "Compare with ACE for cross-validation at L1",
            ],
        },
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
        "profile": {
            "description": "Four-spacecraft constellation studying magnetic reconnection in Earth's magnetosphere",
            "coordinate_systems": ["GSE", "GSM"],
            "typical_cadence": "Survey: 4.5s (FGM), Fast: 30ms (burst)",
            "data_caveats": [
                "Magnetospheric orbit — not solar wind data",
                "Very high cadence burst data available for reconnection events",
            ],
            "analysis_patterns": [
                "Reconnection signatures: look for B reversal + jet in V",
                "Magnetopause crossings: rapid B direction changes",
                "Multi-spacecraft timing using MMS1-4",
            ],
        },
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
        "profile": {
            "description": "Off-Sun-Earth-line observatory in heliocentric orbit ahead of Earth",
            "coordinate_systems": ["RTN"],
            "typical_cadence": "1-minute",
            "data_caveats": ["Off-Sun-Earth line; longitude separation from Earth varies over time"],
            "analysis_patterns": [
                "Multi-point solar wind: compare with L1 monitors for same CME/CIR",
                "Longitude-separated observations for heliospheric structure",
            ],
        },
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
}


def list_spacecraft() -> list[dict]:
    """List all supported spacecraft.

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    return [
        {"id": sc_id, "name": info["name"]}
        for sc_id, info in SPACECRAFT.items()
    ]


def list_instruments(spacecraft: str) -> list[dict]:
    """List instruments for a spacecraft.

    Args:
        spacecraft: Spacecraft ID (e.g., "PSP", "ACE")

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return []
    return [
        {"id": inst_id, "name": info["name"]}
        for inst_id, info in sc["instruments"].items()
    ]


def get_datasets(spacecraft: str, instrument: str) -> list[str]:
    """Get dataset IDs for a spacecraft/instrument combination.

    Args:
        spacecraft: Spacecraft ID
        instrument: Instrument ID

    Returns:
        List of CDAWeb dataset IDs.
    """
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return []
    inst = sc["instruments"].get(instrument)
    if not inst:
        return []
    return inst["datasets"]


def match_spacecraft(query: str) -> Optional[str]:
    """Match a query string to a spacecraft using keywords.

    Args:
        query: User's search query

    Returns:
        Spacecraft ID or None if no match.
    """
    query_lower = query.lower()

    for sc_id, info in SPACECRAFT.items():
        # Check exact match on ID
        if query_lower == sc_id.lower():
            return sc_id
        # Check keywords
        for kw in info["keywords"]:
            if kw in query_lower:
                return sc_id

    return None


def match_instrument(spacecraft: str, query: str) -> Optional[str]:
    """Match a query string to an instrument using keywords.

    Args:
        spacecraft: Spacecraft ID to search within
        query: User's search query

    Returns:
        Instrument ID or None if no match.
    """
    sc = SPACECRAFT.get(spacecraft)
    if not sc:
        return None

    query_lower = query.lower()

    for inst_id, info in sc["instruments"].items():
        # Check exact match on ID
        if query_lower == inst_id.lower():
            return inst_id
        # Check keywords
        for kw in info["keywords"]:
            if kw in query_lower:
                return inst_id

    return None


def search_by_keywords(query: str) -> Optional[dict]:
    """Combined search: find spacecraft, instrument, and datasets from a query.

    This is the main entry point for natural language dataset search.

    Args:
        query: User's natural language query (e.g., "parker magnetic field")

    Returns:
        Dict with spacecraft, instrument, datasets, or None if no match.
        Example: {
            "spacecraft": "PSP",
            "spacecraft_name": "Parker Solar Probe",
            "instrument": "FIELDS/MAG",
            "instrument_name": "FIELDS Magnetometer",
            "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"]
        }
    """
    # Step 1: Match spacecraft
    spacecraft = match_spacecraft(query)
    if not spacecraft:
        return None

    sc_info = SPACECRAFT[spacecraft]

    # Step 2: Match instrument
    instrument = match_instrument(spacecraft, query)
    if not instrument:
        # No instrument match - return spacecraft info but no datasets
        return {
            "spacecraft": spacecraft,
            "spacecraft_name": sc_info["name"],
            "instrument": None,
            "instrument_name": None,
            "datasets": [],
            "available_instruments": list_instruments(spacecraft),
        }

    inst_info = sc_info["instruments"][instrument]
    datasets = inst_info["datasets"]

    return {
        "spacecraft": spacecraft,
        "spacecraft_name": sc_info["name"],
        "instrument": instrument,
        "instrument_name": inst_info["name"],
        "datasets": datasets,
    }
