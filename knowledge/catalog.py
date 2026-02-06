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
