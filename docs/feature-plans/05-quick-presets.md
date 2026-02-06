# Feature 05: Quick Plot Presets

## Summary

Recognize high-level requests like "show me a solar wind overview" and automatically execute a multi-parameter visualization (|B|, V, N, T) — like a one-command dashboard.

## Motivation

Scientists commonly want a "standard view" of solar wind conditions. Currently this requires multiple fetch/compute/plot steps. A preset collapses this into a single natural language command, making demos impressive and daily use faster.

## Approach

This feature is implemented **entirely through the system prompt** — no new tools needed. The LLM already has the ability to call `plot_data` multiple times. We just teach it to recognize preset requests and execute the right sequence.

However, to make it reliable and fast, we add a lightweight `get_preset` tool that returns the parameter list for a named preset, so the LLM doesn't have to memorize dataset IDs.

## Files to Modify

### 1. `agent/tools.py` — Add preset tool

```python
{
    "name": "get_preset",
    "description": """Get a predefined set of parameters for a common visualization.
Use this when the user asks for an "overview", "dashboard", "summary", or "standard view" of a spacecraft or data type.

Returns a list of dataset_id + parameter_id pairs to plot together.
Available presets: solar_wind_overview, magnetic_field_overview, geomagnetic_overview""",
    "parameters": {
        "type": "object",
        "properties": {
            "preset_name": {
                "type": "string",
                "description": "Name of the preset (e.g., 'solar_wind_overview')"
            },
            "spacecraft": {
                "type": "string",
                "description": "Optional spacecraft preference (e.g., 'ACE', 'OMNI'). Defaults to OMNI."
            }
        },
        "required": ["preset_name"]
    }
}
```

### 2. New file: `knowledge/presets.py` — Preset definitions

```python
"""
Predefined parameter sets for common visualization patterns.
"""

PRESETS = {
    "solar_wind_overview": {
        "description": "Standard solar wind overview: magnetic field magnitude, velocity, density, temperature",
        "parameters": {
            "OMNI": [
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "flow_speed", "label": "V (km/s)"},
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "proton_density", "label": "N (cm^-3)"},
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "T", "label": "T (K)"},
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "F", "label": "|B| (nT)"},
            ],
            "ACE": [
                {"dataset_id": "AC_H0_SWE", "parameter_id": "Vp", "label": "V (km/s)"},
                {"dataset_id": "AC_H0_SWE", "parameter_id": "Np", "label": "N (cm^-3)"},
                {"dataset_id": "AC_H2_MFI", "parameter_id": "Magnitude", "label": "|B| (nT)"},
            ],
        },
    },
    "magnetic_field_overview": {
        "description": "Magnetic field components + magnitude",
        "parameters": {
            "ACE": [
                {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc", "label": "B GSE (nT)"},
                {"dataset_id": "AC_H2_MFI", "parameter_id": "Magnitude", "label": "|B| (nT)"},
            ],
            "PSP": [
                {"dataset_id": "PSP_FLD_L2_MAG_RTN_1MIN", "parameter_id": "psp_fld_l2_mag_RTN_1min", "label": "B RTN (nT)"},
            ],
        },
    },
    "geomagnetic_overview": {
        "description": "Geomagnetic indices and IMF conditions",
        "parameters": {
            "OMNI": [
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "SYM_H", "label": "SYM-H (nT)"},
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "BZ_GSM", "label": "Bz GSM (nT)"},
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "flow_speed", "label": "V (km/s)"},
                {"dataset_id": "OMNI_HRO_1MIN", "parameter_id": "proton_density", "label": "N (cm^-3)"},
            ],
        },
    },
}


def get_preset(name: str, spacecraft: str = "OMNI") -> dict | None:
    """Look up a preset by name.

    Args:
        name: Preset name (e.g., 'solar_wind_overview')
        spacecraft: Preferred spacecraft. Falls back to first available.

    Returns:
        Dict with 'description' and 'parameters' list, or None if not found.
    """
    preset = PRESETS.get(name)
    if preset is None:
        return None

    sc = spacecraft.upper()
    params = preset["parameters"].get(sc)
    if params is None:
        # Fall back to first available spacecraft
        first_key = next(iter(preset["parameters"]))
        params = preset["parameters"][first_key]
        sc = first_key

    return {
        "preset_name": name,
        "description": preset["description"],
        "spacecraft": sc,
        "parameters": params,
        "available_spacecraft": list(preset["parameters"].keys()),
    }


def list_presets() -> list[dict]:
    """List all available presets."""
    return [
        {"name": name, "description": info["description"]}
        for name, info in PRESETS.items()
    ]
```

### 3. `agent/core.py` — Add handler

```python
elif tool_name == "get_preset":
    from knowledge.presets import get_preset as lookup_preset
    result = lookup_preset(
        tool_args["preset_name"],
        tool_args.get("spacecraft", "OMNI"),
    )
    if result is None:
        from knowledge.presets import list_presets
        available = list_presets()
        return {
            "status": "error",
            "message": f"Unknown preset '{tool_args['preset_name']}'",
            "available_presets": available,
        }
    return {"status": "success", **result}
```

### 4. `agent/prompts.py` — Add preset instructions

Add to the system prompt:

```
## Quick Presets

When the user asks for an "overview", "dashboard", or "summary view", use `get_preset` to look up the standard parameter set, then plot each parameter using `plot_data` with the user's time range.

Available presets:
- **solar_wind_overview** — |B|, V, N, T (OMNI or ACE)
- **magnetic_field_overview** — B components + magnitude
- **geomagnetic_overview** — SYM-H, Bz, V, N (OMNI)

After getting the preset, call `plot_data` for the first parameter to establish the canvas, then for each subsequent parameter. The user sees a multi-panel view.
```

## Verification

Before implementing, verify each OMNI parameter ID exists:

```python
import requests
r = requests.get("https://cdaweb.gsfc.nasa.gov/hapi/info?id=OMNI_HRO_1MIN")
params = [p["name"] for p in r.json()["parameters"]]
# Check: "flow_speed", "proton_density", "T", "F", "SYM_H", "BZ_GSM" all in params
```

## Testing

```python
def test_get_preset_solar_wind():
    result = get_preset("solar_wind_overview", "OMNI")
    assert result is not None
    assert len(result["parameters"]) == 4

def test_get_preset_fallback_spacecraft():
    result = get_preset("solar_wind_overview", "NONEXISTENT")
    assert result is not None  # Falls back to first available

def test_get_preset_unknown():
    result = get_preset("nonexistent")
    assert result is None

def test_list_presets():
    presets = list_presets()
    assert len(presets) >= 3
```

## Demo Script

```
You: Show me a solar wind overview for last week
Agent: [calls get_preset("solar_wind_overview")]
       [calls plot_data for |B|, V, N, T in sequence]
       Here's the solar wind overview for the past week using OMNI data:
       - Panel 1: Magnetic field magnitude (|B|)
       - Panel 2: Solar wind velocity
       - Panel 3: Proton density
       - Panel 4: Temperature
       The data shows typical quiet solar wind conditions with V~400 km/s.
```

## Notes

- The LLM must call `plot_data` multiple times sequentially. This relies on the multi-panel behavior of Autoplot when plotting different URIs.
- If Autoplot doesn't support automatic multi-panel, this could instead use `fetch_data` + `plot_computed_data` with overplotting (less ideal but works).
- Preset definitions should be verified against the live HAPI catalog before implementation.
