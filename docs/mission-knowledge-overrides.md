# Mission Knowledge Override System

## Problem

All data in `knowledge/missions/` is auto-generated from CDAWeb and disposable (`--force` rebuild). The agent learns knowledge through interaction with data — caveats, notes, corrections, instrument descriptions. This knowledge must survive rebuilds.

## Solution

Learned knowledge is saved as sparse JSON patch files in `{data_dir}/mission_overrides/`. The merge is delayed — it happens at load time, transparently to downstream code.

## Override Locations

- **Mission-level**: `{data_dir}/mission_overrides/{stem}.json`
- **Dataset-level**: `{data_dir}/mission_overrides/{stem}/{dataset_id}.json`

Where `{stem}` is the lowercase mission file stem (e.g., `psp`, `ace`, `solo`).

## Format

Sparse patch — only fields that differ from the auto-generated base. Example mission override:

```json
{
  "profile": {
    "description": "Parker Solar Probe explores the Sun's corona.",
    "data_caveats": ["SPC data unreliable below 0.2 AU"]
  },
  "instruments": {
    "FIELDS": {
      "name": "FIELDS Suite"
    }
  }
}
```

Example dataset override (`psp/PSP_FLD_L2_MAG_RTN_1MIN.json`):

```json
{
  "_custom_note": "Use RTN coordinates for solar wind studies",
  "parameters": [
    {
      "name": "psp_fld_l2_mag_RTN_1min",
      "_note": "3-component RTN magnetic field vector"
    }
  ]
}
```

## Merge Rules

Generic recursive deep-merge (`_deep_merge`):

- If both values are dicts → merge recursively
- Otherwise → patch value replaces base value

No allowlist. Any field can be patched.

## Read Path

### Mission-level (in `mission_loader.py`)

`load_mission(mission_id)`:
1. Load base JSON from `knowledge/missions/{stem}.json`
2. Load override from `{overrides_dir}/{stem}.json` (if exists)
3. Deep-merge override on top of base
4. Cache the merged result

### Dataset-level (in `metadata_client.py`)

`get_dataset_info(dataset_id)`:
1. Load base from local file cache or Master CDF
2. Load dataset override from `{overrides_dir}/{stem}/{dataset_id}.json` (if exists)
3. Deep-merge override on top of base
4. Cache the merged result

## Write Path

Both use read-modify-write pattern:

- `update_mission_override(cache_key, patch)` — merges into existing mission override, writes, invalidates cache
- `update_dataset_override(dataset_id, patch, mission_stem=None)` — merges into existing dataset override, writes, invalidates cache

## Management

```bash
python scripts/manage_overrides.py list              # Show all override files
python scripts/manage_overrides.py show <mission>    # Pretty-print an override
python scripts/manage_overrides.py validate <mission> # Check for issues
```
