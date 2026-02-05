# Dataset Knowledge Base Spike

Validates the HAPI client and catalog approach for Step 3 of the implementation plan.

## What This Tests

1. **HAPI Client** (`test_hapi.py`)
   - Fetch full CDAWeb HAPI catalog
   - Filter datasets by spacecraft pattern (PSP_*, SOLO_*)
   - Fetch parameter metadata for specific datasets
   - Filter to 1D parameters (scalars and small vectors)

2. **Catalog Structure** (`test_catalog.py`)
   - Static catalog for PSP and Solar Orbiter
   - Keyword matching for natural language queries
   - Integration with HAPI for parameter discovery

## Running the Tests

```bash
# Activate virtual environment
cd C:\Users\zhuang\Documents\GitHub\ai-autoplot
venv\Scripts\activate

# Run HAPI client test
python spikes/dataset_knowledge/test_hapi.py

# Run catalog test
python spikes/dataset_knowledge/test_catalog.py
```

## Expected Output

### test_hapi.py
- Should find 100+ PSP datasets and 50+ Solar Orbiter datasets
- Should fetch metadata for `PSP_FLD_L2_MAG_RTN_1MIN`
- Should identify 1D plottable parameters (vectors with size <= 3)

### test_catalog.py
- Should match "parker" -> PSP, "solar orbiter" -> SolO
- Should match "magnetic field" -> FIELDS/MAG instrument
- Should fetch real parameters from HAPI for PSP MAG dataset

## Key Findings to Validate

- [ ] HAPI catalog endpoint works and returns expected structure
- [ ] Dataset filtering by prefix works (PSP_*, SOLO_*)
- [ ] Parameter size field correctly identifies 1D vs 2D data
- [ ] Keyword matching is sufficient for basic NL queries
- [ ] Dataset IDs in catalog match real HAPI dataset IDs
