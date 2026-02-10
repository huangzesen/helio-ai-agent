# SPDF Dataset Size Report

**Generated:** 2026-02-10
**Method:** Queried CDAWeb REST API (`orig_data` endpoint) for every dataset defined in `knowledge/missions/*.json`
**Script:** `scripts/query_dataset_sizes.py`
**Full data:** `dataset_sizes.json` (per-dataset file counts and byte sizes)

## How This Was Generated

The mission JSON files in `knowledge/missions/` contain all HAPI-available datasets hosted on NASA's Space Physics Data Facility (SPDF). Each dataset has a CDAWeb ID and a time range.

For each of the 2,479 unique datasets, we queried:

```
https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/{DATASET_ID}/orig_data/{START},{STOP}
```

This returns a JSON array of every CDF file in the dataset with its exact byte size (`Length` field). We summed these per mission to produce the totals below.

The query completed in ~20 minutes with 8 concurrent workers and 1 timeout error (IMP8 `I1_AV_ALL`). This approach is far more efficient than crawling SPDF directory listings, which can take hours due to the deep directory tree structure (e.g., FAST has 24,000+ per-orbit operations directories).

To reproduce:

```bash
python scripts/query_dataset_sizes.py --workers 8 --output dataset_sizes.json
```

## Summary

| Metric | Value |
|---|---|
| Missions | 52 |
| Datasets queried | 2,479 |
| Total data files | 14,260,445 |
| Total data size | **282.33 TB** |
| Query errors | 1 |

## Per-Mission Breakdown

| # | Mission | Datasets | Files | Size | % of Total |
|---|---|---|---|---|---|
| 1 | MMS | 264 | 7,403,160 | 171.10 TB | 60.6% |
| 2 | CLUSTER | 257 | 1,533,920 | 57.22 TB | 20.3% |
| 3 | RBSP | 136 | 461,407 | 16.72 TB | 5.9% |
| 4 | THEMIS | 272 | 1,135,495 | 7.87 TB | 2.8% |
| 5 | POLAR | 42 | 179,605 | 5.22 TB | 1.8% |
| 6 | Solar Orbiter | 134 | 191,331 | 4.56 TB | 1.6% |
| 7 | MAVEN | 20 | 68,237 | 4.32 TB | 1.5% |
| 8 | PSP | 94 | 146,148 | 4.00 TB | 1.4% |
| 9 | FAST | 7 | 185,369 | 2.71 TB | 1.0% |
| 10 | TIMED | 6 | 320,590 | 2.47 TB | 0.9% |
| 11 | WIND | 66 | 443,935 | 1.47 TB | 0.5% |
| 12 | ARASE | 20 | 58,454 | 1.35 TB | 0.5% |
| 13 | STEREO_A | 37 | 140,422 | 897.95 GB | 0.3% |
| 14 | IMAGE | 19 | 34,155 | 407.05 GB | 0.1% |
| 15 | STEREO_B | 29 | 55,748 | 362.87 GB | 0.1% |
| 16 | DMSP | 17 | 197,613 | 341.88 GB | 0.1% |
| 17 | TWINS | 12 | 39,719 | 230.70 GB | <0.1% |
| 18 | NOAA | 14 | 36,925 | 188.19 GB | <0.1% |
| 19 | IMP8 | 54 | 700,554 | 183.32 GB | <0.1% |
| 20 | DE | 33 | 92,573 | 158.68 GB | <0.1% |
| 21 | CNOFS | 5 | 25,162 | 96.53 GB | <0.1% |
| 22 | ACE | 32 | 189,682 | 83.40 GB | <0.1% |
| 23 | MESSENGER | 3 | 3,145 | 78.61 GB | <0.1% |
| 24 | ISEE | 15 | 23,436 | 69.80 GB | <0.1% |
| 25 | GEOTAIL | 19 | 160,656 | 68.47 GB | <0.1% |
| 26 | GOES | 38 | 24,469 | 40.88 GB | <0.1% |
| 27 | DSCOVR | 5 | 13,245 | 31.23 GB | <0.1% |
| 28 | BARREL | 387 | 4,773 | 26.23 GB | <0.1% |
| 29 | ULYSSES | 42 | 191,785 | 19.75 GB | <0.1% |
| 30 | GPS | 4 | 25,426 | 18.95 GB | <0.1% |
| 31 | ELFIN | 8 | 8,396 | 13.85 GB | <0.1% |
| 32 | SOHO | 16 | 47,940 | 13.16 GB | <0.1% |
| 33 | AMPTE | 1 | 1,233 | 12.11 GB | <0.1% |
| 34 | OMNI | 6 | 2,713 | 9.49 GB | <0.1% |
| 35 | VOYAGER2 | 20 | 23,360 | 9.13 GB | <0.1% |
| 36 | VOYAGER1 | 16 | 24,777 | 8.40 GB | <0.1% |
| 37 | LANL | 14 | 35,641 | 6.23 GB | <0.1% |
| 38 | IBEX | 239 | 3,701 | 5.20 GB | <0.1% |
| 39 | ST5 | 3 | 272 | 3.56 GB | <0.1% |
| 40 | CIRBE | 3 | 1,242 | 2.82 GB | <0.1% |
| 41 | HELIOS | 14 | 6,071 | 2.62 GB | <0.1% |
| 42 | ISS | 3 | 1,767 | 2.10 GB | <0.1% |
| 43 | CRRES | 1 | 43 | 918.47 MB | <0.1% |
| 44 | EQUATOR-S | 7 | 741 | 838.46 MB | <0.1% |
| 45 | SAMPEX | 5 | 11,706 | 624.87 MB | <0.1% |
| 46 | NEW-HORIZONS | 22 | 181 | 223.55 MB | <0.1% |
| 47 | PIONEER | 7 | 557 | 206.15 MB | <0.1% |
| 48 | CASSINI | 3 | 7 | 158.82 MB | <0.1% |
| 49 | CSSWE | 2 | 952 | 153.86 MB | <0.1% |
| 50 | SNOE | 2 | 1,834 | 125.35 MB | <0.1% |
| 51 | PIONEER-VENUS | 2 | 170 | 31.97 MB | <0.1% |
| 52 | JUNO | 2 | 2 | 5.49 MB | <0.1% |

## Key Observations

- **MMS + CLUSTER = 80.9% of all data.** These two magnetospheric missions dominate the archive due to their multi-spacecraft, multi-instrument, burst-mode data products.
- **MMS alone is 60.6%** with 7.4 million CDF files totaling 171 TB across 264 datasets (4 spacecraft x many instruments x burst/fast/slow modes).
- **Top 5 missions account for 91.4%** of total volume (MMS, Cluster, RBSP, THEMIS, Polar).
- The **long tail is small**: missions ranked 20+ collectively hold <1% of total data.
- **File count leaders** differ from size leaders: IMP8 has 700K files but only 183 GB (many small files), while POLAR has 180K files at 5.2 TB (large files).
- This covers only HAPI-available datasets on CDAWeb. The full SPDF archive (including raw telemetry, operations data, and non-HAPI products) is significantly larger.

## Notes

- BARREL has 387 datasets but only 4,773 files (26 GB) because each balloon campaign is a separate dataset with few files each.
- IBEX has 239 datasets but only 3,701 files (5.2 GB) — many small map/flux datasets.
- The 1 error was a read timeout on IMP8 dataset `I1_AV_ALL`; its contribution is negligible.
- Dataset counts differ from the mission JSON totals (3,088 vs 2,479 queried) because duplicate dataset IDs with `@N` suffixes are deduplicated before querying.
