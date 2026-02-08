# Feature Plan: Upcoming Features

Consolidated from: Plan 03 (remaining phases 3-8), Plan 04 (Data Preview Panel).

## What's Already Done

These features are implemented and archived:

| Feature | Implemented |
|---------|-------------|
| Autoplot Interactive GUI Mode (Plan 01) | Feb 2026 |
| Plotly Renderer — drop-in replacement for Autoplot canvas (Plan 03, Phase 1) | Feb 2026 |
| Gradio upgrade — `gr.Plot` with interactive Plotly figures (Plan 03, Phase 2) | Feb 2026 |
| Data Preview Panel in Gradio sidebar (Feature A) | Feb 2026 |
| HAPI-Only Data Path — `plot_cdaweb` → `fetch_data` + `plot_stored_data` (Feature B) | Feb 2026 |

Current state:
- `rendering/plotly_renderer.py` is the active renderer (17 methods, stateful `go.Figure`)
- `gradio_app.py` uses `gr.Plot` for interactive Plotly + data preview dropdown in the sidebar
- `autoplot_bridge/` kept intact but not wired into the default flow
- `autoplot_script` tool hidden in `_legacy` category
- `plot_cdaweb` internally calls `fetch_data` + `plot_stored_data` (all data flows through HAPI → pandas → Plotly)

---

## Feature C: DataEntry Metadata Field

**Effort:** ~30 min | **Priority:** Medium | **Files:** `data_ops/store.py`, `data_ops/fetch.py`

### Goal

Add an optional `metadata` dict to `DataEntry` so HAPI metadata (fill values, dataset_id, parameter_id) persists with the data. Currently `fetch.py` extracts `fill_value` but it's not stored in `DataEntry`.

### Implementation

**`data_ops/store.py`:**
```python
from dataclasses import dataclass, field

@dataclass
class DataEntry:
    label: str
    data: pd.DataFrame
    units: str = ""
    description: str = ""
    source: str = "computed"
    metadata: dict = field(default_factory=dict)  # NEW
```

**`agent/core.py` (fetch_data handler):** Pass metadata when creating DataEntry:
```python
entry = DataEntry(
    label=label,
    data=result["data"],
    units=result.get("units", ""),
    description=result.get("description", ""),
    source="hapi",
    metadata={
        "fill_value": result.get("fill_value"),
        "dataset_id": dataset_id,
        "parameter_id": parameter_id,
    },
)
```

### Files

| File | Change |
|------|--------|
| `data_ops/store.py` | Add `metadata` field to `DataEntry` |
| `agent/core.py` | Pass metadata in `fetch_data` handler |

---

## Feature D: HAPI Response Caching

**Effort:** ~2 hours | **Priority:** Medium | **Files:** `data_ops/fetch.py`

### Goal

Cache HAPI responses locally as parquet files so repeat queries (zoom, re-plot) are instant. No JVM needed.

### Implementation

```python
import hashlib
from pathlib import Path

CACHE_DIR = Path.home() / ".helio-agent" / "data_cache"

def _cache_key(dataset_id, parameter_id, time_min, time_max):
    raw = f"{dataset_id}|{parameter_id}|{time_min}|{time_max}"
    return hashlib.md5(raw.encode()).hexdigest()
```

Before fetching from HAPI, check `CACHE_DIR / f"{key}.parquet"`. After fetching, save to parquet. Add `pyarrow` to requirements.

### Tradeoff

HAPI caches per-parameter, not per-CDF-file. "Show me Bx" then "show me By" is two cache entries. This covers ~90% of use cases without JVM complexity.

### Files

| File | Change |
|------|--------|
| `data_ops/fetch.py` | Add parquet cache layer |
| `requirements.txt` | Add `pyarrow` |

---

## Feature E: Autoplot Data Backend (Optional)

**Effort:** ~3-4 hours | **Priority:** Low | **Files:** new `data_ops/autoplot_fetch.py`, `agent/core.py`, `main.py`

### Goal

For power users with Autoplot installed: use Autoplot's CDF cache (`~/.autoplot/fscache/`) for fast repeat access. Add `--data-backend autoplot` CLI flag.

### Implementation

New file `data_ops/autoplot_fetch.py` uses `ScriptContext.getDataSet()` to load CDAWeb data through Autoplot's CDF cache, converts QDataSet to pandas DataFrame. Wire into `agent/core.py` fetch_data handler with a backend switch.

### Benefit

First request downloads CDF (~2-5s). Repeat access to same CDF file is instant (<100ms), even for different parameters within the same dataset.

### Files

| File | Change |
|------|--------|
| `data_ops/autoplot_fetch.py` | **New** — QDataSet → pandas conversion |
| `agent/core.py` | Backend switch in `fetch_data` handler |
| `main.py` | Add `--data-backend` CLI flag |

---

## Feature F: Gemini Differentiators

**Effort:** ~3-4 hours | **Priority:** Low

### F1: Multimodal Input — "Analyze This Plot"

Let users upload a screenshot and ask "What spacecraft is this? Can you reproduce it?"

| File | Change |
|------|--------|
| `agent/core.py` | Accept image parts in `process_message()` |
| `gradio_app.py` | Enable `multimodal=True` in chat |
| System prompt | Add image analysis instructions |

### F2: Grounding with Google Search

Use Gemini's Google Search grounding for space weather context.

| File | Change |
|------|--------|
| `agent/core.py` | Add `google_search_retrieval` tool config |
| System prompt | Add grounding instructions |

---

## Feature G: Spectrogram / Rank-2 Data (Stretch)

**Effort:** ~3-4 hours | **Priority:** Low

### Goal

Support rank-2 HAPI parameters (energy spectrograms, pitch angle distributions) and render them as `go.Heatmap` in the Plotly renderer.

### Implementation

- Extend `data_ops/fetch.py` to handle multi-dimensional parameters (size=[N])
- Add `go.Heatmap` rendering path in `rendering/plotly_renderer.py`
- Use HAPI bins metadata for y-axis (energy levels)

### Files

| File | Change |
|------|--------|
| `data_ops/fetch.py` | Handle rank-2 parameters |
| `rendering/plotly_renderer.py` | Add `go.Heatmap` for spectrograms |

---

## Implementation Priority

| # | Feature | Effort | Impact | Status |
|---|---------|--------|--------|--------|
| A | Data Preview Panel | ~1h | High — user trust | **Done** |
| B | Fix `plot_cdaweb` | ~1h | High — eliminates dead-end calls | **Done** |
| C | DataEntry metadata | ~30m | Medium — enables future features | Next |
| D | HAPI caching | ~2h | Medium — faster repeat queries | Next |
| E | Autoplot data backend | ~3-4h | Low — power users only | Optional |
| F | Gemini differentiators | ~3-4h | Low — demo value | Optional |
| G | Spectrograms | ~3-4h | Low — niche data types | Stretch |
