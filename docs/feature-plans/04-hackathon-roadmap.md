# Feature Plan: Hackathon Roadmap

**Goal:** Win the Google hackathon by demonstrating that this is NOT another AI chatbot — it's a **data access tool** that gives anyone instant access to NASA's entire heliophysics data catalog through natural language.

**Target users:** Researchers doing quick-look analysis + broader audience exploring spacecraft data for the first time.

**Core pitch:** "Anyone can go from zero to a real scientific plot in one sentence."

---

## What's Already Done

| Feature | Status |
|---------|--------|
| 4-agent architecture (Orchestrator, Mission, DataOps, Visualization) | Done |
| 18 tool schemas, 5 registry methods + custom_visualization | Done |
| 52 spacecraft (8 curated + 44 auto-generated) | Done |
| Plotly renderer (interactive plots, no Java) | Done |
| Gradio web UI with inline plots + data preview | Done |
| Data pipeline: HAPI -> pandas -> compute -> Plotly | Done |
| Multi-step planning with mission-tagged dispatch | Done |
| Custom operations sandbox (LLM-generated pandas/numpy) | Done |
| PNG/PDF export via kaleido | Done |

---

## Priority 1: Full CDAWeb Catalog Access [DONE]

**Effort:** ~4-6 hours | **Impact:** Highest — turns "8 missions" into "all of NASA heliophysics"

### Problem

CDAWeb's HAPI server lists **2000+ datasets** across dozens of missions. We only expose 8 curated missions. If someone asks about Cluster, THEMIS, Voyager, or Ulysses, the agent says "I don't know that spacecraft."

### Solution

Add a dynamic catalog search that queries the full CDAWeb HAPI `/catalog` endpoint. The 8 curated missions remain as fast-path recommendations with rich prompts. Everything else becomes accessible through search.

### Implementation

**New tool: `search_full_catalog`**

Queries CDAWeb HAPI `/catalog` (cached locally after first fetch) and searches dataset names/descriptions by keyword. Returns matching dataset IDs with descriptions.

```python
# In agent/tools.py — new tool schema
{
    "name": "search_full_catalog",
    "category": "discovery",
    "description": "Search the full CDAWeb catalog (2000+ datasets) by keyword. Use when the user asks about a spacecraft or instrument not in the curated list, or wants to browse broadly.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms (spacecraft name, instrument, physical quantity)"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default 20)"
            }
        },
        "required": ["query"]
    }
}
```

**Local catalog cache:**

```python
# In knowledge/cdaweb_catalog.py (new file)
import requests
import json
from pathlib import Path

CATALOG_CACHE = Path.home() / ".helio-agent" / "cdaweb_catalog.json"
HAPI_CATALOG_URL = "https://cdaweb.gsfc.nasa.gov/hapi/catalog"

def get_full_catalog() -> list[dict]:
    """Fetch and cache the full CDAWeb HAPI catalog."""
    if CATALOG_CACHE.exists():
        # Cache for 24 hours
        ...
        return json.loads(CATALOG_CACHE.read_text())["catalog"]

    resp = requests.get(HAPI_CATALOG_URL)
    data = resp.json()
    CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_CACHE.write_text(json.dumps(data))
    return data["catalog"]

def search_catalog(query: str, max_results: int = 20) -> list[dict]:
    """Search full catalog by keyword (case-insensitive substring match on id + title)."""
    catalog = get_full_catalog()
    query_lower = query.lower()
    matches = []
    for entry in catalog:
        text = f"{entry.get('id', '')} {entry.get('title', '')}".lower()
        if query_lower in text:
            matches.append(entry)
        if len(matches) >= max_results:
            break
    return matches
```

**Routing update:**

The orchestrator's system prompt gains a new instruction: "If the user asks about a spacecraft or dataset not in the routing table, use `search_full_catalog` to find matching datasets in CDAWeb's full catalog. Any dataset found can be fetched and plotted with `fetch_data`."

The mission agent is NOT needed for uncurated missions — the orchestrator handles catalog search + fetch + delegate to visualization directly. Curated missions still get their specialist sub-agents with rich prompts.

### Demo Value

- "What magnetospheric data is available?" → searches full catalog, returns Cluster, THEMIS, MMS, Geotail, ...
- "Show me Voyager 2 magnetic field data from 1989" → finds dataset, fetches, plots
- "What datasets does THEMIS have?" → full browse

### Files

| File | Change |
|------|--------|
| `knowledge/cdaweb_catalog.py` | **New** — full catalog fetch + search |
| `agent/tools.py` | Add `search_full_catalog` tool schema |
| `agent/core.py` | Add handler in `_execute_tool()` |
| `knowledge/prompt_builder.py` | Update orchestrator prompt with catalog search instructions |

---

## Priority 2: Multimodal — "Analyze This Plot"

**Effort:** ~2-3 hours | **Impact:** High — Gemini differentiator, judges love it

### Problem

Generic chatbots can't look at a scientific plot and understand it. Gemini can.

### Solution

Let users upload a screenshot of a plot (from a paper, a website, or their own tool) and ask "What is this? Can you reproduce it?" Gemini analyzes the image, identifies spacecraft/instrument/time range, and calls tools to reproduce it.

### Implementation

**`agent/core.py`:**
- `process_message()` accepts an optional `image` parameter (bytes or file path)
- When present, send as a `Part` with `inline_data` (mime_type="image/png") alongside the text
- Gemini natively handles multimodal input — no extra library needed

**`gradio_app.py`:**
- Enable `multimodal=True` in `gr.ChatInterface`
- Extract image from the multimodal message dict, pass to agent

**System prompt addition:**
```
When the user uploads an image of a plot:
1. Describe what you see (axes, data type, time range, spacecraft if identifiable)
2. Identify the likely dataset and parameters
3. Offer to reproduce it using the available tools
4. If you can't identify the exact dataset, suggest the closest match from the catalog
```

### Demo Value

Upload a plot from a published paper → "This appears to be ACE magnetic field data from the January 2024 CME event. Let me reproduce it..." → fetches data, creates matching plot.

### Files

| File | Change |
|------|--------|
| `agent/core.py` | Accept image in `process_message()`, send as multimodal Part |
| `gradio_app.py` | Enable multimodal chat, pass image to agent |
| `knowledge/prompt_builder.py` | Add image analysis instructions to system prompt |

---

## Priority 3: Google Search Grounding [DONE]

**Effort:** ~1-2 hours | **Impact:** Medium — another Gemini-specific feature

**Status:** Implemented as a custom function tool (`google_search`). The Gemini API does not support combining `google_search` with `function_declarations` in the same request, so the implementation uses an isolated Gemini API call with only `GoogleSearch` configured. The orchestrator calls `google_search(query)` via function calling, which triggers a separate `generate_content` call. Returns grounded text with source URLs.

### Files Modified

| File | Change |
|------|--------|
| `agent/tools.py` | Added `google_search` tool schema |
| `agent/core.py` | Added `_google_search()` handler with isolated Gemini API call |
| `knowledge/prompt_builder.py` | Added grounding instructions to system prompt |

---

## Priority 4: Spectrograms (Rank-2 Data)

**Effort:** ~3-4 hours | **Impact:** Medium — visually impressive, shows depth

### Problem

Energy spectrograms and pitch angle distributions are rank-2 HAPI parameters (time x energy). Currently `fetch.py` only handles 1D parameters.

### Solution

Extend the fetch pipeline to handle multi-dimensional HAPI parameters and render them as `go.Heatmap` in Plotly.

### Implementation

**`data_ops/fetch.py`:**
- Detect `size > 1` in HAPI `/info` response (indicates rank-2 parameter)
- Parse CSV with multiple columns per parameter
- Store as DataFrame with energy bin columns
- Attach `bins` metadata from HAPI `/info` to DataEntry

**`rendering/plotly_renderer.py`:**
- New render path for rank-2 data:
```python
if entry.metadata.get("rank") == 2:
    fig.add_trace(go.Heatmap(
        x=entry.data.index,
        y=entry.metadata["bins"],  # energy levels from HAPI
        z=entry.data.values.T,
        colorscale="Viridis",
    ))
```

**`rendering/registry.py`:**
- Update `plot_stored_data` description to mention spectrogram support
- Add note about automatic detection of rank-2 data

### Demo Value

"Show me MMS ion energy spectrogram" → colorful heatmap appears. Visually striking, shows the tool handles complex scientific data types.

### Files

| File | Change |
|------|--------|
| `data_ops/fetch.py` | Handle rank-2 HAPI parameters (size > 1) |
| `data_ops/store.py` | DataEntry metadata for bins + rank |
| `rendering/plotly_renderer.py` | `go.Heatmap` rendering path |
| `rendering/registry.py` | Update descriptions |

---

## Priority 5: Demo Polish

**Effort:** ~2-3 hours | **Impact:** High for judges — smooth demo wins

### Gradio UX Improvements

- **Loading indicators**: Show "Fetching ACE data..." / "Rendering plot..." in real time
- **Example prompts**: Curated quick-start buttons that showcase breadth
  - "Show me ACE magnetic field for last week"
  - "Compare solar wind speed from Wind and DSCOVR"
  - "What magnetospheric datasets are available?" (full catalog)
  - "Compute the magnetic field magnitude for PSP"
- **Token usage display**: Already tracked — show in footer
- **Error recovery UX**: Friendly messages instead of tracebacks

### Demo Script (3 minutes)

| Time | Action | What Judges See |
|------|--------|-----------------|
| 0:00-0:20 | Title + problem | "2000+ NASA datasets. Zero scripting required." |
| 0:20-0:50 | "Show me ACE magnetic field for last week" | Natural language → interactive Plotly plot |
| 0:50-1:20 | "Compute magnitude and overlay" | LLM writes pandas code → plot updates |
| 1:20-1:40 | "What Cluster data is available?" | Full catalog search → 50+ datasets found |
| 1:40-2:10 | Upload screenshot from paper | Multimodal → "This looks like PSP data..." → reproduces it |
| 2:10-2:30 | "What solar storms happened in Jan 2024?" | Google Search grounding → context + data |
| 2:30-3:00 | Architecture slide | 3-agent routing, 2000+ datasets, Gemini multimodal |

### Devpost / Submission Page

```
# Helio AI Agent
> Talk to NASA's spacecraft data. Powered by Gemini.

## The Problem
NASA's heliophysics fleet generates thousands of datasets. Finding, downloading,
processing, and visualizing this data requires hours of scripting in IDL or Python.

## The Solution
A conversational AI agent that gives anyone — from experienced researchers to curious
students — instant access to 2000+ NASA datasets through natural language.

## How It Uses Gemini
- Function calling: 18 tools across 4 specialized agents
- Multimodal: Upload a plot screenshot, get it reproduced
- Google Search grounding: Real-time space weather context
- JSON mode: Structured task decomposition for complex queries
- Multi-agent routing: Orchestrator delegates to domain specialists
```

### Files

| File | Change |
|------|--------|
| `gradio_app.py` | Loading indicators, example prompts, error UX |

---

## Quick Wins (do anytime)

These are small fixes that improve quality independently of the main priorities.

| Task | Effort | Impact |
|------|--------|--------|
| **DataEntry metadata field** | ~30 min | Enables fill-value masking, better axis labels |
| **Fix OMNI empty-string parsing** | ~15 min | Known issue #1 — treat empty strings as NaN in `fetch.py` |
| **Clean up docs** | ~1 hr | Remove autoplot_bridge references from capability-summary.md, roadmap.md; close obsolete known issues #2 and #5 |

---

## Obsolete / Archived

These items from previous plans are no longer relevant:

| Feature | Why |
|---------|-----|
| Feature E: Autoplot data backend | `autoplot_bridge/` archived, no JVM dependencies |
| Phase 4 from Plan 03: Autoplot CDF cache | Same — JVM removed |
| Plan 01: Interactive GUI mode | Was Autoplot Swing window — replaced by Plotly in browser |
| HAPI parquet caching (Feature D) | Lower priority — data fetches are fast enough for demo. Nice-to-have later. |

---

## Implementation Order

```
Quick Wins (anytime)
  |
  v
Priority 1: Full CDAWeb Catalog Access (~4-6 hrs)
  |  "8 missions" → "all of NASA"
  |
  v
Priority 2: Multimodal Input (~2-3 hrs)
  |  Upload a plot, reproduce it
  |
  v
Priority 3: Google Search Grounding (~1-2 hrs)
  |  Space weather context
  |
  v
Priority 4: Spectrograms (~3-4 hrs)
  |  Energy spectrograms, heatmaps
  |
  v
Priority 5: Demo Polish (~2-3 hrs)
     Smooth Gradio UX, demo script, submission page
```

Total estimated effort: **~15-20 hours** for all priorities.

Priorities 1-3 are the hackathon core. Priority 4 adds depth. Priority 5 is the polish that wins judges.

---

## What NOT to Build

- Don't add more curated spacecraft JSONs — the full catalog search covers everything
- Don't build authentication or multi-user support
- Don't build a custom frontend — Gradio is good enough
- Don't optimize for production (Docker, PyPI) — hackathon first
- Don't build an MCP server or REST API — the value is the intelligence layer
- Don't refactor the agent architecture — it's already clean
