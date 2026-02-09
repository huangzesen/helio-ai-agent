# Feature Plan: Hackathon Roadmap

**Goal:** Win the Gemini 3 Hackathon (Marathon Agent track) by demonstrating a **multi-agent autonomous system** that gives anyone instant access to NASA's entire heliophysics data catalog through natural language.

**Track:** Marathon Agent — "Build autonomous systems for tasks spanning hours or days. Use Thought Signatures and Thinking Levels to maintain continuity and self-correct across multi-step tool calls without human supervision."

**Judging criteria:** Technical Execution (40%), Innovation/Wow Factor (30%), Potential Impact (20%), Presentation (10%).

**Deadline:** February 9, 2026 @ 5:00 PM PST

---

## What's Already Done

| Feature | Status |
|---------|--------|
| 5-agent architecture (Orchestrator, Mission, DataOps, DataExtraction, Visualization) | Done |
| 21 tool schemas, 5 registry methods + custom_visualization | Done |
| 52 spacecraft (all auto-generated from CDAWeb) | Done |
| Plotly renderer (interactive plots, no Java) | Done |
| Gradio web UI with inline plots, data preview, multimodal file upload, browse & fetch sidebar | Done |
| Data pipeline: HAPI -> pandas -> compute -> Plotly | Done |
| PlannerAgent with plan-execute-replan loop (up to 5 rounds) | Done |
| Custom operations sandbox (LLM-generated pandas/numpy) | Done |
| PNG/PDF export via kaleido | Done |
| Document conversion (PDF, DOCX, PPTX, XLSX, HTML, images -> Markdown) | Done |
| store_dataframe tool (text -> DataFrame: event lists, catalogs, search results) | Done |
| Session persistence (auto-save, --continue/--session, Gradio sessions sidebar) | Done |
| Mission data refresh (interactive menu + --refresh/--refresh-full/--refresh-all) | Done |
| Full CDAWeb catalog access (2000+ datasets via `search_full_catalog`) | Done |
| Google Search grounding (isolated Gemini API call with GoogleSearch tool) | Done |
| Multimodal file upload (PDF, images via `read_document`) | Done |
| Spectrograms (compute_spectrogram + plot_spectrogram via go.Heatmap) | Done |

---

## Priority 1: Gemini 3 Thinking Config [TODO]

**Effort:** ~2-3 hours | **Impact:** Highest — directly addresses the Marathon Agent track requirement

### Problem

The hackathon Marathon Agent track specifically calls for **Thought Signatures** and **Thinking Levels**. Our project uses Gemini 3 but doesn't explicitly configure these features. Without them, we miss a key differentiator that judges are looking for.

### What Are These Features?

- **Thought Signatures**: Encrypted tokens representing Gemini 3's reasoning state, automatically preserved across function calls via the SDK's chat interface. Enable seamless multi-step reasoning without losing context.
- **Thinking Levels**: Control reasoning depth — `HIGH` for complex planning, `LOW` for simple execution. Configurable per-agent for cost/quality tradeoff.

### Implementation

**Step 1: Add `get_thinking_config()` helper to `config.py`**

```python
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "high")
GEMINI_SUB_AGENT_THINKING_LEVEL = os.getenv("GEMINI_SUB_AGENT_THINKING_LEVEL", "low")

def get_thinking_config(level: str | None = None, include_thoughts: bool = False):
    """Build a ThinkingConfig for Gemini 3. Returns None if level is empty."""
    if not level:
        return None
    from google.genai import types
    return types.ThinkingConfig(
        thinking_level=level,
        include_thoughts=include_thoughts,
    )
```

**Step 2: Update SDK version**

`requirements.txt`: change `google-genai>=1.60.0` -> `google-genai>=1.62.0`

**Step 3: Add ThinkingConfig to all agents**

| Agent | File | Config locations | Thinking Level |
|-------|------|-----------------|----------------|
| OrchestratorAgent | `agent/core.py` | line 94 (main), line 938 (task exec) | HIGH |
| PlannerAgent | `agent/planner.py` | line 167 (planning) | HIGH |
| MissionAgent | `agent/mission_agent.py` | line 74 (task), line 122 (conv) | LOW |
| DataOpsAgent | `agent/data_ops_agent.py` | line 75 (task), line 121 (conv) | LOW |
| DataExtractionAgent | `agent/data_extraction_agent.py` | line 76 (task), line 122 (conv) | LOW |
| VisualizationAgent | `agent/visualization_agent.py` | line 78 (task), line 124 (conv) | LOW |

Skip: google_search config (`core.py` line 186) — no function calling, thinking not beneficial.

**Step 4: Track thinking tokens**

Update `_track_usage()` in all agents to capture `thoughts_token_count`. Update `get_token_usage()` to include `"thinking_tokens"` key and aggregate across sub-agents.

**Step 5: Add `_log_thoughts()` helper to OrchestratorAgent**

```python
def _log_thoughts(self, response):
    if not self.verbose or not response.candidates:
        return
    for part in response.candidates[0].content.parts:
        if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
            self.logger.debug(f"[Thinking] {part.text[:300]}{'...' if len(part.text) > 300 else ''}")
```

**Step 6: Update token displays**
- `main.py` — CLI usage summary (lines 226, 376-380)
- `gradio_app.py` — `_format_tokens()` sidebar (lines 83-89)
- `agent/logging.py` — `log_session_end()` (lines 172-182)

**Step 7: Add tests**

New file `tests/test_thinking_config.py` — test helper function, default env vars, token usage keys.

### Files

| File | Change |
|------|--------|
| `config.py` | +2 env vars, +`get_thinking_config()` helper |
| `requirements.txt` | Bump `google-genai>=1.62.0` |
| `agent/core.py` | ThinkingConfig on 2 configs, thinking token tracking + aggregation, `_log_thoughts()` |
| `agent/planner.py` | ThinkingConfig on 1 config, thinking token tracking |
| `agent/mission_agent.py` | ThinkingConfig on 2 configs, thinking token tracking |
| `agent/data_ops_agent.py` | ThinkingConfig on 2 configs, thinking token tracking |
| `agent/data_extraction_agent.py` | ThinkingConfig on 2 configs, thinking token tracking |
| `agent/visualization_agent.py` | ThinkingConfig on 2 configs, thinking token tracking |
| `main.py` | Show thinking tokens in usage display |
| `gradio_app.py` | Show thinking tokens in sidebar |
| `agent/logging.py` | Include thinking tokens in session log |
| `tests/test_thinking_config.py` | New test file |

### Gotchas
- Thinking is mandatory for Gemini 3 with function calling — can't disable, only adjust level
- SDK chat interface handles thought signatures automatically (no manual extraction needed)
- ThinkingConfig is compatible with `response_schema` (planner) and `mode="ANY"` (sub-agents)
- `getattr(meta, "thoughts_token_count", 0) or 0` handles older SDK gracefully

---

## Priority 2: Demo Polish [TODO]

**Effort:** ~2-3 hours | **Impact:** High for judges — smooth demo wins

### Gradio UX
- **Example prompts**: Curated quick-start buttons showcasing breadth
- **Loading indicators**: Show status during long operations
- **Error recovery UX**: Friendly messages instead of tracebacks

### Demo Script (3 minutes)

| Time | Action | What Judges See |
|------|--------|-----------------|
| 0:00-0:15 | Title + problem | "2000+ NASA datasets. Zero scripting required." |
| 0:15-0:45 | "Show me ACE magnetic field for last week" | Natural language -> interactive Plotly plot |
| 0:45-1:15 | "Compute magnitude and overlay" | LLM writes pandas code -> plot updates |
| 1:15-1:40 | "What Cluster data is available?" | Full catalog search -> 50+ datasets |
| 1:40-2:10 | Complex multi-step query | PlannerAgent decomposes, executes, replans autonomously |
| 2:10-2:30 | "What solar storms happened in Jan 2024?" | Google Search grounding -> context + data |
| 2:30-3:00 | Architecture overlay | 5-agent routing, thinking levels, 2000+ datasets |

### Framing for Judges
- Lead with: **"Autonomous AI Agent for Scientific Data Discovery"**
- Emphasize the **general pattern**: multi-agent orchestration + dynamic replanning + code generation sandboxes
- Heliophysics is the *proof of concept*, not the product
- Key Gemini 3 features to highlight:
  1. Thought Signatures for stateful multi-step reasoning
  2. Thinking Levels (HIGH for planning, LOW for execution)
  3. Function calling with 21 tools across 5 agents
  4. Structured JSON output for planning
  5. Google Search grounding for web context

### Submission Page

```
# Helio AI Agent
> Talk to NASA's spacecraft data. Powered by Gemini 3.

## The Problem
NASA's heliophysics fleet generates thousands of datasets. Finding, downloading,
processing, and visualizing this data requires hours of scripting in IDL or Python.

## The Solution
A conversational AI agent that gives anyone — from experienced researchers to curious
students — instant access to 2000+ NASA datasets through natural language.

## How It Uses Gemini 3
- Thought Signatures: Stateful reasoning across multi-step tool chains
- Thinking Levels: HIGH for orchestration/planning, LOW for fast sub-agent execution
- Function calling: 21 tools across 5 specialized agents
- Google Search grounding: Real-time space weather context
- Structured JSON output: Dynamic task decomposition with replanning
- Multi-agent routing: Orchestrator delegates to domain specialists
```

---

## Quick Wins (do anytime)

| Task | Effort | Impact |
|------|--------|--------|
| **Fix OMNI empty-string parsing** | ~15 min | Known issue — treat empty strings as NaN in `fetch.py` |
| **Clean up docs** | ~1 hr | Remove autoplot_bridge references, close obsolete known issues |

---

## Archived / Done

| Feature | Status |
|---------|--------|
| Full CDAWeb Catalog Access | Done — `search_full_catalog` tool |
| Multimodal File Upload | Done — `read_document` via `gr.MultimodalTextbox` |
| Google Search Grounding | Done — isolated Gemini API call with GoogleSearch tool |
| Spectrograms | Done — `compute_spectrogram` + `plot_spectrogram` via `go.Heatmap` |

---

## What NOT to Build

- Don't add more curated spacecraft JSONs — full catalog search covers everything
- Don't build authentication or multi-user support
- Don't build a custom frontend — Gradio is good enough
- Don't optimize for production (Docker, PyPI) — hackathon first
- Don't build an MCP server or REST API — the value is the intelligence layer
- Don't refactor the agent architecture — it's already clean
- Don't use AI Studio Build tab or Antigravity — public code repo is sufficient

---

## Implementation Order

```
Priority 1: Thinking Config (~2-3 hrs)    <-- Marathon Agent track requirement
  |
  v
Priority 2: Demo Polish (~2-3 hrs)        <-- smooth demo wins judges
```

Total remaining effort: **~4-6 hours**.
