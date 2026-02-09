# Feature Plan: Hackathon Roadmap

**Goal:** Win the Gemini 3 Hackathon (Marathon Agent track) by demonstrating a **multi-agent autonomous system** that gives anyone instant access to NASA's entire heliophysics data catalog through natural language.

**Track:** Marathon Agent — "Build autonomous systems for tasks spanning hours or days. Use Thought Signatures and Thinking Levels to maintain continuity and self-correct across multi-step tool calls without human supervision."

**Judging criteria:** Technical Execution (40%), Innovation/Wow Factor (30%), Potential Impact (20%), Presentation (10%).

**Deadline:** February 9, 2026 @ 5:00 PM PST

---

## What's Already Done

| Feature | Status |
|---------|--------|
| 5-agent architecture (Orchestrator, Mission, DataOps, DataExtraction, Visualization) + passive MemoryAgent | Done |
| 26 tool schemas, 3 declarative viz tools (plot_data, style_plot, manage_plot) | Done |
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
| Long-term memory (cross-session preferences, summaries, pitfalls in `~/.helio-agent/memory.json`) | Done |
| Passive MemoryAgent (auto-triggered session analysis, pitfall extraction, error pattern reports) | Done |
| Automatic model fallback (switch to `GEMINI_FALLBACK_MODEL` on 429 quota errors) | Done |
| Plot self-review metadata (trace summary, warnings, hints for LLM self-assessment) | Done |
| preview_data tool (inspect actual values for debugging/verification) | Done |
| recall_memories tool (search/browse archived memories from past sessions) | Done |
| Thinking levels (HIGH for orchestrator/planner, LOW for sub-agents) + thinking token tracking | Done |
| Empty session auto-cleanup + eager HAPI cache download at startup | Done |

---

## Priority 1: Gemini 3 Thinking Config [DONE]

**Status:** Implemented. Thinking levels configured per agent (HIGH for orchestrator/planner, LOW for sub-agents). Thinking tokens tracked and displayed in CLI + Gradio. Thought previews logged in verbose mode via `agent/thinking.py`.

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
  3. Function calling with 26 tools across 5 agents + passive MemoryAgent
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
- Function calling: 26 tools across 5 specialized agents + passive MemoryAgent
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
| Thinking Config | Done — HIGH for orchestrator/planner, LOW for sub-agents, thinking token tracking |
| Long-term Memory | Done — cross-session preferences, summaries, pitfalls |
| Passive MemoryAgent | Done — auto-triggered session analysis, pitfall extraction |
| Automatic Model Fallback | Done — switch to `GEMINI_FALLBACK_MODEL` on 429 quota errors |
| Plot Self-Review | Done — structured review metadata on every `plot_data` call |

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
Priority 1: Thinking Config               <-- DONE
  |
  v
Priority 2: Demo Polish (~2-3 hrs)        <-- smooth demo wins judges
```

Total remaining effort: **~2-3 hours** (demo polish only).
