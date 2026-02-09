# Helio AI Agent

> Talk to NASA's spacecraft data. Powered by Gemini 3.

A conversational AI agent that gives anyone — from experienced researchers to curious students — instant access to 2,000+ NASA heliophysics datasets through natural language. Ask a question, get an interactive plot.

<!-- TODO: Add a screenshot or GIF of the Gradio UI here -->
<!-- ![Demo](docs/assets/demo.png) -->

## The Problem

NASA's heliophysics fleet (Parker Solar Probe, Solar Orbiter, ACE, MMS, Wind, and 50+ more spacecraft) generates thousands of datasets. Finding the right dataset, downloading it, parsing HAPI/CDAWeb formats, processing the data, and producing publication-quality plots requires hours of scripting in Python or IDL — even for experienced scientists.

## The Solution

Type what you want in plain English. The agent figures out the rest.

```
You: Show me ACE magnetic field data for last week
Agent: [fetches AC_H2_MFI, plots Bx/By/Bz components]

You: Compute the magnitude and overlay it
Agent: [writes pandas code: sqrt(Bx^2+By^2+Bz^2), adds trace to plot]

You: Compare with Wind magnetic field for the same period
Agent: [fetches WI_H2_MFI, creates multi-panel comparison plot]

You: What solar storms happened in January 2024?
Agent: [Google searches for events, creates a DataFrame of storms with dates]
```

## How It Uses Gemini 3

This project is built for the [Gemini 3 Hackathon](https://gemini3.devpost.com/) — **Marathon Agent** track.

| Gemini 3 Feature | How We Use It |
|---|---|
| **Thought Signatures** | Stateful multi-step reasoning across tool chains — the orchestrator maintains context across fetch, compute, and plot steps without losing track |
| **Thinking Levels** | HIGH for orchestration and planning (complex routing decisions), LOW for sub-agent execution (fast tool calling) |
| **Function Calling** | 23 tools across 5 specialized agents — LLM decides which tool to call, no hardcoded routing |
| **Structured JSON Output** | PlannerAgent decomposes complex requests into task batches, observes results, and dynamically replans |
| **Google Search Grounding** | Real-time space weather context — search results get turned into plottable DataFrames |
| **Multimodal Input** | PDF and image upload — Gemini vision extracts tables and text from scientific documents |

## Architecture

Five specialized agents, each with their own tools and system prompts:

```
User Request
    |
    v
OrchestratorAgent (Gemini 3 Pro, HIGH thinking)
    |--- Routes to the right specialist based on conversation context
    |--- Activates PlannerAgent for complex multi-step requests
    |
    +---> MissionAgent (per-spacecraft)     Fetches data from CDAWeb HAPI
    +---> DataOpsAgent                      Transforms data (pandas/numpy sandbox)
    +---> DataExtractionAgent               Turns text into structured DataFrames
    +---> VisualizationAgent                Renders interactive Plotly figures
    +---> PlannerAgent                      Decomposes complex requests, replans
```

**Data pipeline:** User request &rarr; Gemini function calling &rarr; HAPI fetch &rarr; pandas DataFrame &rarr; compute &rarr; Plotly interactive plot

**Key design decisions:**
- **LLM-driven routing** — no regex dispatcher. The orchestrator decides which sub-agent handles each request using conversation context and a routing table.
- **Code generation sandboxes** — the LLM writes pandas/numpy code for data operations and Plotly code for visualization customization. All code is AST-validated before execution (blocks imports, exec/eval, os/sys access).
- **Per-mission knowledge** — 52 auto-generated mission JSON files with instrument groupings, dataset metadata, and time ranges. The orchestrator sees only a routing table; sub-agents get rich domain-specific prompts.

## Features

- **52 spacecraft** with 3,000+ datasets, all auto-generated from CDAWeb
- **Full CDAWeb catalog search** — any of 2,000+ NASA datasets, not just pre-configured ones
- **Interactive Plotly plots** — zoom, pan, hover, multi-panel subplots, WebGL for large datasets
- **Compute pipeline** — magnitude, smoothing, derivatives, resampling, spectrograms, any pandas operation
- **Spectrograms** — compute and render time-frequency heatmaps via scipy.signal
- **Multi-step planning** — "Compare 3 spacecraft over the same period" decomposes into fetch/compute/plot tasks with dynamic replanning (up to 5 rounds)
- **Google Search grounding** — ask about solar storms, ICME events, or any space weather topic
- **Document reading** — upload PDFs or images, extract tables and text via Gemini vision
- **Session persistence** — auto-saves every turn, resume with `--continue`
- **PNG/PDF export** — publication-ready static images via kaleido
- **Gradio web UI** — browser-based chat with inline plots, data sidebar, file upload, browse & fetch panel

## Quick Start

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key with Gemini 3 access

### Setup

```bash
git clone https://github.com/zhuang/ai-autoplot.git
cd ai-autoplot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_API_KEY=your-gemini-api-key
```

On first run, the agent automatically downloads mission metadata from CDAWeb (~1-2 minutes).

### Run (CLI)

```bash
python main.py                  # Interactive mode
python main.py --verbose        # Show tool calls and timing
python main.py --continue       # Resume last session
python main.py "Show me ACE magnetic field for last week"  # Single command
```

### Run (Web UI)

```bash
python gradio_app.py            # Opens at localhost:7860
python gradio_app.py --share    # Generate a public URL
```

## Example Conversations

**Simple fetch and plot:**
> "Show me Parker Solar Probe magnetic field data for January 2024"

**Compute derived quantities:**
> "Compute the magnetic field magnitude and add it to the plot"

**Cross-mission comparison:**
> "Compare ACE and Wind solar wind speed for last month"

**Multi-step analysis:**
> "Fetch PSP magnetic field and plasma data, compute the Alfven speed, and plot everything on separate panels"

**Web-grounded research:**
> "What were the major solar storms in 2024? Plot the geomagnetic indices around the May 2024 event."

**Document extraction:**
> [Upload a PDF table of ICME events] "Plot these events as markers on a timeline"

## Project Structure

```
agent/                  Core agent layer (5 agents + planner + tools)
  core.py                 OrchestratorAgent — routes, dispatches, plans
  mission_agent.py        MissionAgent — per-spacecraft data fetching
  data_ops_agent.py       DataOpsAgent — data transformation
  data_extraction_agent.py DataExtractionAgent — text to DataFrames
  visualization_agent.py  VisualizationAgent — Plotly rendering
  planner.py              PlannerAgent — plan-execute-replan loop
  tools.py                23 tool schemas with category filtering
  session.py              Session persistence
  time_utils.py           Flexible time range parsing

knowledge/              Dataset catalog and prompt generation
  missions/*.json         52 auto-generated mission profiles
  mission_loader.py       Lazy-loading cache and routing table
  prompt_builder.py       Dynamic system prompts from mission data
  hapi_client.py          CDAWeb HAPI client (3-tier cache)
  bootstrap.py            Auto-download mission data from CDAWeb
  cdaweb_catalog.py       Full catalog search (2000+ datasets)

data_ops/               Data pipeline
  fetch.py                HAPI data fetching into pandas DataFrames
  store.py                In-memory DataStore singleton
  custom_ops.py           AST-validated sandbox for LLM-generated code

rendering/              Visualization engine
  plotly_renderer.py      Interactive Plotly figures, multi-panel, export
  registry.py             Method registry (6 core visualization methods)
  custom_viz_ops.py       AST-validated sandbox for Plotly code

main.py                 CLI entry point
gradio_app.py           Gradio web UI
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `GEMINI_MODEL` | `gemini-3-pro-preview` | Model for orchestrator |
| `GEMINI_SUB_AGENT_MODEL` | `gemini-3-flash-preview` | Model for sub-agents |
| `GEMINI_PLANNER_MODEL` | (same as GEMINI_MODEL) | Model for planner |

## Tests

```bash
python -m pytest tests/           # All tests (~650 tests)
python -m pytest tests/ -x -q     # Stop on first failure
```

## Tech Stack

- **Google Gemini 3** (Pro + Flash) — LLM with function calling, thinking levels, structured output
- **Plotly** — Interactive scientific data visualization
- **Gradio** — Browser-based chat UI with inline plots
- **pandas / numpy / scipy** — Data pipeline and computation
- **CDAWeb HAPI** — NASA heliophysics data access protocol
- **kaleido** — Static image export (PNG, PDF)

## License

MIT
