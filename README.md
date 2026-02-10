# Helio AI Agent

> Talk to NASA's spacecraft data. Powered by Gemini 3.

An autonomous AI agent that replaces complex analysis scripts with a single conversation. Ask for spacecraft data in plain English — the agent navigates NASA's heliophysics archive (52 missions, 3,000+ datasets, decades of observations), handles the entire data pipeline, and produces interactive visualizations on demand. It eliminates the tooling barrier that normally takes months to overcome: opaque dataset IDs, mission-specific naming conventions, the HAPI protocol, coordinate systems, unit conversions, and multi-panel plot boilerplate.

## Quick Start

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key with Gemini 3 access

### Setup

```bash
git clone https://github.com/huangzesen/helio-ai-agent.git
cd helio-ai-agent
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

## Features

- **52 spacecraft, 3,000+ datasets** — Parker Solar Probe, Solar Orbiter, ACE, MMS, Wind, DSCOVR, STEREO, Cluster, THEMIS, Van Allen Probes, GOES, Voyager, Ulysses, and more
- **Full CDAWeb catalog search** — access any of NASA's 2,000+ heliophysics datasets, not just pre-configured ones
- **Autonomous multi-step planning** — complex requests decomposed into task batches with dynamic replanning
- **Physics-aware computation** — magnitude, Alfven speed, plasma beta, spectral analysis, smoothing, resampling, derivatives — the LLM writes the code
- **Spectrograms** — time-frequency heatmaps computed via scipy.signal and rendered as interactive Plotly heatmaps
- **Cross-mission comparison** — automatically handles different dataset naming conventions, coordinate systems, and cadences
- **Google Search grounding** — find real-time space weather events, ICME catalogs, solar flare lists and feed them into the analysis pipeline
- **Document ingestion** — upload PDFs or images of data tables, extract structured data via Gemini vision
- **Interactive Plotly plots** — zoom, pan, hover tooltips, multi-panel subplots, WebGL for large datasets
- **Session persistence** — auto-saves every turn, resume with `--continue`
- **PNG/PDF export** — publication-ready static images via kaleido
- **Gradio web UI** — browser-based chat with inline plots, data sidebar, browse & fetch panel, file upload

## Example Workflows

### Basic: Fetch and visualize

> "Show me Parker Solar Probe magnetic field data for its closest perihelion in 2024"

The agent knows PSP's dataset naming convention, finds the right time window, fetches RTN magnetic field components, and plots an interactive 3-component time series.

### Intermediate: Compute derived quantities

> "Fetch ACE magnetic field and solar wind plasma data for last month. Compute the magnetic field magnitude, the Alfven speed, and the plasma beta. Plot everything on separate panels."

Six autonomous steps: two HAPI fetches (different instruments), three physics computations (each requiring the correct formula and unit handling), and a multi-panel Plotly figure.

### Advanced: Cross-mission event analysis

> "What were the major geomagnetic storms in 2024? For the strongest one, compare solar wind conditions at ACE and Wind, showing magnetic field magnitude and proton density on aligned time axes."

The agent searches the web for storm catalogs, identifies the May 2024 event, fetches data from two spacecraft (different dataset IDs, different parameter names), computes magnitudes, aligns time axes, and produces a publication-ready comparison plot — all autonomously through the PlannerAgent's replan loop.

### Research: Document-driven analysis

> [Upload a PDF table of ICME events from Richardson & Cane catalog]
> "Extract the events from 2023-2024 and plot their transit speeds as a time series"

Gemini vision reads the PDF, the DataExtractionAgent converts the table to a structured DataFrame, and the VisualizationAgent renders the result.

## Architecture

Five specialized agents, each with domain-specific tools and system prompts:

```
User Request
    |
    v
OrchestratorAgent (Gemini 3 Pro, HIGH thinking)
    |--- Routing table: 52 missions × instrument types → specialist selection
    |--- Activates PlannerAgent for complex multi-step requests
    |
    +---> MissionAgent (per-spacecraft)     Knows dataset IDs, parameter names,
    |                                        coordinate systems, time ranges
    +---> DataOpsAgent                      Writes pandas/numpy/scipy code in
    |                                        an AST-validated sandbox
    +---> DataExtractionAgent               Converts search results, PDFs,
    |                                        event catalogs into DataFrames
    +---> VisualizationAgent                Interactive Plotly figures with
    |                                        domain-appropriate defaults
    +---> PlannerAgent                      Decomposes, executes, observes,
                                             replans (up to 5 rounds)
```

**Data pipeline:** Natural language &rarr; dataset discovery &rarr; HAPI fetch &rarr; pandas DataFrame &rarr; LLM-generated computation &rarr; interactive Plotly plot

**Key design decisions:**
- **LLM-driven routing** — the orchestrator uses conversation context and a 52-mission routing table to decide which specialist handles each request. No regex dispatching.
- **Code generation sandbox** — the LLM writes pandas/numpy code for data transformations. All generated code is AST-validated before execution (blocks imports, exec/eval, os/sys access). Visualization uses 3 declarative tools (`plot_data`, `style_plot`, `manage_plot`) — no free-form code generation.
- **Per-mission knowledge base** — 52 auto-generated mission JSON files with instrument groupings, dataset IDs, parameter metadata, and time ranges. The orchestrator sees only a routing table; sub-agents receive rich domain-specific prompts with recommended datasets and analysis patterns.
- **Three-tier caching** — HAPI metadata: memory → local JSON file → network. Mission data: auto-downloaded on first run, refreshable via single CDAWeb API call (~3 seconds for all 3,000 datasets).

## Project Structure

```
agent/                  Core agent layer (5 agents + planner + 26 tools)
  core.py                 OrchestratorAgent — routes, dispatches, plans
  mission_agent.py        MissionAgent — per-spacecraft data fetching
  data_ops_agent.py       DataOpsAgent — pandas/numpy/scipy computation
  data_extraction_agent.py DataExtractionAgent — text/PDF to DataFrames
  visualization_agent.py  VisualizationAgent — Plotly rendering
  planner.py              PlannerAgent — plan-execute-replan loop
  tools.py                26 tool schemas with category-based filtering
  session.py              Session persistence (auto-save every turn)

knowledge/              52-mission knowledge base + prompt generation
  missions/*.json         Auto-generated mission profiles from CDAWeb
  mission_loader.py       Lazy-loading cache and routing table
  prompt_builder.py       Dynamic system prompts per agent type
  hapi_client.py          CDAWeb HAPI client (3-tier cache)
  bootstrap.py            Mission data auto-download and refresh

data_ops/               pandas-backed data pipeline
  fetch.py                HAPI data fetching → DataFrames
  store.py                In-memory DataStore singleton
  custom_ops.py           AST-validated sandbox for LLM-generated code

rendering/              Plotly visualization engine
  plotly_renderer.py      Multi-panel figures, WebGL, PNG/PDF export
  registry.py             3 declarative visualization tools (plot_data, style_plot, manage_plot)

main.py                 CLI entry point
gradio_app.py           Gradio web UI with inline plots
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `GEMINI_MODEL` | `gemini-3-pro-preview` | Model for orchestrator + planner |
| `GEMINI_SUB_AGENT_MODEL` | `gemini-3-flash-preview` | Model for sub-agents |
| `GEMINI_PLANNER_MODEL` | (same as GEMINI_MODEL) | Model for planner |

## Tests

```bash
python -m pytest tests/           # ~500 unit tests (no API key needed)
python -m pytest tests/ -x -q     # Stop on first failure
```

## Tech Stack

- **Google Gemini 3** (Pro + Flash) — function calling, thinking levels, structured output, multimodal vision, Google Search grounding
- **Plotly** — interactive scientific data visualization with WebGL
- **Gradio** — browser-based chat UI with inline plots and file upload
- **pandas / numpy / scipy** — data pipeline, computation, spectral analysis
- **CDAWeb HAPI** — NASA Heliophysics Data Application Programmer's Interface
- **kaleido** — static image export (PNG, PDF)

## License

MIT
