# Helio AI Agent

> Talk to NASA's spacecraft data. Powered by Gemini 3.

An autonomous AI agent that replaces months of domain expertise and thousands of lines of analysis scripts with a single conversation. It navigates NASA's entire heliophysics data archive — 52 spacecraft missions, 3,000+ datasets, decades of observations — and produces publication-ready interactive visualizations on demand.

<!-- TODO: Add a screenshot or GIF of the Gradio UI here -->
<!-- ![Demo](docs/assets/demo.png) -->

## The Problem

Heliophysics research has an enormous tooling barrier. To produce a single multi-spacecraft comparison plot, a scientist typically needs to:

1. **Know which datasets exist** — CDAWeb hosts 2,000+ datasets across 50+ missions. Dataset IDs are opaque (`AC_H2_MFI`, `PSP_SWP_SPI_SF0A_L3_MOM`, `WI_H1_SWE`). There is no unified search — you need to already know what you're looking for.
2. **Understand the HAPI protocol** — NASA's data access API requires constructing URIs with exact dataset IDs, parameter names, and ISO 8601 time ranges. Different missions use different naming conventions, coordinate systems, and cadences.
3. **Write data pipeline code** — fetch CSV over HTTP, parse timestamps, handle missing values (each mission has its own quirks), convert units, align time axes across missions with different cadences.
4. **Implement the physics** — computing derived quantities like magnetic field magnitude, Alfven speed, plasma beta, or spectral power density requires domain knowledge that takes years to build.
5. **Build the visualization** — multi-panel plots with shared time axes, proper axis labels with units, log scales for particle data, colormaps for spectrograms, and interactive features for exploration.

A graduate student entering the field spends **months** learning these tools before producing their first meaningful analysis. Even experienced researchers spend hours writing boilerplate scripts for each new study.

## The Solution

Describe what you want to see. The agent handles the entire pipeline — dataset discovery, data access protocol, time range parsing, unit handling, physics computations, and interactive visualization — in seconds.

```
You: Show me ACE magnetic field data for last week
Agent: [searches catalog → finds AC_H2_MFI → fetches via HAPI → parses 3-component
        vector field → plots Bx/By/Bz in RTN coordinates as interactive Plotly figure]

You: Compute the magnitude and overlay it
Agent: [writes pandas code: np.sqrt(Bx**2 + By**2 + Bz**2) → adds |B| trace to plot]

You: Compare with Wind magnetic field for the same period
Agent: [knows Wind uses WI_H2_MFI with different parameter names → fetches, aligns
        time axes → creates multi-panel comparison plot with both spacecraft]

You: What solar storms happened in January 2024?
Agent: [Google Search → finds ICME catalog → extracts event list → creates structured
        DataFrame with dates, speeds, and magnetic field strengths → plots as timeline]
```

Each step above would normally require a separate Python script, knowledge of mission-specific dataset naming conventions, and familiarity with the HAPI API. The agent collapses this entire workflow into a conversation.

## What Makes This Hard

This isn't a chatbot wrapper around an API. The agent solves problems that require genuine domain reasoning:

**Cross-mission analysis** — "Compare Parker Solar Probe and ACE magnetic field during perihelion" requires knowing that PSP uses `PSP_FLD_L2_MAG_RTN_1MIN` while ACE uses `AC_H2_MFI`, that they use different coordinate systems (RTN vs GSE), that PSP's orbit means perihelion dates must be looked up, and that the two spacecraft sample at different cadences that need alignment.

**Physics-aware computation** — "Compute the Alfven speed" requires fetching both magnetic field and plasma density data (from different instruments on the same spacecraft), knowing the formula v_A = B / sqrt(mu_0 * n * m_p), handling unit conversions (nT to T, cm^-3 to m^-3), and producing a result in km/s.

**Multi-step autonomous planning** — "Compare solar wind conditions at L1 during the top 5 geomagnetic storms of 2024" requires: searching the web for storm dates, creating a structured event list, fetching OMNI data for each time window, computing relevant parameters, and producing a multi-panel comparison — all without human intervention. The PlannerAgent decomposes this into task batches, executes them, observes results, and dynamically replans if something fails.

**Spectrogram analysis** — "Show me the MMS ion energy spectrogram" involves fetching rank-2 HAPI parameters (time x energy bins), computing power spectral density via scipy.signal, and rendering as an interactive heatmap with proper log-scaled color axes and frequency labels.

## How It Uses Gemini 3

Built for the [Gemini 3 Hackathon](https://gemini3.devpost.com/) — **Marathon Agent** track.

| Gemini 3 Feature | How We Use It |
|---|---|
| **Thought Signatures** | Preserve reasoning state across multi-tool chains — the orchestrator maintains context through fetch → compute → plot sequences spanning 10+ tool calls without losing track of what it's building toward |
| **Thinking Levels** | HIGH for the orchestrator and planner (complex routing: "which of 52 missions has this data? which instrument? which coordinate system?"), LOW for sub-agents (fast execution of well-defined subtasks) |
| **Function Calling** | 23 tools across 5 specialized agents — the LLM decides which spacecraft agent to delegate to, what computation to run, and how to visualize results. No hardcoded routing. |
| **Structured JSON Output** | PlannerAgent decomposes "compare 3 spacecraft and compute derived quantities" into ordered task batches, observes intermediate results, and replans up to 5 rounds |
| **Google Search Grounding** | Real-time space weather context — "what CMEs hit Earth in 2024?" becomes a structured dataset that feeds into the analysis pipeline |
| **Multimodal Input** | Upload a PDF of an ICME catalog or a screenshot of a data table — Gemini vision extracts the data, the agent converts it to a plottable DataFrame |

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
- **Code generation sandboxes** — the LLM writes pandas/numpy code for data transformations and Plotly code for visualization customization. All generated code is AST-validated before execution (blocks imports, exec/eval, os/sys access).
- **Per-mission knowledge base** — 52 auto-generated mission JSON files with instrument groupings, dataset IDs, parameter metadata, and time ranges. The orchestrator sees only a routing table; sub-agents receive rich domain-specific prompts with recommended datasets and analysis patterns.
- **Three-tier caching** — HAPI metadata: memory → local JSON file → network. Mission data: auto-downloaded on first run, refreshable via single CDAWeb API call (~3 seconds for all 3,000 datasets).

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

## Project Structure

```
agent/                  Core agent layer (5 agents + planner + 23 tools)
  core.py                 OrchestratorAgent — routes, dispatches, plans
  mission_agent.py        MissionAgent — per-spacecraft data fetching
  data_ops_agent.py       DataOpsAgent — pandas/numpy/scipy computation
  data_extraction_agent.py DataExtractionAgent — text/PDF to DataFrames
  visualization_agent.py  VisualizationAgent — Plotly rendering
  planner.py              PlannerAgent — plan-execute-replan loop
  tools.py                23 tool schemas with category-based filtering
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
  registry.py             6 core visualization methods
  custom_viz_ops.py       AST-validated sandbox for Plotly code

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
python -m pytest tests/           # ~650 unit tests (no API key needed)
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
