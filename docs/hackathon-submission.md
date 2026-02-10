# Helio-AI-Agent — Hackathon Submission

## Inspiration

Space scientists spend more time writing Python scripts to wrangle APIs, parse data formats, and configure plots than actually doing science. CDAWeb has 2,000+ datasets across 52 spacecraft missions, but accessing them requires knowing exact dataset IDs, parameter names, and time formats. We wanted to make space data as easy to explore as asking a question.

## What it does

Helio-AI-Agent lets you talk to spacecraft data in plain English. Say "Compare Parker Solar Probe and ACE magnetic field data for January 2024" and it searches the right datasets, fetches the data, computes derived quantities, and renders interactive Plotly visualizations — all through conversation. It supports 52 missions, 2,000+ CDAWeb datasets, multi-panel plots, spectrograms, and data export.

## How we built it

We built a multi-agent system powered by Google Gemini with function calling. An orchestrator agent routes requests to 5 specialized sub-agents: per-mission data fetchers, a data ops agent with an AST-validated pandas sandbox, a visualization agent using declarative Plotly tools, a data extraction agent for text-to-DataFrame conversion, and a planner agent that decomposes complex multi-step requests. The knowledge base is auto-generated from CDAWeb HAPI metadata. The frontend is a Gradio web UI with inline interactive plots and a browse-and-fetch sidebar.

## Challenges we ran into

- **Tool separation**: Getting the LLM to reliably route between 26 tools across 5 agents required careful prompt engineering and category-based tool filtering — each agent only sees its own toolset.
- **Sandboxed code execution**: Letting the LLM write pandas/numpy code for arbitrary transformations while blocking dangerous operations (imports, exec, dunder access) via AST validation.
- **Multi-step planning**: Coordinating parallel data fetches from different spacecraft, then sequencing computation and visualization, with up to 5 rounds of replanning on failure.
- **Data format wrangling**: HAPI CSV parsing edge cases — fill values, mixed types, error responses masquerading as data, datetime index alignment across missions with different cadences.

## Accomplishments that we're proud of

- **Zero-to-plot in one sentence**: A user with no programming knowledge can produce publication-quality interactive plots from NASA data.
- **52 missions auto-generated**: The entire knowledge base bootstraps from CDAWeb metadata — adding a new spacecraft is one command.
- **Self-correcting plots**: Every plot call returns structured review metadata (trace counts, y-ranges, gap detection), and the LLM self-corrects issues before responding.
- **Long-term memory**: A background daemon thread learns user preferences and operational pitfalls across sessions, making the agent smarter over time.

## What we learned

- LLM-driven routing outperforms regex-based routing — the model understands context and intent far better than pattern matching.
- Declarative tool interfaces (key-value params) are more reliable than letting the LLM generate free-form code for visualization.
- AST validation is a practical middle ground between full sandboxing and unrestricted code execution for data transformations.
- Thinking levels matter — HIGH thinking for orchestration/planning, LOW thinking for fast execution in sub-agents — balances quality with latency and cost.

## What's next for Helio-AI-Agent

- **Local CDF file loading** for offline analysis without network access
- **Spectral analysis tools** — FFT, wavelets, dynamic spectra
- **Jupyter notebook integration** for embedding the agent in research workflows
- **Event detection** — automatic identification of discontinuities, threshold crossings, and solar wind structures
- **Docker packaging and PyPI publication** for easy deployment
- **REST API** for programmatic access from other tools and pipelines
