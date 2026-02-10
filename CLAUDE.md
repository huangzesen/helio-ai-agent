# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

helio-ai-agent is an AI-powered natural language interface for spacecraft and heliophysics data visualization. Users type conversational commands (e.g., "Show me ACE magnetic field data for last week") and the agent translates them into data operations and Plotly visualizations.

**Current status:** Fully functional. See `docs/capability-summary.md` for a detailed breakdown of all implemented features, tools, and architecture. Keep that file updated when adding new capabilities.

## Architecture

The system has four layers:

1. **Agent layer** (`agent/`) — Gemini 2.5-Flash with function calling. Three agent types:
   - `core.py` **OrchestratorAgent** — routes to sub-agents, handles data ops directly. Tools defined in `tools.py` (16 tool schemas).
   - `mission_agent.py` **MissionAgent** — per-spacecraft data specialists (discovery + data_ops tools only).
   - `visualization_agent.py` **VisualizationAgent** — visualization specialist using 3 declarative tools (`plot_data`, `style_plot`, `manage_plot`) + tool catalog in prompt. No free-form code generation.

2. **Rendering** (`rendering/`) — Pure-Python Plotly renderer (`plotly_renderer.py`) and tool registry (`registry.py`). The `PlotlyRenderer` class provides interactive Plotly figures with vector decomposition, multi-panel subplots, WebGL for large datasets, and PNG/PDF export via kaleido. The `registry.py` describes 3 declarative visualization tools (`plot_data`, `style_plot`, `manage_plot`).

3. **Knowledge base** (`knowledge/`) — Static dataset catalog (`catalog.py`) with mission profiles for keyword-based spacecraft/instrument search. Prompt builder (`prompt_builder.py`) generates system and planner prompts dynamically from the catalog — single source of truth. HAPI client (`hapi_client.py`) for fetching parameter metadata from CDAWeb.

4. **Data operations** (`data_ops/`) — Python-side data pipeline. Fetches HAPI data into pandas DataFrames (`fetch.py`), stores them in an in-memory singleton (`store.py`), and provides an AST-validated sandbox (`custom_ops.py`) for LLM-generated pandas/numpy code — handles magnitude, arithmetic, smoothing, resampling, derivatives, and any other transformation.

Data flows: User input → Gemini function calling → tool execution → result fed back to Gemini → natural language response. For computed data: fetch → compute → plot through Plotly renderer.

## Key Technologies

- **Python 3** with virtualenv
- **Google Gemini** (`google-genai`) — LLM with function calling for tool routing
- **Plotly** (`plotly`) — Interactive scientific data visualization
- **Kaleido** (`kaleido`) — Static image export for Plotly (PNG, PDF)
- **NumPy** — Array operations for data pipeline

## Commands

**Shell note:** On this Windows machine, the Bash tool runs Git Bash (`/usr/bin/bash`), so always use forward slashes for paths. Use `./venv/Scripts/python.exe` (not `venv\Scripts\python.exe`).

```bash
# Setup
python -m venv venv
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# On Windows (Git Bash), use the venv Python with forward slashes:
#   ./venv/Scripts/python.exe -m pytest tests/
#   ./venv/Scripts/python.exe main.py

# Run the agent
python main.py                 # Normal mode (auto-saves session)
python main.py --verbose       # Show tool calls, timing, errors
python main.py --continue      # Resume most recent session
python main.py --session ID    # Resume specific session by ID

# Unit tests (fast, no API key needed)
python -m pytest tests/test_store.py tests/test_custom_ops.py  # Data ops tests
python -m pytest tests/test_session.py                          # Session persistence tests
python -m pytest tests/                                         # All tests
```

## Gradio Web UI

`gradio_app.py` provides a browser-based chat interface for the agent. It wraps the same `OrchestratorAgent` used by `main.py` with inline plot display, a data table sidebar, and token usage tracking.

```bash
python gradio_app.py                # Launch on localhost:7860
python gradio_app.py --share        # Generate a public Gradio URL
python gradio_app.py --port 8080    # Custom port
python gradio_app.py --quiet        # Hide live progress log
python gradio_app.py --model gemini-2.5-pro  # Override model
```

The app displays interactive Plotly figures in the sidebar via `gr.Plot`. The agent's `web_mode` flag suppresses auto-opening exported files in the OS viewer.

## Interactive Agent Testing

Use `scripts/agent_server.py` to drive multi-turn conversations with the agent programmatically. It keeps an `OrchestratorAgent` alive in a background process and accepts commands over a TCP socket — this is the primary way to test interactive agent behavior without a human at the terminal.

```bash
# Start the server (initializes agent, listens on localhost)
python scripts/agent_server.py serve           # headless mode
python scripts/agent_server.py serve --verbose  # with tool call logging

# Send messages from another terminal (or script)
python scripts/agent_server.py send "Show me ACE magnetic field data for last week"
python scripts/agent_server.py send "Zoom in to January 10-15"
python scripts/agent_server.py reset   # clear conversation history
python scripts/agent_server.py stop    # shut down server

# Run the automated test suite (starts/stops server automatically)
python scripts/run_agent_tests.py              # all 6 scenarios
python scripts/run_agent_tests.py --test 4     # single scenario
python scripts/run_agent_tests.py --no-server  # use already-running server
```

The server uses `agent.core.create_agent()` directly (same `OrchestratorAgent` as `main.py`). Responses include the agent's text reply, tool calls made, timing, and token usage. Session logs are saved to `~/.helio-agent/sessions/`.

## Configuration

Requires a `.env` file at project root with:
- `GOOGLE_API_KEY` — Gemini API key

## Supported Spacecraft

PSP (Parker Solar Probe), Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, and STEREO-A. Each mission has a JSON file in `knowledge/missions/` with keywords, profile, and datasets. The catalog in `knowledge/catalog.py` loads from these JSON files. Prompts are auto-generated from the JSON data.

## CDAWeb URI Format

CDAWeb URIs follow the pattern: `vap+cdaweb:ds={DATASET_ID}&id={PARAMETER}&timerange={TIME_RANGE}`

Time ranges use `YYYY-MM-DD to YYYY-MM-DD` format. The agent accepts flexible input ("last week", "January 2024", "2024-01-15T06:00 to 2024-01-15T18:00") and converts to this format via `agent/time_utils.py`.

## Logging Conventions

**Never use `print()` for diagnostic output. Always use the project's logging system.**

- **Central config**: `agent/logging.py` — `setup_logging(verbose)` creates a shared `"helio-agent"` logger.
- **Get the logger**: `from agent.logging import get_logger` then `logger = get_logger()`. Do NOT use `logging.getLogger(__name__)`.
- **Custom helpers** (prefer these over raw `logger.debug()`):
  - `log_error(message, exc, context)` — full error with stack trace and context dict
  - `log_tool_call(tool_name, tool_args)` — tool invocation (DEBUG)
  - `log_tool_result(tool_name, result, success)` — tool completion (DEBUG/WARNING)
  - `log_plan_event(event, plan_id, details)` — planning milestones (INFO)
  - `log_session_end(token_usage)` — session closure stats (INFO)
- **Log levels**: DEBUG for internal tracing (tool calls, Gemini responses), INFO for milestones (session start, plan events), WARNING for recoverable errors, ERROR for failures with stack traces. Do not use CRITICAL.
- **Verbose mode** (`--verbose`): console shows DEBUG; normal mode shows WARNING+. File handler (`~/.helio-agent/logs/agent_YYYYMMDD.log`) always captures DEBUG.
- **Allowed `print()` exceptions**: User-facing CLI output in `main.py` (welcome message, interactive prompts), `gradio_app.py` (startup status), and `scripts/` (test harness progress). These are intentional UI, not diagnostics.

## For Future Sessions

- Read `docs/capability-summary.md` first to understand what has been implemented.
- Read `docs/planning-workflow.md` for the detailed planning & data fetch pipeline (candidate_datasets design).
- Read `docs/known-issues.md` for tracked bugs and their status.
- Most Plotly customizations (titles, labels, scales, render types, etc.) are handled by `style_plot` via declarative key-value params — no code changes needed. For new visualization capabilities: add to `rendering/registry.py`, implement in `rendering/plotly_renderer.py`, add handler in `agent/core.py:_execute_tool()`. For non-visualization tools: add schema in `agent/tools.py`, handler in `agent/core.py:_execute_tool()`. Update `docs/capability-summary.md` either way.
- When adding new spacecraft: create a JSON file in `knowledge/missions/` (copy an existing one as template). Include `id`, `name`, `keywords`, `profile`, and `instruments` with `datasets` dict. Then run `python scripts/generate_mission_data.py --mission <id>` to populate HAPI metadata. The catalog, prompts, and routing table are all auto-generated from the JSON files.
- Data operations (`data_ops/custom_ops.py`) use an AST-validated sandbox for LLM-generated pandas/numpy code — easy to test.
- Plotting always goes through the Plotly renderer (`rendering/plotly_renderer.py`), not matplotlib.
- Session data is saved to `~/.helio-agent/sessions/` (conversation history, figure JSON, metadata).
- Debug logs are at `~/.helio-agent/logs/` (one file per session, always captures DEBUG level).
