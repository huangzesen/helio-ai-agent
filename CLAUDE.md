# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

helio-ai-agent is an AI-powered natural language interface for [Autoplot](https://autoplot.org/), a Java-based scientific data visualization tool for spacecraft/heliophysics data. Users type conversational commands (e.g., "Show me ACE magnetic field data for last week") and the agent translates them into Autoplot operations and Python-side data computations.

**Current status:** Fully functional. See `docs/capability-summary.md` for a detailed breakdown of all implemented features, tools, and architecture. Keep that file updated when adding new capabilities.

## Architecture

The system has four layers:

1. **Agent layer** (`agent/`) — Gemini 2.5-Flash with function calling. Three agent types:
   - `core.py` **OrchestratorAgent** — routes to sub-agents, handles data ops directly. Tools defined in `tools.py` (14 tool schemas).
   - `mission_agent.py` **MissionAgent** — per-spacecraft data specialists (discovery + data_ops tools only).
   - `autoplot_agent.py` **AutoplotAgent** — visualization specialist using registry-driven dispatch via a single `execute_autoplot` tool + method catalog in prompt.

2. **Autoplot bridge** (`autoplot_bridge/`) — Python-to-Java bridge via JPype. `connection.py` starts the JVM with the Autoplot JAR on the classpath. `commands.py` wraps Autoplot's `ScriptContext` API (plot, set time range, export PNG/PDF, render types, color tables, canvas sizing). `registry.py` describes 16 Autoplot methods as structured data — the single source of truth for visualization capabilities. Uses a singleton pattern to maintain plot state and color assignments across the session.

3. **Knowledge base** (`knowledge/`) — Static dataset catalog (`catalog.py`) with mission profiles for keyword-based spacecraft/instrument search. Prompt builder (`prompt_builder.py`) generates system and planner prompts dynamically from the catalog — single source of truth. HAPI client (`hapi_client.py`) for fetching parameter metadata from CDAWeb.

4. **Data operations** (`data_ops/`) — Python-side data pipeline. Fetches HAPI data into pandas DataFrames (`fetch.py`), stores them in an in-memory singleton (`store.py`), and provides an AST-validated sandbox (`custom_ops.py`) for LLM-generated pandas/numpy code — handles magnitude, arithmetic, smoothing, resampling, derivatives, and any other transformation.

Data flows: User input → Gemini function calling → tool execution → result fed back to Gemini → natural language response. For computed data: fetch → compute → plot through Autoplot canvas.

## Key Technologies

- **Python 3** with virtualenv
- **Google Gemini** (`google-genai`) — LLM with function calling for tool routing
- **JPype** (`jpype1`) — Java-Python bridge to control Autoplot
- **Autoplot** — Java JAR, requires Java runtime and a display (or Xvfb for headless)
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
python main.py                 # Normal mode
python main.py --verbose       # Show tool calls, timing, errors

# Test Autoplot connection
python -m autoplot_bridge.connection

# Unit tests (fast, no API key or JVM needed)
python -m pytest tests/test_store.py tests/test_custom_ops.py  # Data ops tests
python -m pytest tests/                                         # All tests
```

## Gradio Web UI

`gradio_app.py` provides a browser-based chat interface for the agent. It wraps the same `OrchestratorAgent` used by `main.py` with inline plot display, a data table sidebar, and token usage tracking.

```bash
python gradio_app.py                # Launch on localhost:7860
python gradio_app.py --share        # Generate a public Gradio URL
python gradio_app.py --port 8080    # Custom port
python gradio_app.py --verbose      # Show tool call details
python gradio_app.py --model gemini-2.5-pro  # Override model
```

The app uses `concurrency_limit=1` (single-user singleton) and exports plot snapshots to a temp directory after each message. Plot changes are detected via MD5 hash and displayed both inline in the chat and in the sidebar. The agent's `web_mode` flag suppresses auto-opening exported files in the OS viewer.

## Interactive Agent Testing

Use `scripts/agent_server.py` to drive multi-turn conversations with the agent programmatically. It keeps an `OrchestratorAgent` alive in a background process and accepts commands over a TCP socket — this is the primary way to test interactive agent behavior without a human at the terminal.

```bash
# Start the server (initializes JVM + agent, listens on localhost)
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
- `AUTOPLOT_JAR` — path to the Autoplot single-JAR (download from https://autoplot.org/latest/)

## Supported Spacecraft

PSP (Parker Solar Probe), Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, and STEREO-A. Each mission has a JSON file in `knowledge/missions/` with keywords, profile, and datasets. The catalog in `knowledge/catalog.py` loads from these JSON files. Prompts are auto-generated from the JSON data.

## Autoplot URI Format

CDAWeb URIs follow the pattern: `vap+cdaweb:ds={DATASET_ID}&id={PARAMETER}&timerange={TIME_RANGE}`

Time ranges use `YYYY-MM-DD to YYYY-MM-DD` format. The agent accepts flexible input ("last week", "January 2024", "2024-01-15T06:00 to 2024-01-15T18:00") and converts to this format via `agent/time_utils.py`.

## For Future Sessions

- Read `docs/capability-summary.md` first to understand what has been implemented.
- Read `docs/roadmap.md` for planned future development.
- Read `tests/issue-log-20260207/ISSUE_SUMMARY.md` for known bugs from the latest test session (15 issues, 2 critical JVM crashes). Priority fixes: relative path resolution, rolling window DatetimeIndex, CDAWeb parameter validation, 4-panel plot guard.
- When adding new Autoplot capabilities: add entry to `autoplot_bridge/registry.py`, implement bridge method in `commands.py`, add handler in `core.py:_dispatch_autoplot_method()`. No tool schema changes needed. For non-Autoplot tools: add schema in `tools.py`, handler in `core.py:_execute_tool()`. Update `docs/capability-summary.md` either way.
- When adding new spacecraft: create a JSON file in `knowledge/missions/` (copy an existing one as template). Include `id`, `name`, `keywords`, `profile`, and `instruments` with `datasets` dict. Then run `python scripts/generate_mission_data.py --mission <id>` to populate HAPI metadata. The catalog, prompts, and routing table are all auto-generated from the JSON files.
- Data operations (`data_ops/custom_ops.py`) use an AST-validated sandbox for LLM-generated pandas/numpy code — easy to test.
- Plotting always goes through Autoplot (`autoplot_bridge/commands.py`), not matplotlib.
- **Ignore `docs/archive/`** — contains outdated historical documents that are no longer relevant.
