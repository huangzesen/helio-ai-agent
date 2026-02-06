# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

helio-ai-agent is an AI-powered natural language interface for [Autoplot](https://autoplot.org/), a Java-based scientific data visualization tool for spacecraft/heliophysics data. Users type conversational commands (e.g., "Show me ACE magnetic field data for last week") and the agent translates them into Autoplot operations and Python-side data computations.

**Current status:** Fully functional. See `docs/capability-summary.md` for a detailed breakdown of all implemented features, tools, and architecture. Keep that file updated when adding new capabilities.

## Architecture

The system has four layers:

1. **Agent layer** (`agent/`) — Gemini 2.5-Flash with function calling decides which tools to invoke based on user input. `core.py` orchestrates the conversation loop and tool execution. Tools are defined declaratively in `tools.py` as JSON schemas (15 tools total). Token usage is tracked per session. For multi-mission requests, `mission_agent.py` provides specialized sub-agents with focused system prompts per spacecraft mission.

2. **Autoplot bridge** (`autoplot_bridge/`) — Python-to-Java bridge via JPype. `connection.py` starts the JVM with the Autoplot JAR on the classpath. `commands.py` wraps Autoplot's `ScriptContext` API (plot, set time range, export PNG, plot computed data as QDataSets). Uses a singleton pattern to maintain plot state and color assignments across the session.

3. **Knowledge base** (`knowledge/`) — Static dataset catalog (`catalog.py`) with mission profiles for keyword-based spacecraft/instrument search. Prompt builder (`prompt_builder.py`) generates system and planner prompts dynamically from the catalog — single source of truth. HAPI client (`hapi_client.py`) for fetching parameter metadata from CDAWeb.

4. **Data operations** (`data_ops/`) — Python-side data pipeline. Fetches HAPI data into numpy arrays (`fetch.py`), stores them in an in-memory singleton (`store.py`), and provides pure numpy operations (`operations.py`): magnitude, arithmetic, running average, resample, delta/derivative.

Data flows: User input → Gemini function calling → tool execution → result fed back to Gemini → natural language response. For computed data: fetch → compute → plot through Autoplot canvas.

## Key Technologies

- **Python 3** with virtualenv
- **Google Gemini** (`google-genai`) — LLM with function calling for tool routing
- **JPype** (`jpype1`) — Java-Python bridge to control Autoplot
- **Autoplot** — Java JAR, requires Java runtime and a display (or Xvfb for headless)
- **NumPy** — Array operations for data pipeline

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Run the agent
python main.py                 # Normal mode
python main.py --verbose       # Show tool calls, timing, errors

# Test Autoplot connection
python -m autoplot_bridge.connection

# Run tests
python -m pytest tests/test_store.py tests/test_operations.py  # Data ops tests (41 tests)
python -m pytest tests/                                         # All tests
```

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
- When adding new tools: add schema in `agent/tools.py`, handler in `agent/core.py`, and update `docs/capability-summary.md`. The system prompt is auto-generated from the catalog.
- When adding new spacecraft: create a JSON file in `knowledge/missions/` (copy an existing one as template). Include `id`, `name`, `keywords`, `profile`, and `instruments` with `datasets` dict. Then run `python scripts/generate_mission_data.py --mission <id>` to populate HAPI metadata. The catalog, prompts, and routing table are all auto-generated from the JSON files.
- Data operations (`data_ops/operations.py`) are pure numpy functions with no side effects — easy to test.
- Plotting always goes through Autoplot (`autoplot_bridge/commands.py`), not matplotlib.
- **Ignore `docs/archive/`** — contains outdated historical documents that are no longer relevant.
