# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ai-autoplot is an AI-powered natural language interface for [Autoplot](https://autoplot.org/), a Java-based scientific data visualization tool for spacecraft/heliophysics data. Users type conversational commands (e.g., "Show me ACE magnetic field data for last week") and the agent translates them into Autoplot operations.

**Current status:** Specification phase. The full spec lives in `docs/autoplot-agent-spec.md` — no source code has been implemented yet.

## Architecture

The system has three layers:

1. **Agent layer** (`agent/`) — Gemini LLM with function calling decides which tools to invoke based on user input. `core.py` orchestrates the conversation loop and tool execution. Tools are defined declaratively in `tools.py` as JSON schemas.

2. **Autoplot bridge** (`autoplot_bridge/`) — Python-to-Java bridge via JPype. `connection.py` starts the JVM with the Autoplot JAR on the classpath. `commands.py` wraps Autoplot's `ScriptContext` API (plot, set time range, export PNG). Uses a singleton pattern to maintain plot state across the session.

3. **Knowledge base** (`knowledge/`) — Static dataset catalog (CDAWeb dataset IDs, parameters, keywords). The agent searches this to resolve natural language references like "magnetic field" to specific dataset IDs like `AC_H2_MFI`.

Data flows: User input → Gemini function calling → tool execution (dataset search or Autoplot command) → result fed back to Gemini → natural language response.

## Key Technologies

- **Python 3** with virtualenv
- **Google Gemini** (`google-generativeai`) — LLM with function calling for tool routing
- **JPype** (`jpype1`) — Java-Python bridge to control Autoplot
- **Autoplot** — Java JAR, requires Java runtime and a display (or Xvfb for headless)

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run the agent
python main.py

# Test Autoplot connection
python -m autoplot_bridge.connection

# Run tests
python -m pytest tests/
python -m pytest tests/test_agent.py::test_search_datasets_ace  # single test
```

## Configuration

Requires a `.env` file at project root with:
- `GOOGLE_API_KEY` — Gemini API key
- `AUTOPLOT_JAR` — path to the Autoplot single-JAR (download from https://autoplot.org/latest/)

## Phase 1 Scope

Limited to 3 CDAWeb datasets (`AC_H2_MFI`, `AC_H0_SWE`, `OMNI_HRO_1MIN`) and 5 operations: search datasets, plot data, change time range, export PNG, get plot info. The `ask_clarification` tool lets the agent ask the user follow-up questions instead of guessing.

## Autoplot URI Format

CDAWeb URIs follow the pattern: `vap+cdaweb:ds={DATASET_ID}&id={PARAMETER}&timerange={TIME_RANGE}`

Time ranges use `YYYY-MM-DD to YYYY-MM-DD` format. The agent should accept flexible input ("last week", "January 2024") and convert to this format.
