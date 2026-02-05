# macOS Compatibility Testing

**Goal**: Verify that all Phase 1 capabilities work correctly on macOS (project was initially developed on Windows).

## Test Plan

### 1. Environment Setup
- [ ] Python 3.x installed and working
- [ ] Java runtime installed (required for Autoplot/JPype)
- [ ] Virtual environment created
- [ ] Dependencies installed from requirements.txt

### 2. Configuration
- [ ] .env file created with GOOGLE_API_KEY
- [ ] Autoplot JAR downloaded for macOS
- [ ] AUTOPLOT_JAR path configured

### 3. Core Components
- [ ] JPype initializes JVM correctly on macOS
- [ ] Autoplot JAR loads without errors
- [ ] ScriptContext accessible from Python

### 4. Knowledge Modules
- [ ] `knowledge/catalog.py` - keyword search works
- [ ] `knowledge/hapi_client.py` - HAPI API calls succeed
- [ ] Catalog returns PSP, SolO, ACE, OMNI datasets

### 5. Autoplot Bridge
- [ ] Connection initializes without errors
- [ ] Plot command executes (vap+cdaweb URI)
- [ ] Time range change works
- [ ] PNG export creates valid files
- [ ] State tracking maintains context

### 6. Agent Tools (all 7)
- [ ] `search_datasets` - finds spacecraft/instruments
- [ ] `list_parameters` - fetches HAPI metadata
- [ ] `plot_data` - displays Autoplot window
- [ ] `change_time_range` - updates plot time axis
- [ ] `export_plot` - saves PNG file
- [ ] `get_plot_info` - returns current state
- [ ] `ask_clarification` - prompts user for input

### 7. Full Integration
- [ ] Agent conversation loop runs
- [ ] Gemini function calling works
- [ ] Multi-step workflows complete successfully

## Known macOS Considerations

1. **Display**: Autoplot requires a display. On headless systems, use Xvfb or run with GUI.
2. **Java**: macOS may prompt for JDK installation on first run.
3. **File Paths**: macOS uses forward slashes (should work, but verify).
4. **JPype**: Version 1.5.0 specified in requirements.txt (has macOS support).

## Test Scripts

- `test_environment.py` - Check Python, Java, dependencies
- `test_jpype_mac.py` - Verify JVM initialization on macOS
- `test_autoplot_mac.py` - Test Autoplot bridge commands
- `test_knowledge_mac.py` - Test catalog and HAPI client
- `test_agent_mac.py` - Test full agent flow
- `run_all_tests.sh` - Execute complete test suite

## Results

Test results will be logged here after execution.

---

**Status**: In Progress
**Platform**: macOS (Darwin 24.5.0)
**Started**: 2026-02-05
