# macOS Compatibility Testing - Quick Start

This guide helps you test all Phase 1 capabilities on your Mac.

## Prerequisites

Before running tests, ensure you have:

1. **Python 3.x** - Check with: `python3 --version`
2. **Java Runtime** - Check with: `java -version`
   - If not installed: `brew install openjdk`
3. **Autoplot JAR** - Download from: https://autoplot.org/jnlp/latest/autoplot.jar

## Setup Steps

### 1. Install Dependencies

```bash
# From project root
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# .env file
GOOGLE_API_KEY=your_gemini_api_key_here
AUTOPLOT_JAR=/path/to/autoplot.jar
```

Get Gemini API key from: https://ai.google.dev/

### 3. Download Autoplot

```bash
# Download to home directory
wget -O ~/autoplot.jar https://autoplot.org/jnlp/latest/autoplot.jar

# Or using curl
curl -o ~/autoplot.jar https://autoplot.org/jnlp/latest/autoplot.jar
```

Then update your `.env`:
```
AUTOPLOT_JAR=/Users/YOUR_USERNAME/autoplot.jar
```

## Running Tests

### Option 1: Run All Tests (Recommended)

```bash
cd spikes/mac_compatibility
./run_all_tests.sh
```

This runs all 5 test suites in sequence and reports overall results.

### Option 2: Run Individual Tests

Run tests one at a time to diagnose specific issues:

```bash
# Test 1: Environment verification
python3 test_environment.py

# Test 2: JPype + JVM initialization
python3 test_jpype_mac.py

# Test 3: Autoplot bridge commands
python3 test_autoplot_mac.py

# Test 4: Knowledge modules
python3 test_knowledge_mac.py

# Test 5: Agent integration (all tools)
python3 test_agent_mac.py
```

### Option 3: Run Existing Unit Tests

```bash
# From project root
python -m pytest tests/ -v
```

## What Each Test Does

### test_environment.py
- Checks Python version
- Verifies Java installation
- Confirms all dependencies are installed
- Validates .env configuration
- Verifies Autoplot JAR exists

### test_jpype_mac.py
- Tests JPype imports
- Initializes JVM with Autoplot JAR
- Loads Autoplot ScriptContext class
- Creates ScriptContext instance

### test_autoplot_mac.py
- Opens Autoplot connection
- Plots CDAWeb data (opens window)
- Changes time range
- Exports PNG file
- Verifies state tracking

### test_knowledge_mac.py
- Tests catalog keyword search
- Connects to CDAWeb HAPI API
- Lists available parameters
- Tests HAPI response caching

### test_agent_mac.py
- Creates Gemini agent instance
- Tests all 7 agent tools:
  - search_datasets
  - list_parameters
  - plot_data
  - change_time_range
  - export_plot
  - get_plot_info
  - ask_clarification (implicit)

## Expected Output

All tests should show:
```
============================================================
SUMMARY
============================================================
Passed: X/X

✓ All tests work on macOS!
```

Test outputs (PNG files) are saved to `spikes/mac_compatibility/output/`

## Troubleshooting

### Java not found
```bash
brew install openjdk
# Follow brew instructions to link Java
```

### JPype errors
- Make sure Java is installed first
- Check Java version: `java -version`
- Try reinstalling JPype: `pip install --force-reinstall jpype1==1.5.0`

### Autoplot window doesn't appear
- macOS requires a display/GUI session
- Check if running in a proper terminal (not SSH without X11)
- Grant terminal accessibility permissions if prompted

### HAPI connection fails
- Check internet connection
- CDAWeb might be temporarily down
- Try again after a few minutes

### Gemini API errors
- Verify API key is correct in .env
- Check quota at https://aistudio.google.com/
- Ensure billing is enabled if required

## Next Steps

After all tests pass:

1. **Try the full agent**: `python main.py`
2. **Run conversation tests**: See `tests/test_agent.py`
3. **Test with real queries**: Try "Show me Parker magnetic field data for last week"

## Notes

- Tests open Autoplot windows - this is expected
- PNG exports are saved to `output/` folder
- HAPI tests require internet connection
- Agent tests require valid Gemini API key
