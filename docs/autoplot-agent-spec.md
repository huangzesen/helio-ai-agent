# Autoplot Agentic Natural Language Interface

## Project Overview

Build an interactive, agentic natural language interface for [Autoplot](https://autoplot.org/) that allows users to visualize spacecraft data using conversational commands. The agent translates natural language requests into Autoplot operations.

### What is Autoplot?

Autoplot is a Java-based interactive browser for scientific data on the web. Key capabilities:
- Reads multiple formats: CDF, NetCDF, HDF5, ASCII tables, FITS, Excel, etc.
- Accesses data servers: CDAWeb (NASA/Goddard), HAPI servers, Das2 servers
- Uses URIs to identify datasets (e.g., `vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01`)
- Scriptable via Jython (Python on JVM)
- Can be controlled programmatically via Python bridge using JPype

### What We're Building

A Python-based agent that:
1. Accepts natural language commands about spacecraft data
2. Uses Google Gemini to interpret requests and decide actions
3. Executes Autoplot commands via the Python bridge
4. Handles ambiguous requests by asking clarifying questions
5. Maintains conversation context for follow-up commands

---

## Scope: Minimal Working Example (Phase 1)

For the first working prototype, we limit scope to:

### Supported Data Sources
- **CDAWeb** (NASA's Coordinated Data Analysis Web) - most common source for heliophysics data
- Focus on 2-3 well-known datasets:
  - ACE magnetic field data (`AC_H2_MFI`)
  - ACE solar wind data (`AC_H0_SWE`)
  - OMNI combined data (`OMNI_HRO_1MIN`)

### Supported Operations
1. **Load and plot** a dataset for a time range
2. **Change time range** of current plot
3. **Export** current plot to PNG
4. **Get info** about what's currently displayed

### Out of Scope for Phase 1
- Multiple panels/layouts
- Data processing (FFT, smoothing, etc.)
- Non-CDAWeb data sources
- Complex styling customization
- Multi-agent architectures

---

## Project Structure

```
autoplot-agent/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration (API keys, defaults)
├── main.py                   # Entry point - conversation loop
├── agent/
│   ├── __init__.py
│   ├── core.py               # Main agent logic
│   ├── tools.py              # Tool definitions for Gemini
│   └── prompts.py            # System prompts and templates
├── autoplot_bridge/
│   ├── __init__.py
│   ├── connection.py         # Autoplot/JPype initialization
│   └── commands.py           # Autoplot command wrappers
├── knowledge/
│   ├── __init__.py
│   ├── datasets.py           # Dataset catalog (CDAWeb IDs, descriptions)
│   └── missions.py           # Spacecraft/mission info
└── tests/
    ├── test_agent.py
    └── test_autoplot.py
```

---

## Implementation Steps

### Step 1: Environment Setup

**Goal**: Get Autoplot working from Python.

#### 1.1 Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install google-genai          # Gemini API (unified SDK)
pip install jpype1               # Java-Python bridge
pip install requests             # For HAPI/HTTP requests
pip install python-dotenv        # Environment variables
```

#### 1.2 Install Autoplot

Download Autoplot single-jar from https://autoplot.org/latest/

```bash
# Create a directory for Autoplot
mkdir -p ~/autoplot
cd ~/autoplot

# Download the single jar (update version as needed)
wget https://autoplot.org/jnlp/latest/autoplot.jar
```

#### 1.3 Set Up Configuration

Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
AUTOPLOT_JAR=/path/to/autoplot.jar
```

Create `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AUTOPLOT_JAR = os.getenv("AUTOPLOT_JAR", "~/autoplot/autoplot.jar")
DEFAULT_TIME_RANGE = "2024-01-01 to 2024-01-07"
```

#### 1.4 Verify Autoplot Connection

Create `autoplot_bridge/connection.py` and test:
```python
"""
Autoplot connection via JPype.
Run this file directly to test: python -m autoplot_bridge.connection
"""
import jpype
import jpype.imports
from pathlib import Path
import sys

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import AUTOPLOT_JAR
else:
    from config import AUTOPLOT_JAR


def init_autoplot():
    """Initialize JVM with Autoplot on classpath."""
    if not jpype.isJVMStarted():
        jar_path = Path(AUTOPLOT_JAR).expanduser().resolve()
        if not jar_path.exists():
            raise FileNotFoundError(f"Autoplot JAR not found: {jar_path}")
        
        jpype.startJVM(classpath=[str(jar_path)])
    
    # Import Autoplot classes
    from org.autoplot import ScriptContext
    return ScriptContext


def get_script_context():
    """Get the Autoplot ScriptContext for issuing commands."""
    ScriptContext = init_autoplot()
    return ScriptContext


if __name__ == "__main__":
    print("Testing Autoplot connection...")
    try:
        ctx = get_script_context()
        print(f"SUCCESS: Connected to Autoplot")
        print(f"ScriptContext: {ctx}")
    except Exception as e:
        print(f"FAILED: {e}")
```

**Verification**: Run `python -m autoplot_bridge.connection` - should print success message.

---

### Step 2: Autoplot Command Wrappers

**Goal**: Create Python functions that execute Autoplot operations.

Create `autoplot_bridge/commands.py`:

```python
"""
Wrapper functions for Autoplot operations.
Each function translates Python calls to Autoplot/Jython commands.
"""
from .connection import get_script_context
import jpype


class AutoplotCommands:
    """Wrapper for common Autoplot operations."""
    
    def __init__(self):
        self._ctx = None
        self._current_uri = None
        self._current_time_range = None
    
    @property
    def ctx(self):
        """Lazy initialization of ScriptContext."""
        if self._ctx is None:
            self._ctx = get_script_context()
        return self._ctx
    
    def plot_cdaweb(self, dataset_id: str, parameter_id: str, time_range: str) -> dict:
        """
        Plot data from CDAWeb.
        
        Args:
            dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI")
            parameter_id: Parameter within dataset (e.g., "Magnitude")
            time_range: Time range string (e.g., "2024-01-01 to 2024-01-07")
        
        Returns:
            dict with status and details
        """
        # Construct Autoplot URI for CDAWeb
        uri = f"vap+cdaweb:ds={dataset_id}&id={parameter_id}&timerange={time_range.replace(' ', '+')}"
        
        try:
            # Execute plot command
            self.ctx.plot(uri)
            
            # Store current state
            self._current_uri = uri
            self._current_time_range = time_range
            
            return {
                "status": "success",
                "uri": uri,
                "dataset": dataset_id,
                "parameter": parameter_id,
                "time_range": time_range
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "uri": uri
            }
    
    def set_time_range(self, time_range: str) -> dict:
        """
        Change the time range of the current plot.
        
        Args:
            time_range: New time range string
        
        Returns:
            dict with status and details
        """
        try:
            # Import Java classes for time range manipulation
            from org.das2.datum import DatumRangeUtil
            
            # Parse and set the time range
            tr = DatumRangeUtil.parseTimeRange(time_range)
            self.ctx.dom.timeRange = tr
            
            self._current_time_range = time_range
            
            return {
                "status": "success",
                "time_range": time_range
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def export_png(self, filepath: str) -> dict:
        """
        Export current plot to PNG file.
        
        Args:
            filepath: Output file path
        
        Returns:
            dict with status and filepath
        """
        try:
            self.ctx.writeToPng(filepath)
            return {
                "status": "success",
                "filepath": filepath
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_current_state(self) -> dict:
        """
        Get information about current plot state.
        
        Returns:
            dict with current plot details
        """
        return {
            "current_uri": self._current_uri,
            "current_time_range": self._current_time_range,
            "has_plot": self._current_uri is not None
        }


# Singleton instance
_commands = None

def get_commands() -> AutoplotCommands:
    """Get or create the AutoplotCommands singleton."""
    global _commands
    if _commands is None:
        _commands = AutoplotCommands()
    return _commands
```

**Verification**: Create a test script to plot sample data.

---

### Step 3: Dataset Knowledge Base

**Goal**: Give the agent knowledge about available datasets.

Create `knowledge/datasets.py`:

```python
"""
Dataset catalog for CDAWeb and other sources.
This provides the agent with knowledge about available data.
"""

CDAWEB_DATASETS = {
    "AC_H2_MFI": {
        "name": "ACE Magnetic Field 1-Hour",
        "spacecraft": "ACE",
        "instrument": "MAG (Magnetometer)",
        "description": "ACE magnetic field data at 1-hour resolution",
        "parameters": {
            "Magnitude": "Magnetic field magnitude (nT)",
            "BGSEc": "Magnetic field vector in GSE coordinates",
            "BGSM": "Magnetic field vector in GSM coordinates"
        },
        "cadence": "1 hour",
        "start_date": "1998-01-01",
        "keywords": ["magnetic field", "mag", "ace", "imf", "interplanetary"]
    },
    "AC_H0_SWE": {
        "name": "ACE Solar Wind Electron 64-Second",
        "spacecraft": "ACE",
        "instrument": "SWEPAM",
        "description": "ACE solar wind plasma data",
        "parameters": {
            "Np": "Proton density (n/cc)",
            "Vp": "Solar wind proton speed (km/s)",
            "Tpr": "Proton temperature (K)"
        },
        "cadence": "64 seconds",
        "start_date": "1998-01-01",
        "keywords": ["solar wind", "plasma", "density", "velocity", "temperature", "ace"]
    },
    "OMNI_HRO_1MIN": {
        "name": "OMNI High-Resolution 1-Minute",
        "spacecraft": "Multi-spacecraft (OMNI)",
        "instrument": "Combined",
        "description": "Combined multi-spacecraft solar wind data propagated to bow shock nose",
        "parameters": {
            "BX_GSE": "IMF Bx GSE (nT)",
            "BY_GSM": "IMF By GSM (nT)",
            "BZ_GSM": "IMF Bz GSM (nT)",
            "flow_speed": "Solar wind flow speed (km/s)",
            "proton_density": "Proton density (n/cc)",
            "SYM_H": "SYM-H index (nT)"
        },
        "cadence": "1 minute",
        "start_date": "1995-01-01",
        "keywords": ["omni", "solar wind", "imf", "sym-h", "geomagnetic", "bow shock"]
    }
}


def search_datasets(query: str) -> list:
    """
    Search datasets by keyword.
    
    Args:
        query: Search string
    
    Returns:
        List of matching dataset IDs with relevance info
    """
    query_lower = query.lower()
    results = []
    
    for dataset_id, info in CDAWEB_DATASETS.items():
        score = 0
        matches = []
        
        # Check name
        if query_lower in info["name"].lower():
            score += 3
            matches.append("name")
        
        # Check spacecraft
        if query_lower in info["spacecraft"].lower():
            score += 2
            matches.append("spacecraft")
        
        # Check keywords
        for kw in info["keywords"]:
            if query_lower in kw or kw in query_lower:
                score += 1
                matches.append(f"keyword:{kw}")
        
        # Check description
        if query_lower in info["description"].lower():
            score += 1
            matches.append("description")
        
        if score > 0:
            results.append({
                "dataset_id": dataset_id,
                "name": info["name"],
                "score": score,
                "matches": matches,
                "parameters": list(info["parameters"].keys())
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_dataset_info(dataset_id: str) -> dict | None:
    """Get full info for a specific dataset."""
    return CDAWEB_DATASETS.get(dataset_id)


def list_parameters(dataset_id: str) -> dict | None:
    """List available parameters for a dataset."""
    info = CDAWEB_DATASETS.get(dataset_id)
    if info:
        return info["parameters"]
    return None
```

---

### Step 4: Agent Tools Definition

**Goal**: Define tools the LLM can call using Gemini's function calling.

Create `agent/tools.py`:

```python
"""
Tool definitions for the Gemini agent.
These define what actions the agent can take.
"""

# Tool schemas for Gemini function calling
TOOLS = [
    {
        "name": "search_datasets",
        "description": "Search for available spacecraft datasets by keyword. Use this when the user asks about data but doesn't specify an exact dataset ID. Returns matching datasets with their parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'magnetic field', 'ACE', 'solar wind')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "plot_data",
        "description": "Load and plot spacecraft data from CDAWeb. Use this when the user wants to visualize data and you know the dataset ID, parameter, and time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "CDAWeb dataset ID (e.g., 'AC_H2_MFI', 'OMNI_HRO_1MIN')"
                },
                "parameter_id": {
                    "type": "string",
                    "description": "Parameter to plot (e.g., 'Magnitude', 'BZ_GSM')"
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range in format 'YYYY-MM-DD to YYYY-MM-DD' or 'YYYY-MM-DD HH:MM to YYYY-MM-DD HH:MM'"
                }
            },
            "required": ["dataset_id", "parameter_id", "time_range"]
        }
    },
    {
        "name": "change_time_range",
        "description": "Change the time range of the current plot. Use this when the user wants to zoom in/out or look at a different time period.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_range": {
                    "type": "string",
                    "description": "New time range in format 'YYYY-MM-DD to YYYY-MM-DD'"
                }
            },
            "required": ["time_range"]
        }
    },
    {
        "name": "export_plot",
        "description": "Export the current plot to a PNG image file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Output filename (will be saved as PNG)"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "get_plot_info",
        "description": "Get information about what is currently plotted. Use this to understand the current state before making changes.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "ask_clarification",
        "description": "Ask the user for clarification when the request is ambiguous or missing required information. Use this instead of guessing.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices to present to the user"
                },
                "context": {
                    "type": "string",
                    "description": "Brief explanation of why you need this information"
                }
            },
            "required": ["question"]
        }
    }
]


def get_tool_schemas():
    """Return tool schemas in Gemini format."""
    return TOOLS
```

---

### Step 5: System Prompts

**Goal**: Define how the agent should behave.

Create `agent/prompts.py`:

```python
"""
System prompts and templates for the agent.
"""

SYSTEM_PROMPT = """You are an intelligent assistant for Autoplot, a scientific data visualization tool for spacecraft and heliophysics data.

## Your Role
Help users visualize spacecraft data by translating their natural language requests into Autoplot operations. You can search for datasets, plot data, change time ranges, and export plots.

## Available Data Sources
You have access to data from CDAWeb (NASA's Coordinated Data Analysis Web), including:
- ACE spacecraft: magnetic field (AC_H2_MFI), solar wind plasma (AC_H0_SWE)
- OMNI: combined multi-spacecraft solar wind data (OMNI_HRO_1MIN)

## Guidelines

### When to Ask for Clarification
Ask clarifying questions when:
- The user mentions a general concept (e.g., "solar wind") but you need a specific dataset/parameter
- The time range is not specified or ambiguous
- Multiple datasets could satisfy the request
- The user's intent is unclear

### When to Proceed
Proceed without asking when:
- The user provides specific dataset ID, parameter, and time range
- The request is a clear follow-up to a previous action (e.g., "zoom in" on current plot)
- You can make a reasonable default choice (e.g., use last 7 days if no time specified for a quick look)

### Response Style
- Be concise but informative
- When you perform an action, briefly confirm what you did
- If offering choices, keep them to 3-4 options
- Use plain language, but you can include technical terms when relevant

### Time Range Formats
Accept flexible time inputs and convert to standard format:
- "last week" → calculate actual dates
- "January 2024" → "2024-01-01 to 2024-01-31"
- "2024-01-15" → "2024-01-15 to 2024-01-16" (single day)

## Current State
You maintain awareness of what is currently plotted so users can say things like "zoom in" or "export this".
"""


def get_system_prompt():
    """Return the system prompt for the agent."""
    return SYSTEM_PROMPT


def format_tool_result(tool_name: str, result: dict) -> str:
    """Format a tool result for inclusion in conversation."""
    if result.get("status") == "error":
        return f"Error executing {tool_name}: {result.get('message', 'Unknown error')}"
    
    if tool_name == "plot_data":
        return f"Successfully plotted {result['dataset']} / {result['parameter']} for {result['time_range']}"
    
    if tool_name == "search_datasets":
        if not result.get("results"):
            return "No matching datasets found."
        matches = result["results"][:5]  # Top 5
        lines = ["Found these datasets:"]
        for m in matches:
            lines.append(f"- {m['dataset_id']}: {m['name']} (parameters: {', '.join(m['parameters'])})")
        return "\n".join(lines)
    
    if tool_name == "change_time_range":
        return f"Time range changed to {result['time_range']}"
    
    if tool_name == "export_plot":
        return f"Plot exported to {result['filepath']}"
    
    if tool_name == "get_plot_info":
        if not result.get("has_plot"):
            return "No plot is currently displayed."
        return f"Currently showing: {result['current_uri']} for time range {result['current_time_range']}"
    
    return str(result)
```

---

### Step 6: Agent Core Logic

**Goal**: Implement the main agent that orchestrates LLM calls and tool execution.

Create `agent/core.py`:

```python
"""
Core agent logic - orchestrates Gemini calls and tool execution.
"""
from google import genai
from google.genai import types
from typing import Generator
import json

from config import GOOGLE_API_KEY
from .tools import get_tool_schemas
from .prompts import get_system_prompt, format_tool_result
from knowledge.datasets import search_datasets, get_dataset_info, list_parameters
from autoplot_bridge.commands import get_commands


class AutoplotAgent:
    """Main agent class that handles conversation and tool execution."""
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize model with tools
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",  # Fast and cost-effective
            system_instruction=get_system_prompt(),
            tools=self._convert_tools_to_gemini_format()
        )
        
        # Conversation history
        self.chat = self.model.start_chat(history=[])
        
        # Autoplot commands
        self.autoplot = get_commands()
    
    def _convert_tools_to_gemini_format(self):
        """Convert our tool schemas to Gemini's expected format."""
        tools = get_tool_schemas()
        
        # Gemini expects function declarations
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        
        return function_declarations
    
    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a tool and return the result."""
        
        if tool_name == "search_datasets":
            results = search_datasets(tool_args["query"])
            return {"status": "success", "results": results}
        
        elif tool_name == "plot_data":
            return self.autoplot.plot_cdaweb(
                dataset_id=tool_args["dataset_id"],
                parameter_id=tool_args["parameter_id"],
                time_range=tool_args["time_range"]
            )
        
        elif tool_name == "change_time_range":
            return self.autoplot.set_time_range(tool_args["time_range"])
        
        elif tool_name == "export_plot":
            filename = tool_args["filename"]
            if not filename.endswith(".png"):
                filename += ".png"
            return self.autoplot.export_png(filename)
        
        elif tool_name == "get_plot_info":
            return self.autoplot.get_current_state()
        
        elif tool_name == "ask_clarification":
            # This is handled specially - return the question to show user
            return {
                "status": "clarification_needed",
                "question": tool_args["question"],
                "options": tool_args.get("options", []),
                "context": tool_args.get("context", "")
            }
        
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def process_message(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        Handles tool calls automatically.
        
        Args:
            user_message: The user's input
        
        Returns:
            The agent's text response
        """
        # Send message to Gemini
        response = self.chat.send_message(user_message)
        
        # Check if the model wants to call tools
        while response.candidates[0].content.parts:
            # Check for function calls
            function_calls = [
                part.function_call 
                for part in response.candidates[0].content.parts 
                if hasattr(part, 'function_call') and part.function_call.name
            ]
            
            if not function_calls:
                # No function calls, just text response
                break
            
            # Execute each function call
            function_responses = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args)
                
                print(f"  [Executing tool: {tool_name}]")  # Debug output
                
                result = self._execute_tool(tool_name, tool_args)
                
                # Handle clarification specially
                if result.get("status") == "clarification_needed":
                    # Return the clarification question directly
                    question = result["question"]
                    if result.get("options"):
                        question += "\n\nOptions:\n" + "\n".join(
                            f"  {i+1}. {opt}" for i, opt in enumerate(result["options"])
                        )
                    return question
                
                function_responses.append({
                    "name": tool_name,
                    "response": result
                })
            
            # Send function results back to the model
            response = self.chat.send_message(
                [{"function_response": fr} for fr in function_responses]
            )
        
        # Extract text response
        text_parts = [
            part.text 
            for part in response.candidates[0].content.parts 
            if hasattr(part, 'text')
        ]
        
        return "\n".join(text_parts) if text_parts else "I've completed the action."
    
    def reset(self):
        """Reset conversation history."""
        self.chat = self.model.start_chat(history=[])


def create_agent() -> AutoplotAgent:
    """Factory function to create a new agent instance."""
    return AutoplotAgent()
```

---

### Step 7: Main Entry Point

**Goal**: Create the interactive conversation loop.

Create `main.py`:

```python
#!/usr/bin/env python3
"""
Autoplot Agentic Interface - Main Entry Point

Run this to start an interactive conversation with the Autoplot agent.
"""
import sys
from agent.core import create_agent


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("  Autoplot Natural Language Interface")
    print("=" * 60)
    print()
    print("I can help you visualize spacecraft data. Try commands like:")
    print("  - 'Show me ACE magnetic field data for last week'")
    print("  - 'Plot solar wind speed from January 2024'")
    print("  - 'What datasets are available for the Sun?'")
    print()
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'reset' to clear conversation history.")
    print("-" * 60)
    print()


def main():
    """Main conversation loop."""
    print_welcome()
    
    try:
        agent = create_agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure GOOGLE_API_KEY is set in .env file")
        sys.exit(1)
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset()
                print("Conversation reset.")
                continue
            
            # Process the message
            print()  # Blank line before response
            response = agent.process_message(user_input)
            print(f"Agent: {response}")
            print()  # Blank line after response
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("You can continue the conversation or type 'reset' to start fresh.")


if __name__ == "__main__":
    main()
```

---

### Step 8: Requirements File

Create `requirements.txt`:

```
google-genai>=1.60.0
jpype1>=1.4.0
python-dotenv>=1.0.0
requests>=2.28.0
```

---

### Step 9: Testing

Create `tests/test_agent.py`:

```python
"""
Basic tests for the agent.
Run with: python -m pytest tests/
"""
import pytest
from knowledge.datasets import search_datasets, get_dataset_info


def test_search_datasets_ace():
    """Test searching for ACE datasets."""
    results = search_datasets("ACE")
    assert len(results) > 0
    # Should find ACE magnetic field data
    dataset_ids = [r["dataset_id"] for r in results]
    assert "AC_H2_MFI" in dataset_ids


def test_search_datasets_magnetic_field():
    """Test searching by topic."""
    results = search_datasets("magnetic field")
    assert len(results) > 0


def test_get_dataset_info():
    """Test getting specific dataset info."""
    info = get_dataset_info("AC_H2_MFI")
    assert info is not None
    assert "Magnitude" in info["parameters"]


def test_search_no_results():
    """Test search with no matches."""
    results = search_datasets("xyz_nonexistent_12345")
    assert len(results) == 0
```

---

## Running the Project

### First-Time Setup

```bash
# 1. Clone/create the project directory
mkdir autoplot-agent && cd autoplot-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Autoplot
mkdir -p ~/autoplot
wget -O ~/autoplot/autoplot.jar https://autoplot.org/jnlp/latest/autoplot.jar

# 5. Create .env file with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
echo "AUTOPLOT_JAR=~/autoplot/autoplot.jar" >> .env

# 6. Run the agent
python main.py
```

### Example Conversation

```
You: Show me ACE magnetic field data
Agent: I found the ACE magnetic field dataset. What time range would you like to view?

Options:
  1. Last 7 days
  2. Last month
  3. Specify a custom range

You: Last week
Agent: I've plotted ACE magnetic field magnitude (AC_H2_MFI/Magnitude) for the past 7 days.

You: Can you zoom in on just the last 2 days?
Agent: Done - the plot now shows the last 2 days of data.

You: Export this to a file called ace_mag
Agent: Plot exported to ace_mag.png
```

---

## Extension Points (Future Phases)

### Phase 2: More Data Sources
- Add HAPI server support
- Add more CDAWeb datasets
- Support local CDF files

### Phase 3: Advanced Visualization
- Multiple panels
- Spectrograms
- Custom color tables

### Phase 4: Data Processing
- Filtering
- FFT/power spectra
- Statistics

### Phase 5: Model Routing for Cost Optimization
- Use smaller model for simple commands
- Use larger model for complex interpretation

---

## Troubleshooting

### Common Issues

**JVM fails to start**
- Ensure Java is installed: `java -version`
- Check AUTOPLOT_JAR path is correct
- Try absolute path instead of `~`

**Gemini API errors**
- Verify API key is correct
- Check quota/billing in Google Cloud Console
- Try `gemini-1.5-flash` if `pro` has issues

**Autoplot plot doesn't appear**
- Autoplot needs a display; run on machine with GUI or use Xvfb
- For headless: `Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99`

**Import errors**
- Make sure you're running from project root
- Check virtual environment is activated
- Verify all `__init__.py` files exist

---

## Notes for Claude Code

When implementing this project:

1. **Start with Step 1** - verify Autoplot connection before building agent logic
2. **Test incrementally** - each step should be testable in isolation
3. **Keep the knowledge base simple** - start with just 3 datasets
4. **The ask_clarification tool is key** - it enables the interactive, conversational experience
5. **Gemini's function calling** handles most of the complexity - you define tools, it decides when to use them

The minimal working example should be achievable with the code provided. Focus on getting the basic conversation loop working before adding features.
