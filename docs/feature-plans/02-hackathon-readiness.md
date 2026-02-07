# Feature 2: Gemini Hackathon Readiness

## Goal

Make helio-ai-agent a compelling, demo-ready submission for a Devpost Gemini API hackathon. The engineering is already strong — what's missing is presentation, a web interface, Gemini-specific differentiators, and a narrative that makes judges care.

This plan covers four workstreams ranked by impact-per-hour.

---

## Workstream 1: Gradio Web UI (highest impact, ~4-6 hours)

### Why

Judges can't `pip install jpype1`. A browser-based demo is table stakes. Gradio's `ChatInterface` + `share=True` gives a public URL with zero deployment.

### Architecture

```
gradio_app.py  (new, project root)
  |
  v
agent.core.create_agent(gui_mode=False)   # headless, PNG export
  |
  v
gr.ChatInterface  ←→  OrchestratorAgent.process_message()
  |
  +--- gr.Image (latest plot PNG, auto-updated)
  +--- gr.Dataframe (last fetched data preview)
  +--- gr.File (CSV download)
  +--- gr.Textbox (status: plan progress, token usage)
```

### Files to Create/Modify

| File | Change |
|------|--------|
| `gradio_app.py` (new) | Main Gradio app |
| `requirements.txt` | Add `gradio>=4.0.0` |
| `agent/core.py` | Minor: expose last-exported PNG path, add `get_last_plot_path()` method |
| `data_ops/store.py` | Minor: add `get_last_entry_preview()` → first 10 rows as dict |

### Implementation

```python
# gradio_app.py — minimal skeleton
import gradio as gr
from agent.core import create_agent

agent = create_agent(verbose=False, gui_mode=False)

def respond(message, history):
    response = agent.process_message(message)
    return response

def get_latest_plot():
    path = agent.get_last_plot_path()
    return path  # gr.Image handles None gracefully

with gr.Blocks(title="Helio AI Agent") as demo:
    gr.Markdown("# Helio AI Agent\nExplore spacecraft data with natural language.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(fn=respond)
        with gr.Column(scale=1):
            plot_img = gr.Image(label="Latest Plot", every=3)  # polls
            status = gr.Textbox(label="Status", interactive=False)

    plot_img.attach_load_event(get_latest_plot, every=3)

demo.launch(share=True)  # public URL for judges
```

### Key Decisions

- **Headless mode** (not `--gui`): Gradio runs server-side, no Swing window needed.
- **Plot delivery**: Agent exports PNG to a temp path; Gradio polls and displays it. No WebSocket complexity.
- **State**: Single agent instance, single-user. Fine for a hackathon demo.
- **`share=True`**: Free Gradio tunnel — gives a `https://xxxxx.gradio.live` URL valid for 72 hours.

### Stretch: Inline Plot in Chat

Instead of a sidebar image, return the plot as a chat message:

```python
def respond(message, history):
    response = agent.process_message(message)
    plot_path = agent.get_last_plot_path()
    if plot_path:
        return [response, gr.Image(plot_path)]
    return response
```

This makes the conversation flow feel more natural — text + plot interleaved.

---

## Workstream 2: Demo Scenario & Video (~2-3 hours)

### Why

Hackathons are 70% demo, 30% code. A polished 3-minute video with a real science scenario is worth more than any feature.

### The Narrative

> *"During a coronal mass ejection, scientists need to compare solar wind data across multiple spacecraft to track the event as it propagates through the solar system. This normally requires hours of manual scripting — finding datasets, downloading data, aligning time ranges, computing derived quantities, and building comparison plots. With Helio AI Agent, it takes one sentence."*

### Demo Script (3 minutes)

| Time | Action | What Judges See |
|------|--------|-----------------|
| 0:00-0:20 | Title card + problem statement | "Scientists spend hours scripting. We made it conversational." |
| 0:20-0:50 | "Show me ACE magnetic field data for last week" | Chat message → plot appears in UI |
| 0:50-1:20 | "Compute the magnetic field magnitude and overlay it" | Data pipeline: fetch → compute → overplot |
| 1:20-1:50 | "Now compare with Parker Solar Probe for the same period" | Multi-mission: two datasets on one plot |
| 1:50-2:20 | "Describe the data — any interesting features?" | `describe_data` → statistical summary in chat |
| 2:20-2:40 | "Export the plot and save the data as CSV" | PNG + CSV download links appear |
| 2:40-3:00 | Architecture slide + Gemini callout | 3-agent routing, function calling, 8 spacecraft |

### Recording Tips

- Use OBS Studio or Loom
- Record the Gradio web UI in a browser (not the CLI)
- 1080p, clean desktop, large font
- Add captions for the agent's responses
- Background music optional but helps

### Files to Create

| File | Purpose |
|------|---------|
| `demo/demo_script.md` | Exact prompts to type, expected outputs, timing |
| `demo/slides.md` | Architecture diagram, tech stack, Gemini usage |

---

## Workstream 3: Gemini-Specific Differentiators (~3-4 hours)

### Why

For a **Gemini** hackathon, judges want to see you pushing Gemini's unique capabilities, not just using it as a generic chat API. The current function-calling usage is solid but common.

### 3A: Multimodal Input — "Analyze This Plot" (high impact)

Let users upload a screenshot of a plot and ask questions about it.

```
User: [uploads screenshot] "What spacecraft is this data from? Can you reproduce it?"
Agent: "This looks like ACE magnetic field data from January 2024. Let me fetch and plot it..."
```

**Implementation:**
- Gradio already supports image upload in chat
- Send the image to Gemini as a `Part` with `inline_data` (Gemini is natively multimodal)
- Add a system prompt section: "When the user uploads an image of a plot, analyze it..."
- Gemini identifies the spacecraft, parameter, and time range, then calls tools to reproduce it

**Files to modify:**
| File | Change |
|------|--------|
| `agent/core.py` | Accept image parts in `process_message()`, pass to Gemini as multimodal content |
| `gradio_app.py` | Enable `multimodal=True` in ChatInterface |
| System prompt | Add instructions for image analysis |

**Effort:** ~2 hours. Very high "wow factor" for judges.

### 3B: Grounding with Google Search — Space Weather Context (medium impact)

Use Gemini's built-in Google Search grounding to provide real-world context.

```
User: "What was happening with the Sun in January 2024?"
Agent: [uses Google Search grounding] "In January 2024, several X-class flares were observed..."
Agent: "Would you like me to pull PSP and ACE data from that period to see the solar wind response?"
```

**Implementation:**
- Add `google_search` as a tool in the Gemini API config
- The LLM decides when web context is useful (no code changes to routing)
- Bridges the gap between "data tool" and "science assistant"

**Files to modify:**
| File | Change |
|------|--------|
| `agent/core.py` | Add `google_search_retrieval` tool config to Gemini client |
| System prompt | Add: "Use Google Search when the user asks about solar events, space weather, or context" |

**Effort:** ~1 hour. Easy win.

### 3C: Structured JSON Output for Plans (already done, highlight it)

The multi-step planner already uses Gemini's JSON mode (`response_mime_type: "application/json"`). This is a Gemini differentiator — highlight it in the submission.

### 3D: Context Caching for Mission Knowledge (stretch)

Use Gemini's context caching API to cache the large mission knowledge base, reducing cost and latency for repeated queries about the same spacecraft.

**Effort:** ~2 hours. Good for cost optimization narrative.

---

## Workstream 4: Devpost Submission Polish (~2 hours)

### README / Devpost Page

Structure for maximum judge impact:

```markdown
# Helio AI Agent

> Talk to spacecraft data. Powered by Gemini.

## The Problem
Scientists spend hours writing scripts to find, download, process,
and visualize heliophysics data from NASA's spacecraft fleet.

## The Solution
A conversational AI agent that handles the entire workflow — from
"Show me Parker Solar Probe magnetic field data" to a publication-ready
plot — in seconds.

## How It Uses Gemini
- **Function calling**: 13 tools across 3 specialized agents
- **Multi-agent routing**: Orchestrator delegates to mission specialists
- **Multimodal**: Upload a plot screenshot, get it reproduced
- **Google Search grounding**: Real-time space weather context
- **JSON mode**: Structured task decomposition for complex queries

## Demo
[3-minute video link]

## Try It
[Gradio share link]

## Architecture
[diagram]

## Built With
Python, Gemini 2.5-Flash, Autoplot, JPype, HAPI, Gradio
```

### Key Judging Criteria to Hit

| Criterion | How We Score |
|-----------|-------------|
| **Technical complexity** | 3-agent architecture, Java bridge, AST sandboxing, 461 tests |
| **Gemini integration depth** | Function calling, multimodal, grounding, JSON mode |
| **Real-world impact** | Actual NASA data, real science workflows |
| **Demo quality** | Live Gradio URL + polished video |
| **Completeness** | End-to-end: discovery → fetch → compute → plot → export |

### Quick Wins to Include

**Loading/Thinking Indicators**: Show "Fetching ACE data..." or "Computing magnitude..." in the Gradio UI while the agent works. Already have tool call info from verbose mode — surface it.

**Example Prompts**: Pre-populate the Gradio UI with clickable example prompts:

```python
gr.ChatInterface(
    fn=respond,
    examples=[
        "Show me ACE magnetic field data for last week",
        "Compare solar wind speed from PSP and Wind for January 2024",
        "Fetch OMNI data and compute a 1-hour running average",
    ],
)
```

**Token Usage Display**: Show token count and cost estimate in the UI footer. Already tracked in `agent.get_token_usage()`.

**Error Recovery UX**: When a tool fails, show a friendly message instead of a traceback. The agent already handles this in conversation — just make sure Gradio surfaces it cleanly.

---

## Priority Order

For maximum hackathon ROI, work in this order:

| Priority | Workstream | Hours | Impact |
|----------|-----------|-------|--------|
| 1 | **Gradio Web UI** (WS1) | 4-6 | Must-have. No UI = no demo. |
| 2 | **Demo scenario + video** (WS2) | 2-3 | Must-have. Judges watch the video first. |
| 3 | **Multimodal input** (WS3A) | 2 | High wow factor. "Upload a plot" is memorable. |
| 4 | **Google Search grounding** (WS3B) | 1 | Easy Gemini checkbox. |
| 5 | **Devpost page + quick wins** (WS4) | 3-4 | Frames everything for judges. |

**Total: ~12-16 hours** to go from "impressive CLI tool" to "hackathon contender."

**Minimum viable submission (~7 hours):** WS1 (Gradio) + WS2 (demo video) + WS4 (Devpost page).

---

## What NOT to Build

- Don't add more spacecraft or data sources (judges won't notice 8 vs 12)
- Don't refactor the architecture (it's already good)
- Don't add authentication or multi-user support (single-user is fine for demo)
- Don't write more tests (461 is already impressive — just mention the number)
- Don't build a custom frontend (Gradio is faster and good enough)
- Don't build an MCP server — the value of this project is the intelligence layer (mission knowledge, dataset curation, multi-agent routing), not the raw HAPI API calls. A thin MCP wrapper over search/fetch loses the smart dataset selection that makes the agent useful. Not worth the time for a hackathon.

---

## Verification

After completing WS1-WS4:

1. `python gradio_app.py` launches, shows chat + plot panel
2. Gradio `share=True` produces a working public URL
3. The demo script (WS2) runs through without errors
4. Uploading a plot screenshot triggers Gemini multimodal analysis (WS3A)
5. Asking "What solar events happened in Jan 2024?" uses Google Search (WS3B)
6. The Devpost page has: video, live link, architecture diagram, Gemini usage section
