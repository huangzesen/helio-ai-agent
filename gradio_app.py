#!/usr/bin/env python3
"""
Gradio Web UI for the Helio AI Agent.

Provides a browser-based chat interface with interactive Plotly plots,
data table sidebar, and token usage tracking.

Usage:
    python gradio_app.py                # Launch on localhost:7860
    python gradio_app.py --share        # Generate public URL
    python gradio_app.py --port 8080    # Custom port
    python gradio_app.py --verbose      # Show tool call details
"""

import argparse
import gc
import io
import sys
import threading
from datetime import datetime, timedelta, timezone

import gradio as gr
import pandas as pd


# ---------------------------------------------------------------------------
# Globals (initialized in main())
# ---------------------------------------------------------------------------
_agent = None
_verbose = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_current_figure():
    """Return the current Plotly figure from the renderer, or None."""
    if _agent is None:
        return None
    return _agent.get_plotly_figure()


def _build_data_table() -> list[list]:
    """Build rows for the sidebar data table from the in-memory store."""
    from data_ops.store import get_store

    store = get_store()
    entries = store.list_entries()
    if not entries:
        return []
    rows = []
    for e in entries:
        rows.append([
            e["label"],
            e["shape"],
            e["num_points"],
            e.get("units", ""),
            e.get("time_min", "")[:19] if e.get("time_min") else "",
            e.get("time_max", "")[:19] if e.get("time_max") else "",
            e.get("source", ""),
        ])
    return rows


def _format_tokens() -> str:
    """Format token usage and memory as markdown for the sidebar."""
    from data_ops.store import get_store
    store = get_store()
    mem_bytes = store.memory_usage_bytes()
    if mem_bytes < 1024 * 1024:
        mem_str = f"{mem_bytes / 1024:.0f} KB"
    else:
        mem_str = f"{mem_bytes / (1024 * 1024):.1f} MB"
    mem_line = f"**Data in RAM:** {mem_str} ({len(store)} entries)"

    if _agent is None:
        return mem_line
    usage = _agent.get_token_usage()
    if usage["api_calls"] == 0:
        return f"{mem_line}  \n*No API calls yet*"
    return (
        f"{mem_line}  \n"
        f"**Input:** {usage['input_tokens']:,}  \n"
        f"**Output:** {usage['output_tokens']:,}  \n"
        f"**Total:** {usage['total_tokens']:,}  \n"
        f"**API calls:** {usage['api_calls']}"
    )


def _get_label_choices() -> list[str]:
    """Return list of labels currently in the DataStore."""
    from data_ops.store import get_store
    return [e["label"] for e in get_store().list_entries()]


def _get_mission_choices() -> list[str]:
    """Return list of mission IDs for the browse dropdown."""
    from knowledge.mission_loader import get_mission_ids
    return get_mission_ids()


def _on_mission_change(mission_id: str):
    """Populate dataset dropdown when a mission is selected."""
    from knowledge.hapi_client import browse_datasets

    if not mission_id:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            "",
        )

    datasets = browse_datasets(mission_id) or []
    choices = []
    for d in datasets:
        n_params = d.get("parameter_count", "?")
        start = d.get("start_date", "?")[:4]
        stop = d.get("stop_date", "?")[:4]
        label = f"{d['id']}  ({n_params} params, {start}--{stop})"
        choices.append((label, d["id"]))

    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=[], value=None),
        "",
    )


def _on_dataset_change(dataset_id: str):
    """Populate parameter dropdown, info, and time pickers when a dataset is selected."""
    from knowledge.hapi_client import list_parameters, get_dataset_time_range

    if not dataset_id:
        return gr.update(choices=[], value=None), "", gr.skip(), gr.skip()

    params = list_parameters(dataset_id)
    choices = []
    for p in params:
        desc = p.get("description", "")
        units = p.get("units", "")
        parts = [p["name"]]
        if desc:
            parts.append(desc)
        if units:
            parts.append(f"[{units}]")
        choices.append((" -- ".join(parts), p["name"]))

    time_range = get_dataset_time_range(dataset_id)
    info_parts = [f"**{dataset_id}** — {len(params)} parameters"]

    # Default: end = min(dataset stop, now), start = end - 7 days
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    end_dt = now
    start_dt = now - timedelta(days=7)
    if time_range:
        start_avail = time_range["start"][:10]
        stop_avail = time_range["stop"][:10]
        info_parts.append(f"Available: {start_avail} to {stop_avail}")
        try:
            stop_date = datetime.fromisoformat(
                time_range["stop"][:19]
            ).replace(tzinfo=timezone.utc)
            end_dt = min(stop_date, now)
            start_dt = end_dt - timedelta(days=7)
        except ValueError:
            pass

    return (
        gr.update(choices=choices, value=None),
        "\n\n".join(info_parts),
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _on_fetch_click(mission, dataset, param, start_time, end_time, history):
    """Directly fetch HAPI data into the store, then notify the agent."""
    if not dataset or not param:
        return history, gr.skip(), gr.skip(), gr.skip(), "", gr.skip(), gr.skip(), gr.skip()

    from data_ops.fetch import fetch_hapi_data
    from data_ops.store import get_store, DataEntry

    # Convert picker strings to ISO format for HAPI
    start_iso = (start_time or "").replace(" ", "T") + "Z"
    end_iso = (end_time or "").replace(" ", "T") + "Z"

    # Direct fetch — exact IDs, no LLM interpretation
    try:
        result = fetch_hapi_data(
            dataset_id=dataset,
            parameter_id=param,
            time_min=start_iso,
            time_max=end_iso,
        )
    except Exception as e:
        error_msg = f"Fetch failed: {e}"
        history = history + [
            {"role": "assistant", "content": error_msg},
        ]
        return history, gr.skip(), gr.skip(), gr.skip(), "", gr.skip(), gr.skip(), gr.skip()

    label = f"{dataset}.{param}"
    entry = DataEntry(
        label=label,
        data=result["data"],
        units=result["units"],
        description=result["description"],
        source="hapi",
    )
    get_store().put(entry)
    n_points = len(result["data"])
    del result  # free fetch intermediates
    gc.collect()

    # Notify the agent so it has context for follow-up questions
    notify_msg = (
        f"[User fetched data via sidebar] "
        f"{param} from {dataset} ({start_time} to {end_time}) "
        f"is now in memory as '{label}' with {n_points} points."
    )
    try:
        agent_reply, verbose_text = _capture_agent_call(_agent.process_message, notify_msg)
    except Exception:
        agent_reply = f"Fetched **{label}** — {n_points} points."
        verbose_text = ""

    history = history + [
        {"role": "user", "content": notify_msg},
        {"role": "assistant", "content": agent_reply},
    ]

    # Refresh sidebar state
    fig = _get_current_figure()
    data_rows = _build_data_table()
    token_text = _format_tokens()
    label_choices = _get_label_choices()
    selected = label if label in label_choices else (
        label_choices[-1] if label_choices else None
    )
    preview = _preview_data(selected) if selected else None

    return (
        history, fig, data_rows, token_text, "",
        gr.update(choices=label_choices, value=selected),
        preview,
        verbose_text,
    )


def _preview_data(label: str) -> list[list] | None:
    """Return head+tail preview rows for a DataStore label."""
    if not label:
        return None
    from data_ops.store import get_store
    entry = get_store().get(label)
    if entry is None:
        return None
    df = entry.data
    n = len(df)
    # Slice FIRST to avoid copying the entire DataFrame
    if n <= 20:
        subset = df
    else:
        subset = pd.concat([df.head(10), df.tail(10)])
    # Only copy the small subset for formatting
    subset = subset.copy()
    subset.insert(0, "timestamp", subset.index.strftime("%Y-%m-%d %H:%M:%S"))
    subset = subset.reset_index(drop=True)
    num_cols = subset.select_dtypes(include="number").columns
    subset[num_cols] = subset[num_cols].round(4)
    if n <= 20:
        return subset.values.tolist()
    rows = subset.values.tolist()
    sep = [["..."] * len(subset.columns)]
    return rows[:10] + sep + rows[10:]


# ---------------------------------------------------------------------------
# Core response function
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Writes to both the original stream and a StringIO buffer."""

    def __init__(self, original, buffer):
        self.original = original
        self.buffer = buffer

    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


_capture_lock = threading.Lock()


def _capture_agent_call(fn, *args, **kwargs):
    """Call a function, capturing all terminal output if verbose mode is on.

    Tees stdout and stderr to a buffer so every print() and logging message
    that would appear in the terminal also gets collected for the browser UI.

    Returns (result, verbose_text). verbose_text is empty if not verbose.
    """
    if not _verbose:
        return fn(*args, **kwargs), ""

    buf = io.StringIO()

    with _capture_lock:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeWriter(old_stdout, buf)
        sys.stderr = _TeeWriter(old_stderr, buf)
        try:
            result = fn(*args, **kwargs)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    verbose_text = buf.getvalue().rstrip()
    return result, verbose_text


def respond(message: str, history: list[dict]) -> tuple:
    """Process a user message and return updated UI state.

    Args:
        message: The user's chat input.
        history: Chat history in messages format [{role, content}, ...].

    Returns:
        Tuple of (history, plotly_figure, data_table, token_text, textbox_value,
                  label_dropdown_update, preview_data, verbose_text).
    """
    if not message.strip():
        return history, gr.skip(), gr.skip(), gr.skip(), "", gr.skip(), gr.skip(), gr.skip()

    # Append user message
    history = history + [{"role": "user", "content": message}]

    # Call the agent (capture verbose output)
    try:
        response_text, verbose_text = _capture_agent_call(_agent.process_message, message)
    except Exception as e:
        response_text = f"Error: {e}"
        verbose_text = ""

    # Append assistant text response
    history = history + [{"role": "assistant", "content": response_text}]

    # Get current plotly figure (may be None if nothing plotted yet)
    fig = _get_current_figure()

    # Build sidebar state
    data_rows = _build_data_table()
    token_text = _format_tokens()

    # Update data preview dropdown
    label_choices = _get_label_choices()
    selected = label_choices[-1] if label_choices else None
    preview = _preview_data(selected) if selected else None

    return (
        history, fig, data_rows, token_text, "",
        gr.update(choices=label_choices, value=selected),
        preview,
        verbose_text,
    )


def reset_session() -> tuple:
    """Reset the agent, data store, and all UI state."""
    if _agent is not None:
        _agent.reset()

    from data_ops.store import get_store
    get_store().clear()

    return [], None, [], "*Session reset*", "", gr.update(choices=[], value=None), None, ""


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

EXAMPLES = [
    "What spacecraft data is available?",
    "Show me ACE magnetic field data for last week",
    "Plot Solar Orbiter proton density for January 2024",
    "Fetch Wind magnetic field and compute the magnitude",
    "Compare ACE and Wind magnetic field magnitude for 2024-01-10 to 2024-01-17",
    "Zoom in to January 12-14",
    "Describe the data",
    "Export the plot as a PNG",
]


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(title="Helio AI Agent") as app:
        # ---- Header ----
        gr.Markdown(
            "# Helio AI Agent\n"
            "Natural language interface for spacecraft data visualization "
            "powered by Plotly and Gemini."
        )

        # ---- Full-width plot (hero element) ----
        plotly_plot = gr.Plot(label="Interactive Plot")

        with gr.Row():
            # ---- Main column: Chat ----
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=400,
                    label="Chat",
                    placeholder=(
                        "Ask me about spacecraft data! Try:\n"
                        "\"Show me ACE magnetic field data for last week\""
                    ),
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg_input,
                    label="Try these examples",
                )

                # Verbose debug output (only when --verbose)
                if _verbose:
                    verbose_output = gr.Textbox(
                        label="Debug Log",
                        value="Verbose mode enabled. Debug output will appear here after each message.",
                        lines=6,
                        max_lines=20,
                        interactive=False,
                    )

            # ---- Sidebar: data & controls ----
            with gr.Column(scale=1):
                with gr.Accordion("Browse & Fetch", open=False):
                    mission_dropdown = gr.Dropdown(
                        label="Mission",
                        choices=_get_mission_choices(),
                        interactive=True,
                    )
                    dataset_dropdown = gr.Dropdown(
                        label="Dataset",
                        choices=[],
                        interactive=True,
                    )
                    param_dropdown = gr.Dropdown(
                        label="Parameter",
                        choices=[],
                        interactive=True,
                    )
                    browse_info = gr.Markdown(value="")
                    _default_end = datetime.now(tz=timezone.utc).replace(
                        microsecond=0,
                    )
                    _default_start = _default_end - timedelta(days=7)
                    start_dt_picker = gr.DateTime(
                        label="Start",
                        value=_default_start.strftime("%Y-%m-%d %H:%M:%S"),
                        include_time=True,
                        type="string",
                    )
                    end_dt_picker = gr.DateTime(
                        label="End",
                        value=_default_end.strftime("%Y-%m-%d %H:%M:%S"),
                        include_time=True,
                        type="string",
                    )
                    fetch_btn = gr.Button(
                        "Fetch", variant="primary",
                    )
                data_table = gr.Dataframe(
                    headers=["Label", "Shape", "Points", "Units",
                             "Start", "End", "Source"],
                    label="Data in Memory",
                    interactive=False,
                    wrap=True,
                    row_count=(0, "dynamic"),
                )
                label_dropdown = gr.Dropdown(
                    label="Preview Data in Memory",
                    choices=[],
                    interactive=True,
                )
                data_preview = gr.Dataframe(
                    label="Data Preview",
                    interactive=False,
                    wrap=True,
                    row_count=(0, "dynamic"),
                )
                token_display = gr.Markdown(
                    value="*No API calls yet*",
                    label="Token Usage",
                )
                reset_btn = gr.Button("Reset Session", variant="secondary")

        if not _verbose:
            verbose_output = gr.State("")  # hidden placeholder

        # ---- Event wiring ----
        all_outputs = [chatbot, plotly_plot, data_table, token_display,
                       msg_input, label_dropdown, data_preview, verbose_output]

        send_event_args = dict(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=all_outputs,
        )
        send_btn.click(**send_event_args)
        msg_input.submit(**send_event_args)

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=all_outputs,
        )

        label_dropdown.change(
            fn=_preview_data,
            inputs=[label_dropdown],
            outputs=[data_preview],
        )

        # Browse & Plot cascade
        mission_dropdown.change(
            fn=_on_mission_change,
            inputs=[mission_dropdown],
            outputs=[dataset_dropdown, param_dropdown, browse_info],
        )
        dataset_dropdown.change(
            fn=_on_dataset_change,
            inputs=[dataset_dropdown],
            outputs=[param_dropdown, browse_info, start_dt_picker, end_dt_picker],
        )
        fetch_btn.click(
            fn=_on_fetch_click,
            inputs=[mission_dropdown, dataset_dropdown, param_dropdown,
                    start_dt_picker, end_dt_picker, chatbot],
            outputs=all_outputs,
        )

    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    global _agent, _verbose

    parser = argparse.ArgumentParser(description="Helio AI Agent — Gradio Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool call details in browser UI")
    parser.add_argument("--model", "-m", default=None, help="Gemini model name")
    args = parser.parse_args()

    _verbose = args.verbose

    # Initialize agent
    print("Initializing agent...")
    try:
        from agent.core import create_agent
        _agent = create_agent(verbose=args.verbose, model=args.model)
        _agent.web_mode = True  # Suppress auto-open of exported files
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure .env has GOOGLE_API_KEY set.")
        sys.exit(1)
    print(f"Agent ready (model: {_agent.model_name})")

    # Build and launch the app
    app = create_app()

    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        footer { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
