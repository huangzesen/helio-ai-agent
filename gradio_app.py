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
import sys

import gradio as gr


# ---------------------------------------------------------------------------
# Globals (initialized in main())
# ---------------------------------------------------------------------------
_agent = None


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
    """Format token usage as markdown for the sidebar."""
    if _agent is None:
        return ""
    usage = _agent.get_token_usage()
    if usage["api_calls"] == 0:
        return "*No API calls yet*"
    return (
        f"**Input:** {usage['input_tokens']:,}  \n"
        f"**Output:** {usage['output_tokens']:,}  \n"
        f"**Total:** {usage['total_tokens']:,}  \n"
        f"**API calls:** {usage['api_calls']}"
    )


def _get_label_choices() -> list[str]:
    """Return list of labels currently in the DataStore."""
    from data_ops.store import get_store
    return [e["label"] for e in get_store().list_entries()]


def _preview_data(label: str) -> list[list] | None:
    """Return head+tail preview rows for a DataStore label."""
    if not label:
        return None
    from data_ops.store import get_store
    entry = get_store().get(label)
    if entry is None:
        return None
    df = entry.data.copy()
    df.insert(0, "timestamp", df.index.strftime("%Y-%m-%d %H:%M:%S"))
    df = df.reset_index(drop=True)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(4)
    n = len(df)
    if n <= 20:
        return df.values.tolist()
    head = df.head(10)
    tail = df.tail(10)
    sep = [["..."] * len(df.columns)]
    return head.values.tolist() + sep + tail.values.tolist()


# ---------------------------------------------------------------------------
# Core response function
# ---------------------------------------------------------------------------

def respond(message: str, history: list[dict]) -> tuple:
    """Process a user message and return updated UI state.

    Args:
        message: The user's chat input.
        history: Chat history in messages format [{role, content}, ...].

    Returns:
        Tuple of (history, plotly_figure, data_table, token_text, textbox_value,
                  label_dropdown_update, preview_data).
    """
    if not message.strip():
        return history, gr.skip(), gr.skip(), gr.skip(), "", gr.skip(), gr.skip()

    # Append user message
    history = history + [{"role": "user", "content": message}]

    # Call the agent
    try:
        response_text = _agent.process_message(message)
    except Exception as e:
        response_text = f"Error: {e}"

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
    )


def reset_session() -> tuple:
    """Reset the agent, data store, and all UI state."""
    if _agent is not None:
        _agent.reset()

    from data_ops.store import get_store
    get_store().clear()

    return [], None, [], "*Session reset*", "", gr.update(choices=[], value=None), None


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

        with gr.Row():
            # ---- Main column: Chat ----
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
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

            # ---- Sidebar ----
            with gr.Column(scale=1, elem_classes=["plot-sidebar"]):
                plotly_plot = gr.Plot(
                    label="Interactive Plot",
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
                    label="Select Dataset",
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

        # ---- Event wiring ----
        all_outputs = [chatbot, plotly_plot, data_table, token_display,
                       msg_input, label_dropdown, data_preview]

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

    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    global _agent

    parser = argparse.ArgumentParser(description="Helio AI Agent — Gradio Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool call details")
    parser.add_argument("--model", "-m", default=None, help="Gemini model name")
    args = parser.parse_args()

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
