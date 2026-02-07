#!/usr/bin/env python3
"""
Gradio Web UI for the Helio AI Agent.

Provides a browser-based chat interface with inline plot display,
data table sidebar, and token usage tracking.

Usage:
    python gradio_app.py                # Launch on localhost:7860
    python gradio_app.py --share        # Generate public URL
    python gradio_app.py --port 8080    # Custom port
    python gradio_app.py --verbose      # Show tool call details
"""

import argparse
import hashlib
import os
import sys
import tempfile

import gradio as gr


# ---------------------------------------------------------------------------
# Globals (initialized in main())
# ---------------------------------------------------------------------------
_agent = None
_plot_dir = None
_last_plot_hash = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _snapshot_plot() -> tuple[str | None, bool]:
    """Export the current Autoplot canvas to a temp PNG and detect changes.

    Returns:
        (filepath_or_None, changed_bool)
    """
    global _last_plot_hash

    if _agent is None or _agent._autoplot is None:
        return None, False

    filepath = os.path.join(_plot_dir, "current_plot.png")
    try:
        result = _agent.autoplot.export_png(filepath)
        if result.get("status") != "success":
            return None, False
    except Exception:
        return None, False

    # Hash-based change detection
    try:
        with open(filepath, "rb") as f:
            new_hash = hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None, False

    changed = new_hash != _last_plot_hash
    _last_plot_hash = new_hash
    return filepath, changed


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


# ---------------------------------------------------------------------------
# Core response function
# ---------------------------------------------------------------------------

def respond(message: str, history: list[dict]) -> tuple:
    """Process a user message and return updated UI state.

    Args:
        message: The user's chat input.
        history: Chat history in messages format [{role, content}, ...].

    Returns:
        Tuple of (history, plot_image, data_table, token_text, textbox_value).
    """
    if not message.strip():
        return history, gr.skip(), gr.skip(), gr.skip(), ""

    # Append user message
    history = history + [{"role": "user", "content": message}]

    # Call the agent
    try:
        response_text = _agent.process_message(message)
    except Exception as e:
        response_text = f"Error: {e}"

    # Append assistant text response
    history = history + [{"role": "assistant", "content": response_text}]

    # Snapshot plot and detect changes
    plot_path, plot_changed = _snapshot_plot()

    # If the plot changed, add the image inline in chat
    if plot_changed and plot_path:
        history = history + [
            {"role": "assistant", "content": gr.FileData(path=plot_path)}
        ]

    # Build sidebar state
    data_rows = _build_data_table()
    token_text = _format_tokens()

    return history, plot_path, data_rows, token_text, ""


def reset_session() -> tuple:
    """Reset the agent, data store, and all UI state."""
    global _last_plot_hash

    if _agent is not None:
        _agent.reset()

    from data_ops.store import get_store
    get_store().clear()

    _last_plot_hash = None

    return [], None, [], "*Session reset*", ""


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
            "powered by [Autoplot](https://autoplot.org/) and Gemini."
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
                plot_image = gr.Image(
                    label="Current Plot",
                    type="filepath",
                    interactive=False,
                    height=300,
                )
                data_table = gr.Dataframe(
                    headers=["Label", "Shape", "Points", "Units",
                             "Start", "End", "Source"],
                    label="Data in Memory",
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
        send_event_args = dict(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, plot_image, data_table, token_display, msg_input],
        )
        send_btn.click(**send_event_args)
        msg_input.submit(**send_event_args)

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[chatbot, plot_image, data_table, token_display, msg_input],
        )

    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    global _agent, _plot_dir

    parser = argparse.ArgumentParser(description="Helio AI Agent — Gradio Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool call details")
    parser.add_argument("--model", "-m", default=None, help="Gemini model name")
    args = parser.parse_args()

    # Create temp directory for plot snapshots
    _plot_dir = tempfile.mkdtemp(prefix="helio_plots_")
    print(f"Plot snapshots: {_plot_dir}")

    # Initialize agent (this starts the JVM — may take a few seconds)
    print("Initializing agent...")
    try:
        from agent.core import create_agent
        _agent = create_agent(verbose=args.verbose, model=args.model)
        _agent.web_mode = True  # Suppress auto-open of exported files
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure .env has GOOGLE_API_KEY and AUTOPLOT_JAR set.")
        sys.exit(1)
    print(f"Agent ready (model: {_agent.model_name})")

    # Build and launch the app
    app = create_app()
    app.queue(default_concurrency_limit=1)

    try:
        app.launch(
            server_port=args.port,
            share=args.share,
            show_error=True,
            theme=gr.themes.Soft(),
            css="""
            .plot-sidebar img { max-height: 400px; object-fit: contain; }
            footer { display: none !important; }
            """,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
