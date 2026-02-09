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
import logging
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
        t_min = e.get("time_min", "")[:10] if e.get("time_min") else ""
        t_max = e.get("time_max", "")[:10] if e.get("time_max") else ""
        time_range = f"{t_min} to {t_max}" if t_min and t_max else ""
        rows.append([
            e["label"],
            e["num_points"],
            e.get("units", ""),
            time_range,
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
        f"**Thinking:** {usage.get('thinking_tokens', 0):,}  \n"
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
        return history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

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
        return history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

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
        agent_reply = _agent.process_message(notify_msg)
    except Exception:
        agent_reply = f"Fetched **{label}** — {n_points} points."

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
        history, fig, data_rows, token_text, None,
        gr.update(choices=label_choices, value=selected),
        preview,
        "",
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
# Session management helpers
# ---------------------------------------------------------------------------

def _get_session_choices() -> list[tuple[str, str]]:
    """Return dropdown choices for saved sessions: (display_label, session_id)."""
    from agent.session import SessionManager
    sm = SessionManager()
    sessions = sm.list_sessions()[:20]
    choices = []
    for s in sessions:
        turns = s.get("turn_count", 0)
        preview = s.get("last_message_preview", "")[:30]
        updated = s.get("updated_at", "")[:16].replace("T", " ")
        label = f"{updated} ({turns} turns) {preview}"
        choices.append((label, s["id"]))
    return choices


def _extract_display_history(contents: list[dict]) -> list[dict]:
    """Convert Gemini Content dicts into Gradio chatbot messages.

    Skips entries that only contain function_call or function_response parts
    (no user-facing text).
    """
    messages = []
    for content in contents:
        role = content.get("role", "")
        if role not in ("user", "model"):
            continue
        gradio_role = "user" if role == "user" else "assistant"

        # Extract text from parts
        parts = content.get("parts", [])
        text_parts = []
        for p in parts:
            if isinstance(p, dict):
                if "text" in p and p["text"]:
                    text_parts.append(p["text"])
            elif isinstance(p, str):
                text_parts.append(p)

        if not text_parts:
            continue

        messages.append({"role": gradio_role, "content": "\n".join(text_parts)})
    return messages


def _on_load_session(session_id: str):
    """Load a saved session and restore chat + data."""
    if not session_id or _agent is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    # First, read the saved history for display (before _agent.load_session
    # which may fail on Gemini chat recreation but should still restore data)
    try:
        history_dicts, _, _ = _agent._session_manager.load_session(session_id)
    except Exception as e:
        # Session files are missing/corrupt — show error in chat
        return (
            [{"role": "assistant", "content": f"Failed to read session: {e}"}],
            None, [], _format_tokens(),
            gr.update(choices=[], value=None),
            None,
            gr.update(choices=_get_session_choices(), value=None),
        )

    display = _extract_display_history(history_dicts)

    # Now restore the agent state (chat + DataStore)
    try:
        meta = _agent.load_session(session_id)
    except Exception as e:
        # Agent restore failed — still show the saved history as read-only
        display.append({
            "role": "assistant",
            "content": f"*Session history loaded but agent state could not be fully restored: {e}. "
            f"Data and chat context may be missing — try re-fetching data.*",
        })

    fig = _get_current_figure()
    data_rows = _build_data_table()
    token_text = _format_tokens()
    label_choices = _get_label_choices()
    selected = label_choices[-1] if label_choices else None
    preview = _preview_data(selected) if selected else None

    return (
        display, fig, data_rows, token_text,
        gr.update(choices=label_choices, value=selected),
        preview,
        gr.update(choices=_get_session_choices(), value=session_id),
    )


def _on_new_session():
    """Start a fresh session (reset + create new)."""
    if _agent is not None:
        _agent.reset()

    from data_ops.store import get_store
    get_store().clear()

    return (
        [], None, [], "*New session started*", None,
        gr.update(choices=[], value=None), None,
        gr.update(choices=_get_session_choices(), value=_agent.get_session_id() if _agent else None),
    )


def _on_delete_session(session_id: str):
    """Delete a saved session."""
    if not session_id:
        return gr.update(choices=_get_session_choices())
    from agent.session import SessionManager
    sm = SessionManager()
    # Don't delete the currently active session
    if _agent and session_id == _agent.get_session_id():
        return gr.update(choices=_get_session_choices(), value=session_id)
    sm.delete_session(session_id)
    return gr.update(choices=_get_session_choices(), value=None)


# ---------------------------------------------------------------------------
# Core response function
# ---------------------------------------------------------------------------

class _ListHandler(logging.Handler):
    """Logging handler that appends formatted messages to a list (thread-safe)."""

    def __init__(self, target_list: list):
        super().__init__(level=logging.DEBUG)
        self.setFormatter(logging.Formatter("%(message)s"))
        self._target = target_list

    def emit(self, record: logging.LogRecord) -> None:
        self._target.append(self.format(record))


def respond(message, history: list[dict]):
    """Process a user message, streaming live progress in verbose mode.

    ``message`` may be a plain string (text-only) or a dict from
    ``gr.MultimodalTextbox`` with ``{"text": ..., "files": [...]}``.

    This is a *generator* — Gradio calls it repeatedly, and each ``yield``
    pushes an incremental UI update to the browser.

    Yields:
        Tuple of (history, plotly_figure, data_table, token_text, textbox_value,
                  label_dropdown_update, preview_data, verbose_state).
    """
    # Extract text and files from multimodal input
    if isinstance(message, dict):
        text = (message.get("text") or "").strip()
        files = message.get("files") or []
    else:
        text = str(message).strip()
        files = []

    # Build message for the agent: text + file path references
    parts = []
    if text:
        parts.append(text)
    for f in files:
        path = f if isinstance(f, str) else (f.get("path", "") if isinstance(f, dict) else str(f))
        name = Path(path).name if path else "unknown"
        parts.append(f"[Uploaded file: {name} — path: {path}]")

    agent_message = "\n".join(parts)
    if not agent_message:
        yield history, gr.skip(), gr.skip(), gr.skip(), None, gr.skip(), gr.skip(), gr.skip()
        return

    # Display user message with file indicators
    display_parts = []
    if text:
        display_parts.append(text)
    for f in files:
        path = f if isinstance(f, str) else (f.get("path", "") if isinstance(f, dict) else str(f))
        name = Path(path).name if path else "unknown"
        display_parts.append(f"Uploaded: {name}")

    # Append user message
    message = agent_message  # used by the agent below
    history = history + [{"role": "user", "content": "\n".join(display_parts)}]

    if not _verbose:
        # Non-verbose: simple blocking call, no streaming
        try:
            response_text = _agent.process_message(message)
        except Exception as e:
            response_text = f"Error: {e}"
        history = history + [{"role": "assistant", "content": response_text}]
        fig = _get_current_figure()
        data_rows = _build_data_table()
        token_text = _format_tokens()
        label_choices = _get_label_choices()
        selected = label_choices[-1] if label_choices else None
        preview = _preview_data(selected) if selected else None
        yield (
            history, fig, data_rows, token_text, None,
            gr.update(choices=label_choices, value=selected),
            preview, "",
        )
        return

    # --- Verbose streaming mode ---
    # Capture log messages into a thread-safe list
    log_lines: list[str] = []
    handler = _ListHandler(log_lines)
    logger = logging.getLogger("helio-agent")
    saved_level = logger.level
    if logger.getEffectiveLevel() > logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Run the agent in a background thread
    result_box: list = [None]  # [response_text]
    error_box: list = [None]

    def _run():
        try:
            result_box[0] = _agent.process_message(message)
        except Exception as exc:
            error_box[0] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Show immediate "Working..." feedback
    yield (
        history + [{"role": "assistant", "content": "*Working...*"}],
        gr.skip(), gr.skip(), gr.skip(), None,
        gr.skip(), gr.skip(), "",
    )

    # Stream live progress while the agent works
    prev_count = 0
    while thread.is_alive():
        thread.join(timeout=0.4)
        if len(log_lines) > prev_count:
            prev_count = len(log_lines)
            log_text = "\n".join(log_lines)
            thinking = (
                f"*Working...*\n\n"
                f"<details open><summary>Live Log ({len(log_lines)} lines)</summary>\n\n"
                f"```\n{log_text}\n```\n\n</details>"
            )
            yield (
                history + [{"role": "assistant", "content": thinking}],
                gr.skip(), gr.skip(), gr.skip(), None,
                gr.skip(), gr.skip(), "",
            )

    # Remove handler and restore level
    logger.removeHandler(handler)
    logger.setLevel(saved_level)

    # Build final response
    response_text = result_box[0] if error_box[0] is None else f"Error: {error_box[0]}"
    verbose_text = "\n".join(log_lines)

    if verbose_text:
        full_response = (
            f"{response_text}\n\n"
            f"<details><summary>Debug Log ({len(log_lines)} lines)</summary>\n\n"
            f"```\n{verbose_text}\n```\n\n</details>"
        )
    else:
        full_response = response_text

    history = history + [{"role": "assistant", "content": full_response}]
    fig = _get_current_figure()
    data_rows = _build_data_table()
    token_text = _format_tokens()
    label_choices = _get_label_choices()
    selected = label_choices[-1] if label_choices else None
    preview = _preview_data(selected) if selected else None

    yield (
        history, fig, data_rows, token_text, None,
        gr.update(choices=label_choices, value=selected),
        preview, "",
    )


def reset_session() -> tuple:
    """Reset the agent, data store, and all UI state."""
    if _agent is not None:
        _agent.reset()

    from data_ops.store import get_store
    get_store().clear()

    return [], None, [], "*Session reset*", None, gr.update(choices=[], value=None), None, ""


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

EXAMPLES = [
    {"text": "Show me ACE magnetic field data for last week"},
    {"text": "Compare ACE and Wind magnetic field for Jan 10-17, 2024"},
    {"text": "Compute the magnitude and overlay on the plot"},
    {"text": "What major solar storms happened in 2024?"},
]


# ---------------------------------------------------------------------------
# Theme, CSS, and JS for light/dark mode support
# ---------------------------------------------------------------------------

def _build_theme():
    """Build Gradio theme with distinct light and dark mode palettes."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e0f7ff", c100="#b3ecff", c200="#80dfff",
            c300="#4dd2ff", c400="#1ac5ff", c500="#00d9ff",
            c600="#00b8d9", c700="#0097b2", c800="#00768c",
            c900="#005566", c950="#003d4d",
        ),
        secondary_hue=gr.themes.Color(
            c50="#fff8e1", c100="#ffecb3", c200="#ffe082",
            c300="#ffd54f", c400="#ffca28", c500="#ffa500",
            c600="#fb8c00", c700="#f57c00", c800="#ef6c00",
            c900="#e65100", c950="#bf360c",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0",
            c300="#cbd5e1", c400="#94a3b8", c500="#64748b",
            c600="#475569", c700="#334155", c800="#1e293b",
            c900="#0f172a", c950="#020617",
        ),
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        # --- Light mode ---
        body_background_fill="#f8fafc",
        background_fill_primary="#ffffff",
        background_fill_secondary="#f1f5f9",
        border_color_primary="#e2e8f0",
        border_color_accent="#00b8d9",
        body_text_color="#0f172a",
        body_text_color_subdued="#64748b",
        button_primary_background_fill="#00b8d9",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_hover="#0097b2",
        button_secondary_background_fill="#f1f5f9",
        button_secondary_text_color="#0f172a",
        button_secondary_border_color="#e2e8f0",
        input_background_fill="#ffffff",
        input_border_color="#e2e8f0",
        input_placeholder_color="#94a3b8",
        panel_background_fill="#ffffff",
        panel_border_color="#e2e8f0",
        table_even_background_fill="#ffffff",
        table_odd_background_fill="#f8fafc",
        table_border_color="#e2e8f0",
        shadow_drop="0 1px 3px rgba(0, 0, 0, 0.08)",
        shadow_drop_lg="0 4px 12px rgba(0, 0, 0, 0.1)",
        block_label_text_color="#64748b",
        block_title_text_color="#0f172a",
        checkbox_label_text_color="#0f172a",
        # --- Dark mode ---
        body_background_fill_dark="#0a0e1a",
        background_fill_primary_dark="#141824",
        background_fill_secondary_dark="#1a1f2e",
        border_color_primary_dark="#2e3a5a",
        border_color_accent_dark="#00d9ff",
        body_text_color_dark="#e8eaf0",
        body_text_color_subdued_dark="#a2a9bc",
        button_primary_background_fill_dark="#00d9ff",
        button_primary_text_color_dark="#0a0e1a",
        button_primary_background_fill_hover_dark="#1ac5ff",
        button_secondary_background_fill_dark="#1a1f2e",
        button_secondary_text_color_dark="#e8eaf0",
        button_secondary_border_color_dark="#2e3a5a",
        input_background_fill_dark="#141824",
        input_border_color_dark="#2e3a5a",
        input_placeholder_color_dark="#5c6888",
        panel_background_fill_dark="#141824",
        panel_border_color_dark="#2e3a5a",
        table_even_background_fill_dark="#141824",
        table_odd_background_fill_dark="#1a1f2e",
        table_border_color_dark="#2e3a5a",
        block_label_text_color_dark="#a2a9bc",
        block_title_text_color_dark="#e8eaf0",
        checkbox_label_text_color_dark="#e8eaf0",
    )


CUSTOM_CSS = """
/* ---- Hide footer ---- */
footer { display: none !important; }

/* ---- Header ---- */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 1.5rem 1rem;
    background: linear-gradient(135deg, #f8fafc, #f1f5f9, #e2e8f0);
    border-bottom: 2px solid var(--border-color-primary);
    border-radius: 12px;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.dark .app-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #141824 50%, #1a1f2e 100%);
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00b8d9, #ffa500, #00b8d9);
    background-size: 200% 100%;
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.header-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0097b2;
    margin: 0;
}
.dark .header-title {
    color: #00d9ff;
    text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
}
.header-subtitle {
    color: var(--body-text-color-subdued);
    font-size: 0.9rem;
    margin: 0.2rem 0 0 0;
}
.header-controls {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.header-badge {
    background: var(--background-fill-secondary);
    color: #f57c00;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    border: 1px solid #ffa500;
    white-space: nowrap;
}
.theme-toggle {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.1rem !important;
    transition: all 0.2s ease !important;
    padding: 0 !important;
    color: var(--body-text-color) !important;
    line-height: 1 !important;
}
.theme-toggle:hover {
    border-color: var(--border-color-accent) !important;
    transform: scale(1.1) !important;
}
/* Light mode: show moon (click to go dark) */
.theme-toggle .icon-sun { display: none; }
.theme-toggle .icon-moon { display: inline; }
/* Dark mode: show sun (click to go light) */
.dark .theme-toggle .icon-sun { display: inline; }
.dark .theme-toggle .icon-moon { display: none; }

/* ---- Plot container ---- */
.plot-container {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    background: var(--background-fill-primary) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    padding: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}
.dark .plot-container {
    box-shadow: 0 4px 20px rgba(0, 217, 255, 0.05) !important;
}

/* ---- Chatbot ---- */
.chat-window .message-row .message {
    border-radius: 10px !important;
}
.chat-window .message-row.user-row .message {
    background: var(--background-fill-secondary) !important;
    border-left: 3px solid #00b8d9 !important;
}
.dark .chat-window .message-row.user-row .message {
    border-left-color: #00d9ff !important;
}
.chat-window .message-row.bot-row .message {
    background: var(--background-fill-primary) !important;
    border-left: 3px solid #ffa500 !important;
}
.chat-window {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
}

/* ---- Input textbox ---- */
.chat-input textarea {
    border-radius: 10px !important;
}
.chat-input textarea:focus {
    border-color: #00b8d9 !important;
    box-shadow: 0 0 0 2px rgba(0, 184, 217, 0.15) !important;
}
.dark .chat-input textarea:focus {
    border-color: #00d9ff !important;
    box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.2) !important;
}

/* ---- Examples as compact pills ---- */
#example-pills .gr-samples-table {
    gap: 0.4rem !important;
}
#example-pills button.gr-sample-btn,
#example-pills .gr-sample {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 20px !important;
    color: var(--body-text-color-subdued) !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.2s ease !important;
}
#example-pills button.gr-sample-btn:hover,
#example-pills .gr-sample:hover {
    border-color: #00b8d9 !important;
    color: #00b8d9 !important;
}
.dark #example-pills button.gr-sample-btn:hover,
.dark #example-pills .gr-sample:hover {
    border-color: #00d9ff !important;
    color: #00d9ff !important;
}

/* ---- Sidebar ---- */
.sidebar .gr-accordion {
    border-color: var(--border-color-primary) !important;
}
.sidebar {
    border-left: 1px solid var(--border-color-primary);
    padding-left: 0.5rem;
}

/* ---- Data tables ---- */
.data-table table th {
    background: var(--background-fill-secondary) !important;
    color: #00b8d9 !important;
    font-weight: 600 !important;
    border-color: var(--border-color-primary) !important;
}
.dark .data-table table th {
    color: #00d9ff !important;
}
.data-table table td {
    border-color: var(--border-color-primary) !important;
}

/* ---- Token display ---- */
.token-display {
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
}

/* ---- Buttons ---- */
button.primary {
    transition: all 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 184, 217, 0.2) !important;
}
.dark button.primary:hover {
    box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3) !important;
}

/* ---- Dark mode scrollbars ---- */
.dark ::-webkit-scrollbar { width: 8px; height: 8px; }
.dark ::-webkit-scrollbar-track { background: #0a0e1a; }
.dark ::-webkit-scrollbar-thumb { background: #2e3a5a; border-radius: 4px; }
.dark ::-webkit-scrollbar-thumb:hover { background: #3a486e; }

/* ---- Accordion headers ---- */
.dark .gr-accordion .label-wrap {
    color: #e8eaf0 !important;
}

/* ---- Dropdown styling ---- */
.gr-dropdown {
    border-color: var(--border-color-primary) !important;
}
"""

TOGGLE_JS = """
() => {
    const saved = localStorage.getItem('helio-theme');
    if (saved === 'light') {
        document.body.classList.remove('dark');
    } else if (saved === 'dark') {
        document.body.classList.add('dark');
    }
    function attachToggle() {
        const btn = document.querySelector('.theme-toggle');
        if (btn) {
            btn.addEventListener('click', () => {
                const isDark = document.body.classList.toggle('dark');
                localStorage.setItem('helio-theme', isDark ? 'dark' : 'light');
            });
        } else {
            setTimeout(attachToggle, 200);
        }
    }
    attachToggle();
}
"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(title="Helio AI Agent") as app:
        # ---- Header ----
        gr.HTML(
            """
            <div class="app-header">
                <div class="header-content">
                    <h1 class="header-title">Helio AI Agent</h1>
                    <p class="header-subtitle">
                        Talk to NASA's spacecraft data &mdash; 52 missions, 3,000+ datasets
                    </p>
                </div>
                <div class="header-controls">
                    <div class="header-badge">Powered by Gemini</div>
                    <button class="theme-toggle" title="Toggle light/dark mode">
                        <span class="icon-sun">&#9788;</span>
                        <span class="icon-moon">&#9789;</span>
                    </button>
                </div>
            </div>
            """
        )

        # ---- Full-width plot (hero element) ----
        with gr.Group(elem_classes="plot-container"):
            plotly_plot = gr.Plot(label="Interactive Plot", elem_classes="plot-area")

        with gr.Row():
            # ---- Main column: Chat ----
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    placeholder=(
                        "Ask about spacecraft data — e.g. "
                        "\"Show me ACE magnetic field data for last week\" "
                        "or \"Compare solar wind speed across missions\""
                    ),
                    elem_classes="chat-window",
                )
                msg_input = gr.MultimodalTextbox(
                    placeholder="Ask about spacecraft data...",
                    show_label=False,
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".xls",
                                ".html", ".csv", ".json", ".xml", ".zip",
                                ".jpg", ".jpeg", ".png", ".gif", ".bmp",
                                ".epub", ".txt", ".md"],
                    submit_btn="Send",
                    stop_btn=False,
                    elem_classes="chat-input",
                )

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg_input,
                    label="Try these",
                    examples_per_page=4,
                    elem_id="example-pills",
                )

            # ---- Sidebar: data & controls ----
            with gr.Column(scale=1, elem_classes="sidebar"):
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
                    headers=["Label", "Points", "Units", "Time Range", "Source"],
                    label="Data in Memory",
                    interactive=False,
                    wrap=True,
                    row_count=0,
                    elem_classes="data-table",
                )
                label_dropdown = gr.Dropdown(
                    label="Preview",
                    choices=[],
                    interactive=True,
                )
                data_preview = gr.Dataframe(
                    label="Data Preview",
                    interactive=False,
                    wrap=True,
                    row_count=0,
                    elem_classes="data-table",
                )
                token_display = gr.Markdown(
                    value="*No API calls yet*",
                    label="Token Usage",
                    elem_classes="token-display",
                )
                with gr.Accordion("Sessions", open=False):
                    session_dropdown = gr.Dropdown(
                        label="Saved Sessions",
                        choices=_get_session_choices(),
                        interactive=True,
                    )
                    with gr.Row():
                        load_session_btn = gr.Button("Load", size="sm")
                        new_session_btn = gr.Button("New", size="sm")
                        delete_session_btn = gr.Button("Delete", size="sm", variant="stop")
                reset_btn = gr.Button("Reset Session", variant="secondary")

        verbose_output = gr.State("")  # captured text, embedded in chat

        # ---- Event wiring ----
        all_outputs = [chatbot, plotly_plot, data_table, token_display,
                       msg_input, label_dropdown, data_preview, verbose_output]

        send_event_args = dict(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=all_outputs,
        )
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

        # Session management
        session_outputs = [chatbot, plotly_plot, data_table, token_display,
                           label_dropdown, data_preview, session_dropdown]
        load_session_btn.click(
            fn=_on_load_session,
            inputs=[session_dropdown],
            outputs=session_outputs,
        )
        new_session_btn.click(
            fn=_on_new_session,
            inputs=[],
            outputs=[chatbot, plotly_plot, data_table, token_display,
                     msg_input, label_dropdown, data_preview, session_dropdown],
        )
        delete_session_btn.click(
            fn=_on_delete_session,
            inputs=[session_dropdown],
            outputs=[session_dropdown],
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
    parser.add_argument("--refresh", action="store_true", help="Refresh dataset time ranges (fast — updates start/stop dates only)")
    parser.add_argument("--refresh-full", action="store_true", help="Full rebuild of primary mission data (re-download everything)")
    parser.add_argument("--refresh-all", action="store_true", help="Download ALL missions from CDAWeb (full rebuild)")
    args = parser.parse_args()

    _verbose = args.verbose

    # Mission data menu (runs in terminal before Gradio launches)
    from knowledge.startup import resolve_refresh_flags
    resolve_refresh_flags(
        refresh=args.refresh,
        refresh_full=args.refresh_full,
        refresh_all=args.refresh_all,
    )

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

    # Start a session for auto-save
    _agent.start_session()

    # Build and launch the app
    app = create_app()

    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=_build_theme(),
        css=CUSTOM_CSS,
        js=TOGGLE_JS,
    )


if __name__ == "__main__":
    main()
