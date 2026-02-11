#!/usr/bin/env python3
"""
Gradio Web UI for the Helio AI Agent.

Provides a browser-based chat interface with interactive Plotly plots,
data table sidebar, and token usage tracking.

Usage:
    python gradio_app.py                # Launch on localhost:7860
    python gradio_app.py --share        # Generate public URL
    python gradio_app.py --port 8080    # Custom port
    python gradio_app.py --quiet        # Hide live progress log
"""

import argparse
import gc
import logging
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from agent.logging import GRADIO_VISIBLE_TAGS


# ---------------------------------------------------------------------------
# Globals (initialized in main())
# ---------------------------------------------------------------------------
_agent = None
_verbose = True


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

    # load_mission() ensures HAPI cache exists (downloads if needed)
    from knowledge.mission_loader import load_mission as _load_mission
    try:
        _load_mission(mission_id)
    except FileNotFoundError:
        pass
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
    """Directly fetch data into the store, then notify the agent."""
    if not dataset or not param:
        return history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(visible=False, choices=[], value=None)

    import config
    from data_ops.fetch import fetch_data
    from data_ops.store import get_store, DataEntry

    # Convert picker strings to ISO format
    start_iso = (start_time or "").replace(" ", "T") + "Z"
    end_iso = (end_time or "").replace(" ", "T") + "Z"

    # Direct fetch — exact IDs, no LLM interpretation
    try:
        result = fetch_data(
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
        return history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(visible=False, choices=[], value=None)

    label = f"{dataset}.{param}"
    entry = DataEntry(
        label=label,
        data=result["data"],
        units=result["units"],
        description=result["description"],
        source=config.DATA_BACKEND,
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
        history,
        gr.update(visible=fig is not None, value=fig),
        data_rows, token_text, None,
        gr.update(choices=label_choices, value=selected),
        preview, "",
        gr.skip(),  # session_radio — no change
        gr.update(visible=False, choices=[], value=None),  # followup_radio — hide
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
    subset.insert(0, "timestamp", subset.index.strftime("%Y-%m-%d %H:%M"))
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

def _get_active_session_id() -> str | None:
    """Return the ID of the currently active agent session, or None."""
    return _agent.get_session_id() if _agent else None


def _get_session_choices() -> list[tuple[str, str]]:
    """Return choices for the session list: (display_label, session_id).

    The Radio's selected state visually indicates the active session.
    Ensures the currently active session always appears in the list.
    """
    from agent.session import SessionManager
    sm = SessionManager()
    sessions = sm.list_sessions()[:20]
    choices = []
    choice_ids = set()
    for s in sessions:
        preview = s.get("last_message_preview", "").strip()[:40] or "New chat"
        date_str = s.get("updated_at", "")[:10]  # YYYY-MM-DD
        turns = s.get("turn_count", 0)
        label = f"{preview}\n{date_str} · {turns} turns"
        choices.append((label, s["id"]))
        choice_ids.add(s["id"])

    # Ensure the active session is always in the list (avoids Radio value errors)
    active_id = _get_active_session_id()
    if active_id and active_id not in choice_ids:
        choices.insert(0, ("New chat\nJust started · 0 turns", active_id))

    return choices


def _strip_memory_context(text: str) -> str:
    """Strip or collapse the long-term memory context block from a user message.

    The agent prepends [CONTEXT FROM LONG-TERM MEMORY]...[END MEMORY CONTEXT]
    to user messages.  For display, collapse this into a hidden <details> block
    so it doesn't dominate the chat bubble.
    """
    marker_start = "[CONTEXT FROM LONG-TERM MEMORY]"
    marker_end = "[END MEMORY CONTEXT]"
    idx_start = text.find(marker_start)
    idx_end = text.find(marker_end)
    if idx_start == -1 or idx_end == -1:
        return text

    memory_block = text[idx_start + len(marker_start):idx_end].strip()
    after = text[idx_end + len(marker_end):].strip()

    if not memory_block:
        return after

    collapsed = (
        f"<details><summary>Memory context</summary>\n\n"
        f"{memory_block}\n\n</details>\n\n"
    )
    return collapsed + after


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

        display_text = "\n".join(text_parts)
        # Collapse memory context in user messages so it doesn't dominate the bubble
        if gradio_role == "user":
            display_text = _strip_memory_context(display_text)

        messages.append({"role": gradio_role, "content": display_text})
    return messages


def _on_session_radio_change(session_id: str | None):
    """Load a saved session when clicked in the Radio list.

    Takes a single session_id string (Radio value).
    Returns the full 10-element all_outputs tuple.
    """
    if not session_id or _agent is None:
        return (gr.skip(),) * 10

    # Guard: don't reload the already-active session
    if session_id == _get_active_session_id():
        return (gr.skip(),) * 10

    # Save current session before switching
    try:
        _agent.save_session()
    except Exception:
        pass

    # First, read the saved history for display (before _agent.load_session
    # which may fail on Gemini chat recreation but should still restore data)
    try:
        history_dicts, _, _, _ = _agent._session_manager.load_session(session_id)
    except Exception as e:
        # Session files are missing/corrupt — show error in chat
        return (
            [{"role": "assistant", "content": f"Failed to read session: {e}"}],
            gr.update(visible=False),  # plotly_plot
            [], _format_tokens(), None,
            gr.update(choices=[], value=None),
            None, "",
            gr.update(choices=_get_session_choices(), value=_get_active_session_id()),
            gr.update(visible=False, choices=[], value=None),  # followup_radio
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
        display,
        gr.update(visible=fig is not None, value=fig),
        data_rows, token_text, None,
        gr.update(choices=label_choices, value=selected),
        preview, "",
        gr.update(choices=_get_session_choices(), value=_get_active_session_id()),
        gr.update(visible=False, choices=[], value=None),  # followup_radio
    )


def _on_enter_manage_mode():
    """Switch sidebar to manage mode (CheckboxGroup for batch deletion)."""
    return (
        gr.update(visible=False),   # normal_mode_group
        gr.update(visible=True),    # manage_mode_group
        gr.update(choices=_get_session_choices(), value=[]),  # session_checklist
    )


def _on_exit_manage_mode():
    """Switch sidebar back to normal mode (Radio for click-to-load)."""
    return (
        gr.update(visible=True),    # normal_mode_group
        gr.update(visible=False),   # manage_mode_group
        gr.update(value=[]),        # session_checklist — clear selection
    )


def _on_new_session():
    """Start a fresh session (reset + create new).

    Saves the current session first, then resets everything.
    Returns the full 10-element all_outputs tuple.
    """
    if _agent is not None:
        # Save current session before resetting
        try:
            _agent.save_session()
        except Exception:
            pass
        _agent.reset()

    from data_ops.store import get_store
    get_store().clear()

    return (
        [],  # chatbot
        gr.update(visible=False),  # plotly_plot
        [],  # data_table
        "*New session started*",  # token_display
        None,  # msg_input
        gr.update(choices=[], value=None),  # label_dropdown
        None,  # data_preview
        "",  # verbose_output
        gr.update(choices=_get_session_choices(), value=_get_active_session_id()),
        gr.update(visible=False, choices=[], value=None),  # followup_radio
    )


def _on_select_all(current_selection: list[str]):
    """Toggle select all / deselect all sessions."""
    choices = _get_session_choices()
    all_ids = [sid for _, sid in choices]
    if set(current_selection) == set(all_ids) and all_ids:
        # Already all selected — deselect all
        return gr.update(value=[])
    # Select all
    return gr.update(value=all_ids)


def _on_delete_sessions(session_ids: list[str]):
    """Delete selected sessions and exit manage mode.

    Returns 13 elements:
      normal_mode_group, manage_mode_group, session_checklist, session_radio,
      chatbot, plotly_plot, data_table, token_display,
      msg_input, label_dropdown, data_preview, verbose_output, followup_radio
    """
    if not session_ids:
        # Nothing selected — just exit manage mode
        choices = _get_session_choices()
        return (
            gr.update(visible=True),    # normal_mode_group
            gr.update(visible=False),   # manage_mode_group
            gr.update(value=[]),        # session_checklist
            gr.update(choices=choices, value=_get_active_session_id()),  # session_radio
            *(gr.skip(),) * 9,          # chatbot..followup_radio unchanged
        )

    from agent.session import SessionManager
    sm = SessionManager()

    active_id = _get_active_session_id()
    need_reset = False

    for sid in session_ids:
        if sid == active_id:
            need_reset = True
        sm.delete_session(sid)

    if need_reset and _agent is not None:
        # Active session was deleted — reset creates a fresh session
        _agent.reset()
        from data_ops.store import get_store
        get_store().clear()
        # Fetch choices AFTER reset so the new session appears in the list
        choices = _get_session_choices()
        return (
            gr.update(visible=True),    # normal_mode_group
            gr.update(visible=False),   # manage_mode_group
            gr.update(value=[]),        # session_checklist
            gr.update(choices=choices, value=_get_active_session_id()),  # session_radio
            [],                          # chatbot
            gr.update(visible=False),    # plotly_plot
            [],                          # data_table
            "*Session deleted — new session started*",  # token_display
            None,                        # msg_input
            gr.update(choices=[], value=None),  # label_dropdown
            None,                        # data_preview
            "",                          # verbose_output
            gr.update(visible=False, choices=[], value=None),  # followup_radio
        )

    # Active session not deleted — just refresh sidebar and exit manage mode
    choices = _get_session_choices()
    return (
        gr.update(visible=True),    # normal_mode_group
        gr.update(visible=False),   # manage_mode_group
        gr.update(value=[]),        # session_checklist
        gr.update(choices=choices, value=_get_active_session_id()),  # session_radio
        *(gr.skip(),) * 9,          # chatbot..followup_radio unchanged
    )


# ---------------------------------------------------------------------------
# Long-term memory helpers
# ---------------------------------------------------------------------------

def _get_memory_global_enabled() -> bool:
    """Return whether the memory system is globally enabled."""
    if _agent is None:
        return True
    return _agent.memory_store.is_global_enabled()


def _get_memory_choices() -> list[tuple[str, str]]:
    """Return (display_label, memory_id) pairs for the CheckboxGroup."""
    if _agent is None:
        return []
    memories = _agent.memory_store.get_all()
    choices = []
    for m in memories:
        tag = "[P]" if m.type == "preference" else "[S]"
        date_str = m.created_at[:10] if m.created_at else ""
        content_preview = m.content[:60] + "..." if len(m.content) > 60 else m.content
        label = f"{tag} {content_preview} ({date_str})"
        choices.append((label, m.id))
    return choices


def _on_memory_global_toggle(enabled: bool):
    """Toggle the global memory setting."""
    if _agent is not None:
        _agent.memory_store.toggle_global(enabled)


def _on_memory_delete(selected_ids: list[str]):
    """Delete selected memories and return updated choices."""
    if _agent is not None and selected_ids:
        for mid in selected_ids:
            _agent.memory_store.remove(mid)
    return gr.update(choices=_get_memory_choices(), value=[])


def _on_memory_clear():
    """Clear all memories and return updated choices."""
    if _agent is not None:
        _agent.memory_store.clear_all()
    return gr.update(choices=_get_memory_choices(), value=[])


def _on_memory_refresh():
    """Refresh the memory list from disk."""
    if _agent is not None:
        _agent.memory_store.load()
    return gr.update(choices=_get_memory_choices(), value=[])


# ---------------------------------------------------------------------------
# Autocomplete helpers
# ---------------------------------------------------------------------------

def _get_autocomplete_candidates() -> list[str]:
    """Collect user messages from recent sessions for Tab autocomplete."""
    from agent.session import SessionManager
    sm = SessionManager()
    sessions = sm.list_sessions()[:10]
    seen = set()
    candidates = []
    for s in sessions:
        try:
            history_dicts, _, _, _ = sm.load_session(s["id"])
        except Exception:
            continue
        for content in history_dicts:
            if content.get("role") != "user":
                continue
            for p in content.get("parts", []):
                text = (p.get("text", "") if isinstance(p, dict) else "").strip()
                if not text or text.startswith("["):
                    continue
                # Strip memory-context prefix if present
                if text.startswith("Memory context:"):
                    parts = text.split("\n\n", 1)
                    text = parts[1].strip() if len(parts) > 1 else ""
                if text and text not in seen:
                    seen.add(text)
                    candidates.append(text)
    return candidates


# ---------------------------------------------------------------------------
# Core response function
# ---------------------------------------------------------------------------

class _ListHandler(logging.Handler):
    """Logging handler that filters log lines for Gradio display.

    Uses tag-based filtering exclusively: only records whose ``log_tag``
    is in ``GRADIO_VISIBLE_TAGS`` are shown — regardless of log level.
    To add a new category, tag the logger call with
    ``extra=tagged("my_tag")`` and add the tag to ``GRADIO_VISIBLE_TAGS``
    in ``agent/logging.py``.
    """

    def __init__(self, target_list: list):
        super().__init__(level=logging.DEBUG)
        self.setFormatter(logging.Formatter("%(message)s"))
        self._target = target_list

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        if not msg.strip():
            return
        tag = getattr(record, "log_tag", "")
        if tag in GRADIO_VISIBLE_TAGS:
            self._target.append(msg)


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
        yield history, gr.skip(), gr.skip(), gr.skip(), None, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
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
        # Non-verbose: background thread + yield loop (enables stop button)
        result_box: list = [None]
        error_box: list = [None]
        t0 = time.monotonic()

        def _run_quiet():
            try:
                result_box[0] = _agent.process_message(message)
            except Exception as exc:
                error_box[0] = exc

        thread = threading.Thread(target=_run_quiet, daemon=True)
        thread.start()

        # Show immediate "Working..." feedback
        yield (
            history + [{"role": "assistant", "content": "*Working...*"}],
            gr.skip(), gr.skip(), gr.skip(), None,
            gr.skip(), gr.skip(), "",
            gr.skip(),
            gr.update(visible=False, choices=[], value=None),
        )

        # Poll until thread finishes (generator stays alive for stop btn)
        while thread.is_alive():
            thread.join(timeout=0.5)

        elapsed = time.monotonic() - t0
        if error_box[0] is not None:
            response_text = f"Error: {error_box[0]}"
        elif result_box[0]:
            response_text = result_box[0]
        else:
            response_text = "Done."
        response_text += f"\n\n*{elapsed:.1f}s*"
        history = history + [{"role": "assistant", "content": response_text}]
        try:
            fig = _get_current_figure()
            data_rows = _build_data_table()
            token_text = _format_tokens()
            label_choices = _get_label_choices()
            selected = label_choices[-1] if label_choices else None
            preview = _preview_data(selected) if selected else None
            session_update = gr.update(
                choices=_get_session_choices(), value=_get_active_session_id(),
            )
        except Exception as exc:
            logging.getLogger("helio-agent").warning(
                f"[GradioApp] Sidebar update failed after response: {exc}"
            )
            fig = None
            data_rows = gr.skip()
            token_text = gr.skip()
            label_choices = []
            selected = None
            preview = gr.skip()
            session_update = gr.skip()
        # Generate follow-ups (skip if cancelled)
        suggestions = []
        if not _agent.is_cancelled():
            try:
                suggestions = _agent.generate_follow_ups()
            except Exception:
                suggestions = []
        followup_update = (
            gr.update(choices=suggestions, value=None, visible=True)
            if suggestions
            else gr.update(visible=False, choices=[], value=None)
        )
        yield (
            history,
            gr.update(visible=fig is not None, value=fig),
            data_rows, token_text, None,
            gr.update(choices=label_choices, value=selected),
            preview, "",
            session_update,
            followup_update,
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
    t0 = time.monotonic()

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
        gr.skip(),
        gr.update(visible=False, choices=[], value=None),
    )

    # Stream live progress while the agent works
    MAX_STREAM_LINES = 80  # keep streaming payload small
    prev_count = 0
    while thread.is_alive():
        thread.join(timeout=0.4)
        elapsed_so_far = time.monotonic() - t0
        if len(log_lines) > prev_count:
            prev_count = len(log_lines)
            tail = log_lines[-MAX_STREAM_LINES:]
            truncated = len(log_lines) - len(tail)
            log_text = "\n".join(tail)
            if truncated:
                log_text = f"... ({truncated} earlier lines hidden)\n{log_text}"
            thinking = (
                f"*Working... ({elapsed_so_far:.1f}s)*\n\n"
                f"<details open><summary>Live Log ({len(log_lines)} lines)</summary>\n\n"
                f"```\n{log_text}\n```\n\n</details>"
            )
            yield (
                history + [{"role": "assistant", "content": thinking}],
                gr.skip(), gr.skip(), gr.skip(), None,
                gr.skip(), gr.skip(), "",
                gr.skip(),
                gr.skip(),
            )

    # Remove handler and restore level
    logger.removeHandler(handler)
    logger.setLevel(saved_level)

    # Build final response
    elapsed = time.monotonic() - t0
    if error_box[0] is not None:
        response_text = f"Error: {error_box[0]}"
    elif result_box[0]:
        response_text = result_box[0]
    else:
        response_text = "Done."
    verbose_text = "\n".join(log_lines)

    if verbose_text:
        full_response = (
            f"{response_text}\n\n"
            f"*{elapsed:.1f}s*\n\n"
            f"<details><summary>Activity Log ({len(log_lines)} events)</summary>\n\n"
            f"```\n{verbose_text}\n```\n\n</details>"
        )
    else:
        full_response = f"{response_text}\n\n*{elapsed:.1f}s*"

    history = history + [{"role": "assistant", "content": full_response}]

    # Build sidebar state — wrapped in try/except so the response is ALWAYS
    # shown even if sidebar helpers fail (prevents stuck "Working..." state).
    try:
        fig = _get_current_figure()
        data_rows = _build_data_table()
        token_text = _format_tokens()
        label_choices = _get_label_choices()
        selected = label_choices[-1] if label_choices else None
        preview = _preview_data(selected) if selected else None
        session_update = gr.update(
            choices=_get_session_choices(), value=_get_active_session_id(),
        )
    except Exception as exc:
        logging.getLogger("helio-agent").warning(
            f"[GradioApp] Sidebar update failed after response: {exc}"
        )
        fig = None
        data_rows = gr.skip()
        token_text = gr.skip()
        label_choices = []
        selected = None
        preview = gr.skip()
        session_update = gr.skip()

    # Generate follow-up suggestions BEFORE the final yield so the
    # response and suggestions arrive together atomically.  If follow-ups
    # were a separate yield, a new user message could cancel the generator
    # between the two yields, causing the response to be lost.
    # Skip if cancelled — no point generating follow-ups for an interrupted request.
    suggestions = []
    if not _agent.is_cancelled():
        try:
            suggestions = _agent.generate_follow_ups()
        except Exception:
            suggestions = []
    followup_update = (
        gr.update(choices=suggestions, value=None, visible=True)
        if suggestions
        else gr.update(visible=False, choices=[], value=None)
    )

    yield (
        history,
        gr.update(visible=fig is not None, value=fig),
        data_rows, token_text, None,
        gr.update(choices=label_choices, value=selected),
        preview, "",
        session_update,
        followup_update,
    )


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

EXAMPLES = [
    {"text": "How did scientists prove Voyager 1 left the solar system? Show me the data."},
    {"text": "When did Parker Solar Probe first enter the solar corona? Show me what happened."},
    {"text": "Show me a powerful coronal mass ejection hitting Earth. What did it look like in the data?"},
    {"text": "Compare solar wind speed and density from ACE and Wind for the last month. Show them on separate panels and compute a 1-hour running average for each."},
    {"text": "Fetch PSP magnetic field data from its closest perihelion in 2024. Compute the field magnitude, take the derivative, and plot both on separate panels with a log scale on the magnitude."},
    {"text": "Find a geomagnetic storm from 2024 using OMNI Dst index, then show ACE solar wind speed, density, and IMF Bz during that storm on a multi-panel plot."},
]


# ---------------------------------------------------------------------------
# Theme, CSS, and JS for light/dark mode support
# ---------------------------------------------------------------------------

def _build_theme():
    """Build Gradio theme with distinct light and dark mode palettes."""
    return gr.themes.Base(
        text_size=gr.themes.sizes.text_lg,
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
        shadow_drop="none",
        shadow_drop_lg="none",
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
        chatbot_text_size="*text_md",
    )


CUSTOM_CSS = """
/* ---- Hide footer ---- */
footer { display: none !important; }

/* ---- Disable all Gradio animations/transitions globally ---- */
*, *::before, *::after {
    animation: none !important;
    transition: none !important;
}

/* ---- Header (slim, single-line) ---- */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 1.2rem;
    background: var(--background-fill-secondary);
    border-bottom: 1px solid var(--border-color-primary);
    border-radius: 8px;
    margin-bottom: 0.75rem;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0097b2;
    margin: 0;
    white-space: nowrap;
}
.dark .header-title {
    color: #00d9ff;
}
.header-subtitle {
    color: var(--body-text-color-subdued);
    font-size: 0.9rem;
    margin: 0;
    white-space: nowrap;
}
.header-controls {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.header-badge {
    background: var(--background-fill-secondary);
    color: #f57c00;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    border: 1px solid #ffa500;
    white-space: nowrap;
}
.theme-toggle {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1rem !important;
    padding: 0 !important;
    color: var(--body-text-color) !important;
    line-height: 1 !important;
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
    border-radius: 14px !important;
    background: var(--background-fill-primary) !important;
    padding: 0.75rem !important;
    margin-bottom: 0.75rem !important;
}

/* ---- Chatbot bubbles ---- */
.chat-window .message-row .message {
    padding: 0.75rem 1rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.chat-window .message-row.user-row .message {
    background: rgba(0,184,217,0.08) !important;
    border: 1px solid rgba(0,184,217,0.18) !important;
    border-radius: 14px 14px 4px 14px !important;
}
.chat-window .message-row.bot-row .message {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 14px 14px 14px 4px !important;
}
/* Dark mode bubble variants */
.dark .chat-window .message-row.user-row .message {
    background: rgba(0,217,255,0.06) !important;
    border-color: rgba(0,217,255,0.15) !important;
}
.dark .chat-window .message-row.bot-row .message {
    background: var(--background-fill-secondary) !important;
    border-color: var(--border-color-primary) !important;
}
.chat-window {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 14px !important;
    resize: vertical !important;
    overflow: auto !important;
    min-height: 200px !important;
    max-height: 90vh !important;
}

/* ---- Log code blocks: wrap long lines ---- */
.chat-window .message-row details pre,
.chat-window .message-row details code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* ---- Input textbox ---- */
.chat-input textarea {
    border-radius: 10px !important;
}
.chat-input textarea:focus {
    border-color: #00b8d9 !important;
    box-shadow: 0 0 0 2px rgba(0, 184, 217, 0.1) !important;
}
.dark .chat-input textarea:focus {
    border-color: #00d9ff !important;
    box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.1) !important;
}

/* ---- Examples as card-style prompts ---- */
#example-pills .gr-samples-table {
    gap: 0.6rem !important;
}
#example-pills button.gr-sample-btn,
#example-pills .gr-sample {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    color: var(--body-text-color) !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
}
#example-pills button.gr-sample-btn:hover,
#example-pills .gr-sample:hover {
    border-color: #00b8d9 !important;
    background: rgba(0,184,217,0.04) !important;
}
.dark #example-pills button.gr-sample-btn:hover,
.dark #example-pills .gr-sample:hover {
    border-color: #00d9ff !important;
    background: rgba(0,217,255,0.04) !important;
}

/* ---- Left session sidebar: shared styles ---- */
.session-sidebar .normal-mode-group,
.session-sidebar .manage-mode-group {
    max-height: calc(100vh - 200px);
    overflow-y: auto;
    border: none !important;
    background: transparent !important;
}

/* ---- Radio list (normal mode) ---- */
.session-radio-list {
    border: none !important;
}
/* Remove Gradio's default container border/shadow */
.session-radio-list > div {
    border: none !important;
    box-shadow: none !important;
}
/* Hide radio dots */
.session-radio-list input[type="radio"] {
    display: none !important;
}
/* Style each label as a clickable list item */
.session-radio-list label {
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
    margin-bottom: 3px !important;
    cursor: pointer !important;
    font-size: 0.9rem !important;
    line-height: 1.35 !important;
    white-space: pre-line !important;
    border-left: 3px solid transparent !important;
    border-top: none !important;
    border-right: none !important;
    border-bottom: none !important;
}
.session-radio-list label:hover {
    background: rgba(0, 184, 217, 0.06) !important;
}
.dark .session-radio-list label:hover {
    background: rgba(0, 217, 255, 0.06) !important;
}
/* Active session: accent left border + background highlight */
.session-radio-list label:has(input:checked) {
    background: rgba(0, 184, 217, 0.12) !important;
    border-left: 3px solid #00b8d9 !important;
}
.dark .session-radio-list label:has(input:checked) {
    background: rgba(0, 217, 255, 0.1) !important;
    border-left: 3px solid #00d9ff !important;
}

/* ---- CheckboxGroup (manage mode) ---- */
.session-checklist {
    border: none !important;
}
.session-checklist > div {
    border: none !important;
    box-shadow: none !important;
}
.session-checklist label {
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
    margin-bottom: 3px !important;
    cursor: pointer !important;
    font-size: 0.9rem !important;
    line-height: 1.35 !important;
    white-space: pre-line !important;
    border: none !important;
}
/* Red-tinted background on checked items (deletion visual) */
.session-checklist label:has(input:checked) {
    background: rgba(239, 68, 68, 0.1) !important;
}
.dark .session-checklist label:has(input:checked) {
    background: rgba(239, 68, 68, 0.15) !important;
}

/* ---- Manage toggle button ---- */
.manage-toggle-btn {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--body-text-color-subdued) !important;
    text-decoration: underline !important;
    font-size: 0.85rem !important;
    padding: 0.2rem 0.4rem !important;
    min-height: unset !important;
}
.manage-toggle-btn:hover {
    color: var(--body-text-color) !important;
}

/* ---- Right data sidebar ---- */
.data-sidebar {
    min-width: 280px !important;
    max-width: 80vw !important;
    /* NOTE: Do NOT set overflow on the sidebar container — Gradio's toggle
       button is position:absolute outside the sidebar bounds.  Any overflow
       other than 'visible' (the default) clips the toggle and makes the
       sidebar impossible to reopen after collapsing. */
}
.data-sidebar .gr-accordion {
    border-color: var(--border-color-primary) !important;
    border-radius: 10px !important;
    margin-bottom: 0.75rem !important;
}

/* ---- Data tables ---- */
.data-table {
    border-radius: 8px !important;
    overflow-x: auto !important;
    border: 1px solid var(--border-color-primary) !important;
}
.data-table table {
    font-size: 0.8rem !important;
}
.data-table table th {
    background: var(--background-fill-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-bottom: 2px solid var(--border-color-primary) !important;
    padding: 0.4rem 0.6rem !important;
}
.data-table table td {
    border-color: var(--border-color-primary) !important;
    padding: 0.4rem 0.6rem !important;
    font-family: var(--font-mono) !important;
}
.data-table table tr:hover td {
    background: rgba(0,184,217,0.04) !important;
}
.dark .data-table table tr:hover td {
    background: rgba(0,217,255,0.04) !important;
}

/* ---- Token display ---- */
.token-display {
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin-top: 0.75rem !important;
    line-height: 1.5 !important;
}

/* ---- Buttons ---- */

/* ---- Dark mode scrollbars ---- */
.dark ::-webkit-scrollbar { width: 8px; height: 8px; }
.dark ::-webkit-scrollbar-track { background: #0a0e1a; }
.dark ::-webkit-scrollbar-thumb { background: #2e3a5a; border-radius: 4px; }
.dark ::-webkit-scrollbar-thumb:hover { background: #2e3a5a; }

/* ---- Accordion headers ---- */
.dark .gr-accordion .label-wrap {
    color: #e8eaf0 !important;
}

/* ---- Dropdown styling ---- */
.gr-dropdown {
    border-color: var(--border-color-primary) !important;
}

/* ---- Follow-up suggestion pills ---- */
#followup-pills {
    border: none !important;
    padding: 0 !important;
    margin-top: 0.4rem !important;
    background: transparent !important;
}
#followup-pills > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
}
#followup-pills input[type="radio"] {
    display: none !important;
}
#followup-pills label {
    display: inline-flex !important;
    align-items: center !important;
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 20px !important;
    padding: 0.45rem 1rem !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    color: var(--body-text-color) !important;
}
#followup-pills label:hover {
    border-color: #00b8d9 !important;
    background: rgba(0,184,217,0.06) !important;
}
.dark #followup-pills label:hover {
    border-color: #00d9ff !important;
    background: rgba(0,217,255,0.06) !important;
}
#followup-pills label:has(input:checked) {
    border-color: #00b8d9 !important;
    background: rgba(0,184,217,0.1) !important;
}
.dark #followup-pills label:has(input:checked) {
    border-color: #00d9ff !important;
    background: rgba(0,217,255,0.1) !important;
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

    /* ---- Tab autocomplete ---- */
    function initTabComplete() {
        const inputEl = document.querySelector('.chat-input textarea');
        if (!inputEl) { setTimeout(initTabComplete, 500); return; }

        let matches = [];
        let matchIdx = -1;

        function getCandidates() {
            let candidates = [];
            const jsonEl = document.querySelector('#autocomplete-data');
            if (jsonEl) {
                try {
                    const raw = jsonEl.querySelector('textarea, pre');
                    if (raw) candidates = JSON.parse(raw.textContent || raw.value || '[]');
                } catch(e) {}
            }
            document.querySelectorAll('.chat-window .message-row.user-row .message').forEach(el => {
                const t = (el.textContent || '').trim();
                if (t && !candidates.includes(t)) candidates.push(t);
            });
            return candidates;
        }

        function setInputValue(val) {
            const nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            nativeSetter.call(inputEl, val);
            inputEl.dispatchEvent(new Event('input', { bubbles: true }));
        }

        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && matches.length > 0) {
                matches = []; matchIdx = -1; return;
            }
            if (e.key !== 'Tab') {
                if (matches.length > 0) { matches = []; matchIdx = -1; }
                return;
            }
            const val = inputEl.value || '';
            if (!val.trim()) return;

            if (matches.length > 0 && val === matches[matchIdx]) {
                e.preventDefault(); e.stopPropagation();
                matchIdx = (matchIdx + 1) % matches.length;
                setInputValue(matches[matchIdx]);
                return;
            }

            const prefix = val.toLowerCase();
            const candidates = getCandidates();
            matches = candidates.filter(c =>
                c.toLowerCase().startsWith(prefix) && c.toLowerCase() !== prefix
            );
            if (matches.length === 0) return;

            e.preventDefault(); e.stopPropagation();
            matchIdx = 0;
            setInputValue(matches[0]);
        });
    }
    initTabComplete();
}
"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(title="Helio AI Agent") as app:

        # ---- Left Sidebar: Session History ----
        with gr.Sidebar(position="left", width=280, open=True,
                         elem_classes="session-sidebar"):
            new_session_btn = gr.Button("+ New Chat", variant="primary", size="sm")

            # Normal mode — click-to-load via Radio
            with gr.Group(visible=True, elem_classes="normal-mode-group") as normal_mode_group:
                session_radio = gr.Radio(
                    choices=_get_session_choices(),
                    value=_get_active_session_id(),
                    label=None, show_label=False,
                    interactive=True,
                    elem_classes="session-radio-list",
                )

            # Manage mode — multi-select for batch deletion
            with gr.Group(visible=False, elem_classes="manage-mode-group") as manage_mode_group:
                session_checklist = gr.CheckboxGroup(
                    choices=_get_session_choices(),
                    label=None, show_label=False,
                    interactive=True,
                    elem_classes="session-checklist",
                )
                with gr.Row():
                    select_all_btn = gr.Button("Select All", size="sm")
                    delete_btn = gr.Button("Delete", variant="stop", size="sm")
                    cancel_btn = gr.Button("Cancel", size="sm")

            manage_btn = gr.Button(
                "Manage", size="sm", variant="secondary",
                elem_classes="manage-toggle-btn",
            )

        # ---- Right Sidebar: Data Tools ----
        with gr.Sidebar(position="right", width=420, open=True,
                         elem_classes="data-sidebar"):
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
                max_height=200,
                column_widths=["30%", "12%", "12%", "30%", "16%"],
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
                max_height=300,
                elem_classes="data-table data-preview-table",
            )
            token_display = gr.Markdown(
                value="*No API calls yet*",
                label="Token Usage",
                elem_classes="token-display",
            )
            with gr.Accordion("Long-term Memory", open=False):
                memory_toggle = gr.Checkbox(
                    label="Enable long-term memory",
                    value=_get_memory_global_enabled(),
                )
                memory_list = gr.CheckboxGroup(
                    choices=_get_memory_choices(),
                    label="Memories",
                    interactive=True,
                )
                with gr.Row():
                    memory_refresh_btn = gr.Button("Refresh", size="sm")
                    memory_delete_btn = gr.Button(
                        "Delete Selected", variant="stop", size="sm",
                    )
                memory_clear_btn = gr.Button(
                    "Clear All", variant="stop", size="sm",
                )

        # ---- Center: Header + Plot + Chat ----
        gr.HTML(
            """
            <div class="app-header">
                <div class="header-left">
                    <h1 class="header-title">Helio AI Agent</h1>
                    <span class="header-subtitle">
                        52 missions &middot; 3,000+ datasets
                    </span>
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

        plotly_plot = gr.Plot(
            label="Interactive Plot",
            elem_classes="plot-container",
            visible=False,
        )

        chatbot = gr.Chatbot(
            height="65vh",
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
            stop_btn="Stop",
            elem_classes="chat-input",
        )

        followup_radio = gr.Radio(
            choices=[], show_label=False, visible=False,
            elem_id="followup-pills", elem_classes="followup-pills",
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=msg_input,
            label="Try these",
            examples_per_page=6,
            elem_id="example-pills",
        )

        autocomplete_data = gr.JSON(value=[], visible=False, elem_id="autocomplete-data")

        verbose_output = gr.State("")  # captured text, embedded in chat

        # ---- Event wiring ----
        all_outputs = [chatbot, plotly_plot, data_table, token_display,
                       msg_input, label_dropdown, data_preview, verbose_output,
                       session_radio, followup_radio]

        send_event_args = dict(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=all_outputs,
        )
        submit_event = msg_input.submit(**send_event_args)

        # Follow-up pill click → fill input and auto-submit
        def _on_followup_click(suggestion):
            if not suggestion:
                return gr.skip()
            return {"text": suggestion, "files": []}

        followup_event = followup_radio.change(
            fn=_on_followup_click,
            inputs=[followup_radio],
            outputs=[msg_input],
        ).then(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=all_outputs,
        )

        # Stop button: signal the agent to cancel, append interruption message
        def _on_stop(history):
            if _agent is not None:
                _agent.request_cancel()
            history = history + [
                {"role": "assistant", "content": "*Interrupted by user.*"},
            ]
            return history

        msg_input.stop(
            fn=_on_stop,
            inputs=[chatbot],
            outputs=[chatbot],
            cancels=[submit_event, followup_event],
        )

        label_dropdown.change(
            fn=_preview_data,
            inputs=[label_dropdown],
            outputs=[data_preview],
        )

        # Browse & Fetch cascade
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

        # Session management — left sidebar
        # Click-to-load in normal mode
        session_radio.change(
            fn=_on_session_radio_change,
            inputs=[session_radio],
            outputs=all_outputs,
        )
        new_session_btn.click(
            fn=_on_new_session,
            inputs=[],
            outputs=all_outputs,
        )

        # Enter/exit manage mode
        manage_btn.click(
            fn=_on_enter_manage_mode,
            outputs=[normal_mode_group, manage_mode_group, session_checklist],
        )
        cancel_btn.click(
            fn=_on_exit_manage_mode,
            outputs=[normal_mode_group, manage_mode_group, session_checklist],
        )

        # Manage mode actions
        select_all_btn.click(
            fn=_on_select_all,
            inputs=[session_checklist],
            outputs=[session_checklist],
        )

        delete_outputs = [
            normal_mode_group, manage_mode_group, session_checklist, session_radio,
            chatbot, plotly_plot, data_table, token_display,
            msg_input, label_dropdown, data_preview, verbose_output, followup_radio,
        ]
        delete_btn.click(
            fn=_on_delete_sessions,
            inputs=[session_checklist],
            outputs=delete_outputs,
        )

        # Autocomplete: populate candidates on load and after each response
        app.load(fn=_get_autocomplete_candidates, outputs=[autocomplete_data])
        msg_input.submit(
            fn=_get_autocomplete_candidates, outputs=[autocomplete_data],
            trigger_mode="always_last",
        )

        # Memory UI — self-contained, no effect on all_outputs
        memory_toggle.change(
            fn=_on_memory_global_toggle,
            inputs=[memory_toggle],
            outputs=[],
        )
        memory_refresh_btn.click(
            fn=_on_memory_refresh,
            outputs=[memory_list],
        )
        memory_delete_btn.click(
            fn=_on_memory_delete,
            inputs=[memory_list],
            outputs=[memory_list],
        )
        memory_clear_btn.click(
            fn=_on_memory_clear,
            outputs=[memory_list],
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
    parser.add_argument("--quiet", "-q", action="store_true", help="Hide live progress log in browser UI (verbose is on by default)")
    parser.add_argument("--model", "-m", default=None, help="Gemini model name")
    parser.add_argument("--refresh", action="store_true", help="Refresh dataset time ranges (fast — updates start/stop dates only)")
    parser.add_argument("--refresh-full", action="store_true", help="Full rebuild of primary mission data (re-download everything)")
    parser.add_argument("--refresh-all", action="store_true", help="Download ALL missions from CDAWeb (full rebuild)")
    parser.add_argument("--download-hapi-cache", action="store_true", help="Pre-download detailed HAPI parameter cache for all missions")
    args = parser.parse_args()

    _verbose = not args.quiet

    # Mission data menu (runs in terminal before Gradio launches)
    from knowledge.startup import resolve_refresh_flags
    resolve_refresh_flags(
        refresh=args.refresh,
        refresh_full=args.refresh_full,
        refresh_all=args.refresh_all,
        download_hapi_cache=args.download_hapi_cache,
    )

    # Check HAPI availability and auto-fallback to CDF if needed
    import config
    from data_ops.fetch import check_hapi_status

    if config.DATA_BACKEND == "hapi":
        print("Checking HAPI service availability...")
        if check_hapi_status():
            print("HAPI service is online.")
        else:
            print(
                "WARNING: CDAWeb HAPI service is unreachable. "
                "Falling back to direct CDF file download backend."
            )
            config.DATA_BACKEND = "cdf"

    if config.DATA_BACKEND == "cdf":
        print(f"Data backend: CDF (direct file download)")
    else:
        print(f"Data backend: HAPI")

    # Initialize agent
    print("Initializing agent...")
    try:
        from agent.core import create_agent
        _agent = create_agent(verbose=_verbose, model=args.model)
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
