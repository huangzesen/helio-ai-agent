#!/usr/bin/env python3
"""
Panel Web UI for the Helio AI Agent.

Two-page layout served at separate URLs:
  /      — Chat page (sessions sidebar + header + plot + chat + follow-ups + examples)
  /data  — Data Tools page (browse & fetch + data table + preview + tokens + memory)

Both pages share the same process-wide agent and DataStore singleton.

Usage:
    python panel_app.py                # Launch on localhost:5006
    python panel_app.py --port 8080    # Custom port
    python panel_app.py --quiet        # Hide live progress log
    python panel_app.py --model gemini-2.5-pro  # Override model
"""

import argparse
import asyncio
import gc
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import panel as pn
import param



# ---------------------------------------------------------------------------
# Globals (initialized in main())
# ---------------------------------------------------------------------------
_agent = None
_verbose = True
_executor = ThreadPoolExecutor(max_workers=2)
_agent_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging handler (same pattern as gradio_app.py)
# ---------------------------------------------------------------------------

class _ListHandler(logging.Handler):
    """Logging handler that replicates terminal 'simple' console output.

    Shows ALL log records (not just Gradio-curated tags):
    - DEBUG/INFO: bare message (e.g. "  [Gemini] Sending...")
    - WARNING+:   "[LEVEL] message"
    Skips records marked with skip_file=True (Gradio-only preview snippets).
    """

    def __init__(self, target_list: list):
        super().__init__(level=logging.DEBUG)
        self._target = target_list

    def emit(self, record: logging.LogRecord) -> None:
        # Skip Gradio-only preview snippets (same as console handler)
        if getattr(record, "skip_file", False):
            return
        msg = record.getMessage()
        if not msg.strip():
            return
        if record.levelno >= logging.WARNING:
            self._target.append(f"[{record.levelname}] {msg}")
        else:
            self._target.append(f"  {msg}")


# ---------------------------------------------------------------------------
# Helper functions (ported from gradio_app.py)
# ---------------------------------------------------------------------------

def _get_current_figure():
    """Return the current Plotly figure from the renderer, or None."""
    if _agent is None:
        return None
    return _agent.get_plotly_figure()


def _build_data_table() -> pd.DataFrame:
    """Build a DataFrame for the data-in-memory table."""
    from data_ops.store import get_store

    store = get_store()
    entries = store.list_entries()
    if not entries:
        return pd.DataFrame(columns=["Label", "Points", "Units", "Time Range", "Source"])
    rows = []
    for e in entries:
        t_min = e.get("time_min", "")[:10] if e.get("time_min") else ""
        t_max = e.get("time_max", "")[:10] if e.get("time_max") else ""
        time_range = f"{t_min} to {t_max}" if t_min and t_max else ""
        rows.append({
            "Label": e["label"],
            "Points": e["num_points"],
            "Units": e.get("units", ""),
            "Time Range": time_range,
            "Source": e.get("source", ""),
        })
    return pd.DataFrame(rows)


def _format_tokens() -> str:
    """Format token usage and memory as markdown."""
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
        return f"{mem_line}\n\n*No API calls yet*"
    return (
        f"{mem_line}\n\n"
        f"**Input:** {usage['input_tokens']:,}\n\n"
        f"**Output:** {usage['output_tokens']:,}\n\n"
        f"**Thinking:** {usage.get('thinking_tokens', 0):,}\n\n"
        f"**Total:** {usage['total_tokens']:,}\n\n"
        f"**API calls:** {usage['api_calls']}"
    )


def _get_label_choices() -> list[str]:
    """Return list of labels currently in the DataStore."""
    from data_ops.store import get_store
    return [e["label"] for e in get_store().list_entries()]


def _get_mission_choices() -> list[str]:
    """Return list of mission IDs."""
    from knowledge.mission_loader import get_mission_ids
    return get_mission_ids()


def _preview_data(label: str) -> pd.DataFrame | None:
    """Return head+tail preview DataFrame for a DataStore label."""
    if not label:
        return None
    from data_ops.store import get_store
    entry = get_store().get(label)
    if entry is None:
        return None
    if entry.is_xarray:
        return None
    df = entry.data
    n = len(df)
    if n <= 20:
        subset = df
    else:
        subset = pd.concat([df.head(10), df.tail(10)])
    subset = subset.copy()
    subset.insert(0, "timestamp", subset.index.strftime("%Y-%m-%d %H:%M"))
    subset = subset.reset_index(drop=True)
    num_cols = subset.select_dtypes(include="number").columns
    subset[num_cols] = subset[num_cols].round(4)
    return subset


def _strip_memory_context(text: str) -> str:
    """Strip or collapse the long-term memory context block."""
    marker_start = "[CONTEXT FROM LONG-TERM MEMORY]"
    marker_end = "[END MEMORY CONTEXT]"
    idx_start = text.find(marker_start)
    idx_end = text.find(marker_end)
    if idx_start == -1 or idx_end == -1:
        return text
    after = text[idx_end + len(marker_end):].strip()
    return after


def _extract_display_history(contents: list[dict]) -> list[dict]:
    """Convert Content dicts into chat messages."""
    messages = []
    for content in contents:
        role = content.get("role", "")
        if role not in ("user", "model"):
            continue
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
        if role == "user":
            display_text = _strip_memory_context(display_text)
        messages.append({
            "role": "User" if role == "user" else "Assistant",
            "content": display_text,
        })
    return messages


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------

def _get_active_session_id() -> str | None:
    return _agent.get_session_id() if _agent else None


def _get_session_choices() -> dict[str, str]:
    """Return {display_label: session_id} for session selector."""
    from agent.session import SessionManager
    sm = SessionManager()
    sessions = sm.list_sessions()[:20]
    choices = {}
    choice_ids = set()
    for s in sessions:
        preview = s.get("last_message_preview", "").strip()[:40] or "New chat"
        date_str = s.get("updated_at", "")[:10]
        turns = s.get("turn_count", 0)
        label = f"{preview} ({date_str}, {turns} turns)"
        choices[label] = s["id"]
        choice_ids.add(s["id"])

    active_id = _get_active_session_id()
    if active_id and active_id not in choice_ids:
        choices["New chat (just started)"] = active_id

    return choices


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Header styling */
.helio-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    background: var(--panel-surface-color, #f8fafc);
    border: 1px solid var(--panel-border-color, #e2e8f0);
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
.helio-header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0097b2;
    margin: 0;
}
.helio-header .badge {
    background: #fff8e1;
    color: #f57c00;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    border: 1px solid #ffa500;
}
.helio-header .nav-link {
    color: #0097b2;
    font-size: 0.9rem;
    font-weight: 600;
    text-decoration: none;
    padding: 0.3rem 0.75rem;
    border: 1px solid #0097b2;
    border-radius: 20px;
    transition: background 0.2s;
}
.helio-header .nav-link:hover {
    background: rgba(0, 151, 178, 0.1);
}

/* Chat messages */
.chat-feed .message {
    margin-bottom: 0.5rem;
}

/* Follow-up pills */
.followup-btn {
    border-radius: 20px !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.88rem !important;
    margin: 0.2rem !important;
}

/* Example prompt buttons */
.example-btn {
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    white-space: normal !important;
}

/* Session list items */
.session-btn {
    text-align: left !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 0.8rem !important;
    margin-bottom: 2px !important;
    border-radius: 8px !important;
    width: 100% !important;
}
.session-btn-active {
    border-left: 3px solid #00b8d9 !important;
    background: rgba(0, 184, 217, 0.1) !important;
}

/* Data tables */
.data-table-widget {
    font-size: 0.8rem !important;
}

/* Token display */
.token-card {
    background: var(--panel-surface-color, #ffffff);
    border: 1px solid var(--panel-border-color, #e2e8f0);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}

/* Status banner */
.status-banner {
    margin: 0.5rem 0;
}


"""


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

EXAMPLES = [
    "How did scientists prove Voyager 1 left the solar system? Show me the data.",
    "When did Parker Solar Probe first enter the solar corona? Show me what happened.",
    "Show me a powerful coronal mass ejection hitting Earth. What did it look like in the data?",
    "Show me electron pitch angle distribution along with Br and |B| and radial solar wind speed for a recent Parker Solar Probe perihelion, +/- 3 days from the perihelion date.",
    "Compare ACE and Wind magnetic field and solar wind proton density during the September 2017 solar storm.",
    "Show me MMS burst-mode ion energy spectrogram, electron energy spectrogram, and magnetic field components around a magnetopause crossing on 2024-01-02.",
]


# ---------------------------------------------------------------------------
# Agent execution helper
# ---------------------------------------------------------------------------

def _run_agent_sync(message: str) -> str:
    """Run agent.process_message synchronously (for thread pool), with lock."""
    with _agent_lock:
        return _agent.process_message(message)


# ---------------------------------------------------------------------------
# ChatPage — the / route
# ---------------------------------------------------------------------------

class ChatPage(param.Parameterized):
    """Chat page: sessions sidebar + header + plot + chat + follow-ups + examples."""

    def __init__(self, **params):
        super().__init__(**params)

        # --- Chat ---
        self.chat_interface = pn.chat.ChatInterface(
            callback=self._chat_callback,
            show_rerun=False,
            show_undo=False,
            show_clear=False,
            show_button_name=False,
            sizing_mode="stretch_both",
            min_height=400,
        )

        # --- Plot ---
        self.plot_pane = pn.pane.Plotly(
            None,
            sizing_mode="stretch_width",
            min_height=100,
            visible=False,
            config={"responsive": True},
        )

        # --- Follow-up pills ---
        self.followup_row = pn.Row(sizing_mode="stretch_width", visible=False)

        # --- Right sidebar: stats + log (uses template's right_sidebar) ---
        self._log_lines: list[str] = []
        self._log_handler = None
        self._saved_log_level = logging.WARNING
        self._log_start_time = 0.0
        self._log_running = False
        self._log_periodic_cb = None
        self._log_prev_count = 0
        self.stats_pane = pn.pane.HTML(
            self._render_stats_html(),
            sizing_mode="stretch_width",
            height=250,
        )
        self.log_terminal = pn.pane.HTML(
            self._render_terminal_html(),
            sizing_mode="stretch_both",
            min_height=300,
        )

        # --- Session sidebar widgets ---
        self.new_session_btn = pn.widgets.Button(
            name="+ New Chat",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.session_select = pn.widgets.RadioBoxGroup(
            name="Sessions",
            options=self._build_session_options(),
            sizing_mode="stretch_width",
        )
        # Manage mode widgets
        self.manage_btn = pn.widgets.Button(name="Manage", button_type="default", width=80)
        self.session_checklist = pn.widgets.CheckBoxGroup(
            name="Select sessions to delete",
            options=self._build_session_options(),
            sizing_mode="stretch_width",
        )
        self.select_all_btn = pn.widgets.Button(name="Select All", button_type="default", width=80)
        self.delete_sessions_btn = pn.widgets.Button(name="Delete", button_type="danger", width=80)
        self.cancel_manage_btn = pn.widgets.Button(name="Cancel", button_type="default", width=80)

        self.normal_mode_col = pn.Column(self.session_select, sizing_mode="stretch_width")
        self.manage_mode_col = pn.Column(
            self.session_checklist,
            pn.Row(self.select_all_btn, self.delete_sessions_btn, self.cancel_manage_btn),
            sizing_mode="stretch_width",
            visible=False,
        )

        # --- Wire up events ---
        self.new_session_btn.on_click(self._on_new_session)
        self.session_select.param.watch(self._on_session_select, "value")
        self.manage_btn.on_click(self._on_enter_manage)
        self.cancel_manage_btn.on_click(self._on_exit_manage)
        self.select_all_btn.on_click(self._on_select_all)
        self.delete_sessions_btn.on_click(self._on_delete_sessions)

    # ----- Chat callback -----

    async def _chat_callback(self, contents: str, user: str, instance: pn.chat.ChatInterface):
        """Process a user message through the agent.

        Uses run_in_executor so the event loop stays free for UI updates
        (log panel toggle, periodic log refresh, etc.).
        """
        if not contents or not contents.strip():
            return

        message = contents.strip()
        t0 = time.monotonic()

        if _verbose:
            # Setup log capture (append to existing log with separator)
            if self._log_lines:
                self._log_lines.append("")
                self._log_lines.append(f"{'─' * 40}")
                self._log_lines.append("")
            self._log_handler = _ListHandler(self._log_lines)
            logger = logging.getLogger("helio-agent")
            self._saved_log_level = logger.level
            if logger.getEffectiveLevel() > logging.DEBUG:
                logger.setLevel(logging.DEBUG)
            logger.addHandler(self._log_handler)

            # Start periodic refresh
            self._log_start_time = t0
            self._log_running = True
            self._refresh_log_panel()
            self._start_log_refresh()

        # Run agent in executor — event loop stays free for UI
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(_executor, _run_agent_sync, message)
        except Exception as exc:
            response = f"Error: {exc}"

        elapsed = time.monotonic() - t0

        if _verbose:
            # Stop log capture and periodic refresh
            logger = logging.getLogger("helio-agent")
            logger.removeHandler(self._log_handler)
            logger.setLevel(self._saved_log_level)
            self._log_running = False
            self._stop_log_refresh()
            self._refresh_log_panel()

        self._refresh_sidebar()
        self._update_followups()
        yield f"{response}\n\n*{elapsed:.1f}s*"

    # ----- Stats display -----

    def _render_stats_html(self) -> str:
        """Render detailed stats panel with per-agent breakdown."""
        import html as html_mod
        from data_ops.store import get_store

        store = get_store()
        mem_bytes = store.memory_usage_bytes()
        if mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.0f} KB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.1f} MB"
        n_entries = len(store)
        entries = store.list_entries()

        lines = []
        lines.append(f"Data in RAM: {mem_str} ({n_entries} entries)")
        if entries:
            for e in entries:
                pts = f"{e['num_points']:,}" if e.get("num_points") else "?"
                lines.append(f"  {e['label']}  ({pts} pts)")

        if _agent is not None:
            lines.append("")
            lines.append(f"Model: {_agent.model_name}")
            sid = _agent.get_session_id()
            if sid:
                lines.append(f"Session: {sid[:20]}")

            usage = _agent.get_token_usage()
            if usage["api_calls"] > 0:
                lines.append("")
                lines.append("Token Usage (Total)")
                lines.append(f"  Input:    {usage['input_tokens']:>10,}")
                lines.append(f"  Output:   {usage['output_tokens']:>10,}")
                thinking = usage.get("thinking_tokens", 0)
                if thinking:
                    lines.append(f"  Thinking: {thinking:>10,}")
                lines.append(f"  Total:    {usage['total_tokens']:>10,}")
                lines.append(f"  API calls: {usage['api_calls']}")

                # Per-agent breakdown grouped by category
                breakdown = _agent.get_token_usage_breakdown()
                if breakdown:
                    # Group: Orchestrator, Planner, DataOps/DataExtraction, Viz, Missions
                    groups = {
                        "Orchestrator": [],
                        "Planner": [],
                        "Data": [],
                        "Visualization": [],
                        "Missions": [],
                    }
                    for row in breakdown:
                        name = row["agent"]
                        if name == "Orchestrator":
                            groups["Orchestrator"].append(row)
                        elif name == "Planner":
                            groups["Planner"].append(row)
                        elif name in ("DataOps", "DataExtraction"):
                            groups["Data"].append(row)
                        elif name == "Visualization":
                            groups["Visualization"].append(row)
                        elif name.startswith("Mission/"):
                            groups["Missions"].append(row)

                    for group_name, rows in groups.items():
                        if not rows:
                            continue
                        group_in = sum(r["input"] for r in rows)
                        group_out = sum(r["output"] for r in rows)
                        group_think = sum(r["thinking"] for r in rows)
                        group_total = group_in + group_out + group_think
                        group_calls = sum(r["calls"] for r in rows)
                        lines.append("")
                        lines.append(f"{group_name}")
                        lines.append(f"  In/Out/Think: {group_in:,} / {group_out:,} / {group_think:,}")
                        lines.append(f"  Total: {group_total:,}  ({group_calls} calls)")
                        if group_name == "Missions" and len(rows) > 1:
                            for r in rows:
                                t = r["input"] + r["output"] + r["thinking"]
                                lines.append(f"    {r['agent']}: {t:,} ({r['calls']} calls)")

        content = html_mod.escape("\n".join(lines))
        return (
            f'<div style="padding:6px 8px;background:#1a1a1a;border-radius:6px;'
            f'margin-top:4px;height:100%;overflow-y:auto;">'
            f'<pre style="margin:0;font-family:Menlo,Consolas,monospace;'
            f'font-size:0.72rem;line-height:1.4;color:#888;">{content}</pre>'
            f'</div>'
        )

    # ----- Sidebar refresh -----

    def _refresh_sidebar(self):
        """Update plot, session list, and stats after agent response."""
        try:
            fig = _get_current_figure()
            if fig is not None:
                self.plot_pane.object = fig
                self.plot_pane.visible = True
            else:
                self.plot_pane.visible = False

            # Update session list
            self.session_select.options = self._build_session_options()
            active = _get_active_session_id()
            if active:
                self.session_select.value = active

            # Update stats
            self.stats_pane.object = self._render_stats_html()
        except Exception:
            pass

    def _update_followups(self):
        """Generate and display follow-up suggestions."""
        if _agent is None or _agent.is_cancelled():
            self.followup_row.visible = False
            return

        try:
            suggestions = _agent.generate_follow_ups()
        except Exception:
            suggestions = []

        self.followup_row.clear()
        if suggestions:
            for text in suggestions:
                btn = pn.widgets.Button(
                    name=text,
                    button_type="light",
                    css_classes=["followup-btn"],
                )
                btn.on_click(lambda event, t=text: self._on_followup_click(t))
                self.followup_row.append(btn)
            self.followup_row.visible = True
        else:
            self.followup_row.visible = False

    def _on_followup_click(self, text: str):
        """Send a follow-up suggestion as a new message."""
        self.chat_interface.send(text, respond=True)

    # ----- Log panel (terminal-style) -----

    def _render_terminal_html(self) -> str:
        """Render log lines as a terminal-style scrollable block.

        Always scrolls to the bottom to show the latest output.
        """
        import html as html_mod
        if not self._log_lines:
            content = ""
        else:
            tail = self._log_lines[-500:]
            truncated = len(self._log_lines) - len(tail)
            lines = []
            if truncated:
                lines.append(f"... ({truncated} earlier lines hidden)")
            lines.extend(tail)
            content = html_mod.escape("\n".join(lines))

        if self._log_running:
            elapsed = time.monotonic() - self._log_start_time
            status = html_mod.escape(f"Working... {elapsed:.1f}s")
        elif self._log_lines:
            status = html_mod.escape(f"{len(self._log_lines)} lines")
        else:
            status = ""

        status_html = ""
        if status:
            status_html = (
                f'<div style="padding:2px 8px;font-size:0.72rem;color:#888;'
                f'border-top:1px solid #333;font-family:Menlo,Consolas,monospace;">'
                f'{status}</div>'
            )

        # Use flex-direction:column-reverse so the container naturally
        # shows the bottom (latest) content. The <pre> is wrapped in a
        # div with column-reverse, which flips the overflow anchor to
        # the bottom — no JS scrolling needed.
        return (
            f'<div style="display:flex;flex-direction:column;height:100%;'
            f'background:#1a1a1a;border-radius:6px;overflow:hidden;">'
            f'<div style="flex:1;overflow-y:auto;padding:6px 8px;'
            f'display:flex;flex-direction:column-reverse;">'
            f'<pre style="margin:0;font-family:Menlo,Consolas,monospace;'
            f'font-size:0.75rem;line-height:1.35;color:#ccc;'
            f'white-space:pre-wrap;word-break:break-word;">{content}</pre>'
            f'</div>'
            f'{status_html}'
            f'</div>'
        )

    def _refresh_log_panel(self, status: str = ""):
        """Update the terminal log content."""
        self.log_terminal.object = self._render_terminal_html()

    def _start_log_refresh(self):
        """Start periodic callback to update log panel every 500ms."""
        self._stop_log_refresh()
        self._log_periodic_cb = pn.state.add_periodic_callback(
            self._periodic_log_update, period=500
        )

    def _stop_log_refresh(self):
        """Stop the periodic log refresh callback."""
        if self._log_periodic_cb is not None:
            self._log_periodic_cb.stop()
            self._log_periodic_cb = None

    def _periodic_log_update(self):
        """Called every 500ms to refresh the log panel while agent is running."""
        if not self._log_running:
            self._stop_log_refresh()
            return
        self._refresh_log_panel()
        self.stats_pane.object = self._render_stats_html()

    # ----- Session management -----

    def _build_session_options(self) -> dict[str, str]:
        """Build {label: id} dict for session widgets."""
        return _get_session_choices()

    def _on_new_session(self, event):
        """Start a fresh session."""
        if _agent is not None:
            try:
                _agent.save_session()
            except Exception:
                pass
            _agent.reset()

        from data_ops.store import get_store
        get_store().clear()

        self.chat_interface.clear()
        self.plot_pane.object = None
        self.plot_pane.visible = False
        self.followup_row.visible = False

        self.session_select.options = self._build_session_options()

    def _on_session_select(self, event):
        """Load a saved session when clicked."""
        session_id = event.new
        if not session_id:
            return

        if session_id == _get_active_session_id():
            return

        if _agent is not None:
            try:
                _agent.save_session()
            except Exception:
                pass

        # Load display history
        try:
            history_dicts, _, _, _, _ = _agent._session_manager.load_session(session_id)
        except Exception as e:
            self.chat_interface.clear()
            self.chat_interface.send(
                f"Failed to read session: {e}",
                user="System", respond=False,
            )
            return

        display = _extract_display_history(history_dicts)

        # Restore agent state
        try:
            _agent.load_session(session_id)
        except Exception as e:
            display.append({
                "role": "Assistant",
                "content": f"*Session history loaded but agent state could not be fully restored: {e}*",
            })

        # Rebuild chat
        self.chat_interface.clear()
        for msg in display:
            user = msg["role"]
            self.chat_interface.send(msg["content"], user=user, respond=False)

        self._refresh_sidebar()
        self.followup_row.visible = False

    def _on_enter_manage(self, event):
        """Switch to manage mode."""
        self.normal_mode_col.visible = False
        self.manage_mode_col.visible = True
        self.session_checklist.options = self._build_session_options()
        self.session_checklist.value = []

    def _on_exit_manage(self, event):
        """Exit manage mode."""
        self.normal_mode_col.visible = True
        self.manage_mode_col.visible = False

    def _on_select_all(self, event):
        """Toggle select all sessions."""
        all_ids = list(self.session_checklist.options.values()) if isinstance(
            self.session_checklist.options, dict
        ) else list(self.session_checklist.options)
        if set(self.session_checklist.value) == set(all_ids):
            self.session_checklist.value = []
        else:
            self.session_checklist.value = all_ids

    def _on_delete_sessions(self, event):
        """Delete selected sessions."""
        selected_ids = self.session_checklist.value
        if not selected_ids:
            self._on_exit_manage(event)
            return

        from agent.session import SessionManager
        sm = SessionManager()

        active_id = _get_active_session_id()
        need_reset = False

        for sid in selected_ids:
            if sid == active_id:
                need_reset = True
            sm.delete_session(sid)

        if need_reset and _agent is not None:
            _agent.reset()
            from data_ops.store import get_store
            get_store().clear()
            self.chat_interface.clear()
            self.plot_pane.visible = False

        self._on_exit_manage(event)
        self.session_select.options = self._build_session_options()

    # ----- Build layout -----

    def build(self) -> pn.template.FastListTemplate:
        """Construct and return the chat page layout."""

        # --- Header with "Open Data Tools" link and Log toggle ---
        header_html = pn.pane.HTML(
            """
            <div class="helio-header">
                <h1>Helio AI Agent</h1>
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <span style="color:#64748b; font-size:0.9rem;">52 missions &middot; 3,000+ datasets</span>
                    <span class="badge">Powered by Gemini</span>
                    <a href="/data" target="_blank" class="nav-link">Open Data Tools</a>
                </div>
            </div>
            """,
            sizing_mode="stretch_width",
        )

        header_row = pn.Row(
            header_html,
            sizing_mode="stretch_width",
        )

        # --- Example buttons (one per row, full width) ---
        example_btns = []
        for ex_text in EXAMPLES:
            btn = pn.widgets.Button(
                name=ex_text,
                button_type="light",
                css_classes=["example-btn"],
                sizing_mode="stretch_width",
            )
            btn.on_click(lambda event, t=ex_text: self.chat_interface.send(t, respond=True))
            example_btns.append(btn)

        examples_section = pn.Column(
            pn.pane.Markdown("**Try these:**", margin=(10, 0, 5, 0)),
            *example_btns,
            sizing_mode="stretch_width",
        )

        # --- Center column ---
        center_col = pn.Column(
            header_row,
            self.plot_pane,
            self.chat_interface,
            self.followup_row,
            examples_section,
            sizing_mode="stretch_both",
            min_width=500,
        )

        # --- Right sidebar content: stats (fixed) + log (fills rest) ---
        right_sidebar_content = pn.Column(
            pn.pane.Markdown("### Session Stats", margin=(0, 0, 5, 0)),
            self.stats_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Activity Log", margin=(0, 0, 5, 0)),
            self.log_terminal,
            sizing_mode="stretch_both",
        )

        # --- Left sidebar: Sessions ---
        sidebar_content = pn.Column(
            pn.pane.Markdown("### Chat History", margin=(0, 0, 5, 0)),
            self.new_session_btn,
            self.normal_mode_col,
            self.manage_mode_col,
            pn.layout.Divider(),
            self.manage_btn,
            sizing_mode="stretch_width",
        )

        # --- Template ---
        template = pn.template.FastListTemplate(
            title="Helio AI Agent",
            sidebar=[sidebar_content],
            main=[center_col],
            right_sidebar=[right_sidebar_content],
            accent_base_color="#00b8d9",
            header_background="#0097b2",
            sidebar_width=280,
            collapsed_sidebar=True,
            right_sidebar_width=550,
            collapsed_right_sidebar=True,
            theme="default",
            theme_toggle=False,
            raw_css=[CUSTOM_CSS],
        )

        return template


# ---------------------------------------------------------------------------
# DataPage — the /data route
# ---------------------------------------------------------------------------

class DataPage(param.Parameterized):
    """Data Tools page: browse & fetch + data table + preview + tokens + memory."""

    def __init__(self, **params):
        super().__init__(**params)

        # --- Browse & Fetch widgets ---
        self.mission_select = pn.widgets.Select(
            name="Mission",
            options=_get_mission_choices(),
            value=None,
            sizing_mode="stretch_width",
        )
        self.dataset_select = pn.widgets.Select(
            name="Dataset",
            options=[],
            value=None,
            sizing_mode="stretch_width",
        )
        self.param_select = pn.widgets.Select(
            name="Parameter",
            options=[],
            value=None,
            sizing_mode="stretch_width",
        )
        self.browse_info = pn.pane.Markdown("", sizing_mode="stretch_width")

        now = datetime.now(tz=timezone.utc).replace(microsecond=0)
        self.start_picker = pn.widgets.DatetimePicker(
            name="Start",
            value=now - timedelta(days=7),
            sizing_mode="stretch_width",
        )
        self.end_picker = pn.widgets.DatetimePicker(
            name="End",
            value=now,
            sizing_mode="stretch_width",
        )
        self.fetch_btn = pn.widgets.Button(
            name="Fetch",
            button_type="primary",
            sizing_mode="stretch_width",
        )

        # --- Status banner ---
        self.status_banner = pn.pane.Alert(
            "",
            alert_type="info",
            visible=False,
            sizing_mode="stretch_width",
            css_classes=["status-banner"],
        )

        # --- Data in memory table ---
        self.data_table_widget = pn.widgets.Tabulator(
            _build_data_table(),
            name="Data in Memory",
            sizing_mode="stretch_width",
            height=250,
            disabled=True,
            show_index=False,
        )

        # --- Preview ---
        self.preview_select = pn.widgets.Select(
            name="Preview",
            options=_get_label_choices() or [],
            value=None,
            sizing_mode="stretch_width",
        )
        self.preview_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            name="Data Preview",
            sizing_mode="stretch_width",
            height=300,
            disabled=True,
            show_index=False,
            visible=False,
        )

        # --- Token display ---
        self.token_display = pn.pane.Markdown(
            _format_tokens(),
            sizing_mode="stretch_width",
            css_classes=["token-card"],
        )

        # --- Memory widgets ---
        self.memory_toggle = pn.widgets.Checkbox(
            name="Enable long-term memory",
            value=self._get_memory_enabled(),
        )
        self.memory_list = pn.widgets.CheckBoxGroup(
            name="Memories",
            options=self._get_memory_options(),
            sizing_mode="stretch_width",
        )
        self.memory_refresh_btn = pn.widgets.Button(name="Refresh", button_type="default", width=80)
        self.memory_delete_btn = pn.widgets.Button(name="Delete Selected", button_type="danger", width=120)
        self.memory_clear_btn = pn.widgets.Button(name="Clear All", button_type="danger", width=80)

        # --- Wire up events ---
        self.mission_select.param.watch(self._on_mission_change, "value")
        self.dataset_select.param.watch(self._on_dataset_change, "value")
        self.fetch_btn.on_click(self._on_fetch_click)
        self.preview_select.param.watch(self._on_preview_change, "value")

        self.memory_toggle.param.watch(self._on_memory_toggle, "value")
        self.memory_refresh_btn.on_click(self._on_memory_refresh)
        self.memory_delete_btn.on_click(self._on_memory_delete)
        self.memory_clear_btn.on_click(self._on_memory_clear)

    # ----- Browse & Fetch -----

    def _on_mission_change(self, event):
        """Cascade: mission -> datasets."""
        mission_id = event.new
        if not mission_id:
            self.dataset_select.options = []
            self.param_select.options = []
            self.browse_info.object = ""
            return

        from knowledge.mission_loader import load_mission as _load_mission
        try:
            _load_mission(mission_id)
        except FileNotFoundError:
            pass

        from knowledge.metadata_client import browse_datasets
        datasets = browse_datasets(mission_id) or []
        options = {}
        for d in datasets:
            n_params = d.get("parameter_count", "?")
            start = d.get("start_date", "?")[:4]
            stop = d.get("stop_date", "?")[:4]
            label = f"{d['id']}  ({n_params} params, {start}--{stop})"
            options[label] = d["id"]

        self.dataset_select.options = options
        self.dataset_select.value = None
        self.param_select.options = []
        self.browse_info.object = ""

    def _on_dataset_change(self, event):
        """Cascade: dataset -> parameters + time range."""
        dataset_id = event.new
        if not dataset_id:
            self.param_select.options = []
            self.browse_info.object = ""
            return

        from knowledge.metadata_client import list_parameters, get_dataset_time_range

        params = list_parameters(dataset_id)
        options = {}
        for p in params:
            desc = p.get("description", "")
            units = p.get("units", "")
            parts = [p["name"]]
            if desc:
                parts.append(desc)
            if units:
                parts.append(f"[{units}]")
            options[" -- ".join(parts)] = p["name"]

        self.param_select.options = options
        self.param_select.value = None

        time_range = get_dataset_time_range(dataset_id)
        info_parts = [f"**{dataset_id}** — {len(params)} parameters"]

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

        self.browse_info.object = "\n\n".join(info_parts)
        self.start_picker.value = start_dt
        self.end_picker.value = end_dt

    def _on_fetch_click(self, event):
        """Fetch data via data page, store it, and notify agent in background."""
        dataset = self.dataset_select.value
        param_name = self.param_select.value
        if not dataset or not param_name:
            self.status_banner.object = "Please select a dataset and parameter first."
            self.status_banner.alert_type = "warning"
            self.status_banner.visible = True
            return

        import config
        from data_ops.fetch import fetch_data
        from data_ops.store import get_store, DataEntry

        start_val = self.start_picker.value
        end_val = self.end_picker.value

        # Convert datetime to ISO string
        if isinstance(start_val, datetime):
            start_iso = start_val.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start_iso = str(start_val).replace(" ", "T") + "Z"
        if isinstance(end_val, datetime):
            end_iso = end_val.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end_iso = str(end_val).replace(" ", "T") + "Z"

        self.status_banner.object = f"Fetching {param_name} from {dataset}..."
        self.status_banner.alert_type = "info"
        self.status_banner.visible = True

        try:
            result = fetch_data(
                dataset_id=dataset,
                parameter_id=param_name,
                time_min=start_iso,
                time_max=end_iso,
            )
        except Exception as e:
            self.status_banner.object = f"**Fetch failed:** {e}"
            self.status_banner.alert_type = "danger"
            return

        label = f"{dataset}.{param_name}"
        entry = DataEntry(
            label=label,
            data=result["data"],
            units=result["units"],
            description=result["description"],
            source=config.DATA_BACKEND,
        )
        get_store().put(entry)
        n_points = len(result["data"])
        del result
        gc.collect()

        # Update status banner
        self.status_banner.object = (
            f"Fetched **{label}** — {n_points:,} points "
            f"({start_val} to {end_val})"
        )
        self.status_banner.alert_type = "success"

        # Refresh data widgets immediately
        self._refresh_data_widgets()

        # Notify agent in background thread (agent reply is discarded)
        notify_msg = (
            f"[User fetched data via data tools page] "
            f"{param_name} from {dataset} ({start_val} to {end_val}) "
            f"is now in memory as '{label}' with {n_points} points."
        )

        def _notify():
            try:
                with _agent_lock:
                    _agent.process_message(notify_msg)
            except Exception:
                pass

        threading.Thread(target=_notify, daemon=True).start()

    # ----- Preview -----

    def _on_preview_change(self, event):
        """Update preview table when label selection changes."""
        label = event.new
        if not label:
            self.preview_table.visible = False
            return

        preview_df = _preview_data(label)
        if preview_df is not None:
            self.preview_table.value = preview_df
            self.preview_table.visible = True
        else:
            self.preview_table.visible = False

    # ----- Periodic refresh -----

    def _refresh_data_widgets(self):
        """Refresh data table, preview options, and token display."""
        try:
            self.data_table_widget.value = _build_data_table()
            self.token_display.object = _format_tokens()

            labels = _get_label_choices()
            old_labels = self.preview_select.options
            if labels != old_labels:
                self.preview_select.options = labels
                if labels:
                    self.preview_select.value = labels[-1]
        except Exception:
            pass

    # ----- Memory -----

    def _get_memory_enabled(self) -> bool:
        if _agent is None:
            return True
        return _agent.memory_store.is_global_enabled()

    def _get_memory_options(self) -> dict[str, str]:
        if _agent is None:
            return {}
        memories = _agent.memory_store.get_all()
        options = {}
        for m in memories:
            tag = "[P]" if m.type == "preference" else "[S]"
            date_str = m.created_at[:10] if m.created_at else ""
            preview = m.content[:60] + "..." if len(m.content) > 60 else m.content
            label = f"{tag} {preview} ({date_str})"
            options[label] = m.id
        return options

    def _on_memory_toggle(self, event):
        if _agent is not None:
            _agent.memory_store.toggle_global(event.new)

    def _on_memory_refresh(self, event):
        if _agent is not None:
            _agent.memory_store.load()
        self.memory_list.options = self._get_memory_options()

    def _on_memory_delete(self, event):
        selected_ids = self.memory_list.value
        if _agent is not None and selected_ids:
            for mid in selected_ids:
                _agent.memory_store.remove(mid)
        self.memory_list.options = self._get_memory_options()

    def _on_memory_clear(self, event):
        if _agent is not None:
            _agent.memory_store.clear_all()
        self.memory_list.options = self._get_memory_options()

    # ----- Build layout -----

    def build(self) -> pn.template.FastListTemplate:
        """Construct and return the data tools page layout."""

        # --- Header with "Back to Chat" link ---
        header_html = pn.pane.HTML(
            """
            <div class="helio-header">
                <h1>Data Tools</h1>
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <span style="color:#64748b; font-size:0.9rem;">Browse, fetch & inspect data</span>
                    <a href="/" class="nav-link">Back to Chat</a>
                </div>
            </div>
            """,
            sizing_mode="stretch_width",
        )

        # --- Browse & Fetch section ---
        browse_section = pn.Card(
            self.mission_select,
            self.dataset_select,
            self.param_select,
            self.browse_info,
            pn.Row(self.start_picker, self.end_picker),
            self.fetch_btn,
            self.status_banner,
            title="Browse & Fetch",
            collapsed=False,
            sizing_mode="stretch_width",
        )

        # --- Memory section ---
        memory_section = pn.Card(
            self.memory_toggle,
            self.memory_list,
            pn.Row(self.memory_refresh_btn, self.memory_delete_btn),
            self.memory_clear_btn,
            title="Long-term Memory",
            collapsed=True,
            sizing_mode="stretch_width",
        )

        # --- Left column: browse/fetch + memory ---
        left_col = pn.Column(
            browse_section,
            memory_section,
            sizing_mode="stretch_width",
        )

        # --- Right column: data table + preview + tokens ---
        right_col = pn.Column(
            pn.pane.Markdown("**Data in Memory**", margin=(10, 0, 5, 0)),
            self.data_table_widget,
            self.preview_select,
            self.preview_table,
            pn.layout.Divider(),
            self.token_display,
            sizing_mode="stretch_width",
        )

        # --- Two-column layout ---
        main_area = pn.Row(
            left_col,
            right_col,
            sizing_mode="stretch_both",
        )

        # --- Template (no sidebar) ---
        template = pn.template.FastListTemplate(
            title="Helio AI Agent — Data Tools",
            main=[header_html, main_area],
            accent_base_color="#00b8d9",
            header_background="#0097b2",
            theme="default",
            theme_toggle=False,
            raw_css=[CUSTOM_CSS],
        )

        # --- Periodic refresh every 4 seconds ---
        pn.state.add_periodic_callback(self._refresh_data_widgets, period=4000)

        return template


# ---------------------------------------------------------------------------
# Page factory functions (one instance per browser session)
# ---------------------------------------------------------------------------

def _create_chat_page():
    return ChatPage().build()


def _create_data_page():
    return DataPage().build()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    global _agent, _verbose

    pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

    parser = argparse.ArgumentParser(description="Helio AI Agent — Panel Web UI")
    parser.add_argument("--port", type=int, default=5006, help="Port to listen on")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Hide live progress log")
    parser.add_argument("--model", "-m", default=None, help="LLM model name")
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh dataset time ranges")
    parser.add_argument("--refresh-full", action="store_true",
                        help="Full rebuild of primary mission data")
    parser.add_argument("--refresh-all", action="store_true",
                        help="Download ALL missions from CDAWeb")

    args = parser.parse_args()
    _verbose = not args.quiet

    # Mission data refresh
    from knowledge.startup import resolve_refresh_flags
    resolve_refresh_flags(
        refresh=args.refresh,
        refresh_full=args.refresh_full,
        refresh_all=args.refresh_all,
    )

    print("Data backend: CDF (direct file download)")

    # Initialize agent
    print("Initializing agent...")
    try:
        from agent.core import create_agent
        _agent = create_agent(verbose=_verbose, model=args.model)
        _agent.web_mode = True
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure .env has LLM_API_KEY set.")
        sys.exit(1)
    print(f"Agent ready (model: {_agent.model_name})")

    # Start a session for auto-save
    _agent.start_session()

    # Ensure Ctrl+C always kills the process.
    # os._exit bypasses Tornado's cleanup which can hang on background threads.
    def _force_exit(*_args):
        print("\nShutting down...")
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    # Re-install after Tornado starts (it may override signal handlers)
    def _reinstall_signals():
        signal.signal(signal.SIGINT, _force_exit)
        signal.signal(signal.SIGTERM, _force_exit)

    pn.state.execute(_reinstall_signals)

    # Serve two pages
    pn.serve(
        {"/": _create_chat_page, "/data": _create_data_page},
        port=args.port,
        show=True,
        title="Helio AI Agent",
        websocket_origin="*",
    )


if __name__ == "__main__":
    main()
