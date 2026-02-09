"""
Session persistence for chat history and data store.

Saves and restores Gemini chat history + DataStore entries so users
can resume conversations across process restarts.

Storage layout:
    ~/.helio-agent/sessions/{session_id}/
        metadata.json     — session info (model, turn_count, timestamps, etc.)
        history.json      — Gemini Content dicts (from Content.model_dump)
        data/
            {label}.pkl   — pickled DataFrames
            _index.json   — label -> {filename, units, description, source}
"""

import base64
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


# Characters unsafe for filenames on Windows
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')


def _safe_filename(label: str) -> str:
    """Convert a DataStore label to a safe filename (without extension)."""
    return _UNSAFE_CHARS.sub("_", label)


_B64_PREFIX = "b64:"


def _encode_bytes_fields(history: list[dict]) -> list[dict]:
    """Encode bytes fields (e.g. thought_signature) as base64 for JSON safety.

    The Gemini SDK's Content.model_dump() produces raw bytes for
    thought_signature fields.  json.dump(default=str) would mangle these
    into Python repr strings like "b'\\x12\\x9c...'" which can't be
    deserialized back to bytes.  We base64-encode them instead.
    """
    for entry in history:
        for part in entry.get("parts", []):
            if isinstance(part, dict) and "thought_signature" in part:
                val = part["thought_signature"]
                if isinstance(val, bytes):
                    part["thought_signature"] = _B64_PREFIX + base64.b64encode(val).decode("ascii")
    return history


def _decode_bytes_fields(history: list[dict]) -> list[dict]:
    """Decode base64-encoded bytes fields back to raw bytes for the SDK.

    Also handles legacy sessions where bytes were mangled by ``default=str``
    into Python repr strings like ``"b'\\x12\\x9c...'"`` — these are
    re-parsed via ``ast.literal_eval``.
    """
    import ast

    for entry in history:
        for part in entry.get("parts", []):
            if isinstance(part, dict) and "thought_signature" in part:
                val = part["thought_signature"]
                if isinstance(val, (bytes, bytearray)):
                    continue  # already bytes
                if isinstance(val, str):
                    if val.startswith(_B64_PREFIX):
                        part["thought_signature"] = base64.b64decode(val[len(_B64_PREFIX):])
                    elif val.startswith("b'") or val.startswith('b"'):
                        # Legacy format from default=str mangling
                        try:
                            part["thought_signature"] = ast.literal_eval(val)
                        except (ValueError, SyntaxError):
                            pass
    return history


class SessionManager:
    """Manages session directories for chat history persistence."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path.home() / ".helio-agent" / "sessions"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, model_name: str = "") -> str:
        """Create a new session directory with initial metadata.

        Returns:
            The session_id string.
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "data").mkdir(exist_ok=True)

        metadata = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "model": model_name,
            "turn_count": 0,
            "last_message_preview": "",
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "api_calls": 0},
        }
        self._write_json(session_dir / "metadata.json", metadata)
        self._write_json(session_dir / "history.json", [])

        return session_id

    def save_session(
        self,
        session_id: str,
        chat_history: list[dict],
        data_store,
        metadata_updates: Optional[dict] = None,
        figure_state: Optional[dict] = None,
    ) -> None:
        """Save chat history, DataStore, and plot figure to disk.

        Args:
            session_id: The session to save.
            chat_history: List of Content dicts (from Content.model_dump).
            data_store: A DataStore instance to persist.
            metadata_updates: Optional dict to merge into metadata
                (e.g. token_usage, turn_count).
            figure_state: Optional renderer state dict from
                ``PlotlyRenderer.save_state()``.
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_id}")

        # Save history (encode bytes fields so JSON round-trips cleanly)
        self._write_json(session_dir / "history.json", _encode_bytes_fields(chat_history))

        # Save DataStore
        data_dir = session_dir / "data"
        data_dir.mkdir(exist_ok=True)
        data_store.save_to_directory(data_dir)

        # Save figure state (or remove stale file if no figure)
        figure_path = session_dir / "figure.json"
        if figure_state:
            self._write_json(figure_path, figure_state)
        elif figure_path.exists():
            figure_path.unlink()

        # Update metadata
        metadata = self._read_json(session_dir / "metadata.json") or {}
        metadata["updated_at"] = datetime.now().isoformat()
        if metadata_updates:
            metadata.update(metadata_updates)
        self._write_json(session_dir / "metadata.json", metadata)

    def load_session(self, session_id: str) -> tuple[list[dict], Path, dict, Optional[dict]]:
        """Load a session from disk.

        Args:
            session_id: The session to load.

        Returns:
            Tuple of (history_dicts, data_dir_path, metadata, figure_state).
            ``figure_state`` is the dict saved by ``PlotlyRenderer.save_state()``,
            or ``None`` if no figure was saved.

        Raises:
            FileNotFoundError: If the session does not exist.
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        history = self._read_json(session_dir / "history.json") or []
        history = _decode_bytes_fields(history)
        metadata = self._read_json(session_dir / "metadata.json") or {}
        data_dir = session_dir / "data"
        figure_state = self._read_json(session_dir / "figure.json")

        return history, data_dir, metadata, figure_state

    def list_sessions(self) -> list[dict]:
        """List all sessions, sorted by updated_at descending.

        Returns:
            List of metadata dicts (with id, created_at, updated_at, etc.).
        """
        sessions = []
        for d in self.base_dir.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = self._read_json(meta_path)
                if meta and "id" in meta:
                    sessions.append(meta)
            except Exception:
                continue

        sessions.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session directory.

        Returns:
            True if deleted, False if not found.
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            return False
        shutil.rmtree(session_dir)
        return True

    def get_most_recent_session(self) -> Optional[str]:
        """Return the session_id of the most recently updated session, or None."""
        sessions = self.list_sessions()
        if not sessions:
            return None
        return sessions[0]["id"]

    # ---- Internal helpers ----

    @staticmethod
    def _write_json(path: Path, data) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _read_json(path: Path):
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
