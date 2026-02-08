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
    ) -> None:
        """Save chat history and DataStore to disk.

        Args:
            session_id: The session to save.
            chat_history: List of Content dicts (from Content.model_dump).
            data_store: A DataStore instance to persist.
            metadata_updates: Optional dict to merge into metadata
                (e.g. token_usage, turn_count).
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_id}")

        # Save history
        self._write_json(session_dir / "history.json", chat_history)

        # Save DataStore
        data_dir = session_dir / "data"
        data_dir.mkdir(exist_ok=True)
        data_store.save_to_directory(data_dir)

        # Update metadata
        metadata = self._read_json(session_dir / "metadata.json") or {}
        metadata["updated_at"] = datetime.now().isoformat()
        if metadata_updates:
            metadata.update(metadata_updates)
        self._write_json(session_dir / "metadata.json", metadata)

    def load_session(self, session_id: str) -> tuple[list[dict], Path, dict]:
        """Load a session from disk.

        Args:
            session_id: The session to load.

        Returns:
            Tuple of (history_dicts, data_dir_path, metadata).

        Raises:
            FileNotFoundError: If the session does not exist.
        """
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        history = self._read_json(session_dir / "history.json") or []
        metadata = self._read_json(session_dir / "metadata.json") or {}
        data_dir = session_dir / "data"

        return history, data_dir, metadata

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
