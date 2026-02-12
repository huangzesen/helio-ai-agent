import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Secret — stays in .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# User config — loaded from ~/.helio-agent/config.json (primary)
# or project-root config.json (fallback).
CONFIG_PATH = Path.home() / ".helio-agent" / "config.json"
_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_user_config: dict = {}


def _load_config() -> dict:
    # Project-local config.json as base, user home config overlaid on top
    merged: dict = {}
    for path in (_LOCAL_CONFIG_PATH, CONFIG_PATH):
        if path is not None and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    merged.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return merged


def get(key: str, default=None):
    """Get a config value by dot-separated key. E.g. get('memory.max_preferences', 15)"""
    keys = key.split(".")
    val = _user_config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


_user_config = _load_config()


# ---- Data directory -----------------------------------------------------------
# Single source of truth for the base data directory (logs, sessions, memory, etc.).
# Priority: HELIO_AGENT_DIR env var > "data_dir" config key > ~/.helio-agent

_data_dir: Optional[Path] = None


def get_data_dir() -> Path:
    """Return the resolved base data directory.

    Resolution order:
    1. ``HELIO_AGENT_DIR`` environment variable (highest — useful for CI/Docker)
    2. ``"data_dir"`` key in config.json
    3. ``~/.helio-agent`` (default)
    """
    global _data_dir
    if _data_dir is not None:
        return _data_dir
    env_val = os.environ.get("HELIO_AGENT_DIR")
    if env_val:
        _data_dir = Path(env_val).expanduser().resolve()
    else:
        configured = get("data_dir")
        if configured:
            _data_dir = Path(configured).expanduser().resolve()
        else:
            _data_dir = Path.home() / ".helio-agent"
    return _data_dir


def _reset_data_dir() -> None:
    """Reset the cached data directory (for testing only)."""
    global _data_dir
    _data_dir = None


# Flat aliases for backward compatibility (existing code does `from config import GEMINI_MODEL`)
GEMINI_MODEL = get("model", "gemini-3-flash-preview")
GEMINI_SUB_AGENT_MODEL = get("sub_agent_model", "gemini-3-flash-preview")
GEMINI_PLANNER_MODEL = get("planner_model", GEMINI_MODEL)
GEMINI_FALLBACK_MODEL = get("fallback_model", "gemini-3-flash-preview")
DATA_BACKEND = get("data_backend", "cdf")  # "cdf" or "hapi"
CATALOG_SEARCH_METHOD = get("catalog_search_method", "semantic")  # "semantic" or "substring"
