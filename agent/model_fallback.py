"""
Automatic model fallback for Gemini API calls.

When any model hits quota or rate limits (429 RESOURCE_EXHAUSTED), all agents
switch to GEMINI_FALLBACK_MODEL for the remainder of the session.
"""

from google.genai import errors as genai_errors

from .logging import get_logger

logger = get_logger()

# Session-level flag: once set, ALL subsequent API calls use the fallback model.
_fallback_active = False
_fallback_model = None


def activate_fallback(fallback_model: str) -> None:
    """Activate fallback mode — all future calls use *fallback_model*."""
    global _fallback_active, _fallback_model
    _fallback_active = True
    _fallback_model = fallback_model
    logger.warning(f"[Fallback] Activated — all models switching to {fallback_model}")


def is_fallback_active() -> bool:
    return _fallback_active


def get_active_model(requested_model: str) -> str:
    """Return the model to actually use: fallback if active, else requested."""
    if _fallback_active and _fallback_model:
        return _fallback_model
    return requested_model


def is_quota_error(exc: Exception) -> bool:
    """Return True if *exc* is a 429 / RESOURCE_EXHAUSTED error."""
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)
    return False
