"""Utilities for Gemini ThinkingConfig — thought extraction and token tracking."""


def extract_thoughts(response) -> list[str]:
    """Extract thought text from Gemini response parts where part.thought=True.

    response.text already filters these out, so this is for verbose logging only.
    """
    thoughts = []
    if not response.candidates:
        return thoughts
    content = response.candidates[0].content
    if not content:
        return thoughts
    for part in content.parts:
        if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
            thoughts.append(part.text)
    return thoughts


def get_thinking_tokens(response) -> int:
    """Extract thoughts_token_count from response metadata (0 if unavailable)."""
    meta = getattr(response, "usage_metadata", None)
    if meta:
        return getattr(meta, "thoughts_token_count", 0) or 0
    return 0
