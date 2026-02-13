"""
Logging configuration for helio-ai-agent.

Three logging destinations, two tiers:

Tier 1 — File + Console (identical by default):
  - Full detail, no truncation
  - File: always DEBUG level
  - Console: DEBUG if --verbose, WARNING+ otherwise
  - Filter: both drop records with extra={'skip_file': True}
  - Format: "timestamp | level | name | session_id | message"
  - Config console_format options:
    - "full"   — (default) same structured format as the file handler
    - "simple" — bare messages for DEBUG/INFO, [LEVEL] prefix for WARNING+
    - "gui"    — curated tagged records only (mirrors Gradio live-log sidebar)
    - "clean"  — no console output at all (file logging still active)

Tier 2 — Gradio live log (curated):
  - Only records tagged with a key in GRADIO_VISIBLE_TAGS
  - Handler lives in gradio_app.py (_ListHandler)
  - Preview snippets use extra={**tagged("x"), "skip_file": True}
    so they reach Gradio but not file/console

Log files are stored in ~/.helio-agent/logs/ with one file per session.
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import get_data_dir


# Log directory
LOG_DIR = get_data_dir() / "logs"

# Tags that the Gradio live-log handler will display.
# To show a new category in Gradio, tag the log call with
# ``extra=tagged("my_tag")`` and add ``"my_tag"`` here.
GRADIO_VISIBLE_TAGS = frozenset({
    "delegation",       # "[Router] Delegating to X specialist"
    "delegation_done",  # "[Router] X specialist finished"
    "plan_event",       # Plan created / completed / failed
    "plan_task",        # Plan task executing / round progress
    "progress",         # Milestone updates during planning/discovery
    "data_fetched",     # "[DataOps] Stored 'label' (N points)"
    "thinking",         # "[Thinking] ..." (truncated preview)
    "error",            # log_error() — real errors with context/stack traces
})


def tagged(tag: str) -> dict:
    """Return ``extra`` dict for logger calls: ``logger.debug("...", extra=tagged("x"))``."""
    return {"log_tag": tag}


# Module-level state (shared across re-inits)
_session_filter: Optional["_SessionFilter"] = None
_current_log_file: Optional[Path] = None
_token_log_file: Optional[Path] = None


class _SessionFilter(logging.Filter):
    """Injects session_id into every log record."""

    def __init__(self) -> None:
        super().__init__()
        self.session_id = ""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = self.session_id or "-"
        if not hasattr(record, "log_tag"):
            record.log_tag = ""
        return True


class _SkipDuplicateFilter(logging.Filter):
    """Drops records marked with ``extra={'skip_file': True}``.

    Applied to both file and console handlers so that Gradio-only preview
    snippets (tagged + skip_file) don't appear in either persistent destination.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "skip_file", False)


class _ConsoleFormatter(logging.Formatter):
    """Console formatter: shows [LEVEL] prefix only for WARNING and above.

    DEBUG/INFO messages print bare (e.g. ``  [Gemini] Sending...``).
    WARNING/ERROR messages include the level (e.g. ``  [WARNING] ...``).
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            return f"  [{record.levelname}] {record.getMessage()}"
        return f"  {record.getMessage()}"


class _GradioStyleFilter(logging.Filter):
    """Pass only records tagged with a key in GRADIO_VISIBLE_TAGS.

    Mirrors the curated output shown in the Gradio live-log sidebar,
    but routed to the console instead.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tag = getattr(record, "log_tag", "")
        return tag in GRADIO_VISIBLE_TAGS


def setup_token_log(session_timestamp: str) -> Path:
    """Create the per-API-call token usage log file.

    Args:
        session_timestamp: Timestamp string (e.g. '20260210_211534') shared
            with the main agent log for easy correlation.

    Returns:
        Path to the token log file.
    """
    global _token_log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"token_{session_timestamp}.log"
    _token_log_file = path
    # Write header line
    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp | agent | tool_context | in out think | cum_in cum_out cum_think | calls\n")
    return path


def log_token_usage(
    agent_name: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int,
    cumulative_input: int,
    cumulative_output: int,
    cumulative_thinking: int,
    api_calls: int,
    tool_context: str = "send_message",
    token_log_path: Optional[Path] = None,
) -> None:
    """Append one line to the token usage log.

    Args:
        agent_name: Name of the agent (e.g. 'OrchestratorAgent').
        input_tokens: Tokens consumed in this API call (prompt).
        output_tokens: Tokens produced in this API call (candidates).
        thinking_tokens: Thinking tokens in this API call.
        cumulative_input: Running total of input tokens for this agent.
        cumulative_output: Running total of output tokens for this agent.
        cumulative_thinking: Running total of thinking tokens for this agent.
        api_calls: Running total of API calls for this agent.
        tool_context: What triggered this API call (tool name, 'initial_message', etc.).
        token_log_path: Explicit path to the token log file. If None, falls back
            to the module-global ``_token_log_file`` (legacy behaviour).
    """
    target = token_log_path or _token_log_file
    if target is None:
        return
    # Truncate tool_context to 60 chars
    ctx = tool_context[:60] if tool_context else "unknown"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts} | {agent_name} | {ctx} | "
        f"in:{input_tokens} out:{output_tokens} think:{thinking_tokens} | "
        f"cum_in:{cumulative_input} cum_out:{cumulative_output} cum_think:{cumulative_thinking} | "
        f"calls:{api_calls}\n"
    )
    try:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass  # Don't let token logging break the agent


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the agent.

    Args:
        verbose: If True, show DEBUG level on console; otherwise INFO only

    Returns:
        Configured logger instance
    """
    global _session_filter, _current_log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("helio-agent")
    logger.setLevel(logging.DEBUG)  # Capture everything, filter at handler level

    # Clear existing handlers (in case of re-init)
    logger.handlers.clear()

    # Session filter — reuse existing instance to preserve session_id across re-inits
    if _session_filter is None:
        _session_filter = _SessionFilter()
    logger.addFilter(_session_filter)

    # File handler - one log file per session
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"agent_{session_timestamp}.log"
    _current_log_file = log_file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(_SkipDuplicateFilter())
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(session_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler - less verbose unless --verbose flag
    import config as _config
    console_format = _config.get("console_format", "simple")

    if console_format != "clean":
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.addFilter(_SkipDuplicateFilter())

        if console_format == "gui":
            # Curated output: same tagged records as the Gradio live-log sidebar
            console_handler.setLevel(logging.DEBUG)
            console_handler.addFilter(_GradioStyleFilter())
            console_handler.setFormatter(_ConsoleFormatter())
        elif console_format == "simple":
            console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
            console_handler.setFormatter(_ConsoleFormatter())
        else:
            # "full" (default)
            console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
            console_handler.setFormatter(file_format)  # identical to file handler

        logger.addHandler(console_handler)
    # "clean" — no console handler at all (file logging still active)

    # Token usage log (shares the same timestamp suffix)
    setup_token_log(session_timestamp)

    logger.info("=" * 60)
    logger.info(f"Session started at {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")

    return logger


def get_logger() -> logging.Logger:
    """Get the agent logger instance.

    Returns:
        The helio-agent logger (creates with defaults if not configured)
    """
    logger = logging.getLogger("helio-agent")
    if not logger.handlers:
        # Not configured yet, set up with defaults
        return setup_logging(verbose=False)
    return logger


def set_session_id(session_id: str) -> None:
    """Set the session ID that will be included in all subsequent log lines.

    Args:
        session_id: The session identifier (e.g. '20260209_223120_4b7103d5')
    """
    global _session_filter
    if _session_filter is None:
        # Logger not set up yet — create filter so it's ready when logging starts
        _session_filter = _SessionFilter()
    _session_filter.session_id = session_id


def log_error(
    message: str,
    exc: Optional[Exception] = None,
    context: Optional[dict] = None,
) -> None:
    """Log an error with full details including stack trace.

    Args:
        message: Error description
        exc: Optional exception to include stack trace from
        context: Optional dict of additional context (tool name, args, etc.)
    """
    logger = get_logger()

    # Build detailed message for the file log
    lines = [message]

    if context:
        lines.append("Context:")
        for key, value in context.items():
            lines.append(f"  {key}: {value}")

    if exc:
        lines.append(f"Exception type: {type(exc).__name__}")
        lines.append(f"Exception message: {exc}")
        lines.append("Stack trace:")
        lines.append(traceback.format_exc())

    full_message = "\n".join(lines)

    # Full details to file log only (no tag → not shown in Gradio)
    logger.error(full_message)

    # Short summary to Gradio live-log (truncate long messages)
    short = message if len(message) <= 200 else message[:200] + "..."
    logger.error(short, extra=tagged("error"))


def log_tool_call(tool_name: str, tool_args: dict) -> None:
    """Log a tool call for debugging.

    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
    """
    logger = get_logger()
    args_str = str(tool_args)
    logger.debug(f"Tool call: {tool_name}({args_str})")


def log_tool_result(tool_name: str, result: dict, success: bool) -> None:
    """Log a tool result.

    Args:
        tool_name: Name of the tool
        result: Result dict from the tool
        success: Whether the tool succeeded
    """
    logger = get_logger()
    if success:
        logger.debug(f"Tool result: {tool_name} -> success")
    else:
        error_msg = result.get("message", "Unknown error")
        logger.warning(f"Tool result: {tool_name} -> error: {error_msg}")


def log_plan_event(event: str, plan_id: str, details: Optional[str] = None) -> None:
    """Log a planning/execution event.

    Args:
        event: Event type (created, executing, completed, failed, etc.)
        plan_id: ID of the plan
        details: Optional additional details
    """
    logger = get_logger()
    msg = f"Plan {event}: {plan_id[:8]}..."
    if details:
        msg += f" - {details}"
    logger.info(msg, extra=tagged("plan_event"))


def log_session_end(token_usage: dict) -> None:
    """Log session end with usage stats.

    Args:
        token_usage: Dict with input_tokens, output_tokens, api_calls
    """
    logger = get_logger()
    logger.info(
        f"Session ended. Tokens: {token_usage.get('total_tokens', 0):,} "
        f"(in: {token_usage.get('input_tokens', 0):,}, "
        f"out: {token_usage.get('output_tokens', 0):,}), "
        f"API calls: {token_usage.get('api_calls', 0)}"
    )
    logger.info("=" * 60)


def get_token_log_path() -> Optional[Path]:
    """Return the path to the current session's token log file (or None)."""
    return _token_log_file


def get_current_log_path() -> Path:
    """Return the path to the current session's log file."""
    if _current_log_file is not None:
        return _current_log_file
    # Fallback: find most recent log file in the directory
    logs = sorted(LOG_DIR.glob("agent_*.log"))
    if logs:
        return logs[-1]
    return LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def get_log_size(path: Path) -> int:
    """Return the size of a log file in bytes, or 0 if it doesn't exist."""
    try:
        return path.stat().st_size
    except (OSError, ValueError):
        return 0


def get_recent_errors(days: int = 7, limit: int = 50) -> list[dict]:
    """Retrieve recent errors from log files.

    Args:
        days: How many days back to search
        limit: Maximum number of errors to return

    Returns:
        List of error entries with timestamp, message, and details
    """
    errors = []
    cutoff = datetime.now().timestamp() - days * 86400
    # Collect all log files, newest first
    log_files = sorted(LOG_DIR.glob("agent_*.log"), reverse=True)

    for log_file in log_files:
        if log_file.stat().st_mtime < cutoff:
            break

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                current_error = None
                for line in f:
                    if "| ERROR" in line or "| WARNING" in line:
                        if current_error:
                            errors.append(current_error)
                        # Parse the log line
                        # Format: timestamp | level | name | session_id | message
                        parts = line.split(" | ", 4)
                        if len(parts) >= 5:
                            current_error = {
                                "timestamp": parts[0].strip(),
                                "level": parts[1].strip(),
                                "session_id": parts[3].strip(),
                                "message": parts[4].strip(),
                                "details": [],
                            }
                        elif len(parts) >= 4:
                            # Backwards-compat with old 4-field format
                            current_error = {
                                "timestamp": parts[0].strip(),
                                "level": parts[1].strip(),
                                "session_id": "-",
                                "message": parts[3].strip(),
                                "details": [],
                            }
                    elif current_error and line.startswith("  "):
                        # Continuation of error details
                        current_error["details"].append(line.rstrip())

                if current_error:
                    errors.append(current_error)

        except Exception:
            continue

        if len(errors) >= limit:
            break

    return errors[:limit]


def print_recent_errors(days: int = 7, limit: int = 10) -> None:
    """Print recent errors to console for review.

    Args:
        days: How many days back to search
        limit: Maximum number of errors to show
    """
    errors = get_recent_errors(days=days, limit=limit)

    if not errors:
        print(f"No errors found in the last {days} days.")
        return

    print(f"Recent errors (last {days} days, showing up to {limit}):")
    print("-" * 60)

    for i, error in enumerate(errors, 1):
        print(f"\n{i}. [{error['timestamp']}] {error['level']}")
        print(f"   {error['message']}")
        if error["details"]:
            for detail in error["details"][:5]:  # Limit detail lines
                print(f"   {detail}")
            if len(error["details"]) > 5:
                print(f"   ... and {len(error['details']) - 5} more lines")

    print("-" * 60)
    print(f"Full logs available at: {LOG_DIR}")
