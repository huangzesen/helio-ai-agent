"""
Logging configuration for helio-ai-agent.

Provides structured logging to both console and file, with detailed
error information including stack traces for debugging.

Log files are stored in ~/.helio-agent/logs/ with daily rotation.
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


# Log directory
LOG_DIR = Path.home() / ".helio-agent" / "logs"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the agent.

    Args:
        verbose: If True, show DEBUG level on console; otherwise INFO only

    Returns:
        Configured logger instance
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("helio-agent")
    logger.setLevel(logging.DEBUG)  # Capture everything, filter at handler level

    # Clear existing handlers (in case of re-init)
    logger.handlers.clear()

    # File handler - detailed logging with rotation by date
    log_file = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler - less verbose unless --verbose flag
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_format = logging.Formatter("  [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

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

    # Build detailed message
    lines = [message]

    if context:
        lines.append("Context:")
        for key, value in context.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:500] + "..."
            lines.append(f"  {key}: {str_value}")

    if exc:
        lines.append(f"Exception type: {type(exc).__name__}")
        lines.append(f"Exception message: {exc}")
        lines.append("Stack trace:")
        lines.append(traceback.format_exc())

    full_message = "\n".join(lines)
    logger.error(full_message)


def log_tool_call(tool_name: str, tool_args: dict) -> None:
    """Log a tool call for debugging.

    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
    """
    logger = get_logger()
    # Truncate large args
    args_str = str(tool_args)
    if len(args_str) > 200:
        args_str = args_str[:200] + "..."
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
    logger.info(msg)


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


def get_recent_errors(days: int = 7, limit: int = 50) -> list[dict]:
    """Retrieve recent errors from log files.

    Args:
        days: How many days back to search
        limit: Maximum number of errors to return

    Returns:
        List of error entries with timestamp, message, and details
    """
    errors = []
    today = datetime.now()

    for i in range(days):
        date = today.replace(day=today.day - i) if today.day > i else today
        log_file = LOG_DIR / f"agent_{date.strftime('%Y%m%d')}.log"

        if not log_file.exists():
            continue

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                current_error = None
                for line in f:
                    if "| ERROR" in line or "| WARNING" in line:
                        if current_error:
                            errors.append(current_error)
                        # Parse the log line
                        parts = line.split(" | ", 3)
                        if len(parts) >= 4:
                            current_error = {
                                "timestamp": parts[0].strip(),
                                "level": parts[1].strip(),
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
