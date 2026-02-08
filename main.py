#!/usr/bin/env python3
"""
Helio AI Agent - Main Entry Point

Run this to start an interactive conversation with the heliophysics data agent.

Usage:
    python main.py           # Normal mode
    python main.py --verbose # Show tool execution details

Commands:
    status       - Show current plan progress
    retry        - Retry a failed task
    cancel       - Cancel the current plan
    errors       - Show recent errors from logs
    capabilities - Show detailed capability summary
    reset        - Clear conversation history
    help         - Show help message
    quit         - Exit the program
"""

import sys
import argparse
from pathlib import Path

# readline is optional (not available on Windows without pyreadline3)
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

HISTORY_FILE = Path.home() / ".helio_agent_history"


def setup_readline():
    """Configure readline for input history."""
    if not READLINE_AVAILABLE:
        return
    readline.set_history_length(500)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("  Helio AI Agent")
    print("=" * 60)
    print()
    print("I can help you explore and visualize spacecraft data")
    print("from 8 missions: PSP, Solar Orbiter, ACE, OMNI, Wind,")
    print("DSCOVR, MMS, and STEREO-A.")
    print()
    print("What I can do:")
    print("  Search & plot    - Find datasets and plot them instantly")
    print("  Compute          - Magnitude, smoothing, derivatives, etc.")
    print("  Describe data    - Statistical summaries of fetched data")
    print("  Export           - Save plots to PNG, data to CSV")
    print("  Multi-step tasks - Complex requests broken into steps")
    print()
    print("Examples:")
    print("  'Show me ACE magnetic field data for last week'")
    print("  'Fetch Parker solar wind data and compute the magnitude'")
    print("  'Describe the data'")
    print("  'Save the data to a CSV file'")
    print("  'Compare Wind and ACE magnetic field for January 2024'")
    print()
    print("Commands: quit, reset, status, retry, cancel, errors,")
    print("          capabilities, help")
    print("-" * 60)
    print()


def print_capabilities():
    """Print detailed capability summary from docs."""
    docs_path = Path(__file__).parent / "docs" / "capability-summary.md"
    if not docs_path.exists():
        print("Capability summary not found.")
        return
    print()
    print("=" * 60)
    with open(docs_path, "r", encoding="utf-8") as f:
        print(f.read())
    print("=" * 60)
    print()


def check_incomplete_plans(agent, verbose: bool):
    """Check for incomplete plans from previous sessions and offer to resume."""
    from agent.tasks import get_task_store

    store = get_task_store()
    incomplete = store.get_incomplete_plans()

    if not incomplete:
        return

    # Get the most recent incomplete plan
    plan = sorted(incomplete, key=lambda p: p.created_at, reverse=True)[0]

    print("-" * 60)
    print("Found incomplete plan from previous session:")
    print(f"  Request: {plan.user_request[:60]}...")
    print(f"  Status: {plan.progress_summary()}")
    print()

    while True:
        choice = input("Resume (r), discard (d), or ignore (i)? ").strip().lower()
        if choice in ("r", "resume"):
            print()
            result = agent.resume_plan(plan)
            print(f"Agent: {result}")
            print()
            break
        elif choice in ("d", "discard"):
            result = agent.discard_plan(plan)
            print(result)
            print()
            break
        elif choice in ("i", "ignore"):
            print("Ignoring incomplete plan.")
            print()
            break
        else:
            print("Please enter 'r' to resume, 'd' to discard, or 'i' to ignore.")


def main():
    """Main conversation loop."""
    parser = argparse.ArgumentParser(description="Helio AI Agent")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show tool execution details",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch with visible GUI window for interactive plots",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Gemini model name (default: GEMINI_MODEL from .env)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="Single command to execute (non-interactive mode)",
    )
    args = parser.parse_args()

    # Skip welcome message and readline in single-command mode
    if not args.command:
        setup_readline()
        print_welcome()

    # Import here to delay JVM startup until user is ready
    try:
        from agent.core import create_agent
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you've installed dependencies: pip install -r requirements.txt")
        sys.exit(1)

    try:
        agent = create_agent(verbose=args.verbose, gui_mode=args.gui, model=args.model)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure GOOGLE_API_KEY is set in .env file")
        print("  2. Check that google-genai is installed")
        sys.exit(1)

    # Single command mode (non-interactive)
    if args.command:
        print(f"You: {args.command}\n")
        response = agent.process_message(args.command)
        print(f"Agent: {response}\n")

        # Print token usage and exit
        usage = agent.get_token_usage()
        if usage["api_calls"] > 0:
            print("-" * 60)
            print(f"  Tokens: {usage['total_tokens']:,} (in: {usage['input_tokens']:,}, out: {usage['output_tokens']:,})")
            print("-" * 60)

        sys.stdout.flush()
        import os
        os._exit(0)

    if args.gui:
        print("GUI Mode: Plot window will appear when plotting.")

    print(f"Model: {agent.model_name}")
    print()

    # Check for incomplete plans from previous sessions
    check_incomplete_plans(agent, args.verbose)

    print("Agent ready. Type your request:\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset()
                print("Conversation reset.\n")
                continue

            if user_input.lower() == "help":
                print_welcome()
                continue

            if user_input.lower() == "status":
                status = agent.get_plan_status()
                if status:
                    print(status)
                else:
                    print("No active or incomplete plans.")
                print()
                continue

            if user_input.lower() == "retry":
                result = agent.retry_failed_task()
                print(result)
                print()
                continue

            if user_input.lower() == "cancel":
                result = agent.cancel_plan()
                print(result)
                print()
                continue

            if user_input.lower() == "errors":
                from agent.logging import print_recent_errors
                print_recent_errors(days=7, limit=10)
                print()
                continue

            if user_input.lower() in ("capabilities", "caps"):
                print_capabilities()
                continue

            # Process the message
            print()
            response = agent.process_message(user_input)
            print(f"Agent: {response}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("You can continue the conversation or type 'reset' to start fresh.\n")

    # Print token usage summary and log session end
    usage = agent.get_token_usage()
    if usage["api_calls"] > 0:
        print()
        print("-" * 60)
        print(f"  Session token usage:")
        print(f"    Input tokens:  {usage['input_tokens']:,}")
        print(f"    Output tokens: {usage['output_tokens']:,}")
        print(f"    Total tokens:  {usage['total_tokens']:,}")
        print(f"    API calls:     {usage['api_calls']}")
        print("-" * 60)

        # Log session end
        from agent.logging import log_session_end
        log_session_end(usage)

    # Clean shutdown
    readline.write_history_file(HISTORY_FILE)
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
