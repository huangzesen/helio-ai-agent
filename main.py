#!/usr/bin/env python3
"""
Autoplot Natural Language Interface - Main Entry Point

Run this to start an interactive conversation with the Autoplot agent.

Usage:
    python main.py           # Normal mode
    python main.py --verbose # Show tool execution details
"""

import sys
import argparse
import readline
from pathlib import Path

HISTORY_FILE = Path.home() / ".autoplot_history"


def setup_readline():
    """Configure readline for input history."""
    readline.set_history_length(500)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("  Autoplot Natural Language Interface")
    print("=" * 60)
    print()
    print("I can help you visualize spacecraft data. Try commands like:")
    print("  - 'Show me Parker magnetic field data for last week'")
    print("  - 'What data is available for Solar Orbiter?'")
    print("  - 'Plot ACE solar wind velocity for January 2024'")
    print()
    print("Supported time ranges:")
    print("  - Relative:  'last week', 'last 3 days', 'last month', 'last year'")
    print("  - Month:     'January 2024', 'Jan 2024'")
    print("  - Date:      '2024-01-15' (full day)")
    print("  - Range:     '2024-01-15 to 2024-01-20'")
    print("  - Sub-day:   '2024-01-15T06:00 to 2024-01-15T18:00'")
    print()
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'reset' to clear conversation history.")
    print("-" * 60)
    print()


def main():
    """Main conversation loop."""
    parser = argparse.ArgumentParser(description="Autoplot Natural Language Interface")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show tool execution details",
    )
    args = parser.parse_args()

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
        agent = create_agent(verbose=args.verbose)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure GOOGLE_API_KEY is set in .env file")
        print("  2. Check that google-generativeai is installed")
        sys.exit(1)

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

            # Process the message
            print()
            response = agent.process_message(user_input)
            print(f"Agent: {response}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            readline.write_history_file(HISTORY_FILE)
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("You can continue the conversation or type 'reset' to start fresh.\n")

    # Clean shutdown
    readline.write_history_file(HISTORY_FILE)
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
