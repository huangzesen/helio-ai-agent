#!/usr/bin/env python3
"""
Detailed test of function calling in google.genai API.
This is critical for agent migration.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Function Calling Deep Dive - google.genai")
print("=" * 70)

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Define a simple tool function
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The name of the city

    Returns:
        Weather description
    """
    return f"The weather in {location} is 23°C and sunny."

print("\n[Test 1] Simple function call with Python function")
print("-" * 70)

try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="What's the weather in Boston?",
        config=types.GenerateContentConfig(
            tools=[get_weather]
        )
    )

    print(f"Response type: {type(response)}")
    print(f"Response has text: {hasattr(response, 'text')}")

    # Inspect response structure
    print(f"\nResponse attributes:")
    for attr in dir(response):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    # Check for candidates
    if hasattr(response, 'candidates') and response.candidates:
        print(f"\nCandidates count: {len(response.candidates)}")
        candidate = response.candidates[0]
        print(f"Candidate type: {type(candidate)}")

        if hasattr(candidate, 'content'):
            print(f"Content type: {type(candidate.content)}")

            if hasattr(candidate.content, 'parts'):
                print(f"Parts count: {len(candidate.content.parts)}")

                for i, part in enumerate(candidate.content.parts):
                    print(f"\nPart {i}:")
                    print(f"  Type: {type(part)}")
                    print(f"  Attributes: {[a for a in dir(part) if not a.startswith('_')]}")

                    # Check for different part types
                    if hasattr(part, 'text') and part.text:
                        print(f"  Text: {part.text[:100]}")

                    if hasattr(part, 'function_call'):
                        print(f"  function_call: {part.function_call}")
                        if part.function_call:
                            print(f"    name: {part.function_call.name if hasattr(part.function_call, 'name') else 'N/A'}")
                            print(f"    args: {part.function_call.args if hasattr(part.function_call, 'args') else 'N/A'}")

                    if hasattr(part, 'function_response'):
                        print(f"  function_response: {part.function_response}")

    # Try to get the text
    try:
        print(f"\nResponse text: {response.text}")
    except Exception as e:
        print(f"\nCouldn't get response.text: {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("[Test 2] Function call with manual tool schema (like agent/core.py)")
print("-" * 70)

try:
    # This is how we currently define tools in agent/core.py
    # We need to find the equivalent in new API

    tool_schema = {
        "name": "search_datasets",
        "description": "Search for spacecraft datasets by keywords",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (spacecraft name, instrument, parameter)"
                }
            },
            "required": ["query"]
        }
    }

    print("Tool schema (current format):")
    print(json.dumps(tool_schema, indent=2))

    print("\nAttempting to find new API equivalent...")

    # Check what's available in types module
    print("\nAvailable in google.genai.types:")
    type_members = [m for m in dir(types) if not m.startswith('_')]
    for member in type_members:
        if 'function' in member.lower() or 'tool' in member.lower():
            print(f"  - {member}")

    # Try to use types.Tool or types.FunctionDeclaration
    if hasattr(types, 'FunctionDeclaration'):
        print("\n✓ types.FunctionDeclaration exists!")
        print(f"  Type: {types.FunctionDeclaration}")

        # Try to create a function declaration
        try:
            fd = types.FunctionDeclaration(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"]
            )
            print(f"  ✓ Created FunctionDeclaration: {fd}")
        except Exception as e:
            print(f"  ✗ Failed to create: {e}")

    if hasattr(types, 'Tool'):
        print("\n✓ types.Tool exists!")
        print(f"  Type: {types.Tool}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("[Test 3] Multi-turn function calling with responses")
print("-" * 70)

try:
    # Create a chat session with tools
    chat = client.chats.create(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            tools=[get_weather],
            system_instruction="You are a helpful weather assistant."
        )
    )

    # Send a message that should trigger a function call
    print("User: What's the weather in Paris?")
    response1 = chat.send_message("What's the weather in Paris?")

    print(f"\nFirst response type: {type(response1)}")

    # Check if function was called
    function_called = False
    if hasattr(response1, 'candidates') and response1.candidates:
        for part in response1.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_called = True
                print(f"✓ Function called: {part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'}")
                print(f"  Args: {part.function_call.args if hasattr(part.function_call, 'args') else {}}")

                # Execute the function
                location = dict(part.function_call.args).get('location', 'unknown')
                result = get_weather(location)
                print(f"  Executed: {result}")

                # Now we need to send the function response back
                # How to do this in new API?
                print("\n  Need to send function response back...")
                print("  Old API: genai.protos.Part(function_response=...)")
                print("  New API: ???")

    if not function_called:
        print("✗ No function call detected")
        try:
            print(f"Response text: {response1.text}")
        except:
            pass

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("Key findings:")
print("  1. Function calling with Python functions: ✓ Works")
print("  2. Response structure: Need to inspect carefully")
print("  3. Manual tool schemas: Need to find FunctionDeclaration equivalent")
print("  4. Function responses: Need to discover new API for sending back")
print("=" * 70)
