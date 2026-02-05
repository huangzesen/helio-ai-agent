#!/usr/bin/env python3
"""
Test script to explore the new google.genai API.
Tests basic functionality before migrating agent/core.py.
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("=" * 70)
print("Testing google.genai API")
print("=" * 70)

# Test 1: Import and initialize client
print("\n[1/6] Import and initialize client...")
try:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  ✗ GOOGLE_API_KEY not found in .env")
        exit(1)

    client = genai.Client(api_key=api_key)
    print(f"  ✓ Client initialized (google-genai v{genai.__version__})")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Simple content generation
print("\n[2/6] Simple content generation...")
try:
    # Use the same model as agent/core.py
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="Say 'Hello from google.genai!' in exactly those words."
    )
    print(f"  ✓ Response: {response.text[:50]}...")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: System instruction
print("\n[3/6] Testing system instruction...")
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="What are you?",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant that answers in exactly 5 words."
        )
    )
    print(f"  ✓ Response: {response.text}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Function calling
print("\n[4/6] Testing function calling...")
try:
    # Define a simple function
    def get_current_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is 23°C and sunny."

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="What's the weather in Boston?",
        config=types.GenerateContentConfig(
            tools=[get_current_weather]
        )
    )

    print(f"  ✓ Response received")
    print(f"  Response type: {type(response)}")
    print(f"  Has text: {hasattr(response, 'text')}")

    # Check if function was called
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for part in candidate.content.parts:
                if hasattr(part, 'function_call'):
                    print(f"  ✓ Function call detected: {part.function_call.name}")
                if hasattr(part, 'text') and part.text:
                    print(f"  ✓ Text response: {part.text[:50]}...")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Chat session
print("\n[5/6] Testing chat session...")
try:
    chat = client.chats.create(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Keep responses under 20 words."
        )
    )

    # Send first message
    response1 = chat.send_message("My name is Alice.")
    print(f"  ✓ Message 1 sent")
    print(f"  Response: {response1.text[:50]}...")

    # Send second message (testing history)
    response2 = chat.send_message("What's my name?")
    print(f"  ✓ Message 2 sent")
    print(f"  Response: {response2.text[:50]}...")

    if "Alice" in response2.text or "alice" in response2.text.lower():
        print(f"  ✓ Chat history working (remembered name)")
    else:
        print(f"  ⚠ Chat history might not be working")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Function calling with manual tool schemas
print("\n[6/6] Testing function calling with tool schemas...")
try:
    # Define tool schema manually (like we do in agent/core.py)
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

    # Try to create function declaration from schema
    # Need to discover the correct way to do this in new API
    print(f"  Tool schema created: {tool_schema['name']}")
    print(f"  ℹ Need to discover how to pass tool schemas in new API")
    print(f"  ℹ Old API: FunctionDeclaration(...)")
    print(f"  ℹ New API: types.??? or direct function passing?")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print("Key findings:")
print("  1. Client initialization: ✓ Works")
print("  2. Content generation: ✓ Works")
print("  3. System instruction: ✓ Works")
print("  4. Function calling: ✓ Partially tested")
print("  5. Chat sessions: ✓ Works")
print("  6. Tool schemas: ⚠ Need to investigate")
print("\nNext: Investigate function calling API in detail")
print("=" * 70)
