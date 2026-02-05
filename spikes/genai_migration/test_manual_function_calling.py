#!/usr/bin/env python3
"""
Test MANUAL function calling (without automatic execution).
This is what agent/core.py needs.
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Manual Function Calling - google.genai")
print("=" * 70)

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Test 1: Disable automatic function calling
print("\n[Test 1] Disable automatic function calling")
print("-" * 70)

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # This should NOT be called automatically
    print("  !!! FUNCTION WAS CALLED AUTOMATICALLY !!!")
    return f"Weather in {location}: 23°C"

try:
    # Check if we can disable automatic calling
    print("Checking AutomaticFunctionCallingConfig...")

    if hasattr(types, 'AutomaticFunctionCallingConfig'):
        print("  ✓ Found AutomaticFunctionCallingConfig")
        print(f"    Type: {types.AutomaticFunctionCallingConfig}")

        # Try to disable automatic calling
        config = types.GenerateContentConfig(
            tools=[get_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents="What's the weather in Boston?",
            config=config
        )

        print(f"\nResponse type: {type(response)}")

        # Check if we got a function call (not executed)
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    print(f"  ✓ Got function_call (not executed automatically)")
                    print(f"    Name: {part.function_call.name}")
                    print(f"    Args: {dict(part.function_call.args)}")
                elif hasattr(part, 'text') and part.text:
                    print(f"  Text response: {part.text[:100]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Use FunctionDeclaration instead of Python function
print("\n" + "=" * 70)
print("[Test 2] Use FunctionDeclaration (no executable function)")
print("-" * 70)

try:
    # Create a function declaration WITHOUT an executable function
    func_decl = types.FunctionDeclaration(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["location"]
        }
    )

    print(f"Created FunctionDeclaration: {func_decl.name}")

    # Create a Tool with this declaration
    tool = types.Tool(function_declarations=[func_decl])
    print(f"Created Tool with function_declarations")

    # Use it in generate_content
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="What's the weather in Seattle?",
        config=types.GenerateContentConfig(
            tools=[tool]
        )
    )

    print(f"\nResponse type: {type(response)}")

    # Check response
    if hasattr(response, 'candidates') and response.candidates:
        for i, part in enumerate(response.candidates[0].content.parts):
            print(f"\nPart {i}:")
            if hasattr(part, 'function_call') and part.function_call:
                print(f"  ✓ GOT FUNCTION CALL (NOT EXECUTED)")
                print(f"    Name: {part.function_call.name}")
                print(f"    Args: {dict(part.function_call.args)}")
            elif hasattr(part, 'text') and part.text:
                print(f"  Text: {part.text[:100]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Send function response back
print("\n" + "=" * 70)
print("[Test 3] Send function response back to model")
print("-" * 70)

try:
    # Create chat with function declaration
    func_decl = types.FunctionDeclaration(
        name="search_datasets",
        description="Search for spacecraft datasets",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    )

    tool = types.Tool(function_declarations=[func_decl])

    chat = client.chats.create(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            tools=[tool],
            system_instruction="You help users find spacecraft data. Use search_datasets tool."
        )
    )

    # Send message
    print("User: Find Parker Solar Probe magnetic field data")
    response1 = chat.send_message("Find Parker Solar Probe magnetic field data")

    print(f"\nResponse 1 type: {type(response1)}")

    # Check for function call
    function_call_found = None
    if hasattr(response1, 'candidates') and response1.candidates:
        for part in response1.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call_found = part.function_call
                print(f"  ✓ Function call detected: {part.function_call.name}")
                print(f"    Args: {dict(part.function_call.args)}")

    if function_call_found:
        # Execute the function (simulated)
        print("\n  Executing function (simulated)...")
        mock_result = {
            "status": "success",
            "spacecraft": "PSP",
            "datasets": ["PSP_FLD_L2_MAG_RTN_1MIN"]
        }
        print(f"  Result: {mock_result}")

        # Send function response back
        print("\n  Sending function response back...")

        # Try different methods to send response
        try:
            # Method 1: Use Part.from_function_response
            if hasattr(types.Part, 'from_function_response'):
                function_response_part = types.Part.from_function_response(
                    name=function_call_found.name,
                    response=mock_result
                )
                print(f"    ✓ Created Part.from_function_response")

                # Send it back
                response2 = chat.send_message(function_response_part)
                print(f"    ✓ Sent function response")
                print(f"    Response 2: {response2.text[:100] if hasattr(response2, 'text') else 'No text'}...")

        except Exception as e:
            print(f"    ✗ Method 1 failed: {e}")

    else:
        print("  ✗ No function call detected")
        try:
            print(f"  Response text: {response1.text}")
        except:
            pass

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("Key discoveries:")
print("  1. AutomaticFunctionCallingConfig can disable auto-execution")
print("  2. FunctionDeclaration + Tool = manual function calling")
print("  3. Part.from_function_response() to send results back")
print("  4. This matches our agent/core.py needs!")
print("=" * 70)
