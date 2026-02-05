#!/usr/bin/env python3
"""
Simple JPype test - just start JVM without Autoplot
"""

import sys
import os

print("=" * 60)
print("Simple JPype Test")
print("=" * 60)

# Test 1: Import JPype
print("\n[1/3] Importing JPype...")
try:
    import jpype
    print(f"  ✓ JPype {jpype.__version__} imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Start JVM without any JARs
print("\n[2/3] Starting JVM (no JARs)...")
try:
    if not jpype.isJVMStarted():
        print(f"  JAVA_HOME: {os.getenv('JAVA_HOME', 'Not set')}")
        jpype.startJVM()
        print("  ✓ JVM started")
    else:
        print("  ℹ JVM already running")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Import a basic Java class
print("\n[3/3] Testing Java class access...")
try:
    String = jpype.JClass("java.lang.String")
    test_str = String("Hello from Java!")
    print(f"  ✓ Java String created: {test_str}")
    print(f"  ✓ Length: {test_str.length()}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: JPype works!")
print("=" * 60)
