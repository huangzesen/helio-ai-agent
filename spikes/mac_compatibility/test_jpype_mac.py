#!/usr/bin/env python3
"""
Test 2: JPype + JVM Initialization on macOS

Verifies that JPype can start the JVM and load the Autoplot JAR on macOS.
This is a critical test as JPype behavior can vary by platform.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_jpype_import():
    """Test that jpype module can be imported."""
    print("\n[1/4] Testing JPype import...")
    try:
        import jpype
        print(f"  ✓ JPype version: {jpype.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import jpype: {e}")
        return False


def test_jvm_start():
    """Test JVM startup with Autoplot JAR on classpath."""
    print("\n[2/4] Testing JVM initialization...")

    try:
        import jpype
        from config import AUTOPLOT_JAR

        jar_path = Path(AUTOPLOT_JAR).expanduser().resolve()

        if not jar_path.exists():
            print(f"  ✗ Autoplot JAR not found: {jar_path}")
            return False

        print(f"  JAR: {jar_path}")

        # Start JVM with Autoplot on classpath
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[str(jar_path)])
            print("  ✓ JVM started successfully")
        else:
            print("  ℹ JVM already running")

        return True

    except Exception as e:
        print(f"  ✗ JVM startup failed: {e}")
        return False


def test_autoplot_import():
    """Test importing Autoplot ScriptContext class."""
    print("\n[3/4] Testing Autoplot class import...")

    try:
        import jpype

        if not jpype.isJVMStarted():
            print("  ✗ JVM not started (run test_jvm_start first)")
            return False

        # Import Autoplot's main scripting class
        ScriptContext = jpype.JClass("org.autoplot.ScriptContext")
        print(f"  ✓ ScriptContext class loaded: {ScriptContext}")

        return True

    except Exception as e:
        print(f"  ✗ Failed to import ScriptContext: {e}")
        return False


def test_script_context_creation():
    """Test creating a ScriptContext instance."""
    print("\n[4/4] Testing ScriptContext instantiation...")

    try:
        from autoplot_bridge.connection import get_script_context

        ctx = get_script_context()
        print(f"  ✓ ScriptContext created: {type(ctx)}")

        # Try to access a basic property
        dom = ctx.getDocumentModel()
        print(f"  ✓ Document model accessible: {type(dom)}")

        return True

    except Exception as e:
        print(f"  ✗ Failed to create ScriptContext: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all JPype/JVM tests."""
    print("=" * 60)
    print("JPype + JVM Initialization Test (macOS)")
    print("=" * 60)

    results = [
        test_jpype_import(),
        test_jvm_start(),
        test_autoplot_import(),
        test_script_context_creation(),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\n✓ JPype and JVM work correctly on macOS!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
