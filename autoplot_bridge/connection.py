"""
Autoplot connection via JPype.
Run this file directly to test: python -m autoplot_bridge.connection
"""
import sys
from pathlib import Path

import jpype

from config import AUTOPLOT_JAR, JAVA_HOME


def _find_jvm() -> str:
    """Locate the JVM DLL, trying server then client then JPype default."""
    if JAVA_HOME:
        server = Path(JAVA_HOME) / "bin" / "server" / "jvm.dll"
        if server.exists():
            return str(server)
        client = Path(JAVA_HOME) / "bin" / "client" / "jvm.dll"
        if client.exists():
            return str(client)
    return jpype.getDefaultJVMPath()


def init_autoplot(verbose: bool = False):
    """Initialize JVM with Autoplot on classpath. Returns ScriptContext class."""
    if not jpype.isJVMStarted():
        jar_path = Path(AUTOPLOT_JAR).expanduser().resolve()
        if not jar_path.exists():
            raise FileNotFoundError(f"Autoplot JAR not found: {jar_path}")

        jvm_path = _find_jvm()
        if verbose:
            print(f"  [Autoplot] Starting JVM: {jvm_path}")
            print(f"  [Autoplot] JAR: {jar_path}")
        jpype.startJVM(jvm_path, '-Djava.awt.headless=true', classpath=[str(jar_path)])
        if verbose:
            print("  [Autoplot] JVM started.")
    elif verbose:
        print("  [Autoplot] JVM already running.")

    if verbose:
        print("  [Autoplot] Loading ScriptContext class...")
    ScriptContext = jpype.JClass("org.autoplot.ScriptContext")

    # Create a headless application model so plot() doesn't try to create a GUI.
    # Without this, plot() hangs on macOS because Swing needs the main thread.
    if not ScriptContext.isModelInitialized():
        if verbose:
            print("  [Autoplot] Creating headless application model...")
        ScriptContext.createApplicationModel('')
        if verbose:
            print("  [Autoplot] Application model ready.")

    if verbose:
        print("  [Autoplot] ScriptContext ready.")
    return ScriptContext


def get_script_context(verbose: bool = False):
    """Get the Autoplot ScriptContext for issuing commands."""
    return init_autoplot(verbose=verbose)


if __name__ == "__main__":
    print("Testing Autoplot connection...")
    print(f"JAVA_HOME:    {JAVA_HOME}")
    print(f"AUTOPLOT_JAR: {AUTOPLOT_JAR}")
    print(f"JVM path:     {_find_jvm()}")
    print()

    try:
        ctx = get_script_context()

        System = jpype.JClass("java.lang.System")
        java_version = str(System.getProperty("java.version"))
        print(f"Java version: {java_version}")
        print(f"ScriptContext: {ctx}")
        print("SUCCESS")

        sys.stdout.flush()
        import os
        os._exit(0)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.stdout.flush()
        import os
        os._exit(1)
