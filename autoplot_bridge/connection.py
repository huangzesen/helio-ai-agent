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


# Module-level headless mode flag (set once when JVM starts, read via is_headless())
_headless_mode: bool = True


def is_headless() -> bool:
    """Return True if Autoplot is running in headless mode (no GUI window)."""
    return _headless_mode


def init_autoplot(verbose: bool = False, headless: bool = True):
    """Initialize JVM with Autoplot on classpath. Returns ScriptContext class.

    Args:
        verbose: If True, print debug info.
        headless: If True (default), run without GUI. If False, the Autoplot
                  Swing window will appear when plotting.
    """
    global _headless_mode

    if not jpype.isJVMStarted():
        jar_path = Path(AUTOPLOT_JAR).expanduser().resolve()
        if not jar_path.exists():
            raise FileNotFoundError(f"Autoplot JAR not found: {jar_path}")

        jvm_path = _find_jvm()
        if verbose:
            print(f"  [Autoplot] Starting JVM: {jvm_path}")
            print(f"  [Autoplot] JAR: {jar_path}")
            print(f"  [Autoplot] Headless: {headless}")

        jvm_args = [jvm_path]
        if headless:
            jvm_args.append('-Djava.awt.headless=true')
        jpype.startJVM(*jvm_args, classpath=[str(jar_path)])
        _headless_mode = headless

        if verbose:
            print("  [Autoplot] JVM started.")
    elif verbose:
        print("  [Autoplot] JVM already running.")

    if verbose:
        print("  [Autoplot] Loading ScriptContext class...")
    ScriptContext = jpype.JClass("org.autoplot.ScriptContext")

    # Create the application model. In headless mode, createApplicationModel('')
    # is sufficient. In GUI mode, we also need createGui() to spawn the visible
    # Swing window — createApplicationModel alone creates zero AWT frames.
    if not ScriptContext.isModelInitialized():
        if verbose:
            mode_str = "headless" if headless else "GUI"
            print(f"  [Autoplot] Creating {mode_str} application model...")
        ScriptContext.createApplicationModel('')
        if not headless:
            ScriptContext.createGui()
        if verbose:
            print("  [Autoplot] Application model ready.")

    if verbose:
        print("  [Autoplot] ScriptContext ready.")
    return ScriptContext


def get_script_context(verbose: bool = False, headless: bool = True):
    """Get the Autoplot ScriptContext for issuing commands.

    Args:
        verbose: If True, print debug info.
        headless: If True (default), run without GUI.
    """
    return init_autoplot(verbose=verbose, headless=headless)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Test Autoplot connection")
    _parser.add_argument("--gui", action="store_true", help="Launch with GUI (non-headless)")
    _cli_args = _parser.parse_args()

    print("Testing Autoplot connection...")
    print(f"JAVA_HOME:    {JAVA_HOME}")
    print(f"AUTOPLOT_JAR: {AUTOPLOT_JAR}")
    print(f"JVM path:     {_find_jvm()}")
    print(f"Headless:     {not _cli_args.gui}")
    print()

    try:
        ctx = get_script_context(headless=not _cli_args.gui)

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
