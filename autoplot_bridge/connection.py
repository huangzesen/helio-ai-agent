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


def init_autoplot():
    """Initialize JVM with Autoplot on classpath. Returns ScriptContext class."""
    if not jpype.isJVMStarted():
        jar_path = Path(AUTOPLOT_JAR).expanduser().resolve()
        if not jar_path.exists():
            raise FileNotFoundError(f"Autoplot JAR not found: {jar_path}")

        jvm_path = _find_jvm()
        jpype.startJVM(jvm_path, classpath=[str(jar_path)])

    ScriptContext = jpype.JClass("org.autoplot.ScriptContext")
    return ScriptContext


def get_script_context():
    """Get the Autoplot ScriptContext for issuing commands."""
    return init_autoplot()


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
