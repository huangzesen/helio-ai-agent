"""Smoke test: verify JPype can start the JVM using the Autoplot-bundled JRE."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import jpype

# Load .env from project root (two levels up from this file)
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

java_home = os.environ.get("JAVA_HOME")
autoplot_jar = os.environ.get("AUTOPLOT_JAR")

if not java_home:
    print("FAIL: JAVA_HOME not set in .env")
    sys.exit(1)
if not autoplot_jar:
    print("FAIL: AUTOPLOT_JAR not set in .env")
    sys.exit(1)

jvm_path = Path(java_home) / "bin" / "server" / "jvm.dll"
if not jvm_path.exists():
    # Try client JVM as fallback
    jvm_path = Path(java_home) / "bin" / "client" / "jvm.dll"
if not jvm_path.exists():
    # Let JPype find it via JAVA_HOME
    jvm_path = jpype.getDefaultJVMPath()

print(f"JAVA_HOME:    {java_home}")
print(f"AUTOPLOT_JAR: {autoplot_jar}")
print(f"JVM path:     {jvm_path}")

try:
    jpype.startJVM(str(jvm_path), classpath=[autoplot_jar])
    System = jpype.JClass("java.lang.System")
    java_version = str(System.getProperty("java.version"))
    print(f"Java version: {java_version}")
    jpype.shutdownJVM()
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
