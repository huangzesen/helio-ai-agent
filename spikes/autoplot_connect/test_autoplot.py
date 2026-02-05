"""Full Autoplot scripting test: plot data and export to PNG.

Tests two data sources:
  1. Inline synthetic data (fast, no network)
  2. CDAWeb ACE magnetic field data (requires internet, may be slow)

Note: Autoplot spawns background threads that prevent clean JVM shutdown,
so we use os._exit() after verifying success.
"""

import os
import sys
import time
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
    jvm_path = Path(java_home) / "bin" / "client" / "jvm.dll"
if not jvm_path.exists():
    jvm_path = jpype.getDefaultJVMPath()

# Output directory next to this script
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

print(f"JAVA_HOME:    {java_home}")
print(f"AUTOPLOT_JAR: {autoplot_jar}")
print(f"JVM path:     {jvm_path}")
print()


def verify_png(path):
    """Check that a PNG file exists and is non-empty."""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        size_kb = p.stat().st_size / 1024
        print(f"  PNG created: {size_kb:.1f} KB")
        return True
    print(f"  FAIL: PNG not created or is empty at {path}")
    return False


try:
    # Start JVM with Autoplot on classpath
    jpype.startJVM(str(jvm_path), classpath=[autoplot_jar])
    ScriptContext = jpype.JClass("org.autoplot.ScriptContext")

    # --- Test 1: Inline data (no network) ---
    print("Test 1: Inline synthetic data")
    inline_png = str(output_dir / "test_inline.png").replace("\\", "/")
    if Path(inline_png).exists():
        Path(inline_png).unlink()

    t0 = time.time()
    ScriptContext.plot("vap+inline:ripples(200)")
    print(f"  plot() completed in {time.time() - t0:.1f}s")
    ScriptContext.writeToPng(inline_png)
    print(f"  writeToPng() completed in {time.time() - t0:.1f}s")

    if not verify_png(inline_png):
        print("FAIL")
        os._exit(1)
    print("  Test 1 PASS")
    print()

    # --- Test 2: CDAWeb data (requires internet) ---
    print("Test 2: CDAWeb ACE magnetic field data")
    cdaweb_png = str(output_dir / "test_cdaweb.png").replace("\\", "/")
    if Path(cdaweb_png).exists():
        Path(cdaweb_png).unlink()

    uri = "vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01+to+2024-01-02"
    print(f"  URI: {uri}")
    t0 = time.time()
    ScriptContext.plot(uri)
    print(f"  plot() completed in {time.time() - t0:.1f}s")
    ScriptContext.writeToPng(cdaweb_png)
    print(f"  writeToPng() completed in {time.time() - t0:.1f}s")

    if not verify_png(cdaweb_png):
        print("FAIL")
        os._exit(1)
    print("  Test 2 PASS")
    print()

    print("ALL PASS")
    sys.stdout.flush()
    os._exit(0)

except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)
