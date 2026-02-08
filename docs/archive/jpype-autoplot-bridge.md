# JPype + Autoplot Bridge Setup Guide

How to control [Autoplot](https://autoplot.org/) from Python using [JPype](https://jpype.readthedocs.io/). This guide covers setup, usage, and every pitfall encountered during the spike (`spikes/autoplot_connect/`).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Starting the JVM](#starting-the-jvm)
5. [Using the ScriptContext API](#using-the-scriptcontext-api)
6. [Pitfalls & Caveats](#pitfalls--caveats)
7. [Quick Reference](#quick-reference)

---

## Overview

Autoplot is a Java application distributed as a single JAR. JPype starts a JVM inside the CPython process and exposes Java classes as Python objects. This gives us:

- Full CPython ecosystem (pip packages, Gemini SDK, etc.)
- Direct calls into Autoplot's Java API — no IPC, no sockets
- A single process with shared memory

The spike code that proves this works is in `spikes/autoplot_connect/`:

| File | Purpose |
|------|---------|
| `test_jvm.py` | Starts the JVM, prints Java version, shuts down cleanly |
| `test_autoplot.py` | Plots inline data + CDAWeb data, exports PNGs |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.11+ (tested with 3.11.9) |
| **Autoplot JAR** | `autoplot.latest.jar` from <https://autoplot.org/latest/> |
| **Java 8 JRE** | Autoplot's bundled Temurin JRE works: OpenJDK 1.8.0_392 at `C:\Program Files\autoplot\jre\` |
| **`.env` file** | At project root with `AUTOPLOT_JAR` and `JAVA_HOME` set |

Example `.env`:

```ini
AUTOPLOT_JAR=C:\Program Files\autoplot\autoplot.latest.jar
JAVA_HOME=C:\Program Files\autoplot\jre
```

---

## Installation

```bash
pip install jpype1==1.5.0 python-dotenv
```

**The version pin is critical.** JPype 1.6.0 dropped Java 8 support and will fail at startup. See [Pitfall #1](#1-jpype-160-breaks-java-8) for details.

---

## Starting the JVM

### Locating the JVM DLL

Autoplot's bundled JRE ships with a `client` JVM. The code tries `server` first, then falls back to `client`:

```python
from pathlib import Path
import jpype

java_home = os.environ.get("JAVA_HOME")

jvm_path = Path(java_home) / "bin" / "server" / "jvm.dll"
if not jvm_path.exists():
    # Bundled JRE only has client JVM
    jvm_path = Path(java_home) / "bin" / "client" / "jvm.dll"
if not jvm_path.exists():
    # Last resort: let JPype auto-detect
    jvm_path = jpype.getDefaultJVMPath()
```

### Starting the JVM

```python
autoplot_jar = os.environ.get("AUTOPLOT_JAR")
jpype.startJVM(str(jvm_path), classpath=[autoplot_jar])
```

### Verifying the JVM

```python
System = jpype.JClass("java.lang.System")
java_version = str(System.getProperty("java.version"))
print(f"Java version: {java_version}")  # → 1.8.0_392
```

### Complete example (`test_jvm.py`)

```python
import os, sys
from pathlib import Path
from dotenv import load_dotenv
import jpype

load_dotenv()

java_home = os.environ.get("JAVA_HOME")
autoplot_jar = os.environ.get("AUTOPLOT_JAR")

jvm_path = Path(java_home) / "bin" / "server" / "jvm.dll"
if not jvm_path.exists():
    jvm_path = Path(java_home) / "bin" / "client" / "jvm.dll"
if not jvm_path.exists():
    jvm_path = jpype.getDefaultJVMPath()

jpype.startJVM(str(jvm_path), classpath=[autoplot_jar])
System = jpype.JClass("java.lang.System")
print(f"Java version: {System.getProperty('java.version')}")
jpype.shutdownJVM()
print("PASS")
```

---

## Using the ScriptContext API

`org.autoplot.ScriptContext` is a static utility class — you call methods on the class itself.

### Importing ScriptContext

```python
# Option A: jpype.JClass (works immediately after startJVM)
ScriptContext = jpype.JClass("org.autoplot.ScriptContext")

# Option B: jpype.imports (more Pythonic)
import jpype.imports
from org.autoplot import ScriptContext
```

### Plotting inline data (no network)

```python
ScriptContext.plot("vap+inline:ripples(200)")
```

### Plotting CDAWeb data

```python
uri = "vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01+to+2024-01-02"
ScriptContext.plot(uri)
```

### Exporting to PNG

```python
# IMPORTANT: use forward slashes, even on Windows
output_path = str(Path("output/test.png")).replace("\\", "/")
ScriptContext.writeToPng(output_path)
```

### Complete example (`test_autoplot.py`)

```python
import os, sys, time
from pathlib import Path
from dotenv import load_dotenv
import jpype

load_dotenv()
java_home = os.environ.get("JAVA_HOME")
autoplot_jar = os.environ.get("AUTOPLOT_JAR")

jvm_path = Path(java_home) / "bin" / "server" / "jvm.dll"
if not jvm_path.exists():
    jvm_path = Path(java_home) / "bin" / "client" / "jvm.dll"
if not jvm_path.exists():
    jvm_path = jpype.getDefaultJVMPath()

jpype.startJVM(str(jvm_path), classpath=[autoplot_jar])
ScriptContext = jpype.JClass("org.autoplot.ScriptContext")

# Test 1: Inline data (fast, no network)
inline_png = str(Path("output/test_inline.png")).replace("\\", "/")
ScriptContext.plot("vap+inline:ripples(200)")
ScriptContext.writeToPng(inline_png)

# Test 2: CDAWeb data (requires internet, ~4-5s)
cdaweb_png = str(Path("output/test_cdaweb.png")).replace("\\", "/")
uri = "vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01+to+2024-01-02"
ScriptContext.plot(uri)
ScriptContext.writeToPng(cdaweb_png)

print("ALL PASS")
sys.stdout.flush()
os._exit(0)  # Required — see Pitfall #2
```

---

## Pitfalls & Caveats

### 1. JPype 1.6.0 breaks Java 8

**Symptom:**

```
RuntimeError: Can't find org.jpype.jar support library
```

**Cause:** JPype 1.6.0 dropped support for Java 8. The internal support JAR it tries to load requires Java 11+.

**Fix:** Pin to 1.5.x:

```bash
pip install jpype1==1.5.0
```

**Reference:** [jpype-project/jpype#1306](https://github.com/jpype-project/jpype/issues/1306)

---

### 2. JVM shutdown hangs after loading Autoplot

**Symptom:** `jpype.shutdownJVM()` never returns. The process hangs indefinitely.

**Cause:** Autoplot spawns daemon threads (Swing event dispatch, data caching, etc.) that prevent the JVM from terminating cleanly.

**Workaround:** Force-exit the process:

```python
sys.stdout.flush()  # MUST flush before os._exit()
os._exit(0)
```

**Important:** `os._exit()` bypasses Python's normal cleanup (buffer flush, atexit handlers). Always call `sys.stdout.flush()` first, otherwise output may be lost.

**Note:** `test_jvm.py` (which loads the JVM but never imports Autoplot classes) shuts down cleanly with `jpype.shutdownJVM()`. The hang only occurs once Autoplot's Java code has been loaded.

---

### 3. `writeToPng()` needs forward slashes on Windows

**Symptom:** `writeToPng()` fails or writes to the wrong location when given a Windows backslash path like `C:\Users\output\test.png`.

**Fix:** Replace backslashes with forward slashes:

```python
png_path = str(Path("output/test.png")).replace("\\", "/")
ScriptContext.writeToPng(png_path)
```

---

### 4. JVM is a singleton — can only start once per process

**Symptom:**

```
RuntimeError: JVM cannot be started
```

**Cause:** JPype only allows one `startJVM()` call per Python process. If you restart a Jupyter kernel or re-run a script in the same process, it fails.

**Workaround:** Check before starting:

```python
if not jpype.isJVMStarted():
    jpype.startJVM(str(jvm_path), classpath=[autoplot_jar])
```

If you need to restart the JVM, you must restart the Python process.

---

### 5. CDAWeb data fetching is slow

**Symptom:** `ScriptContext.plot()` with a CDAWeb URI takes ~4-5 seconds for one day of data. Longer time ranges take proportionally longer.

**Mitigation for development/testing:** Use inline synthetic data instead:

```python
# ~instant, no network required
ScriptContext.plot("vap+inline:ripples(200)")
```

Only use CDAWeb URIs for integration tests or production.

---

### 6. Stdout buffering with `os._exit()`

**Symptom:** Print output is silently lost when using `os._exit()`.

**Cause:** `os._exit()` terminates the process immediately without flushing Python's I/O buffers.

**Fix:** Always flush before exiting:

```python
print("Done")
sys.stdout.flush()
os._exit(0)
```

---

## Quick Reference

### Common URIs

| URI | Description |
|-----|-------------|
| `vap+inline:ripples(200)` | Synthetic test data (fast, no network) |
| `vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01+to+2024-01-02` | ACE magnetic field magnitude |
| `vap+cdaweb:ds=AC_H0_SWE&id=Np&timerange=2024-01-01+to+2024-01-02` | ACE solar wind proton density |
| `vap+cdaweb:ds=OMNI_HRO_1MIN&id=flow_speed&timerange=2024-01-01+to+2024-01-02` | OMNI solar wind speed |

### ScriptContext method signatures

| Method | Description |
|--------|-------------|
| `ScriptContext.plot(String uri)` | Load and display data |
| `ScriptContext.plot(int plotIndex, String uri)` | Plot to a specific panel |
| `ScriptContext.writeToPng(String filename)` | Export to PNG |
| `ScriptContext.writeToPng(String filename, int width, int height)` | Export with dimensions |
| `ScriptContext.writeToPdf(String filename)` | Export to PDF |
| `ScriptContext.getDocumentModel()` | Get the DOM root object |
| `ScriptContext.setCanvasSize(int w, int h)` | Set canvas size |
| `ScriptContext.reset()` | Reset to default state |
| `ScriptContext.waitUntilIdle()` | Block until all operations complete |

### Startup boilerplate

```python
import os, sys
from pathlib import Path
from dotenv import load_dotenv
import jpype

load_dotenv()

java_home = os.environ["JAVA_HOME"]
autoplot_jar = os.environ["AUTOPLOT_JAR"]

jvm_path = Path(java_home) / "bin" / "server" / "jvm.dll"
if not jvm_path.exists():
    jvm_path = Path(java_home) / "bin" / "client" / "jvm.dll"
if not jvm_path.exists():
    jvm_path = jpype.getDefaultJVMPath()

if not jpype.isJVMStarted():
    jpype.startJVM(str(jvm_path), classpath=[autoplot_jar])

ScriptContext = jpype.JClass("org.autoplot.ScriptContext")
```
