# Autoplot Scripting Guide

A detailed reference for how Autoplot scripting works and how this project interfaces with it.

---

## Table of Contents

1. [What is Autoplot?](#what-is-autoplot)
2. [Autoplot's Two Scripting Models](#autoplots-two-scripting-models)
3. [The ScriptContext API](#the-scriptcontext-api)
4. [URI System](#uri-system)
5. [The JPype Bridge (Python-to-Java)](#the-jpype-bridge-python-to-java)
6. [DOM Model](#dom-model)
7. [Time Ranges](#time-ranges)
8. [QDataSet: The Universal Data Model](#qdataset-the-universal-data-model)
9. [Render Types](#render-types)
10. [Spectrograms and Custom Spectrums](#spectrograms-and-custom-spectrums)
11. [Data Processing Functions](#data-processing-functions)
12. [Data Flow in This Project](#data-flow-in-this-project)
13. [Headless Operation](#headless-operation)
14. [Troubleshooting](#troubleshooting)

---

## What is Autoplot?

Autoplot is a Java-based interactive browser for scientific data, primarily used in the heliophysics and space science communities. It is developed at the University of Iowa and distributed as a single JAR file.

Key characteristics:

- **Pure Java** — runs anywhere a JVM is available.
- **Multi-format reader** — CDF, NetCDF, HDF5, ASCII tables, FITS, Excel, and more.
- **Data server access** — connects to CDAWeb (NASA/Goddard), HAPI servers, Das2 servers, and others over the network.
- **URI-driven** — every data source is identified by a URI, making operations reproducible and scriptable.
- **Scriptable** — exposes a Jython (Python-on-JVM) scripting console and a Java API (`ScriptContext`) that external programs can call.

Autoplot is distributed as a single JAR (`autoplot.jar`) downloadable from <https://autoplot.org/latest/>. This JAR contains the entire application: GUI, data readers, renderers, and the scripting API.

---

## Autoplot's Two Scripting Models

### Model 1: Internal Jython Scripting

Autoplot embeds a **Jython** interpreter (Python 2.7 syntax running on the JVM). Users can open a scripting console inside Autoplot's GUI and write scripts that call Autoplot's Java classes directly.

Example Jython script run inside Autoplot:

```python
# Plot ACE magnetic field data
plot('vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01')

# Wait for the plot to render, then export
writeToPng('/tmp/ace_mag.png')

# Change time range
from org.das2.datum import DatumRangeUtil
tr = DatumRangeUtil.parseTimeRange('2024-02-01 to 2024-02-07')
dom.timeRange = tr
```

In this model, functions like `plot()` and `writeToPng()` are convenience wrappers imported automatically into the Jython namespace from the `ScriptContext` class.

**Limitation:** Jython is Python 2.7 and cannot use CPython libraries (NumPy, SciPy, etc.). This is why our project uses JPype instead.

### Model 2: External Control via JPype (What This Project Uses)

**JPype** is a Python library that starts a JVM inside the CPython process and provides direct access to Java classes as if they were Python objects. This gives us:

- Full CPython ecosystem (pip packages, Gemini SDK, etc.)
- Direct calls into Autoplot's Java API
- A single process — no IPC, no sockets, no serialization overhead

The trade-off is that JPype requires careful lifecycle management (the JVM can only be started once per process) and Java/Python type conversions are handled automatically but must be understood.

---

## The ScriptContext API

`org.autoplot.ScriptContext` is the primary API for programmatic control of Autoplot. It is a **static utility class** — you call methods on the class itself, not on instances.

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `plot` | `plot(String uri)` | Load data from a URI and display it in the plot canvas. This is the most commonly used method. It fetches data from the remote server, creates a renderer, and displays the result. |
| `plot` | `plot(int plotIndex, String uri)` | Plot to a specific panel (0-indexed). Used for multi-panel layouts. |
| `writeToPng` | `writeToPng(String filename)` | Export the current plot canvas to a PNG image file. The entire visible canvas is rendered. |
| `writeToPng` | `writeToPng(String filename, int width, int height)` | Export with specific pixel dimensions. |
| `writeToPdf` | `writeToPdf(String filename)` | Export the current plot to a PDF file. |
| `getDocumentModel` | `getDocumentModel()` | Returns the `Application` DOM object that represents the full state of the Autoplot session (plots, axes, time ranges, data sources, styles). |
| `setCanvasSize` | `setCanvasSize(int width, int height)` | Set the size of the plot canvas in pixels. |
| `reset` | `reset()` | Reset the application to its default state, clearing all plots and data. |
| `load` | `load(String uri)` | Load a `.vap` file (Autoplot's saved-state format), restoring a previous session. |
| `save` | `save(String filename)` | Save the current session state to a `.vap` file. |
| `waitUntilIdle` | `waitUntilIdle()` | Block until all pending data loads and rendering operations complete. Important for scripting to ensure the plot is ready before exporting. |

### Usage from JPype (Python)

```python
import jpype
import jpype.imports

# Start the JVM with Autoplot on the classpath
jpype.startJVM(classpath=['/path/to/autoplot.jar'])

# Import the Java class as if it were a Python module
from org.autoplot import ScriptContext

# Now call static methods
ScriptContext.plot('vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01')
ScriptContext.waitUntilIdle()
ScriptContext.writeToPng('/tmp/output.png')
```

### Important Behavior

- **`plot()` is asynchronous** — it initiates a data load and returns before the data is fully rendered. If you need to export immediately after plotting, call `waitUntilIdle()` first.
- **State is global** — there is one Autoplot application instance per JVM. Calling `plot()` replaces or adds to the current display. There are no separate "sessions."
- **Thread safety** — Autoplot uses Swing (Java's GUI toolkit) internally. From JPype, calls are marshalled to the appropriate thread, but be aware of potential threading issues in complex scenarios.

---

## URI System

Autoplot uses URIs (Uniform Resource Identifiers) to describe data sources. The URI fully specifies what data to load, making operations reproducible.

### URI Structure

```
scheme:source-specific-part
```

### Common URI Schemes

| Scheme | Data Source | Example |
|--------|-----------|---------|
| `vap+cdaweb:` | NASA CDAWeb | `vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01` |
| `vap+cdf:` | Local CDF file | `vap+cdf:file:///data/ac_h2_mfi.cdf?Magnitude` |
| `vap+hapi:` | HAPI server | `vap+hapi:https://cdaweb.gsfc.nasa.gov/hapi?id=AC_H2_MFI&parameters=Magnitude` |
| `vap+das2server:` | Das2 server | `vap+das2server:https://das2.org/server?dataset=...` |
| `vap+txt:` | ASCII text file | `vap+txt:file:///data/table.csv` |
| `vap+inline:` | Inline data | `vap+inline:1,2,3,4,5` |

### CDAWeb URI Format (Primary Format for This Project)

```
vap+cdaweb:ds={DATASET_ID}&id={PARAMETER_ID}&timerange={TIME_RANGE}
```

**Components:**

- **`ds`** — The CDAWeb dataset identifier. These are standardized IDs maintained by NASA's Space Physics Data Facility (SPDF). Examples:
  - `AC_H2_MFI` — ACE Magnetic Field, 1-hour resolution
  - `AC_H0_SWE` — ACE Solar Wind Electrons, 64-second resolution
  - `OMNI_HRO_1MIN` — OMNI combined solar wind, 1-minute resolution

- **`id`** — The parameter (variable) within the dataset. Each dataset contains multiple parameters. Examples for `AC_H2_MFI`:
  - `Magnitude` — scalar magnetic field strength in nT
  - `BGSEc` — magnetic field vector in GSE coordinates (3-component)
  - `BGSM` — magnetic field vector in GSM coordinates (3-component)

- **`timerange`** — The time window to fetch data for. Format: `YYYY-MM-DD+to+YYYY-MM-DD` (spaces encoded as `+`). Examples:
  - `2024-01-01+to+2024-01-07` — one week
  - `2024-01-01+to+2024-02-01` — one month
  - `2024-01-15T00:00+to+2024-01-15T12:00` — 12 hours with time-of-day

### How URIs Are Constructed in This Project

The `autoplot_bridge/commands.py` module builds URIs programmatically:

```python
uri = f"vap+cdaweb:ds={dataset_id}&id={parameter_id}&timerange={time_range.replace(' ', '+')}"
```

The LLM agent determines the `dataset_id`, `parameter_id`, and `time_range` from the user's natural language input (with help from the knowledge base), and the bridge code formats them into a valid URI.

---

## DOM Model

Autoplot uses a **Document Object Model (DOM)** to represent the entire state of the application. The DOM is a tree of Java objects that can be inspected and modified programmatically.

### DOM Hierarchy

```
Application (root)
├── timeRange                    # Global time range (DatumRange)
├── canvases[]                   # Plot canvases
│   └── Canvas
│       ├── rows[]               # Layout rows
│       ├── columns[]            # Layout columns
│       └── marginRow / marginColumn
├── plots[]                      # Individual plot panels
│   └── Plot
│       ├── title                # Plot title string
│       ├── xaxis                # X-axis configuration
│       │   ├── range            # DatumRange for X axis
│       │   ├── label            # Axis label
│       │   └── log              # Boolean: logarithmic scale
│       ├── yaxis                # Y-axis configuration
│       │   ├── range
│       │   ├── label
│       │   └── log
│       └── zaxis                # Z-axis (for spectrograms)
├── plotElements[]               # Data-to-plot bindings
│   └── PlotElement
│       ├── dataSourceFilter     # Reference to DataSourceFilter
│       ├── plotId               # Which Plot this element renders in
│       ├── renderType           # How to draw: "series", "spectrogram", etc.
│       └── style                # Line color, width, symbol, etc.
│           ├── color
│           ├── lineWidth
│           └── symbolSize
└── dataSourceFilters[]          # Data loading configuration
    └── DataSourceFilter
        ├── uri                  # The data source URI
        └── filters              # Optional processing (smoothing, etc.)
```

### Accessing the DOM from Python

```python
from org.autoplot import ScriptContext

# Get the DOM root
dom = ScriptContext.getDocumentModel()

# Read the current time range
print(dom.getTimeRange())        # Returns a DatumRange object

# Change the time range
from org.das2.datum import DatumRangeUtil
new_range = DatumRangeUtil.parseTimeRange('2024-02-01 to 2024-02-07')
dom.setTimeRange(new_range)

# Access plot properties
plot = dom.getPlots(0)           # First plot panel
print(plot.getTitle())
print(plot.getYaxis().getLabel())

# Access the current data source URI
pele = dom.getPlotElements(0)    # First plot element
dsf = dom.getDataSourceFilters(0)
print(dsf.getUri())

# Modify rendering style
style = pele.getStyle()
from java.awt import Color
style.setColor(Color.RED)
```

### DOM vs ScriptContext

- **`ScriptContext`** provides high-level convenience methods (`plot()`, `writeToPng()`). Use this for most operations.
- **DOM** provides fine-grained control over every aspect of the display. Use this when you need to modify axis labels, colors, scales, panel layouts, or other details that `ScriptContext` doesn't expose directly.

For Phase 1 of this project, `ScriptContext` methods plus `dom.timeRange` are sufficient. The DOM becomes important in later phases for multi-panel layouts, custom styling, and advanced rendering.

---

## Time Ranges

Time ranges are a fundamental concept in Autoplot. They specify the window of data to fetch and display.

### DatumRange Objects

Autoplot represents time ranges internally as `org.das2.datum.DatumRange` objects. These are immutable value objects with a start and end time.

### Parsing Time Range Strings

The `DatumRangeUtil.parseTimeRange()` method accepts flexible string formats:

```python
from org.das2.datum import DatumRangeUtil

# Full range
tr = DatumRangeUtil.parseTimeRange('2024-01-01 to 2024-01-07')

# Single day (implied next day as end)
tr = DatumRangeUtil.parseTimeRange('2024-01-15')

# Month
tr = DatumRangeUtil.parseTimeRange('2024-01')

# Year
tr = DatumRangeUtil.parseTimeRange('2024')

# With times
tr = DatumRangeUtil.parseTimeRange('2024-01-15T06:00 to 2024-01-15T18:00')
```

### Time Range in URIs

In URIs, the time range is encoded with `+` for spaces:

```
timerange=2024-01-01+to+2024-01-07
```

### Setting Time Range Programmatically

There are two ways to change the displayed time range:

```python
# Method 1: Via DOM property (changes display without reloading URI)
from org.das2.datum import DatumRangeUtil
tr = DatumRangeUtil.parseTimeRange('2024-02-01 to 2024-02-07')
dom = ScriptContext.getDocumentModel()
dom.setTimeRange(tr)

# Method 2: Re-plot with new URI (reloads everything)
ScriptContext.plot('vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-02-01+to+2024-02-07')
```

Method 1 is faster for the user experience — Autoplot re-fetches only the data needed for the new time range without rebuilding the entire plot.

---

## QDataSet: The Universal Data Model

All data in Autoplot flows through **QDataSet** — a generic N-dimensional array with rich metadata. The rendering pipeline is completely data-source-agnostic: any QDataSet can be plotted with any render type. Whether data comes from CDAWeb, a local CDF file, or is computed from scratch in Python, it all becomes a QDataSet before rendering.

### Rank (Dimensionality)

QDataSet uses "rank" to describe dimensionality:

| Rank | Shape | Typical Use | Default Render |
|------|-------|-------------|----------------|
| **Rank 1** | `[n]` | Time series (e.g., magnetic field magnitude vs. time) | `series` (line plot) |
| **Rank 2** | `[n, m]` | Spectrograms (e.g., flux vs. time and energy) | `spectrogram` (color plot) |
| **Rank 3** | `[n, m, k]` | Volumetric data (rare in heliophysics) | Sliced before plotting |

### Metadata and Dependencies

QDataSet carries metadata that Autoplot uses for labeling and scaling:

- **Units** — physical units (nT, cm⁻³, eV) attached to each axis
- **Labels** — human-readable axis labels
- **Dependencies** — axis linkage (e.g., a Rank 2 dataset has DEPEND_0 for its time axis and DEPEND_1 for its frequency/energy axis)
- **Valid range** — min/max bounds for filtering fill values

This metadata means Autoplot can automatically label axes, choose appropriate scales, and handle unit conversions without manual configuration.

### Data Sources

QDataSets can originate from:

1. **URIs** — loaded via `ScriptContext.plot(uri)` or `getDataSet(uri)` in Jython
2. **Jython scripts** — computed using built-in math functions
3. **Manual construction** — built from scratch via `DDataSet` (useful from JPype)

### Creating a QDataSet from Scratch (JPype)

```python
from org.das2.qds.util import DataSetBuilder

# Create a Rank 1 dataset with 100 points
builder = DataSetBuilder(1, 100)
for i in range(100):
    builder.putValue(-1, float(i) * 0.1)
    builder.nextRecord()

ds = builder.getDataSet()

# Plot it directly
from org.autoplot import ScriptContext
ScriptContext.plot(ds)
```

For Rank 2 data (e.g., a spectrogram), use `DataSetBuilder(2, numTimes, numFreqs)` and populate with `putValue(column, value)`.

---

## Render Types

Autoplot supports multiple render types that control how data is visualized. The render type can be auto-detected from the data's rank or set explicitly.

### Available Render Types

| Render Type | Description |
|-------------|-------------|
| `series` | Line plot connecting data points in order. Default for Rank 1 (time series) data. |
| `scatter` | Individual data points plotted without connecting lines. Useful for sparse or irregularly sampled data. |
| `stairstep` | Stepped line that shows each sample's value as a horizontal segment, emphasizing the discrete sampling intervals. |
| `fillToZero` | Like `stairstep` but filled from the line down to a reference value (typically zero). Good for bar-chart-style displays. |
| `spectrogram` | 2D color plot using bilinear interpolation between data points. Default for Rank 2 data. Produces smooth color gradients. |
| `nnSpectrogram` | 2D color plot using nearest-neighbor interpolation. Each data point fills a rectangular cell with a single color. Better for data with discrete bins. |
| `hugeScatter` | Optimized scatter renderer for datasets with millions of points. Uses density-based rendering to avoid overplotting. |

### Auto-Detection

When you call `ScriptContext.plot(uri)`, Autoplot inspects the loaded QDataSet and picks an appropriate render type:

- **Rank 1** data → `series`
- **Rank 2** data → `spectrogram`

This means most CDAWeb datasets "just work" — time series plot as line charts and energy/frequency spectral data plots as color spectrograms automatically.

### Setting Render Type Programmatically

To override the auto-detected render type, modify the plot element via the DOM:

```python
from org.autoplot import ScriptContext

dom = ScriptContext.getDocumentModel()
pele = dom.getPlotElements(0)

# Change to nearest-neighbor spectrogram
pele.setRenderType(pele.getRenderType().parse('nnSpectrogram'))

# Or for simple types
pele.setRenderType(pele.getRenderType().parse('scatter'))
```

In Jython scripts, the syntax is simpler:

```python
dom.plotElements[0].renderType = 'spectrogram'
```

---

## Spectrograms and Custom Spectrums

Spectrograms are 2D color plots where the X-axis is typically time, the Y-axis is frequency or energy, and color represents intensity or flux. They are essential for visualizing wave data, particle distributions, and other spectral measurements in heliophysics.

### Plotting Existing Spectrogram Data

Many CDAWeb datasets already contain 2D data that Autoplot renders as spectrograms automatically. You just call `plot()` and Autoplot handles the rest:

```python
from org.autoplot import ScriptContext

# OMNI ion density energy spectrogram (example spectrogram dataset)
ScriptContext.plot('vap+cdaweb:ds=OMNI_HRO_1MIN&id=flow_speed&timerange=2024-01-01+to+2024-01-02')

# MMS FPI ion energy spectrogram
# ScriptContext.plot('vap+cdaweb:ds=MMS1_FPI_FAST_L2_DIS-MOMS&id=mms1_dis_energyspectr_omni_fast&timerange=2024-01-01+to+2024-01-02')
```

Autoplot detects the Rank 2 data structure and renders it as a spectrogram with appropriate color scaling, axis labels, and units — all pulled from the CDF metadata.

### Computing a Spectrum from Time Series

In Jython scripts running inside Autoplot, you can compute a power spectrum from time series data using the built-in `fftPower()` function:

```python
# Jython script (runs inside Autoplot's scripting console)
ds = getDataSet('vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01')
result = fftPower(ds, 512)  # 512-point FFT windows
plot(result)
```

From JPype, you would typically do the spectral computation on the Python side using SciPy, then pass the result to Autoplot for visualization (see "Building Custom Data from Python" below).

### Customizing Spectrogram Appearance

After plotting a spectrogram, you can adjust its appearance via the DOM:

```python
from org.autoplot import ScriptContext

dom = ScriptContext.getDocumentModel()
plot = dom.getPlots(0)

# Z-axis (color axis) controls
zaxis = plot.getZaxis()
zaxis.setLabel('Flux (cm⁻² s⁻¹ sr⁻¹ eV⁻¹)')
zaxis.setLog(True)  # Logarithmic color scale

# Set Z-axis range (color range)
from org.das2.datum import DatumRangeUtil, Units
zrange = DatumRangeUtil.parseTimeRange('1e2 to 1e6')
zaxis.setRange(zrange)

# Y-axis (frequency/energy axis) controls
yaxis = plot.getYaxis()
yaxis.setLog(True)  # Log scale for energy axis

# Color table
dom.getOptions().setColortable(
    dom.getOptions().getColortable().parse('viridis')
)

# Canvas size for high-resolution export
ScriptContext.setCanvasSize(1600, 900)
ScriptContext.writeToPng('spectrogram_output.png')
```

### Interpolation: spectrogram vs. nnSpectrogram

Choose between the two spectrogram render types based on your data:

- **`spectrogram`** (bilinear) — Produces smooth color gradients. Best for continuously sampled data where interpolation between bins is physically meaningful.
- **`nnSpectrogram`** (nearest-neighbor) — Each data bin is a discrete colored rectangle. Best for data with well-defined bins (energy channels, frequency bands) where interpolation would be misleading.

To switch:

```python
pele = dom.getPlotElements(0)
pele.setRenderType(pele.getRenderType().parse('nnSpectrogram'))
```

### Building Custom Data from Python (via JPype)

You can compute data entirely in Python (using NumPy, SciPy, etc.) and pass it to Autoplot for visualization. This is powerful for custom spectral analysis:

```python
import numpy as np
from org.das2.qds import DDataSet

# Compute a synthetic spectrogram: 100 time steps x 64 frequency bins
n_times = 100
n_freqs = 64
data = np.random.random((n_times, n_freqs))

# Create an Autoplot-compatible Rank 2 dataset
ds = DDataSet.createRank2(n_times, n_freqs)
for i in range(n_times):
    for j in range(n_freqs):
        ds.putValue(i, j, data[i, j])

# Plot it — Autoplot auto-detects Rank 2 and renders as spectrogram
from org.autoplot import ScriptContext
ScriptContext.plot(ds)
```

For real-world use, you would also attach axis metadata (time tags, frequency values, units) using `putProperty()`:

```python
from org.das2.qds import DDataSet
from org.das2.qds import DataSetUtil

# After creating ds as above...
# Attach DEPEND_0 (time axis) and DEPEND_1 (frequency axis) as separate Rank 1 datasets
```

### Spectrogram Limitations

- **Requires 2D data** — The dataset must be Rank 2 (time x frequency/energy). Rank 1 time series data will render as a line plot, not a spectrogram.
- **Data gaps** — Large gaps in the time axis can produce interpolation artifacts. Autoplot may stretch colors across gaps. Use `nnSpectrogram` to minimize this.
- **Interleaved sweeps** — Some instruments (e.g., MMS FPI) interleave energy sweeps, producing datasets where alternating records cover different energy ranges. Autoplot may display artifacts. Workaround: filter or resample before plotting.
- **Memory** — Large spectrograms (long time ranges at high resolution) consume significant JVM heap. Use `-Xmx` to increase heap size if needed (see Troubleshooting).

---

## Data Processing Functions

Autoplot provides a rich set of built-in functions available in Jython scripts for data processing. These functions operate on QDataSets and return QDataSets, so they integrate naturally with the plotting pipeline.

### Spectral Analysis

| Function | Description |
|----------|-------------|
| `fft(ds)` | Fast Fourier Transform of a dataset |
| `fftPower(ds, length)` | Power spectral density using sliding FFT windows of the given length |
| `fftFilter(ds, cadence)` | Apply FFT-based filtering with a given cutoff cadence |
| `fftWindow(ds, length)` | Apply a windowing function before FFT |

### Smoothing and Filtering

| Function | Description |
|----------|-------------|
| `smooth(ds, size)` | Boxcar smoothing over `size` points |
| `medianFilter(ds, size)` | Median filter over `size` points (robust to outliers) |
| `boxcar(ds, size)` | Boxcar average (similar to `smooth`) |

### Mathematical Operations

| Function | Description |
|----------|-------------|
| `abs(ds)` | Absolute value |
| `sqrt(ds)` | Square root |
| `log(ds)` | Natural logarithm |
| `log10(ds)` | Base-10 logarithm |
| `exp(ds)` | Exponential |
| `sin(ds)`, `cos(ds)`, `tan(ds)` | Trigonometric functions |
| `magnitude(ds)` | Vector magnitude (for multi-component data) |
| `toDegrees(ds)`, `toRadians(ds)` | Angle unit conversion |

### Statistics and Reduction

| Function | Description |
|----------|-------------|
| `reduceMean(ds, dim)` | Mean along dimension `dim` |
| `reduceMax(ds, dim)` | Maximum along dimension `dim` |
| `reduceMin(ds, dim)` | Minimum along dimension `dim` |
| `reduceMedian(ds, dim)` | Median along dimension `dim` |
| `total(ds, dim)` | Sum along dimension `dim` |
| `extent(ds)` | Returns min and max of the dataset |

### Array Operations

| Function | Description |
|----------|-------------|
| `transpose(ds)` | Transpose a Rank 2 dataset |
| `reform(ds, dims)` | Reshape dataset to new dimensions |
| `reverse(ds)` | Reverse the order of elements |
| `collapse1(ds)` | Average over dimension 1 (reduce Rank 2 to Rank 1) |
| `trim(ds, start, end)` | Extract a subset of records |
| `concatenate(ds1, ds2)` | Join two datasets end-to-end |

### Data Linking and Loading

| Function | Description |
|----------|-------------|
| `link(x, y)` | Create a dataset with explicit X-Y dependency |
| `link(x, y, z)` | Create a Rank 2 dataset with X, Y, Z dependencies |
| `synchronize(ds1, ds2)` | Interpolate ds2 onto ds1's time tags |
| `getDataSet(uri)` | Load data from a URI without plotting it |

### Usage from JPype

These functions are available in Jython scripts running inside Autoplot. From JPype (our project's approach), equivalent operations should be done on the Python side:

- **Spectral analysis** — Use `scipy.fft` or `numpy.fft` instead of `fftPower`
- **Smoothing** — Use `scipy.signal` or `pandas.Series.rolling`
- **Math/statistics** — Use `numpy` equivalents
- **Data loading** — Use `getDataSet(uri)` via JPype to pull data into Java, then convert to Python arrays if needed

The typical workflow is: load data via Autoplot (`getDataSet`), convert to NumPy arrays for processing, then create a new QDataSet from the result for visualization.

---

## Data Flow in This Project

End-to-end data flow for a user request:

```
┌──────────────────────────────────────────────────────────────┐
│  User: "Show me ACE magnetic field data for last week"       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  main.py: Conversation Loop                                  │
│  Receives user input, passes to agent.process_message()      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  agent/core.py: AutoplotAgent                                │
│  Sends user message + tool schemas to Gemini API             │
│  Gemini decides to call "plot_data" tool with:               │
│    dataset_id="AC_H2_MFI"                                    │
│    parameter_id="Magnitude"                                  │
│    time_range="2024-01-28 to 2024-02-04"                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  agent/core.py: _execute_tool()                              │
│  Routes the tool call to autoplot.plot_cdaweb()              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  autoplot_bridge/commands.py: AutoplotCommands.plot_cdaweb() │
│  Builds URI:                                                 │
│    "vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=..."     │
│  Calls ScriptContext.plot(uri)                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  JPype Bridge                                                │
│  Marshals the Python string into a Java String               │
│  Calls the Java method on the JVM                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Autoplot (Java, in-process JVM)                             │
│  1. Parses the URI                                           │
│  2. Connects to CDAWeb via HTTP                              │
│  3. Downloads CDF data for the time range                    │
│  4. Reads the requested parameter                            │
│  5. Creates a time-series renderer                           │
│  6. Renders the plot on a canvas (Swing/AWT)                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Result flows back up:                                       │
│  commands.py → returns status dict                           │
│  core.py → sends result to Gemini as function_response       │
│  Gemini → generates natural language confirmation            │
│  main.py → prints: "I've plotted ACE magnetic field..."      │
└──────────────────────────────────────────────────────────────┘
```

### Singleton Pattern

The `AutoplotCommands` class uses a singleton pattern because:

1. **The JVM can only be started once** per Python process (JPype limitation). Restarting requires restarting the process.
2. **Plot state is global** within the JVM — there is one Autoplot application instance.
3. **Follow-up commands** (zoom, export, etc.) must operate on the same plot, so the bridge must maintain state references (`_current_uri`, `_current_time_range`).

---

## Headless Operation

Autoplot uses Java Swing for rendering, which normally requires a display (X11 on Linux, or a native GUI on Windows/macOS).

### Running Without a Display (Linux Servers)

Use **Xvfb** (X Virtual Framebuffer) to provide a fake display:

```bash
# Start a virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Now run the agent — Autoplot renders to the virtual display
python main.py
```

### Running on Windows

On Windows, Autoplot uses the native Windows GUI. No special setup is needed — the plot window will appear on screen. If running as a service without a desktop session, you would need to explore Java's headless mode or alternative rendering approaches.

### Java Headless Mode

For export-only workflows where you don't need to see the plot interactively:

```python
# Before starting the JVM
jpype.startJVM(
    classpath=[str(jar_path)],
    '-Djava.awt.headless=true'
)
```

Note: headless mode may not be fully compatible with all Autoplot rendering features. Test thoroughly.

---

## Troubleshooting

### JVM Won't Start

```
RuntimeError: JVM cannot be started
```

- **Java not installed:** Run `java -version` to verify. Autoplot requires Java 8+.
- **Wrong JAR path:** Verify `AUTOPLOT_JAR` in `.env` points to the actual file. Use an absolute path.
- **JVM already started:** JPype only allows one `startJVM()` per process. If you get this error in a notebook, restart the kernel.

### Import Errors for Autoplot Classes

```
ModuleNotFoundError: No module named 'org.autoplot'
```

- The JVM hasn't been started yet, or the JAR wasn't on the classpath.
- Make sure `init_autoplot()` is called before any `from org.autoplot import ...` statements.

### Plot Doesn't Appear

- On Linux: Ensure `DISPLAY` is set (`echo $DISPLAY`). Use Xvfb if no physical display.
- On Windows: The Autoplot window may be behind other windows. Check the taskbar.
- Data loading can take time for large time ranges or slow network connections. Call `waitUntilIdle()` to block until ready.

### CDAWeb Data Not Found

```
Error: unable to resolve dataset
```

- Double-check the dataset ID (case-sensitive). Use the full ID like `AC_H2_MFI`, not just `ACE`.
- Verify the parameter ID exists in that dataset. Use `list_parameters()` from the knowledge base.
- CDAWeb may be temporarily unavailable. Check <https://cdaweb.gsfc.nasa.gov/> for status.

### Memory Issues with Large Datasets

- Autoplot loads data into JVM heap memory. For large time ranges at high resolution, you may run out of memory.
- Increase JVM heap size when starting:
  ```python
  jpype.startJVM(classpath=[str(jar_path)], '-Xmx2g')  # 2 GB heap
  ```
- Prefer lower-resolution datasets or shorter time ranges for initial exploration.

### JPype Type Conversion Issues

- JPype automatically converts between Python and Java types for common cases (strings, numbers, booleans).
- For Java arrays, use `jpype.JArray` or access elements with standard indexing.
- Java `null` maps to Python `None`.
- If a method expects a specific Java type, you may need explicit casting:
  ```python
  from jpype import JString
  ScriptContext.plot(JString(uri))
  ```

---

## Quick Reference

### Minimal Plotting Script

```python
import jpype
import jpype.imports

# 1. Start JVM
jpype.startJVM(classpath=['autoplot.jar'])

# 2. Import Autoplot
from org.autoplot import ScriptContext as sc

# 3. Plot data
sc.plot('vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01+to+2024-01-07')

# 4. Wait for render
sc.waitUntilIdle()

# 5. Export
sc.writeToPng('output.png')

# 6. Shutdown
jpype.shutdownJVM()
```

### Common Operations Cheat Sheet

```python
# Plot CDAWeb data
sc.plot('vap+cdaweb:ds=DATASET&id=PARAM&timerange=START+to+END')

# Change time range (fast, reuses current plot setup)
from org.das2.datum import DatumRangeUtil
dom = sc.getDocumentModel()
dom.setTimeRange(DatumRangeUtil.parseTimeRange('2024-02-01 to 2024-02-07'))

# Export to image
sc.writeToPng('/path/to/output.png')
sc.writeToPdf('/path/to/output.pdf')

# Set canvas size
sc.setCanvasSize(1200, 800)

# Reset everything
sc.reset()

# Wait for all operations to complete
sc.waitUntilIdle()

# Access current URI
dsf = sc.getDocumentModel().getDataSourceFilters(0)
print(dsf.getUri())
```
