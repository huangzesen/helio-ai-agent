# helio-ai-agent Roadmap

Future development plan for the helio-ai-agent project.

**Last updated**: February 2026

---

## Current Status

### Implemented (Phase 1 Complete)

| Component | Status | Details |
|-----------|--------|---------|
| Agent Core | ✅ | Gemini 2.5-Flash with 15 function-calling tools |
| Autoplot Bridge | ✅ | JPype connection, headless mode, plot/export/time-range |
| Dataset Catalog | ✅ | PSP, Solar Orbiter, ACE, OMNI with keyword search |
| HAPI Client | ✅ | CDAWeb parameter metadata fetching |
| Data Pipeline | ✅ | fetch → store → compute → plot |
| Operations | ✅ | magnitude, arithmetic, running average, resample, delta |
| Time Parsing | ✅ | Relative, absolute, date ranges, sub-day precision |
| Cross-platform | ✅ | Windows + macOS (Python 3.11 required on Mac) |
| Token Tracking | ✅ | Per-session usage statistics |

### Tools (12 Total)

**Dataset Discovery**: `search_datasets`, `list_parameters`, `get_data_availability`

**Visualization**: `plot_data`, `change_time_range`, `export_plot`, `get_plot_info`

**Data Operations**: `fetch_data`, `list_fetched_data`, `custom_operation`, `plot_computed_data`

**Conversation**: `ask_clarification`

---

## Phase 2: Data Source Expansion

**Goal**: Support more spacecraft and data sources.

### New Spacecraft
- [ ] WIND — Solar wind monitor at L1
- [ ] DSCOVR — Real-time solar wind
- [ ] Cluster — Multi-spacecraft magnetospheric mission
- [ ] MMS — Magnetospheric Multiscale
- [ ] Voyager 1/2 — Heliospheric boundary
- [ ] STEREO A/B — Solar imaging and in-situ
- [ ] Ulysses — High-latitude heliosphere

### New Data Sources
- [ ] Local CDF file loading (no network required)
- [ ] Additional HAPI servers (HelioViewer, AMDA, etc.)
- [ ] SPDF Web Services API
- [ ] Real-time data feeds

### Catalog Improvements
- [ ] Fuzzy matching for spacecraft/instrument names
- [ ] Dataset recommendations based on time range
- [ ] Automatic parameter suggestions

---

## Phase 3: Visualization Enhancements

**Goal**: Rich scientific visualizations matching Autoplot's full capabilities.

### Layout
- [ ] Multi-panel stack plots
- [ ] Synchronized time axes
- [ ] Panel add/remove/reorder

### Plot Types
- [ ] Spectrograms for wave/particle data
- [ ] Orbit plots (3D trajectory)
- [ ] Polar plots for directional data
- [ ] Histogram/distribution plots

### Styling
- [ ] Custom color scales (jet, viridis, plasma, etc.)
- [ ] Configurable axis labels and titles
- [ ] Log/linear scale toggle
- [ ] Grid and tick customization

### Annotations
- [ ] Event markers (vertical lines with labels)
- [ ] Shaded regions (e.g., storm intervals)
- [ ] Text annotations
- [ ] Legend customization

---

## Phase 4: Advanced Analysis

**Goal**: In-memory Python-side analysis for derived science products.

### Spectral Analysis
- [ ] FFT / Power spectral density
- [ ] Wavelet transform
- [ ] Dynamic spectra (spectrogram from timeseries)
- [ ] Coherence between signals

### Statistical Operations
- [ ] Min/max/mean/std over intervals
- [ ] Percentiles and distributions
- [ ] Correlation coefficients
- [ ] Trend fitting (linear, polynomial)

### Event Detection
- [ ] Threshold crossings
- [ ] Peak/valley detection
- [ ] Discontinuity identification
- [ ] Automatic interval selection

### Multi-dataset Analysis
- [ ] Cross-correlation with lag
- [ ] Interpolation to common time grid
- [ ] Dataset comparison reports

---

## Phase 5: Production Readiness

**Goal**: Reliable, maintainable, deployable system.

### Testing & CI
- [ ] GitHub Actions CI/CD pipeline
- [ ] Integration tests with mock LLM
- [ ] Autoplot bridge tests (headless)
- [ ] Cross-platform matrix (Windows, macOS, Linux)

### Deployment
- [ ] Docker container with all dependencies
- [ ] pyproject.toml with proper metadata
- [ ] PyPI package publication
- [ ] Conda-forge recipe

### Reliability
- [ ] Retry logic for network failures
- [ ] Graceful degradation when services unavailable
- [ ] Session recovery after crashes
- [ ] Input validation and sanitization

### Cost Optimization
- [ ] Model routing: small model for simple commands
- [ ] Response caching for repeated queries
- [ ] Token usage budgets and alerts
- [ ] Batch API for bulk operations

---

## Phase 6: User Experience

**Goal**: Make the tool accessible to more users.

### Interfaces
- [ ] Web UI (Streamlit or Gradio)
- [ ] Jupyter notebook integration
- [ ] VS Code extension
- [ ] REST API for programmatic access

### Session Management
- [ ] Save/restore conversation state
- [ ] Named sessions with history
- [ ] Export session as script
- [ ] Replay previous analyses

### Batch Processing
- [ ] Script mode (non-interactive)
- [ ] Batch job files (YAML/JSON)
- [ ] Parallel execution for multiple time ranges
- [ ] Scheduled recurring analyses

### Export Formats
- [ ] PDF reports with plots and metadata
- [ ] CSV/ASCII data export
- [ ] NetCDF/CDF output
- [ ] Shareable plot URLs

---

## Contributing

### Adding New Spacecraft

1. Add entry to `knowledge/catalog.py`
2. Add test case in `tests/test_catalog.py`
3. Verify HAPI parameters load correctly
4. Update system prompt if needed

### Adding New Tools

1. Add schema to `agent/tools.py`
2. Add handler to `agent/core.py` `_execute_tool()`
3. Update system prompt in `agent/prompts.py`
4. Update `docs/capability-summary.md`
5. Add tests

### Adding New Operations

1. Add pure function to `data_ops/operations.py`
2. Add tool schema and handler
3. Add tests to `tests/test_operations.py`

---

## Related Documentation

- `docs/capability-summary.md` — Current feature summary
- `docs/autoplot-agent-spec.md` — Original Phase 1 specification
- `docs/autoplot-scripting-guide.md` — Autoplot ScriptContext API
- `docs/jpype-autoplot-bridge.md` — JPype integration details
- `spikes/mac_compatibility/SESSION_SUMMARY.md` — macOS setup guide
