# helio-ai-agent Roadmap

Future development plan for the helio-ai-agent project.

**Last updated**: February 2026

---

## Current Status

### Implemented

| Component | Status | Details |
|-----------|--------|---------|
| Agent Core | Done | Gemini 2.5-Flash with function calling |
| Autoplot Bridge | Done | JPype connection, plot/export/time-range, overplot with color mgmt |
| Dataset Catalog | Done | 8 spacecraft with keyword search |
| HAPI Client | Done | CDAWeb parameter metadata fetching (cached) |
| Data Pipeline | Done | fetch -> store -> custom_operation -> plot (pandas-backed) |
| Custom Operations | Done | LLM-generated pandas/numpy code, AST-validated sandbox |
| Time Parsing | Done | Relative, absolute, date ranges, sub-day precision |
| Multi-step Planning | Done | Regex complexity detection + Gemini task decomposition |
| Logging | Done | Daily rotation to `~/.helio-agent/logs/` |
| Cross-platform | Done | Windows + macOS |
| Token Tracking | Done | Per-session usage statistics |

### Tools (14 Total)

**Dataset Discovery**: `search_datasets`, `list_parameters`, `get_data_availability`

**Visualization**: `plot_data`, `change_time_range`, `export_plot`, `get_plot_info`

**Data Operations**: `fetch_data`, `list_fetched_data`, `custom_operation`, `plot_computed_data`, `describe_data`, `save_data`

**Conversation**: `ask_clarification`

### Supported Spacecraft (8)

PSP, Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, STEREO-A

---

## Next: Mission-Specific Agent Architecture

See `docs/mission-agent-architecture.md` for the full 3-phase plan.

- **Phase 1** (foundation): Rich mission profiles in catalog, dynamic prompt generation from catalog, eliminate hardcoded prompt duplication
- **Phase 2**: Mission sub-agents with specialized prompts, task dispatch by mission
- **Phase 3**: Parallel execution with dependency tracking

---

## Data Source Expansion

### New Spacecraft
- [ ] Cluster — Multi-spacecraft magnetospheric mission
- [ ] Voyager 1/2 — Heliospheric boundary
- [ ] STEREO-B — Off-Sun-Earth-line (if data available)
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

## Visualization Enhancements

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

## Advanced Analysis

### Spectral Analysis
- [ ] FFT / Power spectral density
- [ ] Wavelet transform
- [ ] Dynamic spectra (spectrogram from timeseries)
- [ ] Coherence between signals

### Statistical Operations
- [x] Min/max/mean/std over intervals — via `describe_data`
- [x] Percentiles and distributions — via `describe_data`
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

## Production Readiness

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
- [x] Session recovery after crashes — via TaskStore persistence
- [ ] Input validation and sanitization

### Cost Optimization
- [ ] Model routing: small model for simple commands
- [ ] Response caching for repeated queries
- [ ] Token usage budgets and alerts
- [ ] Batch API for bulk operations

---

## User Experience

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
- [x] Script mode (non-interactive) — `python main.py "request"`
- [ ] Batch job files (YAML/JSON)
- [ ] Parallel execution for multiple time ranges
- [ ] Scheduled recurring analyses

### Export Formats
- [ ] PDF reports with plots and metadata
- [x] CSV/ASCII data export — via `save_data`
- [ ] NetCDF/CDF output
- [ ] Shareable plot URLs

---

## Contributing

### Adding New Spacecraft

1. Add entry to `knowledge/catalog.py`
2. Add test case in `tests/test_catalog.py`
3. Verify HAPI parameters load correctly
4. Update system prompt in `agent/prompts.py`

### Adding New Tools

1. Add schema to `agent/tools.py`
2. Add handler to `agent/core.py` `_execute_tool()`
3. Update system prompt in `agent/prompts.py`
4. Update `docs/capability-summary.md`
5. Add tests

---

## Related Documentation

- `docs/capability-summary.md` — Current feature summary
- `docs/mission-agent-architecture.md` — Multi-agent architecture plan
- `docs/autoplot-scripting-guide.md` — Autoplot ScriptContext API
- `docs/jpype-autoplot-bridge.md` — JPype integration details
- `docs/known-issues.md` — Bug tracker
- `docs/feature-plans/` — Unimplemented feature specs (05-10)
