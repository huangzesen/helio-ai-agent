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

### Tools (12 Tool Schemas)

**Dataset Discovery**: `search_datasets`, `list_parameters`, `get_data_availability`

**Autoplot Visualization**: `execute_autoplot` (dispatches to 16 registry methods)

**Data Operations**: `fetch_data`, `list_fetched_data`, `custom_operation`, `describe_data`, `save_data`

**Conversation**: `ask_clarification`

**Routing**: `delegate_to_mission`, `delegate_to_autoplot`

### Supported Spacecraft (8)

PSP, Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, STEREO-A

---

## Completed: Multi-Agent Architecture

- [x] **Mission sub-agents**: Per-spacecraft data specialists with rich prompts, tiered datasets
- [x] **Autoplot sub-agent**: Visualization specialist with method registry (16 operations)
- [x] **OrchestratorAgent**: LLM-driven routing to mission + autoplot sub-agents
- [x] **Method registry**: Structured data describing all Autoplot capabilities (extensible)
- [x] **Render type switching**: series, scatter, spectrogram, fill_to_zero, staircase_plus, digital
- [x] **Color tables**: viridis, plasma, jet, etc. for spectrograms
- [x] **Canvas sizing**: Custom width/height for exports
- [x] **PDF export**: Alongside existing PNG export

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
- [x] Custom color scales (jet, viridis, plasma, etc.) — via `set_color_table`
- [x] Configurable axis labels and titles — via `set_axis_label`, `set_title`
- [x] Log/linear scale toggle — via `toggle_log_scale`
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

1. Create JSON file in `knowledge/missions/` (copy existing template)
2. Run `python scripts/generate_mission_data.py --mission <id>` to populate HAPI metadata
3. The catalog, prompts, and routing table are auto-generated from JSON

### Adding New Autoplot Capabilities

1. Add entry to `autoplot_bridge/registry.py` (method definition)
2. Implement bridge method in `autoplot_bridge/commands.py`
3. Add dispatch handler in `agent/core.py:_dispatch_autoplot_method()`
4. Update `docs/capability-summary.md`
5. No tool schema changes needed — the registry is the single source of truth

### Adding New Non-Autoplot Tools

1. Add schema to `agent/tools.py`
2. Add handler to `agent/core.py:_execute_tool()`
3. Update `docs/capability-summary.md`
4. Add tests

---

## Related Documentation

- `docs/capability-summary.md` — Current feature summary
- `docs/mission-agent-architecture.md` — Multi-agent architecture plan
- `docs/autoplot-scripting-guide.md` — Autoplot ScriptContext API
- `docs/jpype-autoplot-bridge.md` — JPype integration details
- `docs/known-issues.md` — Bug tracker
- `docs/feature-plans/` — Unimplemented feature specs (05-10)
