# helio-ai-agent Roadmap

Future development plan for the helio-ai-agent project.

**Last updated**: February 2026

---

## Current Status

### Implemented

| Component | Status | Details |
|-----------|--------|---------|
| Agent Core | Done | Gemini 3 Pro/Flash Preview with function calling |
| Plotly Renderer | Done | Interactive Plotly figures, multi-panel, WebGL, PNG/PDF export via kaleido |
| Custom Visualization | Done | LLM-generated Plotly code sandbox for any customization |
| Dataset Catalog | Done | 52 spacecraft (all auto-generated from CDAWeb) with keyword search, per-mission JSON knowledge |
| HAPI Client | Done | CDAWeb parameter metadata fetching (3-tier cache: memory/local/network) |
| Data Pipeline | Done | fetch -> store -> custom_operation -> plot (pandas-backed) |
| Custom Operations | Done | LLM-generated pandas/numpy code, AST-validated sandbox |
| Time Parsing | Done | Relative, absolute, date ranges, sub-day precision |
| Multi-step Planning | Done | PlannerAgent with plan-execute-replan loop (up to 5 rounds) |
| Logging | Done | Daily rotation to `~/.helio-agent/logs/` |
| Cross-platform | Done | Windows + macOS |
| Token Tracking | Done | Per-session usage statistics (includes all sub-agents) |
| Gradio Web UI | Done | Browser-based chat with inline Plotly plots, data sidebar |
| Google Search | Done | Web search grounding via isolated Gemini API call |
| Data Extraction | Done | Text-to-DataFrame via `store_dataframe` + `read_document` |
| Document Reading | Done | PDF and images → text extraction via Gemini vision |
| Multimodal Upload | Done | File upload in Gradio (drag-and-drop, 18+ file types) |
| Mission Data Refresh | Done | Interactive startup menu + `--refresh`/`--refresh-full`/`--refresh-all` CLI flags |
| Session Persistence | Done | Auto-save every turn, `--continue`/`--session` CLI, Gradio sidebar |
| Long-term Memory | Done | Cross-session memory (preferences, summaries, pitfalls) in `~/.helio-agent/memory.json` |
| Passive MemoryAgent | Done | Auto-triggered session analysis, pitfall extraction, error pattern reports |
| Model Fallback | Done | Auto-switch to `GEMINI_FALLBACK_MODEL` on 429 quota errors (session-level) |
| Plot Self-Review | Done | Structured review metadata on every `plot_data` call for LLM self-assessment |

### Tools (26 Tool Schemas)

**Dataset Discovery**: `search_datasets`, `browse_datasets`, `list_parameters`, `get_data_availability`, `get_dataset_docs`, `search_full_catalog`, `google_search`

**Visualization**: `plot_data`, `style_plot`, `manage_plot` (3 declarative tools — no free-form code generation)

**Data Operations**: `fetch_data`, `list_fetched_data`, `custom_operation`, `compute_spectrogram`, `describe_data`, `preview_data`, `save_data`

**Data Extraction**: `store_dataframe`

**Document Reading**: `read_document`

**Memory**: `recall_memories`

**Conversation**: `ask_clarification`

**Routing**: `delegate_to_mission`, `delegate_to_data_ops`, `delegate_to_visualization`, `delegate_to_data_extraction`, `request_planning`

### Supported Spacecraft (52 missions + full CDAWeb catalog)

52 missions total (all auto-generated from CDAWeb): PSP, Solar Orbiter, ACE, OMNI, Wind, DSCOVR, MMS, STEREO-A, Cluster, Voyager, THEMIS, GOES, Van Allen Probes, and more. All 2000+ CDAWeb datasets searchable via `search_full_catalog`.

---

## Completed: Multi-Agent Architecture

- [x] **5-agent architecture**: Orchestrator + Mission + DataOps + DataExtraction + Visualization sub-agents
- [x] **Mission sub-agents**: Per-spacecraft data specialists with rich prompts
- [x] **DataOps sub-agent**: Data transformation specialist (compute, describe, save)
- [x] **DataExtraction sub-agent**: Text-to-DataFrame specialist (store_dataframe, read_document)
- [x] **Visualization sub-agent**: Plotly rendering via 5 core methods + custom Plotly sandbox
- [x] **Method registry**: Structured data describing 5 core visualization operations
- [x] **Custom visualization**: Free-form Plotly code for titles, labels, scales, render types, annotations, etc.
- [x] **Canvas sizing**: Custom width/height for exports
- [x] **PNG/PDF export**: Via kaleido static image export
- [x] **PlannerAgent**: Chat-based plan-execute-replan loop (up to 5 rounds) for complex requests
- [x] **Document reading**: `read_document` tool (PDF, images via Gemini vision)
- [x] **store_dataframe tool**: Create DataFrames from text (event lists, catalogs, search results)

## Completed: Plotly Migration

- [x] **Plotly renderer**: Replaced Java Autoplot bridge with pure-Python Plotly
- [x] **Gradio web UI**: Browser-based chat with inline interactive plots, multimodal file upload
- [x] **Google Search grounding**: Web search via custom function tool
- [x] **custom_visualization tool**: Replaced 10 thin wrapper methods with single Plotly sandbox
- [x] **JPype/Java removal**: All JVM dependencies eliminated
- [x] **Mission data startup**: Interactive refresh menu + CLI flags (`--refresh`, `--refresh-full`, `--refresh-all`)
- [x] **Browse & Fetch sidebar**: Mission → dataset → parameter cascade dropdowns in Gradio UI

---

## Data Source Expansion

### New Spacecraft
- [x] Full CDAWeb catalog access — all 2000+ datasets searchable via `search_full_catalog`
- [x] Shared prefix map — 40+ mission groups mapped via `knowledge/mission_prefixes.py`
- [x] Auto-generated mission skeletons — `scripts/generate_mission_data.py --create-new`
- [x] Cluster, Voyager, STEREO-B, Ulysses, THEMIS, GOES, Van Allen Probes, and more

### New Data Sources
- [ ] Local CDF file loading (no network required)
- [ ] Additional HAPI servers (HelioViewer, AMDA, etc.)
- [ ] SPDF Web Services API
- [ ] Real-time data feeds

### Catalog Improvements
- [x] Full CDAWeb catalog access (2000+ datasets via `search_full_catalog`)
- [ ] Fuzzy matching for spacecraft/instrument names
- [ ] Dataset recommendations based on time range
- [ ] Automatic parameter suggestions

---

## Visualization Enhancements

### Layout
- [x] Multi-panel stack plots — via `plot_stored_data` with panel index
- [ ] Synchronized time axes across panels
- [ ] Panel add/remove/reorder

### Plot Types
- [ ] Spectrograms for wave/particle data
- [ ] Orbit plots (3D trajectory)
- [ ] Polar plots for directional data
- [ ] Histogram/distribution plots

### Styling
- [x] Configurable axis labels and titles — via `custom_visualization`
- [x] Log/linear scale toggle — via `custom_visualization`
- [x] Canvas sizing — via `custom_visualization`
- [x] Per-trace line color/styling — via `custom_visualization`
- [ ] Grid and tick customization

### Annotations
- [x] Event markers, shaded regions, text annotations — via `custom_visualization` (Plotly code)
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
- [ ] Cross-platform matrix (Windows, macOS, Linux)

### Deployment
- [ ] Docker container with all dependencies
- [ ] pyproject.toml with proper metadata
- [ ] PyPI package publication
- [ ] Conda-forge recipe

### Reliability
- [x] Automatic model fallback — switch to `GEMINI_FALLBACK_MODEL` on 429 quota errors (session-level)
- [ ] Retry logic for network failures
- [x] Graceful degradation when services unavailable — auto-clamp time ranges to dataset availability, consecutive error tracking breaks agent loops
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
- [x] Web UI (Gradio) — `gradio_app.py` with inline Plotly plots, live progress streaming on by default
- [ ] Jupyter notebook integration
- [ ] VS Code extension
- [ ] REST API for programmatic access

### Session Management
- [x] Save/restore conversation state — auto-save every turn to `~/.helio-agent/sessions/`
- [x] Named sessions with history — `--continue` / `--session ID` CLI flags
- [x] Gradio sessions sidebar — Load / New / Delete buttons
- [x] Long-term memory — cross-session preferences, summaries, pitfalls in `~/.helio-agent/memory.json`
- [x] Passive MemoryAgent — auto-triggered session analysis with pitfall extraction
- [x] Empty session auto-cleanup — removes abandoned sessions on startup
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

### Adding New Visualization Capabilities

Most Plotly customizations already work via `custom_visualization` — no code changes needed.

For **new core methods** (that need special logic like `plot_stored_data`):
1. Add entry to `rendering/registry.py` (method definition)
2. Implement renderer method in `rendering/plotly_renderer.py`
3. Add dispatch handler in `agent/core.py:_dispatch_viz_method()`
4. Update `docs/capability-summary.md`

### Adding New Non-Visualization Tools

1. Add schema to `agent/tools.py`
2. Add handler to `agent/core.py:_execute_tool()`
3. Update `docs/capability-summary.md`
4. Add tests

---

## Related Documentation

- `docs/capability-summary.md` — Current feature summary
- `docs/known-issues.md` — Bug tracker
- `docs/feature-plans/` — Active and archived feature specs
- `docs/redundancy-report.md` — Code redundancy analysis
