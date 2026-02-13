# Reproducible JSON Specs for DataOps and Mission Agent Workflows

*Research conducted Feb 2026. Analyzed 8+ industry frameworks for workflow serialization patterns.*

## Context

The helio-agent has a Pipeline feature (Phases 1-3 complete) that serializes visualization workflows as JSON — capturing `fetch_data`, `custom_operation`, and `render_spec` steps with dependency graphs, variable substitution, and deterministic replay. This document explores extending the approach so that **DataOps** (data fetching + transformations) and **Mission Agent** (discovery + fetching) workflows are also expressed as reproducible JSON.

## Industry Patterns Summary

| Framework | Key Pattern | Relevance |
|-----------|------------|-----------|
| **dbt** | `manifest.json` — complete DAG as a single artifact; `sources.yml` separates data catalog from transformations | Closest analog to our "data manifest as a view" |
| **CWL** | Explicit `inputs`/`outputs` per step; DAG inferred from connections | Best pattern for data lineage tracking |
| **Kedro** | Separates catalog (data sources) / pipeline (nodes) / parameters (config) | Clean but creates coordination overhead |
| **DVC** | Single `dvc.yaml` with stages, deps, outs — all in one file | Validates "one file, tagged layers" approach |
| **MLflow** | Run-based experiment tracking with parameters, metrics, artifacts | Good for discovery context (metadata) |
| **Dagster** | Assets-first — models data as first-class citizens with lineage | Validates explicit I/O declarations |
| **Airflow** | Serialized DAG persistence (AIP-24); task dependencies via operators | Enterprise-grade but heavy-weight |
| **Snakemake/Nextflow** | Rule/process-based with implicit DAGs from file I/O patterns | Bioinformatics-focused, container-aware |

## Detailed Framework Analysis

### dbt (data build tool)

- **Sources**: `sources.yml` defines upstream data with freshness checks (database, schema, table name)
- **Transforms**: SQL SELECT statements with materialization types (view, table, incremental, ephemeral)
- **References**: `{{ source(...) }}` and `{{ ref(...) }}` for explicit dependencies
- **Artifacts**: `manifest.json` (complete DAG + node definitions), `run_results.json` (execution outcomes), `catalog.json` (table/column structure)
- **Strengths**: Complete DAG visibility, excellent lineage, artifact-driven (manifest-as-contract)
- **Weaknesses**: SQL-only, data warehouse-centric

### CWL (Common Workflow Language)

```yaml
cwlVersion: v1.2
class: Workflow
inputs:
  data_file: File
  threshold: float
outputs:
  result: File
steps:
  process:
    run: tool.cwl
    in:
      input: data_file
      param: threshold
    out: [processed]
```

- **Sources**: Explicit `inputs` section with type definitions; URIs or file paths
- **Transforms**: `CommandLineTool` elements wrap CLI executables; `Workflow` composes them
- **DAG**: Inferred from input/output connections between steps
- **Strengths**: Platform-agnostic, standardized, container-aware, fully reproducible
- **Weaknesses**: Verbose, limited to CLI tools, not Python-native

### Kedro

```yaml
# catalog.yml
raw_data:
  type: pandas.CSVDataSet
  filepath: data/raw/data.csv
  layer: raw

processed_data:
  type: pandas.ParquetDataSet
  filepath: data/processed/data.parquet
  layer: processed
```

- **Sources**: `catalog.yml` defines all data sources/sinks with type, path, credentials
- **Transforms**: Python functions organized as nodes with explicit inputs/outputs
- **Config**: `parameters.yml` for runtime configuration (OmegaConfigLoader)
- **Strengths**: Clean separation, Python-native, modular
- **Weaknesses**: No built-in execution tracking, lighter-weight than Airflow

### DVC (Data Version Control)

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - raw_data.csv
    outs:
      - prepared_data.csv
  train:
    cmd: python train.py
    deps:
      - prepared_data.csv
    params:
      - lr
      - epochs
    outs:
      - model.pkl
    metrics:
      - metrics.json
```

- **Sources**: `dvc.yaml` stages with explicit deps and outs
- **Versioning**: `.dvc` files containing hash information; remote storage for artifacts
- **Strengths**: Lightweight, Git-friendly, clear data/model/code versioning separation
- **Weaknesses**: Not full orchestration, no distributed execution

### MLflow

- **Tracking**: Experiments → runs → parameters + metrics + artifacts + tags
- **Storage**: JSON metadata in local/remote backend; artifacts stored separately
- **Registry**: Model versioning and stage management (staging → production)
- **Strengths**: Lightweight experiment tracking, model lifecycle management
- **Weaknesses**: Not a full pipeline orchestrator

### Dagster

- **Assets-first**: Models data artifacts with schemas and type safety
- **Ops**: Strongly typed operations with explicit inputs/outputs
- **Lineage**: First-class — every data dependency is traceable
- **Config**: YAML-based with type validation
- **Strengths**: Modern architecture, excellent observability, type safety
- **Weaknesses**: Steeper learning curve, smaller ecosystem than Airflow

### Airflow

- **DAGs**: Python-defined task graphs with explicit dependencies (`>>` operator)
- **Serialization**: AIP-24 DAG Persistence — Python DAGs parsed and stored as JSON in metadata DB
- **XComs**: Cross-task communication mechanism for passing data
- **Strengths**: Enterprise-grade, distributed, sophisticated scheduling, web UI
- **Weaknesses**: Heavy infrastructure, steep learning curve

### Snakemake / Nextflow

- **Snakemake**: Declarative rules with inputs/outputs/commands; implicit DAG from file patterns
- **Nextflow**: DSL2 processes with channels for data flow; `nextflow.config` for parameters
- **Both**: Container support (Docker, Singularity, Conda); resumable with caching
- **Strengths**: Simple syntax, scalable, reproducible via containerization
- **Weaknesses**: Bioinformatics-focused ecosystem

## Cross-Cutting Patterns

### Data Sources Representation
- **Declarative** (most reproducible): CWL, dbt, Kedro catalog
- **Programmatic**: Airflow connections, Prefect/Dagster code, Jupyter inline
- **Trend**: Move toward declarative for reproducibility

### Transformation Representation
- **SQL-based**: dbt (native)
- **Code-based**: Kedro, Prefect, Dagster, Jupyter
- **CLI-wrapped**: CWL, Snakemake, Nextflow
- **Key insight**: For LLM-generated code (our custom_operation), the code itself must be stored in the spec

### Parameterization
- **YAML/JSON configs**: dbt vars, Kedro parameters.yml, CWL parameters
- **Runtime injection**: Papermill, MLflow, Dagster config
- **Variable references**: dbt `{{ var(...) }}`, our `$TIME_RANGE`

### Lineage & Reproducibility
- **OpenLineage**: Emerging standard for lineage metadata across tools
- **Artifact-based**: dbt manifest, MLflow runs, DVC .dvc files
- **Key insight**: Track which data versions, parameters, and code versions produced results

## Key Takeaways

1. **CWL's explicit I/O pattern** is the most valuable idea — every step declares what it reads (`inputs`) and writes (`produces`). Our pipeline already has `produces` but lacks `inputs`.

2. **dbt's manifest-as-artifact** pattern maps to a `data_manifest()` derived view — a computed summary of data sources + transforms, not a stored file.

3. **DVC's single-file approach** validates keeping everything in one Pipeline JSON with tagged layers, rather than splitting into separate files.

4. **Separate specs create synchronization burden** — Kedro and dbt handle this with tooling, but our LLM-based workflow makes multi-file coordination error-prone.

## Recommendation: Layered Pipeline (NOT Separate Files)

### Why NOT separate DataOps and Mission Agent JSON files

1. **Redundancy**: The current pipeline already contains `fetch_data` and `custom_operation` steps. A separate DataOps JSON duplicates this.

2. **Synchronization burden**: Three cross-referencing JSON files create staleness when one is updated independently.

3. **Mission Agent discovery is non-deterministic**: The LLM's reasoning about which dataset to choose varies between runs. The *result* (dataset_id/parameter_id) is already captured in `fetch_data` steps. A "reproducible" discovery spec implies false reproducibility.

4. **LLM complexity**: One JSON is easier for the LLM to generate correctly than three.

5. **"Same data, different visualizations" is rare**: LLM-mediated modification already handles this.

### What to do instead

Extend the existing Pipeline JSON with:
- **`layer` tags** on each step (`"data"` or `"visualization"`) — auto-inferred from `tool_name`
- **`inputs` field** on each step (CWL-style explicit I/O)
- **`output_schema`** on data steps (columns, units — captured at recording time)
- **`data_manifest()` method** — computed view extracting just the data layer
- **`discovery_context`** metadata on Pipeline — captures why datasets were chosen
- **Layer-filtered execution** — `run_pipeline(layers=["data"])` to skip visualization

### Example: Full Annotated Pipeline JSON

```json
{
  "id": "ace-bfield-overview",
  "name": "ACE B-field Overview",
  "description": "Fetch ACE mag data, compute magnitude, two-panel plot",
  "variables": {"$TIME_RANGE": {"type": "time_range", "default": "last 7 days"}},
  "discovery_context": {
    "ACE": {
      "datasets_considered": ["AC_H2_MFI", "AC_H0_MFI"],
      "dataset_selected": "AC_H2_MFI",
      "reason": "16-second cadence vs 1-minute"
    }
  },
  "steps": [
    {
      "step_id": 1, "tool_name": "fetch_data",
      "tool_args": {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc", "time_range": "$TIME_RANGE"},
      "intent": "Fetch ACE magnetic field vector in GSE",
      "layer": "data", "inputs": [], "produces": ["AC_H2_MFI.BGSEc"],
      "output_schema": {"columns": ["Bx", "By", "Bz"], "units": "nT"},
      "depends_on": [], "critical": true
    },
    {
      "step_id": 2, "tool_name": "custom_operation",
      "tool_args": {"source_labels": ["AC_H2_MFI.BGSEc"], "pandas_code": "result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')", "output_label": "ACE_Bmag"},
      "intent": "Compute |B| from vector components",
      "layer": "data", "inputs": ["AC_H2_MFI.BGSEc"], "produces": ["ACE_Bmag"],
      "output_schema": {"columns": ["magnitude"], "units": "nT"},
      "depends_on": [1], "critical": true
    },
    {
      "step_id": 3, "tool_name": "render_spec",
      "tool_args": {"spec": {"labels": "AC_H2_MFI.BGSEc,ACE_Bmag", "panels": [["AC_H2_MFI.BGSEc"], ["ACE_Bmag"]], "y_label": {"1": "B (nT)", "2": "|B| (nT)"}}},
      "intent": "Two-panel plot with labeled axes",
      "layer": "visualization", "inputs": ["AC_H2_MFI.BGSEc", "ACE_Bmag"], "produces": [],
      "depends_on": [1, 2], "critical": false
    }
  ]
}
```

### Data Manifest (computed view, not stored)

`pipeline.data_manifest()` returns:
```json
{
  "pipeline_id": "ace-bfield-overview",
  "sources": [
    {"dataset_id": "AC_H2_MFI", "parameter_id": "BGSEc", "label": "AC_H2_MFI.BGSEc",
     "schema": {"columns": ["Bx", "By", "Bz"], "units": "nT"}}
  ],
  "transforms": [
    {"input_labels": ["AC_H2_MFI.BGSEc"],
     "code": "result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')",
     "output_label": "ACE_Bmag", "intent": "Compute |B| from vector components",
     "schema": {"columns": ["magnitude"], "units": "nT"}}
  ],
  "output_labels": ["AC_H2_MFI.BGSEc", "ACE_Bmag"],
  "variables": {"$TIME_RANGE": {"type": "time_range", "default": "last 7 days"}}
}
```

## Implementation Phases

### Phase 1: Data Lineage Fields
- Add `layer`, `inputs`, `output_schema` to `PipelineStep`
- Update `PipelineRecorder.record()` to auto-populate `inputs`
- Add `data_steps()`, `viz_steps()`, `data_manifest()` to `Pipeline`
- Files: `agent/pipeline.py`, `tests/test_pipeline.py`

### Phase 2: Layer-Filtered Execution
- Add `layers` param to `PipelineExecutor.execute()`
- Add optional `layers` param to `run_pipeline` tool schema
- Files: `agent/pipeline.py`, `agent/tools.py`, `agent/core.py`, `tests/test_pipeline.py`

### Phase 3: Discovery Context
- Add `discovery_context` to `Pipeline` dataclass
- Add optional param to `save_pipeline` tool schema
- Files: `agent/pipeline.py`, `agent/tools.py`, `agent/core.py`, `tests/test_pipeline.py`

### Phase 4: Output Validation (future)
- After replay, compare actual DataStore entries against stored `output_schema`
- Warn on regressions (column name changes, zero point counts, etc.)
