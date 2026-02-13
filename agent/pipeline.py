"""
Pipeline data structures, persistence, recording, and execution.

A pipeline is a saved sequence of tool calls (fetch → compute → plot)
that can be replayed deterministically with variable substitution,
or executed with LLM-mediated modifications.

This module provides:
- PipelineStep: A single tool call in a pipeline
- PipelineVariable: A template variable (e.g., $TIME_RANGE)
- Pipeline: An ordered list of steps with variables and metadata
- PipelineStore: JSON persistence to ~/.helio-agent/pipelines/
- PipelineRecorder: Passive tool call buffer for the current session
- PipelineExecutor: Deterministic replay engine with dependency tracking
"""

import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .logging import get_logger

logger = get_logger()

# Tools worth recording in a pipeline (data-producing + visualization)
RECORDABLE_TOOLS = {
    "fetch_data", "custom_operation", "compute_spectrogram",
    "store_dataframe", "plot_data", "style_plot", "manage_plot",
}


@dataclass
class PipelineStep:
    """A single tool call in a pipeline.

    Attributes:
        step_id: Sequential identifier (1-based)
        tool_name: Name of the tool to execute
        tool_args: Arguments dict for the tool
        intent: Natural-language description of what this step does
        produces: DataStore labels created by this step
        depends_on: Step IDs that must succeed before this step runs
        critical: If True, failure aborts all dependent steps
    """
    step_id: int
    tool_name: str
    tool_args: dict
    intent: str = ""
    produces: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    critical: bool = True

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "intent": self.intent,
            "produces": self.produces,
            "depends_on": self.depends_on,
            "critical": self.critical,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineStep":
        return cls(
            step_id=data["step_id"],
            tool_name=data["tool_name"],
            tool_args=data["tool_args"],
            intent=data.get("intent", ""),
            produces=data.get("produces", []),
            depends_on=data.get("depends_on", []),
            critical=data.get("critical", True),
        )


@dataclass
class PipelineVariable:
    """A template variable that gets substituted at execution time.

    Attributes:
        type: Variable type hint (e.g., "time_range", "string", "mission_id")
        description: Human-readable description of the variable
        default: Default value as a string
    """
    type: str
    description: str
    default: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "description": self.description,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineVariable":
        return cls(
            type=data["type"],
            description=data["description"],
            default=data.get("default", ""),
        )


@dataclass
class Pipeline:
    """A saved sequence of tool calls with template variables.

    Attributes:
        id: Unique slug identifier (e.g., "ace-bfield-overview")
        name: Human-readable name
        description: What this pipeline does
        steps: Ordered list of tool call steps
        variables: Template variables (key starts with $)
        created_at: ISO timestamp
        updated_at: ISO timestamp
        source_session: Session ID that created this pipeline
    """
    id: str
    name: str
    description: str
    steps: list[PipelineStep] = field(default_factory=list)
    variables: dict[str, PipelineVariable] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    source_session: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_session": self.source_session,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        variables = {}
        for k, v in data.get("variables", {}).items():
            variables[k] = PipelineVariable.from_dict(v)
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=[PipelineStep.from_dict(s) for s in data.get("steps", [])],
            variables=variables,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            source_session=data.get("source_session", ""),
        )

    def to_llm_context(self) -> str:
        """Format the pipeline as readable text for LLM context injection."""
        lines = [
            f"Pipeline: {self.name}",
            f"Description: {self.description}",
        ]
        if self.variables:
            lines.append("Variables:")
            for var_name, var_def in self.variables.items():
                lines.append(f"  {var_name}: {var_def.description} (default: {var_def.default})")
        lines.append(f"Steps ({len(self.steps)}):")
        for step in self.steps:
            dep_str = f" [depends on step(s) {step.depends_on}]" if step.depends_on else ""
            crit_str = " [critical]" if step.critical else " [non-critical]"
            lines.append(f"  Step {step.step_id}: {step.tool_name}{dep_str}{crit_str}")
            lines.append(f"    Intent: {step.intent}")
            lines.append(f"    Args: {json.dumps(step.tool_args, default=str)}")
            if step.produces:
                lines.append(f"    Produces: {', '.join(step.produces)}")
        return "\n".join(lines)


def _slugify(name: str) -> str:
    """Convert a pipeline name to a filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "pipeline"


class PipelineStore:
    """JSON file persistence for pipelines in ~/.helio-agent/pipelines/."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            from config import get_data_dir
            base_dir = get_data_dir() / "pipelines"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _pipeline_path(self, pipeline_id: str) -> Path:
        return self.base_dir / f"{pipeline_id}.json"

    def save(self, pipeline: Pipeline) -> Path:
        """Save a pipeline to disk. Overwrites if ID already exists."""
        pipeline.updated_at = datetime.now().isoformat()
        if not pipeline.created_at:
            pipeline.created_at = pipeline.updated_at
        path = self._pipeline_path(pipeline.id)
        with open(path, "w") as f:
            json.dump(pipeline.to_dict(), f, indent=2)
        return path

    def load(self, pipeline_id: str) -> Optional[Pipeline]:
        """Load a pipeline by its ID."""
        path = self._pipeline_path(pipeline_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return Pipeline.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load pipeline {pipeline_id}: {e}")
            return None

    def delete(self, pipeline_id: str) -> bool:
        """Delete a pipeline by its ID. Returns True if deleted."""
        path = self._pipeline_path(pipeline_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_pipelines(self) -> list[dict]:
        """List all saved pipelines with summary info."""
        pipelines = []
        for f in sorted(self.base_dir.glob("*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    pipelines.append({
                        "id": data["id"],
                        "name": data["name"],
                        "description": data.get("description", ""),
                        "step_count": len(data.get("steps", [])),
                        "variables": list(data.get("variables", {}).keys()),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        return pipelines


class PipelineRecorder:
    """Passively records tool calls during a session for pipeline creation.

    Always-on: records every tool call that goes through _execute_tool_safe().
    Only captures data-producing and visualization tools (RECORDABLE_TOOLS).
    """

    def __init__(self):
        self._recording: list[dict] = []

    def record(self, tool_name: str, tool_args: dict, result: dict):
        """Record a tool call if it's a recordable tool and succeeded."""
        if tool_name not in RECORDABLE_TOOLS:
            return
        if result.get("status") == "error":
            return
        self._recording.append({
            "tool_name": tool_name,
            "tool_args": copy.deepcopy(tool_args),
            "result_status": result.get("status", "unknown"),
        })

    def get_recording(self) -> list[dict]:
        """Return a copy of the recording buffer."""
        return list(self._recording)

    def clear(self):
        """Clear the recording buffer."""
        self._recording.clear()

    def __len__(self) -> int:
        return len(self._recording)


def _substitute_variables(obj, variables: dict[str, str]):
    """Recursively substitute $VAR_NAME references in tool_args.

    Args:
        obj: The object to process (dict, list, str, or scalar)
        variables: Mapping from variable names (e.g. "$TIME_RANGE") to their resolved values
    """
    if isinstance(obj, str):
        for var_name, var_value in variables.items():
            if var_name in obj:
                obj = obj.replace(var_name, var_value)
        return obj
    elif isinstance(obj, dict):
        return {k: _substitute_variables(v, variables) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_variables(item, variables) for item in obj]
    return obj


class PipelineExecutor:
    """Executes a pipeline deterministically (no LLM).

    Walks steps in order, substitutes template variables,
    calls the tool executor function, and handles dependency/failure propagation.
    """

    def __init__(self, tool_executor: Callable[[str, dict], dict]):
        """
        Args:
            tool_executor: Function that executes a tool call.
                Signature: (tool_name, tool_args) -> result_dict
        """
        self.tool_executor = tool_executor

    def execute(
        self,
        pipeline: Pipeline,
        variable_overrides: Optional[dict[str, str]] = None,
    ) -> dict:
        """Execute all pipeline steps deterministically.

        Args:
            pipeline: The pipeline to execute
            variable_overrides: Override variable values (e.g. {"$TIME_RANGE": "2026-01-01 to 2026-01-31"})

        Returns:
            {"status": "success"|"partial"|"failed",
             "steps_completed": int, "steps_failed": int, "steps_skipped": int,
             "step_results": [...], "summary": str}
        """
        # Resolve variable values: override > default
        resolved_vars = {}
        for var_name, var_def in pipeline.variables.items():
            if variable_overrides and var_name in variable_overrides:
                resolved_vars[var_name] = variable_overrides[var_name]
            else:
                resolved_vars[var_name] = var_def.default

        step_results = []
        failed_steps: set[int] = set()
        completed = 0
        failed = 0
        skipped = 0

        for step in pipeline.steps:
            # Check if any dependency failed
            blocked_by = [d for d in step.depends_on if d in failed_steps]
            if blocked_by:
                step_results.append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "intent": step.intent,
                    "status": "skipped",
                    "reason": f"Dependency step(s) {blocked_by} failed",
                })
                skipped += 1
                if step.critical:
                    failed_steps.add(step.step_id)
                continue

            # Substitute variables in tool_args
            resolved_args = _substitute_variables(
                copy.deepcopy(step.tool_args), resolved_vars
            )

            # Execute the tool
            try:
                result = self.tool_executor(step.tool_name, resolved_args)
                is_error = result.get("status") == "error"
            except Exception as e:
                result = {"status": "error", "message": str(e)}
                is_error = True

            if is_error:
                failed += 1
                if step.critical:
                    failed_steps.add(step.step_id)
                step_results.append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "intent": step.intent,
                    "status": "failed",
                    "error": result.get("message", "Unknown error"),
                })
                logger.warning(
                    f"[Pipeline] Step {step.step_id} ({step.tool_name}) failed: "
                    f"{result.get('message', 'Unknown')}"
                )
            else:
                completed += 1
                step_results.append({
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "intent": step.intent,
                    "status": "success",
                })

        # Determine overall status
        total = len(pipeline.steps)
        if completed == total:
            status = "success"
        elif completed == 0:
            status = "failed"
        else:
            status = "partial"

        summary_parts = [f"{completed}/{total} steps completed"]
        if failed:
            summary_parts.append(f"{failed} failed")
        if skipped:
            summary_parts.append(f"{skipped} skipped")

        return {
            "status": status,
            "steps_completed": completed,
            "steps_failed": failed,
            "steps_skipped": skipped,
            "step_results": step_results,
            "summary": ", ".join(summary_parts),
            "variables_used": resolved_vars,
        }


# ---- Singleton store ----

_store: Optional[PipelineStore] = None


def get_pipeline_store() -> PipelineStore:
    """Get the global PipelineStore instance (creates on first call)."""
    global _store
    if _store is None:
        _store = PipelineStore()
    return _store


def reset_pipeline_store():
    """Reset the global PipelineStore instance (for testing)."""
    global _store
    _store = None
