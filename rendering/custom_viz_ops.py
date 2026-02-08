"""
Custom Plotly visualization executor with AST-based safety validation.

Allows the LLM to generate free-form Plotly code that mutates the current
figure in place. Code is validated via the same AST analysis used by
data_ops/custom_ops.py and executed in a restricted namespace with only
``fig``, ``go`` (plotly.graph_objects), and ``np`` available.

Unlike custom_operation, no ``result`` assignment is required — code
modifies ``fig`` via its methods (update_layout, add_trace, etc.).
"""

import builtins

import numpy as np
import plotly.graph_objects as go

from data_ops.custom_ops import validate_pandas_code, _SAFE_BUILTINS


def validate_plotly_code(code: str) -> list[str]:
    """Validate Plotly code for safety using AST analysis.

    Same safety rules as pandas code (no imports, exec, dunder, etc.)
    but does NOT require a ``result = ...`` assignment.

    Args:
        code: Python code string to validate.

    Returns:
        List of violation descriptions. Empty list means code is safe.
    """
    return validate_pandas_code(code, require_result=False)


def execute_custom_visualization(fig: go.Figure, code: str) -> None:
    """Execute validated Plotly code in a restricted namespace.

    The code runs with ``fig`` (Plotly Figure), ``go`` (graph_objects),
    and ``np`` (numpy) available.  It is expected to mutate ``fig`` in
    place — no return value is captured.

    Args:
        fig: The Plotly Figure to modify (mutated in place).
        code: Validated Python code that operates on ``fig``.

    Raises:
        RuntimeError: If code execution fails.
    """
    namespace = {
        "fig": fig,
        "go": go,
        "np": np,
    }

    safe_builtins = {
        name: getattr(builtins, name)
        for name in _SAFE_BUILTINS
        if hasattr(builtins, name)
    }

    try:
        exec(code, {"__builtins__": safe_builtins}, namespace)  # noqa: S102
    except Exception as e:
        raise RuntimeError(f"Execution error: {type(e).__name__}: {e}") from e


def run_custom_visualization(fig: go.Figure, code: str) -> None:
    """Validate and execute custom Plotly visualization code.

    Convenience function that combines validation and execution.

    Args:
        fig: The Plotly Figure to modify.
        code: Python code that operates on ``fig``.

    Raises:
        ValueError: If validation fails.
        RuntimeError: If execution fails.
    """
    violations = validate_plotly_code(code)
    if violations:
        raise ValueError(
            "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        )
    execute_custom_visualization(fig, code)
