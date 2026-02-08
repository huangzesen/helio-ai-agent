"""
Custom pandas operation executor with AST-based safety validation.

Allows the LLM to generate arbitrary pandas/numpy code that operates on
a DataFrame. Code is validated via AST analysis to block dangerous
constructs (imports, exec, file I/O, dunder access) and executed in a
restricted namespace with only df, pd, and np available.
"""

import ast
import builtins

import numpy as np
import pandas as pd


# Builtins that are safe to use in custom operations
_SAFE_BUILTINS = frozenset({
    "abs", "bool", "dict", "enumerate", "float", "int", "len", "list",
    "max", "min", "print", "range", "round", "sorted", "str", "sum",
    "tuple", "zip", "True", "False", "None", "isinstance", "type",
})

# Builtins that are explicitly dangerous
_DANGEROUS_BUILTINS = frozenset({
    "exec", "eval", "compile", "open", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "breakpoint", "exit",
    "quit", "input", "memoryview", "classmethod", "staticmethod", "super",
    "property",
})

# Names allowed in the execution namespace
_ALLOWED_NAMES = frozenset({"df", "pd", "np", "result"})


def validate_pandas_code(code: str, require_result: bool = True) -> list[str]:
    """Validate pandas code for safety using AST analysis.

    Args:
        code: Python code string to validate.
        require_result: If True (default), require ``result = ...`` assignment.
            Set to False for code that mutates objects in place (e.g., Plotly figures).

    Returns:
        List of violation descriptions. Empty list means code is safe.
    """
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    has_result_assignment = False

    for node in ast.walk(tree):
        # Block imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("Imports are not allowed")

        # Block dangerous builtins
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_BUILTINS:
            violations.append(f"Dangerous builtin '{node.id}' is not allowed")

        # Block dunder attribute access (e.g., __class__, __dict__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"Dunder attribute access '{node.attr}' is not allowed")

        # Block global/nonlocal
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append("global/nonlocal statements are not allowed")

        # Block async constructs
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            violations.append("Async constructs are not allowed")

        # Track result assignment
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    has_result_assignment = True
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "result":
                has_result_assignment = True

    if require_result and not has_result_assignment:
        violations.append("Code must assign to 'result'")

    return violations


def execute_custom_operation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute validated pandas code in a restricted namespace.

    Args:
        df: Input DataFrame (will be copied to prevent mutation).
        code: Validated Python code that assigns to 'result'.

    Returns:
        Result DataFrame with DatetimeIndex.

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a DataFrame/Series or loses DatetimeIndex.
    """
    import gc

    df = df.copy()
    # Ensure DatetimeIndex so time-based rolling windows (e.g., '2H') work
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    namespace = {
        "df": df,
        "pd": pd,
        "np": np,
        "result": None,
    }

    safe_builtins = {name: getattr(builtins, name) for name in _SAFE_BUILTINS if hasattr(builtins, name)}

    try:
        exec(code, {"__builtins__": safe_builtins}, namespace)
    except Exception as e:
        raise RuntimeError(f"Execution error: {type(e).__name__}: {e}") from e

    result = namespace.get("result")

    # Free the working copy and namespace intermediates
    del df, namespace
    gc.collect()

    if result is None:
        raise ValueError("Code did not assign a value to 'result'")

    # Convert Series to DataFrame
    if isinstance(result, pd.Series):
        result = result.to_frame(name="value")

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"Result must be a DataFrame or Series, got {type(result).__name__}"
        )

    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError(
            "Result must have a DatetimeIndex (time axis). "
            "Make sure your operation preserves the DataFrame index."
        )

    return result


def run_custom_operation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Validate and execute a custom pandas operation.

    Convenience function that combines validation and execution.

    Args:
        df: Input DataFrame.
        code: Python code that operates on 'df' and assigns to 'result'.

    Returns:
        Result DataFrame.

    Raises:
        ValueError: If validation fails or result is invalid.
        RuntimeError: If execution fails.
    """
    violations = validate_pandas_code(code)
    if violations:
        raise ValueError(
            "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        )
    return execute_custom_operation(df, code)
