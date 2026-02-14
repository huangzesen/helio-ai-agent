"""
Custom pandas operation executor with AST-based safety validation.

Allows the LLM to generate arbitrary pandas/numpy code that operates on
a DataFrame. Code is validated via AST analysis to block dangerous
constructs (imports, exec, file I/O, dunder access) and executed in a
restricted namespace with only df, pd, and np available.
"""

import ast
import builtins
import gc

import numpy as np
import pandas as pd
import xarray as xr


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
_ALLOWED_NAMES = frozenset({"df", "pd", "np", "xr", "scipy", "pywt", "result"})


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


def _execute_in_sandbox(code: str, namespace: dict) -> object:
    """Execute code in a sandboxed namespace and return the 'result' value.

    Builds safe builtins, runs exec(), extracts result, then cleans up.

    Args:
        code: Python code to execute.
        namespace: Dict with variables available to the code (must include 'result': None).

    Returns:
        The value of 'result' after execution.

    Raises:
        RuntimeError: If code execution fails.
    """
    safe_builtins = {name: getattr(builtins, name) for name in _SAFE_BUILTINS if hasattr(builtins, name)}

    try:
        exec(code, {"__builtins__": safe_builtins}, namespace)
    except Exception as e:
        raise RuntimeError(f"Execution error: {type(e).__name__}: {e}") from e

    result = namespace.get("result")

    del namespace
    gc.collect()

    return result


def _validate_result(result: object) -> pd.DataFrame | xr.DataArray:
    """Validate that the sandbox result is a DataFrame or xarray DataArray.

    - Series → converted to single-column DataFrame
    - xarray DataArray with ``time`` dim → returned as-is (any dimensionality)
    - DataFrame with any index type (DatetimeIndex, numeric, string) → accepted
    - Other types → error

    Args:
        result: The value produced by sandbox execution.

    Returns:
        Validated DataFrame or DataArray.

    Raises:
        ValueError: If result is None or wrong type.
    """
    if result is None:
        raise ValueError("Code did not assign a value to 'result'")

    if isinstance(result, pd.Series):
        result = result.to_frame(name="value")

    # xarray DataArray: accept if it has a time dimension
    if isinstance(result, xr.DataArray):
        if "time" in result.dims:
            return result
        dims = dict(result.sizes)
        raise ValueError(
            f"Result is an xarray DataArray with dims {dims} but no 'time' "
            f"dimension. The result must have a 'time' dimension. "
            f"Use .rename() if the time dimension has a different name, "
            f"or convert to DataFrame with .to_pandas()."
        )

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"Result must be a DataFrame, Series, or xarray DataArray, "
            f"got {type(result).__name__}"
        )

    return result


def execute_multi_source_operation(
    sources: dict[str, pd.DataFrame | xr.DataArray], code: str
) -> pd.DataFrame | xr.DataArray:
    """Execute validated pandas/xarray code with multiple sources.

    Each source is available in the sandbox by its key:
    - ``df_SUFFIX`` for pandas DataFrames
    - ``da_SUFFIX`` for xarray DataArrays

    The first DataFrame source is also aliased as ``df`` for backward
    compatibility.  ``xr`` (xarray) is always available in the namespace.

    Args:
        sources: Mapping of variable names to DataFrames or DataArrays.
        code: Validated Python code that assigns to 'result'.

    Returns:
        Result DataFrame with DatetimeIndex, or xarray DataArray with time dim.

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a valid type or missing time axis.
    """
    import scipy
    import pywt
    namespace = {"pd": pd, "np": np, "xr": xr, "scipy": scipy, "pywt": pywt, "result": None}
    first_df_key = None
    for key, data in sources.items():
        if isinstance(data, xr.DataArray):
            namespace[key] = data.copy()
        else:
            df = data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            namespace[key] = df
            if first_df_key is None:
                first_df_key = key
                namespace["df"] = df

    result = _execute_in_sandbox(code, namespace)
    return _validate_result(result)


def validate_result(
    result: pd.DataFrame | xr.DataArray,
    sources: dict[str, pd.DataFrame | xr.DataArray],
) -> list[str]:
    """Validate a computation result against its sources for common pitfalls.

    For DataFrame results, checks:
    1. NaN-to-zero: result has zeros where sources have NaN (skipna issue)
    2. Row count anomaly: result has significantly more rows than largest source
    3. Constant output: result is constant when sources are not

    For DataArray results, only checks row count anomaly.

    Args:
        result: The computation result (DataFrame or DataArray).
        sources: The source DataFrames/DataArrays used (keyed by sandbox variable name).

    Returns:
        List of warning strings. Empty list means no issues detected.
    """
    warnings = []

    # For DataArray results, only do row-count check
    if isinstance(result, xr.DataArray):
        if sources:
            def _source_len(s):
                return s.sizes["time"] if isinstance(s, xr.DataArray) else len(s)
            result_len = result.sizes["time"]
            max_source_rows = max(_source_len(s) for s in sources.values())
            if max_source_rows > 0 and result_len > max_source_rows * 1.1:
                warnings.append(
                    f"Result has {result_len} time steps vs largest source "
                    f"{max_source_rows} — unexpected expansion"
                )
        return warnings

    # DataFrame-specific checks below
    result_df = result

    # Separate DataFrame sources (skip DataArray sources for DataFrame-specific checks)
    df_sources = {k: v for k, v in sources.items() if isinstance(v, pd.DataFrame)}

    # Check 1: NaN-to-zero — zeros in result coinciding with NaN in sources
    result_zeros = (result_df == 0.0)
    for var_name, src_df in df_sources.items():
        src_nan_times = src_df.index[src_df.isna().any(axis=1)]
        if len(src_nan_times) == 0:
            continue
        overlap = result_df.index.intersection(src_nan_times)
        if len(overlap) == 0:
            continue
        zero_at_nan = result_zeros.loc[overlap].any(axis=1).sum()
        if zero_at_nan > 0:
            warnings.append(
                f"Result has {zero_at_nan} zeros coinciding with NaN in "
                f"source '{var_name}' — possible skipna issue"
            )

    # Check 2: Row count anomaly
    if sources:
        def _source_len(s):
            return s.sizes["time"] if isinstance(s, xr.DataArray) else len(s)
        max_source_rows = max(_source_len(s) for s in sources.values())
        if max_source_rows > 0 and len(result_df) > max_source_rows * 1.1:
            warnings.append(
                f"Result has {len(result_df)} rows vs largest source "
                f"{max_source_rows} — unexpected expansion"
            )

    # Check 3: Constant output from non-constant sources
    for col in result_df.columns:
        col_data = result_df[col].dropna()
        if len(col_data) < 2:
            continue
        if col_data.std() == 0:
            # Check if any source is non-constant
            any_source_varies = any(
                src_df.std().max() > 0 for src_df in df_sources.values()
            )
            if any_source_varies:
                warnings.append(
                    f"Result column '{col}' is constant "
                    f"(value={col_data.iloc[0]}) — possible fill value "
                    f"or collapsed computation"
                )

    return warnings


def run_multi_source_operation(
    sources: dict[str, pd.DataFrame | xr.DataArray], code: str
) -> tuple[pd.DataFrame | xr.DataArray, list[str]]:
    """Validate code, execute with multiple sources, then validate result.

    Convenience function combining code validation, multi-source execution,
    and result validation.

    Args:
        sources: Mapping of variable names to source DataFrames/DataArrays.
        code: Python code that operates on named variables and assigns to 'result'.

    Returns:
        Tuple of (result DataFrame or DataArray, list of warning strings).

    Raises:
        ValueError: If code validation fails or result is invalid.
        RuntimeError: If execution fails.
    """
    violations = validate_pandas_code(code)
    if violations:
        raise ValueError(
            "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        )
    result = execute_multi_source_operation(sources, code)
    warnings = validate_result(result, sources)
    return result, warnings


def execute_custom_operation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute validated pandas code in a restricted namespace.

    Args:
        df: Input DataFrame (will be copied to prevent mutation).
        code: Validated Python code that assigns to 'result'.

    Returns:
        Result DataFrame or xarray DataArray.

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a DataFrame/Series/DataArray.
    """
    import scipy
    import pywt
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    result = _execute_in_sandbox(code, {"df": df, "pd": pd, "np": np, "xr": xr, "scipy": scipy, "pywt": pywt, "result": None})
    return _validate_result(result)


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


def execute_dataframe_creation(code: str) -> pd.DataFrame:
    """Execute validated pandas code to create a DataFrame from scratch.

    Unlike execute_custom_operation(), there is no input DataFrame — the code
    constructs data using pd and np only.  Used by the store_dataframe tool to
    turn text data (event catalogs, search results) into stored DataFrames.

    Args:
        code: Validated Python code that assigns to 'result'.

    Returns:
        Result DataFrame (any index type).

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a DataFrame/Series/DataArray.
    """
    result = _execute_in_sandbox(code, {"pd": pd, "np": np, "xr": xr, "result": None})
    return _validate_result(result)


def execute_spectrogram_computation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute code to compute a spectrogram from a timeseries.

    Like execute_custom_operation() but adds scipy.signal to the namespace
    for FFT, windowing, and spectrogram functions.

    Namespace: df, pd, np, signal (scipy.signal)
    Output: DataFrame with DatetimeIndex x numeric columns (bin values as column names)

    Args:
        df: Input DataFrame with timeseries data.
        code: Validated Python code that assigns to 'result'.

    Returns:
        Result DataFrame or xarray DataArray.

    Raises:
        RuntimeError: If code execution fails.
        ValueError: If result is not a DataFrame/Series/DataArray.
    """
    from scipy import signal

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    result = _execute_in_sandbox(
        code, {"df": df, "pd": pd, "np": np, "xr": xr, "signal": signal, "result": None}
    )
    return _validate_result(result)


def run_spectrogram_computation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Validate and execute spectrogram computation code.

    Convenience function that combines validation and execution.

    Args:
        df: Input DataFrame with timeseries data.
        code: Python code that computes a spectrogram using df, pd, np, signal.

    Returns:
        Result DataFrame (DatetimeIndex x frequency/energy bins).

    Raises:
        ValueError: If validation fails or result is invalid.
        RuntimeError: If execution fails.
    """
    violations = validate_pandas_code(code)
    if violations:
        raise ValueError(
            "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        )
    return execute_spectrogram_computation(df, code)


def run_dataframe_creation(code: str) -> pd.DataFrame:
    """Validate and execute code that creates a DataFrame from scratch.

    Convenience function that combines validation and execution.

    Args:
        code: Python code that constructs data using pd/np and assigns to 'result'.

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
    return execute_dataframe_creation(code)
