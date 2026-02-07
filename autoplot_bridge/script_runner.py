"""
AST-validated sandbox for executing Autoplot ScriptContext/DOM code.

Allows the AutoplotAgent to write and execute code that directly manipulates
Autoplot's ScriptContext and DOM model. Code is validated via AST analysis to
block dangerous constructs (imports, exec, file I/O, dunder access, raw Java
class construction) and executed in a restricted namespace with only
pre-imported Autoplot/Java classes available.

Mirrors the structure of data_ops/custom_ops.py but adapted for Autoplot:
- No `result` assignment required (most Autoplot ops are void/side-effects)
- `result` is optional — if assigned, captured in return dict
- Additional blocked names for Java/system access
"""

import ast
import builtins
import io
import sys

import numpy as np


# Builtins that are safe to use in Autoplot scripts
_SAFE_BUILTINS = frozenset({
    "abs", "bool", "dict", "enumerate", "float", "int", "len", "list",
    "max", "min", "print", "range", "round", "sorted", "str", "sum",
    "tuple", "zip", "True", "False", "None", "isinstance", "type",
    "map", "filter", "reversed", "any", "all", "repr", "format",
    "chr", "ord", "hex", "oct", "bin", "pow", "divmod", "hash",
    "id", "slice",
})

# Builtins that are explicitly dangerous
_DANGEROUS_BUILTINS = frozenset({
    "exec", "eval", "compile", "open", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "breakpoint", "exit",
    "quit", "input", "memoryview", "classmethod", "staticmethod", "super",
    "property",
})

# Names that are blocked in the code to prevent Java/system escape
_BLOCKED_NAMES = frozenset({
    "JClass", "jpype", "os", "sys", "subprocess", "socket", "shutil",
    "pathlib", "importlib",
})


def validate_autoplot_code(code: str) -> list[str]:
    """Validate Autoplot script code for safety using AST analysis.

    Args:
        code: Python code string to validate.

    Returns:
        List of violation descriptions. Empty list means code is safe.
    """
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Block imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("Imports are not allowed")

        # Block dangerous builtins
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_BUILTINS:
            violations.append(f"Dangerous builtin '{node.id}' is not allowed")

        # Block Java/system escape names
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            violations.append(f"Blocked name '{node.id}' is not allowed")

        # Block dunder attribute access (e.g., __class__, __dict__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"Dunder attribute access '{node.attr}' is not allowed")

        # Block global/nonlocal
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append("global/nonlocal statements are not allowed")

        # Block async constructs
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            violations.append("Async constructs are not allowed")

    return violations


def _build_namespace(ctx):
    """Build the restricted execution namespace for Autoplot scripts.

    Args:
        ctx: Autoplot ScriptContext object (from JPype).

    Returns:
        Dict mapping names to objects available in the script namespace.
    """
    import jpype

    namespace = {}

    # ScriptContext — the primary Autoplot API
    namespace["sc"] = ctx

    # DOM model — for fine-grained plot/panel manipulation
    namespace["dom"] = ctx.getDocumentModel()

    # java.awt.Color — for styling
    namespace["Color"] = jpype.JClass("java.awt.Color")

    # Autoplot RenderType enum
    namespace["RenderType"] = jpype.JClass("org.autoplot.RenderType")

    # das2 datum classes — for time ranges and units
    namespace["DatumRangeUtil"] = jpype.JClass("org.das2.datum.DatumRangeUtil")
    namespace["DatumRange"] = jpype.JClass("org.das2.datum.DatumRange")
    namespace["Units"] = jpype.JClass("org.das2.datum.Units")

    # das2 dataset classes — for data creation and property constants
    namespace["DDataSet"] = jpype.JClass("org.das2.qds.DDataSet")
    namespace["QDataSet"] = jpype.JClass("org.das2.qds.QDataSet")

    # In-memory data store access
    from data_ops.store import get_store, DataEntry
    store = get_store()
    namespace["store"] = store

    # --- to_qdataset helper ---
    DDataSet = namespace["DDataSet"]
    QDataSet = namespace["QDataSet"]
    Units = namespace["Units"]

    def to_qdataset(label_or_entry, component=None):
        """Convert stored data to a QDataSet for use with sc.plot().

        Args:
            label_or_entry: A string label (looked up in store) or a DataEntry.
            component: For vector data, specify an integer column index (0, 1, 2)
                or a column name string. Required for multi-column data.

        Returns:
            A rank-1 QDataSet with DEPEND_0 (time), LABEL, and TITLE set.

        Raises:
            ValueError: If the label is not found, or component is needed but
                not specified.
        """
        if isinstance(label_or_entry, str):
            entry = store.get(label_or_entry)
            if entry is None:
                raise ValueError(f"Label '{label_or_entry}' not found in store")
        elif isinstance(label_or_entry, DataEntry):
            entry = label_or_entry
        else:
            raise ValueError("Expected a string label or DataEntry")

        time_arr = entry.time
        values = entry.values

        # Handle vector / multi-column data
        if values.ndim == 2 and values.shape[1] > 1:
            if component is None:
                cols = list(entry.data.columns)
                raise ValueError(
                    f"'{entry.label}' is vector data ({values.shape[1]} columns: {cols}). "
                    f"Specify a component index or name. Examples:\n"
                    f"  to_qdataset('{entry.label}', component=0)  # first component\n"
                    f"  to_qdataset('{entry.label}', component=2)  # third component"
                )
            if isinstance(component, str):
                if component not in entry.data.columns:
                    raise ValueError(f"Column '{component}' not found in '{entry.label}'")
                col_idx = list(entry.data.columns).index(component)
            else:
                col_idx = int(component)
            values = values[:, col_idx]
            label = f"{entry.label}.{component}"
        else:
            values = values.ravel() if values.ndim > 1 else values
            label = entry.label

        n = len(time_arr)

        # Build time dataset (Units.t2000 = seconds since 2000-01-01)
        epoch_2000 = np.datetime64("2000-01-01T00:00:00", "ns")
        time_seconds = (time_arr.astype("datetime64[ns]") - epoch_2000).astype(np.float64) / 1e9

        time_ds = DDataSet.createRank1(n)
        for i in range(n):
            time_ds.putValue(i, float(time_seconds[i]))
        time_ds.putProperty(QDataSet.UNITS, Units.t2000)

        # Build values dataset
        val_ds = DDataSet.createRank1(n)
        for i in range(n):
            val_ds.putValue(i, float(values[i]))
        val_ds.putProperty(QDataSet.DEPEND_0, time_ds)

        # Units
        if entry.units:
            try:
                val_ds.putProperty(QDataSet.UNITS, Units.lookupUnits(entry.units))
            except Exception:
                pass

        val_ds.putProperty(QDataSet.LABEL, label)
        val_ds.putProperty(QDataSet.TITLE, label)

        return val_ds

    namespace["to_qdataset"] = to_qdataset

    # Wrap sc.plot to enforce 3-panel maximum (ISSUE-01)
    _original_sc_plot = ctx.plot

    def _guarded_plot(*args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], (int, float)) and int(args[0]) > 2:
            raise ValueError(
                f"Panel index {int(args[0])} exceeds maximum (0-2). "
                f"Maximum 3 panels supported. Use overlays instead of additional panels."
            )
        return _original_sc_plot(*args, **kwargs)

    # Monkey-patch the guarded plot onto a thin proxy so sc.plot is guarded
    # but all other sc methods (reset, waitUntilIdle, etc.) still work
    class _ScProxy:
        def __init__(self, ctx, guarded_plot):
            object.__setattr__(self, '_ctx', ctx)
            object.__setattr__(self, '_plot', guarded_plot)
        def plot(self, *args, **kwargs):
            return object.__getattribute__(self, '_plot')(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, '_ctx'), name)

    namespace["sc"] = _ScProxy(ctx, _guarded_plot)

    return namespace


def _format_java_exception(e: Exception) -> str:
    """Extract a readable message from a JPype Java exception.

    Args:
        e: The Python exception (may wrap a Java exception).

    Returns:
        A human-readable error string.
    """
    # Try to get the Java exception message
    java_exc = getattr(e, "java_exception", None) or getattr(e, "__cause__", None)
    if java_exc is not None:
        msg = getattr(java_exc, "getMessage", lambda: None)()
        cls = type(java_exc).__name__
        if msg:
            return f"{cls}: {msg}"
        return str(java_exc)
    return str(e)


def execute_autoplot_script(code: str, ctx) -> dict:
    """Execute validated Autoplot script code in a restricted namespace.

    Args:
        code: Validated Python code using sc, dom, Color, etc.
        ctx: Autoplot ScriptContext object.

    Returns:
        Dict with:
        - status: "success" or "error"
        - output: Any print() output captured during execution
        - result: Value of `result` variable if assigned (optional)
        - message: Error message if status is "error"
    """
    namespace = _build_namespace(ctx)

    # Add safe builtins
    safe_builtins = {
        name: getattr(builtins, name)
        for name in _SAFE_BUILTINS
        if hasattr(builtins, name)
    }

    # Capture print output
    output_buffer = io.StringIO()

    # Override print to capture output
    original_print = safe_builtins.get("print", print)

    def captured_print(*args, **kwargs):
        kwargs["file"] = output_buffer
        original_print(*args, **kwargs)

    safe_builtins["print"] = captured_print

    try:
        exec(code, {"__builtins__": safe_builtins}, namespace)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Execution error: {_format_java_exception(e)}",
            "output": output_buffer.getvalue(),
        }

    response = {
        "status": "success",
        "output": output_buffer.getvalue(),
    }

    # Capture optional result variable
    if "result" in namespace:
        result_val = namespace["result"]
        # Convert to string for serialization
        response["result"] = str(result_val) if result_val is not None else None

    return response


def run_autoplot_script(code: str, ctx) -> dict:
    """Validate and execute an Autoplot script.

    Convenience function that combines validation and execution.

    Args:
        code: Python code using sc, dom, Color, etc.
        ctx: Autoplot ScriptContext object.

    Returns:
        Dict with status, output, and optional result.

    The dict always has a "status" key ("success" or "error").
    On validation failure, "message" contains the violation details.
    """
    violations = validate_autoplot_code(code)
    if violations:
        return {
            "status": "error",
            "message": "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations),
        }
    return execute_autoplot_script(code, ctx)
