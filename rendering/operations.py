"""
Operation-based rendering: registry, resolver, and built-in operation definitions.

Each operation is a JSON template that maps parameters to a Plotly JSON patch.
The resolver applies $variable substitution, panel-indexed axis mapping,
trace matching, and append mode to produce the final patch.

Usage:
    registry = OperationRegistry()
    registry.register_builtins()
    for op_dict in spec["operations"]:
        registry.resolve(op_dict, fig, trace_label_map, panel_axis_map)
"""

from __future__ import annotations

import copy
import re
from typing import Any

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _panel_to_axis_key(panel: int, axis_prefix: str = "yaxis") -> str:
    """Map a 1-indexed panel number to a Plotly axis key.

    panel=1 → 'yaxis', panel=2 → 'yaxis2', panel=3 → 'yaxis3', etc.
    Works for any axis prefix: 'yaxis', 'xaxis'.
    """
    if panel <= 0:
        raise ValueError(f"Panel must be >= 1, got {panel}")
    return axis_prefix if panel == 1 else f"{axis_prefix}{panel}"


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge *patch* into *base*, returning a new dict.

    Values in *patch* overwrite values in *base*.  Nested dicts are
    merged recursively rather than replaced wholesale.
    """
    result = dict(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _substitute_vars(obj: Any, params: dict[str, Any]) -> Any:
    """Recursively replace '$varname' strings with values from *params*.

    - A string that is exactly '$varname' is replaced by the param value
      (preserving type — int, list, etc.).
    - A string containing '$varname' as a substring gets string interpolation.
    - Non-string values pass through unchanged.
    """
    if isinstance(obj, str):
        # Exact match: "$text" → params["text"] (preserves type)
        m = re.fullmatch(r"\$(\w+)", obj)
        if m and m.group(1) in params:
            return params[m.group(1)]
        # Substring matches: "hello $name" → "hello World"
        def _replacer(match: re.Match) -> str:
            key = match.group(1)
            return str(params[key]) if key in params else match.group(0)
        return re.sub(r"\$(\w+)", _replacer, obj)
    elif isinstance(obj, dict):
        return {k: _substitute_vars(v, params) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_vars(item, params) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Operation definition
# ---------------------------------------------------------------------------

class OperationDef:
    """A registered operation template.

    Attributes:
        name:  Unique operation name (e.g. 'set_title').
        description:  Human-readable description.
        params:  Schema dict: {param_name: {type, description, required?}}.
        target:  Where to apply the patch: 'layout', 'trace', or 'figure'.
        patch:  JSON patch template with $variable placeholders.
        panel_indexed_axis:  If set, the patch targets a panel-indexed axis
                             (e.g. 'yaxis' → 'yaxis2' for panel 2).
        trace_target:  If True, a 'trace' param selects which trace to patch.
        append_to:  If set, the patch value is appended to this layout array
                    key (e.g. 'shapes', 'annotations') instead of merged.
        composite:  If set, a list of sub-operation dicts to expand into.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        params: dict | None = None,
        target: str = "layout",
        patch: dict | None = None,
        panel_indexed_axis: str | None = None,
        trace_target: bool = False,
        append_to: list[str] | None = None,
        composite: list[dict] | None = None,
    ):
        self.name = name
        self.description = description
        self.params = params or {}
        self.target = target
        self.patch = patch or {}
        self.panel_indexed_axis = panel_indexed_axis
        self.trace_target = trace_target
        self.append_to = append_to
        self.composite = composite

    @classmethod
    def from_dict(cls, d: dict) -> OperationDef:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            params=d.get("params", {}),
            target=d.get("target", "layout"),
            patch=d.get("patch", {}),
            panel_indexed_axis=d.get("panel_indexed_axis"),
            trace_target=d.get("trace_target", False),
            append_to=d.get("append_to"),
            composite=d.get("composite"),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class OperationRegistry:
    """Registry of named operation definitions."""

    def __init__(self):
        self._ops: dict[str, OperationDef] = {}

    def register(self, op_def: OperationDef | dict) -> None:
        """Register an operation definition."""
        if isinstance(op_def, dict):
            op_def = OperationDef.from_dict(op_def)
        self._ops[op_def.name] = op_def

    def get(self, name: str) -> OperationDef | None:
        return self._ops.get(name)

    def list_operations(self) -> list[str]:
        return sorted(self._ops.keys())

    def register_builtins(self) -> None:
        """Register all built-in operation templates."""
        for op_dict in _BUILTIN_OPERATIONS:
            self.register(op_dict)

    # ------------------------------------------------------------------
    # Resolver
    # ------------------------------------------------------------------

    def resolve(
        self,
        op_dict: dict,
        fig: go.Figure,
        trace_label_map: dict[str, int],
        panel_axis_map: dict[int, str] | None = None,
    ) -> None:
        """Apply a single operation to *fig* in place.

        Args:
            op_dict: Operation instance, e.g. {"op": "set_title", "text": "..."}.
            fig: The Plotly figure to modify.
            trace_label_map: Mapping of trace label/name → index in fig.data.
            panel_axis_map: Mapping of panel number → y-axis key in layout.
                           If None, uses default _panel_to_axis_key.
        """
        op_name = op_dict.get("op")
        if op_name is None:
            return

        op_def = self._ops.get(op_name)
        if op_def is None:
            return  # unknown op — skip silently

        # Extract user params (everything except 'op')
        user_params = {k: v for k, v in op_dict.items() if k != "op"}

        # Composite: expand into sub-operations and resolve each
        if op_def.composite:
            for sub_op_template in op_def.composite:
                sub_op = _substitute_vars(sub_op_template, user_params)
                if isinstance(sub_op, dict):
                    self.resolve(sub_op, fig, trace_label_map, panel_axis_map)
            return

        # Substitute $variables into the patch template
        patch = _substitute_vars(copy.deepcopy(op_def.patch), user_params)

        # --- Apply based on target type ---

        if op_def.target == "trace" or op_def.trace_target:
            self._apply_trace_patch(op_dict, op_def, patch, fig, trace_label_map)

        elif op_def.append_to:
            self._apply_append_patch(op_dict, op_def, patch, user_params, fig)

        elif op_def.panel_indexed_axis:
            self._apply_panel_patch(op_dict, op_def, patch, fig, panel_axis_map)

        else:
            # Plain layout merge
            fig.update_layout(**patch)

    def _apply_trace_patch(
        self,
        op_dict: dict,
        op_def: OperationDef,
        patch: dict,
        fig: go.Figure,
        trace_label_map: dict[str, int],
    ) -> None:
        """Apply a patch to one or more traces."""
        trace_selector = op_dict.get("trace")
        if trace_selector is None:
            # Apply to all traces
            for trace in fig.data:
                trace.update(**patch)
            return

        # Find the matching trace index
        idx = trace_label_map.get(trace_selector)
        if idx is not None and 0 <= idx < len(fig.data):
            fig.data[idx].update(**patch)
        else:
            # Try substring matching as fallback
            for label, i in trace_label_map.items():
                if trace_selector in label or label in trace_selector:
                    if 0 <= i < len(fig.data):
                        fig.data[i].update(**patch)
                    break

    def _apply_panel_patch(
        self,
        op_dict: dict,
        op_def: OperationDef,
        patch: dict,
        fig: go.Figure,
        panel_axis_map: dict[int, str] | None,
    ) -> None:
        """Apply a patch to a panel-indexed axis."""
        panel = op_dict.get("panel")
        if panel is None:
            panel = 1

        axis_prefix = op_def.panel_indexed_axis  # e.g. 'yaxis', 'xaxis'

        if panel_axis_map and panel in panel_axis_map:
            axis_key = panel_axis_map[panel]
        else:
            axis_key = _panel_to_axis_key(panel, axis_prefix)

        fig.update_layout(**{axis_key: patch})

    def _apply_append_patch(
        self,
        op_dict: dict,
        op_def: OperationDef,
        patch: dict,
        user_params: dict,
        fig: go.Figure,
    ) -> None:
        """Append items to layout array fields (shapes, annotations)."""
        for array_key in op_def.append_to:
            items = patch.get(array_key)
            if items is None:
                continue
            if not isinstance(items, list):
                items = [items]

            current = list(fig.layout[array_key] or [])
            current.extend(items)
            fig.update_layout(**{array_key: current})


# ---------------------------------------------------------------------------
# Built-in operation definitions
# ---------------------------------------------------------------------------

_BUILTIN_OPERATIONS: list[dict] = [
    # === Layout ===
    {
        "name": "set_title",
        "description": "Set the plot title",
        "params": {
            "text": {"type": "string", "description": "Title text"},
        },
        "target": "layout",
        "patch": {"title": {"text": "$text"}},
    },
    {
        "name": "set_x_label",
        "description": "Set the x-axis label (shared across panels)",
        "params": {
            "text": {"type": "string", "description": "X-axis label text"},
        },
        "target": "layout",
        "patch": {"xaxis": {"title": {"text": "$text"}}},
    },
    {
        "name": "set_y_label",
        "description": "Set the y-axis label for a specific panel",
        "params": {
            "panel": {"type": "int", "description": "Panel number (1-indexed)"},
            "text": {"type": "string", "description": "Y-axis label text"},
        },
        "target": "layout",
        "panel_indexed_axis": "yaxis",
        "patch": {"title": {"text": "$text"}},
    },
    {
        "name": "set_theme",
        "description": "Set the Plotly template/theme",
        "params": {
            "theme": {"type": "string", "description": "Plotly template name"},
        },
        "target": "layout",
        "patch": {"template": "$theme"},
    },
    {
        "name": "set_font_size",
        "description": "Set the global font size",
        "params": {
            "size": {"type": "int", "description": "Font size in points"},
        },
        "target": "layout",
        "patch": {"font": {"size": "$size"}},
    },
    {
        "name": "set_canvas_size",
        "description": "Set figure width and height",
        "params": {
            "width": {"type": "int", "description": "Width in pixels"},
            "height": {"type": "int", "description": "Height in pixels"},
        },
        "target": "layout",
        "patch": {"width": "$width", "height": "$height"},
    },
    {
        "name": "set_legend",
        "description": "Show or hide the legend, and configure legend properties",
        "params": {
            "show": {"type": "bool", "description": "Show/hide legend"},
        },
        "target": "layout",
        "patch": {"showlegend": "$show"},
    },
    {
        "name": "set_x_range",
        "description": "Set the x-axis range (shared across panels via shared_xaxes)",
        "params": {
            "range": {"type": "array", "description": "[min, max] values"},
        },
        "target": "layout",
        "patch": {"xaxis": {"range": "$range"}},
    },
    {
        "name": "set_y_range",
        "description": "Set the y-axis range for a specific panel",
        "params": {
            "panel": {"type": "int", "description": "Panel number (1-indexed)"},
            "range": {"type": "array", "description": "[min, max] values"},
        },
        "target": "layout",
        "panel_indexed_axis": "yaxis",
        "patch": {"range": "$range"},
    },
    {
        "name": "set_y_scale",
        "description": "Set the y-axis scale type for a specific panel",
        "params": {
            "panel": {"type": "int", "description": "Panel number (1-indexed)"},
            "scale": {"type": "string", "description": "'log' or 'linear'"},
        },
        "target": "layout",
        "panel_indexed_axis": "yaxis",
        "patch": {"type": "$scale"},
    },
    {
        "name": "set_margin",
        "description": "Set figure margins",
        "params": {
            "l": {"type": "int", "description": "Left margin (px)"},
            "r": {"type": "int", "description": "Right margin (px)"},
            "t": {"type": "int", "description": "Top margin (px)"},
            "b": {"type": "int", "description": "Bottom margin (px)"},
        },
        "target": "layout",
        "patch": {"margin": {"l": "$l", "r": "$r", "t": "$t", "b": "$b"}},
    },

    # === Traces ===
    {
        "name": "set_trace_color",
        "description": "Set the color of a trace by label",
        "params": {
            "trace": {"type": "string", "description": "Trace label to match"},
            "color": {"type": "string", "description": "CSS color string"},
        },
        "target": "trace",
        "trace_target": True,
        "patch": {"line": {"color": "$color"}},
    },
    {
        "name": "set_line_style",
        "description": "Set line width and dash style for a trace",
        "params": {
            "trace": {"type": "string", "description": "Trace label to match"},
            "width": {"type": "number", "description": "Line width"},
            "dash": {"type": "string", "description": "Dash style: solid, dash, dot, dashdot"},
        },
        "target": "trace",
        "trace_target": True,
        "patch": {"line": {"width": "$width", "dash": "$dash"}},
    },
    {
        "name": "set_line_mode",
        "description": "Set the trace rendering mode",
        "params": {
            "trace": {"type": "string", "description": "Trace label to match"},
            "mode": {"type": "string", "description": "'lines', 'markers', 'lines+markers'"},
        },
        "target": "trace",
        "trace_target": True,
        "patch": {"mode": "$mode"},
    },
    {
        "name": "set_trace_visibility",
        "description": "Show or hide a trace",
        "params": {
            "trace": {"type": "string", "description": "Trace label to match"},
            "visible": {"type": "bool", "description": "True, False, or 'legendonly'"},
        },
        "target": "trace",
        "trace_target": True,
        "patch": {"visible": "$visible"},
    },

    # === Decorations (append mode) ===
    {
        "name": "add_vline",
        "description": "Add a vertical line across all panels",
        "params": {
            "x": {"type": "string", "description": "X position (timestamp)"},
            "color": {"type": "string", "description": "Line color", "default": "red"},
            "width": {"type": "number", "description": "Line width", "default": 1.5},
            "dash": {"type": "string", "description": "Dash style", "default": "solid"},
            "label": {"type": "string", "description": "Optional annotation text"},
        },
        "target": "layout",
        "append_to": ["shapes", "annotations"],
        "patch": {
            "shapes": [
                {
                    "type": "line",
                    "x0": "$x", "x1": "$x",
                    "y0": 0, "y1": 1,
                    "xref": "x", "yref": "paper",
                    "line": {"color": "$color", "width": "$width", "dash": "$dash"},
                }
            ],
            "annotations": [
                {
                    "x": "$x", "y": 1.02,
                    "xref": "x", "yref": "paper",
                    "text": "$label",
                    "showarrow": False,
                    "font": {"size": 11, "color": "$color"},
                }
            ],
        },
    },
    {
        "name": "add_hline",
        "description": "Add a horizontal line to a panel",
        "params": {
            "y": {"type": "number", "description": "Y position"},
            "panel": {"type": "int", "description": "Panel number (1-indexed)", "default": 1},
            "color": {"type": "string", "description": "Line color", "default": "gray"},
            "width": {"type": "number", "description": "Line width", "default": 1},
            "dash": {"type": "string", "description": "Dash style", "default": "dash"},
        },
        "target": "layout",
        "append_to": ["shapes"],
        "patch": {
            "shapes": [
                {
                    "type": "line",
                    "x0": 0, "x1": 1,
                    "y0": "$y", "y1": "$y",
                    "xref": "paper", "yref": "y",
                    "line": {"color": "$color", "width": "$width", "dash": "$dash"},
                }
            ],
        },
    },
    {
        "name": "add_vrect",
        "description": "Add a highlighted vertical region",
        "params": {
            "x0": {"type": "string", "description": "Start x (timestamp)"},
            "x1": {"type": "string", "description": "End x (timestamp)"},
            "color": {"type": "string", "description": "Fill color", "default": "rgba(135,206,250,0.3)"},
            "opacity": {"type": "number", "description": "Fill opacity", "default": 0.3},
            "label": {"type": "string", "description": "Optional annotation text"},
        },
        "target": "layout",
        "append_to": ["shapes", "annotations"],
        "patch": {
            "shapes": [
                {
                    "type": "rect",
                    "x0": "$x0", "x1": "$x1",
                    "y0": 0, "y1": 1,
                    "xref": "x", "yref": "paper",
                    "fillcolor": "$color",
                    "opacity": "$opacity",
                    "line": {"width": 0},
                    "layer": "below",
                }
            ],
            "annotations": [
                {
                    "x": "$x0", "y": 1.02,
                    "xref": "x", "yref": "paper",
                    "text": "$label",
                    "showarrow": False,
                    "font": {"size": 11},
                }
            ],
        },
    },
    {
        "name": "add_annotation",
        "description": "Add a text annotation to the plot",
        "params": {
            "text": {"type": "string", "description": "Annotation text"},
            "x": {"type": "any", "description": "X position"},
            "y": {"type": "any", "description": "Y position"},
            "showarrow": {"type": "bool", "description": "Show arrow", "default": True},
        },
        "target": "layout",
        "append_to": ["annotations"],
        "patch": {
            "annotations": [
                {
                    "text": "$text",
                    "x": "$x",
                    "y": "$y",
                    "showarrow": "$showarrow",
                }
            ],
        },
    },
    {
        "name": "add_shape",
        "description": "Add a generic shape to the plot",
        "params": {
            "type": {"type": "string", "description": "Shape type: line, rect, circle"},
            "x0": {"type": "any", "description": "Start x"},
            "y0": {"type": "any", "description": "Start y"},
            "x1": {"type": "any", "description": "End x"},
            "y1": {"type": "any", "description": "End y"},
            "line_color": {"type": "string", "description": "Line color", "default": "black"},
            "line_width": {"type": "number", "description": "Line width", "default": 1},
            "fillcolor": {"type": "string", "description": "Fill color", "default": ""},
        },
        "target": "layout",
        "append_to": ["shapes"],
        "patch": {
            "shapes": [
                {
                    "type": "$type",
                    "x0": "$x0", "y0": "$y0",
                    "x1": "$x1", "y1": "$y1",
                    "line": {"color": "$line_color", "width": "$line_width"},
                    "fillcolor": "$fillcolor",
                }
            ],
        },
    },

    # === Spectrogram-specific ===
    {
        "name": "set_colorscale",
        "description": "Set the colorscale for heatmap/spectrogram traces",
        "params": {
            "colorscale": {"type": "string", "description": "Plotly colorscale name"},
        },
        "target": "trace",
        "trace_target": True,
        "patch": {"colorscale": "$colorscale"},
    },
    {
        "name": "set_z_range",
        "description": "Set the z-axis (color) range for heatmap/spectrogram traces",
        "params": {
            "trace": {"type": "string", "description": "Trace label to match"},
            "zmin": {"type": "number", "description": "Min z value"},
            "zmax": {"type": "number", "description": "Max z value"},
        },
        "target": "trace",
        "trace_target": True,
        "patch": {"zmin": "$zmin", "zmax": "$zmax"},
    },
    {
        "name": "set_colorbar_title",
        "description": "Set the colorbar title for a heatmap/spectrogram trace",
        "params": {
            "trace": {"type": "string", "description": "Trace label to match"},
            "text": {"type": "string", "description": "Colorbar title text"},
        },
        "target": "trace",
        "trace_target": True,
        "patch": {"colorbar": {"title": {"text": "$text"}}},
    },

    # === Composite ===
    {
        "name": "style_publication",
        "description": "Apply publication-ready styling: white theme, larger font, larger canvas",
        "params": {
            "font_size": {"type": "int", "description": "Font size", "default": 14},
            "width": {"type": "int", "description": "Width", "default": 1200},
            "height": {"type": "int", "description": "Height", "default": 800},
        },
        "composite": [
            {"op": "set_theme", "theme": "plotly_white"},
            {"op": "set_font_size", "size": "$font_size"},
            {"op": "set_canvas_size", "width": "$width", "height": "$height"},
            {"op": "set_legend", "show": True},
        ],
    },
]


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_default_registry: OperationRegistry | None = None


def get_default_registry() -> OperationRegistry:
    """Return the singleton default registry with built-in operations."""
    global _default_registry
    if _default_registry is None:
        _default_registry = OperationRegistry()
        _default_registry.register_builtins()
    return _default_registry
