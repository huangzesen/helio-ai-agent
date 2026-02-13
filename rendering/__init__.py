"""Rendering backend and tool registry."""

from .plotly_renderer import (
    PlotlyRenderer,
    ColorState,
    RenderResult,
    build_figure_from_spec,
)
from .operations import OperationRegistry, get_default_registry
from .registry import TOOLS, get_method, validate_args, render_method_catalog

__all__ = [
    "PlotlyRenderer",
    "ColorState",
    "RenderResult",
    "build_figure_from_spec",
    "OperationRegistry",
    "get_default_registry",
    "TOOLS",
    "get_method",
    "validate_args",
    "render_method_catalog",
]
