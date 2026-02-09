"""Rendering backend and tool registry."""

from .plotly_renderer import PlotlyRenderer
from .registry import TOOLS, get_method, validate_args, render_method_catalog

__all__ = [
    "PlotlyRenderer",
    "TOOLS",
    "get_method",
    "validate_args",
    "render_method_catalog",
]
