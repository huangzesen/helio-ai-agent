"""Rendering backend and method registry."""

from .plotly_renderer import PlotlyRenderer
from .registry import METHODS, get_method, validate_args, render_method_catalog

__all__ = [
    "PlotlyRenderer",
    "METHODS",
    "get_method",
    "validate_args",
    "render_method_catalog",
]
