"""Pluggable rendering backend — pure-Python alternative to Autoplot canvas."""

from .plotly_renderer import PlotlyRenderer

__all__ = ["PlotlyRenderer"]
