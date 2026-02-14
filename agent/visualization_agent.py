"""
Visualization sub-agent with optional think phase.

For plot-creation requests, runs a think phase to inspect data (shapes,
types, units, NaN counts) before constructing render_plotly_json calls.
Style requests skip the think phase to avoid wasting tokens.

Owns all visualization through three tools:
- render_plotly_json — create/update plots via Plotly figure JSON with data_label placeholders
- manage_plot — export, reset, zoom, get state, add/remove traces
- list_fetched_data — discover available data in memory
"""

import re

from .llm import LLMAdapter, FunctionSchema
from .base_agent import BaseSubAgent
from .tasks import Task
from .tools import get_tool_schemas
from .tool_loop import run_tool_loop, extract_text_from_response
from .logging import tagged
from .model_fallback import get_active_model
from knowledge.prompt_builder import (
    build_visualization_prompt,
    build_visualization_think_prompt,
)

# Visualization agent gets its own tool category + list_fetched_data from data_ops
# render_plotly_json and manage_plot are exposed; plot_data and style_plot are
# excluded (legacy primitives superseded by render_plotly_json).
VIZ_TOOL_CATEGORIES = ["visualization"]
VIZ_EXTRA_TOOLS = ["list_fetched_data", "manage_plot"]

# Think phase: data inspection only (no viz tools)
VIZ_THINK_EXTRA_TOOLS = ["list_fetched_data", "describe_data", "preview_data"]

# Keywords for the skip heuristic
_STYLE_MANAGE_KEYWORDS = {
    "title", "zoom", "log scale", "linear scale", "color", "colour",
    "font", "legend", "annotation", "vline", "vrect",
    "bigger", "smaller", "resize", "canvas", "width", "height",
    "theme", "dark", "export", "reset", "get_state",
    "remove trace", "set time", "time range", "dash", "line style",
}

_PLOT_KEYWORDS = {
    "plot", "show", "display", "visualize", "create", "draw",
    "compare", "panel", "spectrogram", "overlay", "side by side",
}


def _extract_labels_from_instruction(instruction: str) -> list[str]:
    """Extract data labels from a task instruction that has store contents appended.

    The orchestrator appends lines like "  - AC_H0_MFI.Magnitude (37800 pts)"
    to the instruction. This extracts the label portion.
    """
    labels = []
    for match in re.finditer(r"^\s+-\s+(\S+)\s+\(", instruction, re.MULTILINE):
        labels.append(match.group(1))
    return labels


class VisualizationAgent(BaseSubAgent):
    """An LLM session specialized for visualization.

    Uses three tools: render_plotly_json (create/update plots via Plotly
    figure JSON with data_label placeholders), manage_plot (export, reset,
    zoom, add/remove traces), and list_fetched_data (discover available data).

    For plot-creation requests via process_request(), runs an optional
    think phase to inspect data before the execute phase.
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        model_name: str,
        tool_executor,
        verbose: bool = False,
        gui_mode: bool = False,
        cancel_event=None,
        pitfalls: list[str] | None = None,
        token_log_path=None,
    ):
        self.gui_mode = gui_mode
        super().__init__(
            adapter=adapter,
            model_name=model_name,
            tool_executor=tool_executor,
            verbose=verbose,
            agent_name="Visualization Agent",
            system_prompt=build_visualization_prompt(gui_mode=gui_mode),
            tool_categories=VIZ_TOOL_CATEGORIES,
            extra_tool_names=VIZ_EXTRA_TOOLS,
            cancel_event=cancel_event,
            pitfalls=pitfalls,
            token_log_path=token_log_path,
        )

        # Build think-phase tool schemas (data inspection only)
        self._think_tool_schemas: list[FunctionSchema] = []
        for tool_schema in get_tool_schemas(
            categories=[],
            extra_names=VIZ_THINK_EXTRA_TOOLS,
        ):
            self._think_tool_schemas.append(FunctionSchema(
                name=tool_schema["name"],
                description=tool_schema["description"],
                parameters=tool_schema["parameters"],
            ))

    def _needs_think_phase(self, request: str) -> bool:
        """Decide whether a request needs data inspection before plotting.

        Plot-creation requests benefit from inspecting data shapes and types.
        Style/manage requests (title changes, zoom, export) do not.

        Returns True if ambiguous (conservative — better to over-inspect).
        """
        req_lower = request.lower()
        has_plot_signal = any(kw in req_lower for kw in _PLOT_KEYWORDS)
        has_style_signal = any(kw in req_lower for kw in _STYLE_MANAGE_KEYWORDS)
        # If only style signals, skip. If any plot signal (or ambiguous), run think.
        if has_style_signal and not has_plot_signal:
            return False
        return True

    def _run_think_phase(self, user_request: str) -> str:
        """Inspect data in memory before creating a visualization.

        Creates an ephemeral chat session with data inspection tools.
        Runs a tool-calling loop to explore data shapes, types, and values,
        then returns a text summary of findings.

        Args:
            user_request: The user's visualization request.

        Returns:
            Text summary of data inspection findings (shapes, panel layout
            recommendations, sizing hints).
        """
        think_prompt = build_visualization_think_prompt()

        self.logger.debug("[Viz] Think phase: inspecting data...")

        chat = self.adapter.create_chat(
            model=get_active_model(self.model_name),
            system_prompt=think_prompt,
            tools=self._think_tool_schemas,
            thinking="high",
        )

        self._last_tool_context = "viz_think_initial"
        response = self._send_with_timeout(chat, user_request)
        self._track_usage(response)

        response = run_tool_loop(
            chat=chat,
            response=response,
            tool_executor=self.tool_executor,
            agent_name="Viz/Think",
            max_total_calls=10,
            max_iterations=4,
            track_usage=self._track_usage,
            cancel_event=self._cancel_event,
            send_fn=lambda msg: self._send_with_timeout(chat, msg),
            adapter=self.adapter,
        )

        text = extract_text_from_response(response)
        if self.verbose and text:
            self.logger.debug(f"[Viz] Think result: {text[:500]}")

        self.logger.debug("[Viz] Think phase complete", extra=tagged("progress"))
        return text or ""

    def process_request(self, user_message: str) -> dict:
        """Conditionally run think→execute for plot-creation requests.

        Style/manage requests skip the think phase and go straight to
        the execute phase (standard BaseSubAgent.process_request).

        Args:
            user_message: The user's visualization request.

        Returns:
            Dict with text, failed, errors (same as BaseSubAgent.process_request).
        """
        if self._needs_think_phase(user_message):
            think_context = self._run_think_phase(user_message)
            if think_context:
                enriched = (
                    f"{user_message}\n\n"
                    f"## Data Inspection Findings\n{think_context}\n\n"
                    f"Now create the visualization using render_plotly_json."
                )
            else:
                enriched = user_message
        else:
            self.logger.debug("[Viz] Skipping think phase (style/manage request)")
            enriched = user_message

        return super().process_request(enriched)

    def _get_task_prompt(self, task: Task) -> str:
        """Build an explicit task prompt with concrete label values.

        Extracts actual data labels from the instruction (injected by
        _execute_plan_task) and constructs the exact render_plotly_json call
        so Gemini Flash sees the precise command to execute.

        Note: Export tasks are handled directly by the orchestrator and
        never reach this method.
        """
        labels = _extract_labels_from_instruction(task.instruction)

        pitfall_section = ""
        if self._pitfalls:
            pitfall_section = (
                "\n\nVisualization operational knowledge:\n"
                + "".join(f"- {p}\n" for p in self._pitfalls)
            )

        no_labels = not labels

        if labels:
            # Build a simple Plotly JSON example with the extracted labels
            traces_example = ", ".join(
                f'{{"type": "scatter", "data_label": "{lbl}"}}'
                for lbl in labels
            )
            first_call = (
                f'render_plotly_json(figure_json={{"data": [{traces_example}], '
                f'"layout": {{}}}})'
            )
        else:
            first_call = "render_plotly_json with the appropriate labels"

        task_prompt = (
            f"Execute this task: {task.instruction}\n\n"
            f"Your FIRST call must be: {first_call}\n\n"
            "RULES:\n"
            "- Call render_plotly_json with the labels shown above.\n"
            "- After plotting, inspect review.sizing_recommendation and adjust\n"
            "  layout width/height if it differs from review.figure_size.\n"
            "- Use manage_plot for export, reset, zoom, or trace operations if needed."
            + pitfall_section
        )

        if no_labels:
            task_prompt += "\n\nNote: Labels were not pre-extracted. Call list_fetched_data first to discover available labels.\n"

        return task_prompt
