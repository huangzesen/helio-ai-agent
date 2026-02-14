# Plan: Undo/Redo for Plot Operations

## Status: Future — blocked on JSON spec maturity

## Motivation

Users frequently want to revert a plot change ("undo that", "go back to how it was"). This is an imperative operation that doesn't fit the `update_plot_spec` paradigm — you'd need to remember what the previous spec was.

## Design

### Core Idea
Maintain a stack of plot states (spec + figure JSON) so `manage_plot(action="undo")` can restore the previous state.

### Components

1. **State stack in PlotlyRenderer**
   - `_state_history: list[dict]` — list of `{"spec": dict, "figure_json": str}` snapshots
   - Max depth configurable (default ~20 to bound memory)
   - Push on every successful `render_from_spec()` and `style()` call
   - `undo()` pops the stack and restores the previous figure + spec
   - `redo()` uses a separate forward stack (cleared on new changes)

2. **manage_plot actions**
   - `action="undo"` — revert to previous state
   - `action="redo"` — re-apply undone change
   - Return the restored spec so the LLM knows what state it's in

3. **Spec tracking**
   - Currently `_current_plot_spec` tracks the latest spec
   - Undo needs to also restore this, so the diff logic in `_handle_update_plot_spec` works correctly against the restored state

### Open Questions

- **Memory cost**: Each snapshot stores a full Plotly figure JSON. For multi-panel plots with large datasets this could be significant. Options:
  - Store only the spec (re-render on undo — slower but memory-efficient)
  - Store figure JSON (instant undo but memory-heavy)
  - Hybrid: store spec + figure for last N, spec-only beyond that
- **Session persistence**: Should undo history survive session save/restore?
- **Granularity**: Is every `render_plotly_json` call one undo step, or should related calls be grouped?

### Prerequisites

- JSON spec implementation needs to be mature and stable first
- The spec must fully capture all plot state (currently it does for layout + style, but edge cases like auto-computed y-ranges, trace ordering, etc. may not round-trip perfectly)
- `_current_plot_spec` must be the single source of truth for plot state

### Estimated Scope

~150-200 lines across:
- `rendering/plotly_renderer.py` — state stack, undo/redo methods
- `agent/core.py` — dispatch for undo/redo actions in `_handle_manage_plot`
- `agent/tools.py` — add undo/redo to manage_plot action enum
- `rendering/registry.py` — update manage_plot action enum
- `knowledge/prompt_builder.py` — update viz agent prompt with undo guidance
