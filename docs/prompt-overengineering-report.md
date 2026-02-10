# Prompt Over-Engineering Report

*Generated: 2026-02-09*
*Scope: All system prompts, tool descriptions, and prompt builders*

---

## Summary

The prompt system wastes an estimated **600-1000 extra tokens per orchestrator call** and **~300 per sub-agent call** due to duplicated instructions, redundant tool descriptions, and defensive rules that are already enforced by tool filtering. Over a multi-turn session with planning, this adds up to several thousand unnecessary tokens.

9 findings, 3 high severity.

---

## HIGH — Massive Duplication

### 1. Computation examples appear 3 times

The same pandas/numpy code examples (magnitude, smoothing, resample, derivative, normalize, clip, log, interpolate, detrend, etc.) appear in:

| Location | File | Lines |
|---|---|---|
| `custom_operation` tool description | `agent/tools.py` | 196-210 |
| DataOps agent system prompt | `knowledge/prompt_builder.py` | 370-382 |
| DataExtraction agent system prompt | `knowledge/prompt_builder.py` | 466-486 (subset) |

The DataOps agent receives **both** the tool schema description AND the system prompt, so it sees the same examples **twice** in its context window. ~300 wasted tokens per DataOps call.

Similarly, spectrogram patterns are duplicated between `tools.py` (lines 302-308) and `build_data_ops_prompt()` (lines 389-400).

**Fix:** Keep examples only in the system prompt (where they have context). Strip them from tool descriptions, which should be concise.

---

### 2. Visualization tools described in 3 layers

The 3 visualization tools (`plot_data`, `style_plot`, `manage_plot`) are fully described in:

1. **`rendering/registry.py`** — source-of-truth registry with parameters, descriptions, and examples (rendered via `render_method_catalog()`)
2. **`agent/tools.py`** (lines 432-590) — full Gemini function-calling schemas duplicating all parameter definitions
3. **`prompt_builder.py` `build_visualization_prompt()`** (lines 543-611) — renders the registry as markdown AND manually adds more examples on top

The visualization agent sees all three: Gemini tool schemas + rendered registry catalog + hand-written examples. The `render_method_catalog()` already generates examples (registry.py lines 174-190), then `build_visualization_prompt()` adds yet more examples manually (lines 547-577). ~400 wasted tokens per visualization call.

**Fix:** Remove the hand-written examples from `build_visualization_prompt()` — the registry catalog already has them. Alternatively, stop rendering the catalog into the system prompt and rely solely on the Gemini tool schemas.

---

### 3. Redundant routing "Do NOT" guidance (3+ locations)

Every routing tool in `tools.py` has extensive "Do NOT delegate" lists telling the LLM what the *other* agents handle. This same information also appears in the orchestrator system prompt under:
- "Workflow" section (lines 677-686)
- "After Data Delegation" section (lines 688-700)
- Each `delegate_to_*` tool description

For example, "don't delegate visualization to mission agents" is stated in the orchestrator workflow section, the `delegate_to_mission` tool's "Do NOT delegate" list, AND the `delegate_to_visualization` tool's description. ~200 wasted tokens per orchestrator call.

**Fix:** Keep routing guidance in one place — either the system prompt OR the tool descriptions, not both.

---

## MEDIUM — Repeated Instructions

### 4. Time range format stated 5 times

The instruction "Use `YYYY-MM-DD to YYYY-MM-DD` format, do NOT use `/` as separator" appears in:

1. `build_system_prompt()` — lines 707-714 (orchestrator prompt)
2. `build_mission_prompt()` — line 323 (mission agent prompt)
3. `build_visualization_prompt()` — lines 579-583 (visualization agent prompt)
4. `tools.py` line 169 (`fetch_data` parameter description)
5. `tools.py` line 577 (`manage_plot` `time_range` parameter description)

**Fix:** Keep only in tool parameter descriptions (where the LLM actually constructs the argument). Remove from system prompts.

---

### 5. Sub-agent negative boundary rules are redundant with tool filtering

Every sub-agent prompt ends with a block of "Do NOT" rules:

- **Mission agent** (lines 333-335): "Do NOT attempt data transformations... Do NOT attempt to plot..."
- **DataOps agent** (lines 426-429): "Do NOT use save_data unless... Do NOT fetch... Do NOT plot... Do NOT create DataFrames..."
- **DataExtraction agent** (lines 506-508): "Do NOT fetch... Do NOT plot... Do NOT compute..."
- **Visualization agent** (lines 591, 614-615): "Do NOT call manage_plot(export)... Always use fetch_data first..."

These agents already can't perform the forbidden actions because `get_tool_schemas(categories=...)` filters tools by category — a mission agent literally doesn't have `plot_data` in its tool list. The "Do NOT" rules add ~50-100 tokens per sub-agent call for no behavioral effect.

**Fix:** Remove most negative boundary rules. Keep only rules that restrict *available* tools (e.g., "Do NOT use save_data unless explicitly requested" in DataOps, since save_data IS in its tool set).

---

### 6. Keyword-to-type classification logic duplicated in code

The keyword-to-data-type classification (magnetic/plasma/particles/electric/waves/indices/ephemeris) is implemented as near-identical if/elif chains in:

1. `generate_dataset_quick_reference()` — lines 66-81
2. `generate_planner_dataset_reference()` — lines 98-113

Both manually classify instruments by checking keyword lists. Any change to classification rules needs to be made in two places.

**Fix:** Extract a `classify_instrument(keywords) -> str` helper function used by both generators.

---

## LOW — Minor Bloat

### 7. Planner prompt's 3-round JSON example is very long

`build_planner_agent_prompt()` includes a full 3-round JSON example spanning lines 937-956 (~500 tokens). The response schema (`PLANNER_RESPONSE_SCHEMA`) already enforces the JSON structure via `response_mime_type="application/json"` and `response_schema=...`.

**Fix:** Shorten to a single 1-round example. The schema enforcement handles structure; the example only needs to show content conventions.

---

### 8. `is_complex_request()` has 24 fragile regex patterns

`planner.py` lines 33-67 contain 24 regex patterns including:
- 6 patterns for pairwise spacecraft name detection (PSP-ACE, PSP-WIND, ACE-WIND, etc.)
- `r"\bthen\b"` and `r"\bfinally\b"` — match conversational English ("I finally got the data working")
- `r"\band\b.*\band\b"` — matches any sentence with two "and"s

These are fragile heuristics with false positives. The orchestrator already has `request_planning` as a tool the LLM can choose to call.

**Fix:** Simplify to 5-8 high-signal patterns (explicit "compare", 2+ spacecraft names, "then"+"and" combos), or remove entirely and let the LLM's `request_planning` tool handle routing.

---

### 9. `rendering/registry.py` parallels `tools.py` for 3 tools

The same 3 visualization tools exist in two different schema formats:
- `rendering/registry.py` — list-of-dicts with `validate_args()` and `render_method_catalog()`
- `agent/tools.py` — Gemini-compatible schema format

The `validate_args()` function in `registry.py` is never used to gate tool calls (the renderer validates internally). The registry only exists to render the catalog markdown for the visualization prompt.

**Fix:** Either generate the visualization tool schemas in `tools.py` from the registry (single source of truth), or delete the registry and put the catalog rendering in the prompt builder.

---

## Token Cost Summary

| Issue | Est. Wasted Tokens/Call | Affects |
|---|---|---|
| Duplicated computation examples (#1) | ~300 | Every DataOps agent call |
| Visualization tools described 3x (#2) | ~400 | Every visualization agent call |
| Redundant routing guidance (#3) | ~200 | Every orchestrator call |
| Time format repeated 5x (#4) | ~100 | Every orchestrator + sub-agent call |
| Sub-agent negative boundary rules (#5) | ~50-100 | Every sub-agent call |
| Long planner example (#7) | ~500 | Every planning call |
| **Total per multi-step session (est.)** | **~3000-5000** | |

---

## Recommended Fixes (by effort)

### Quick wins (30 min)
1. Remove computation examples from `tools.py` tool descriptions — keep only in DataOps system prompt
2. Remove hand-written examples from `build_visualization_prompt()` — registry catalog already has them
3. Remove time format instructions from system prompts — keep only in tool param descriptions
4. Remove negative boundary rules for tools the sub-agents don't have

### Medium effort (1-2 hours)
5. Consolidate routing "Do NOT" guidance into one location
6. Extract `classify_instrument()` helper for keyword-to-type mapping
7. Shorten planner 3-round example to 1-round

### Larger refactor (2-4 hours)
8. Unify `registry.py` and `tools.py` visualization tool definitions (single source of truth)
9. Simplify or remove `is_complex_request()` regex patterns
