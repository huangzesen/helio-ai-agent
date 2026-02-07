"""
Dynamic prompt generation from the spacecraft catalog.

Generates prompt sections for the agent system prompt and planner prompt
from the single source of truth in catalog.py and per-mission JSON files.

The main agent gets a slim routing table (no dataset IDs or analysis tips).
Mission sub-agents get rich, focused prompts with full domain knowledge.
"""

from .catalog import SPACECRAFT
from .mission_loader import load_mission, load_all_missions, get_routing_table, get_mission_datasets
from .hapi_client import list_parameters as _list_parameters
from autoplot_bridge.registry import render_method_catalog


# ---------------------------------------------------------------------------
# Section generators — each produces a markdown string
# ---------------------------------------------------------------------------

def generate_spacecraft_overview() -> str:
    """Generate the spacecraft/instruments/example-data table for the system prompt.

    Kept for backward compatibility but now only used in the slim system prompt.
    """
    lines = [
        "| Spacecraft | Instruments | Example Data |",
        "|------------|-------------|--------------|",
    ]
    for sc_id, sc in SPACECRAFT.items():
        name = sc["name"]
        instruments = ", ".join(
            inst["name"] for inst in sc["instruments"].values()
        )
        # Summarise from profile if available, else from instrument keywords
        profile = sc.get("profile", {})
        example = profile.get("description", "")
        if not example:
            # Fallback: first two instrument keywords
            all_kw = []
            for inst in sc["instruments"].values():
                all_kw.extend(inst["keywords"][:2])
            example = ", ".join(dict.fromkeys(all_kw))  # unique, ordered
        # Truncate to keep table readable
        if len(example) > 60:
            example = example[:57] + "..."
        lines.append(f"| {name} ({sc_id}) | {instruments} | {example} |")
    return "\n".join(lines)


def generate_dataset_quick_reference() -> str:
    """Generate the known-dataset-ID table for the system prompt.

    Lists dataset IDs and types. Parameter details come from HAPI via
    list_parameters at runtime — not hardcoded here.
    """
    lines = [
        "| Spacecraft | Dataset ID | Type | Notes |",
        "|-----------|------------|------|-------|",
    ]
    for sc_id, sc in SPACECRAFT.items():
        name = sc["name"]
        for inst_id, inst in sc["instruments"].items():
            # Determine type from instrument keywords
            kws = inst["keywords"]
            if any(k in kws for k in ("magnetic", "mag", "b-field")):
                dtype = "Magnetic"
            elif any(k in kws for k in ("plasma", "solar wind", "ion")):
                dtype = "Plasma"
            else:
                dtype = "Combined"
            for ds in inst["datasets"]:
                lines.append(f"| {name} | {ds} | {dtype} | use list_parameters |")
    return "\n".join(lines)


def generate_planner_dataset_reference() -> str:
    """Generate the dataset reference block for the planner prompt.

    Only includes primary-tier datasets from JSON files.
    """
    missions = load_all_missions()
    lines = []
    for mission_id, mission in missions.items():
        parts = []
        for inst_id, inst in mission.get("instruments", {}).items():
            kws = inst.get("keywords", [])
            if any(k in kws for k in ("magnetic", "mag")):
                kind = "magnetic"
            elif any(k in kws for k in ("plasma", "solar wind", "ion")):
                kind = "plasma"
            else:
                kind = "combined"
            for ds_id, ds_info in inst.get("datasets", {}).items():
                if ds_info.get("tier") == "primary":
                    parts.append(f"dataset={ds_id} ({kind})")
        lines.append(f"- {mission['name']}: {'; '.join(parts)}")
    return "\n".join(lines)


def generate_mission_profiles() -> str:
    """Generate detailed per-mission context sections.

    Provides domain knowledge (analysis tips, caveats, coordinate systems)
    that HAPI doesn't supply. Parameter-level metadata (units, descriptions)
    comes from HAPI at runtime via list_parameters.
    """
    sections = []
    for sc_id, sc in SPACECRAFT.items():
        profile = sc.get("profile")
        if not profile:
            continue
        lines = [f"### {sc['name']} ({sc_id})"]
        lines.append(f"{profile['description']}")
        lines.append(f"- Coordinates: {', '.join(profile['coordinate_systems'])}")
        lines.append(f"- Typical cadence: {profile['typical_cadence']}")
        if profile.get("data_caveats"):
            lines.append("- Caveats: " + "; ".join(profile["data_caveats"]))
        if profile.get("analysis_patterns"):
            lines.append("- Analysis tips:")
            for tip in profile["analysis_patterns"]:
                lines.append(f"  - {tip}")
        # List instruments and datasets (details from HAPI)
        for inst_id, inst in sc["instruments"].items():
            ds_list = ", ".join(inst["datasets"])
            lines.append(f"  **{inst['name']}** ({inst_id}): {ds_list}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def generate_routing_table_text() -> str:
    """Generate a slim routing table for the main agent's system prompt.

    Shows mission name and capabilities only — no dataset IDs.
    """
    routing = get_routing_table()
    lines = [
        "| Mission | Capabilities |",
        "|---------|-------------|",
    ]
    for entry in routing:
        caps = ", ".join(entry["capabilities"]) if entry["capabilities"] else "various"
        lines.append(f"| {entry['name']} ({entry['id']}) | {caps} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mission-specific prompt builder (for mission sub-agents)
# ---------------------------------------------------------------------------

def _format_parameter_summary(dataset_id: str, max_params: int = 8) -> str:
    """Generate a compact one-liner of parameter names for a dataset.

    Reads from local cache (instant). Returns empty string if no cache.

    Args:
        dataset_id: CDAWeb dataset ID
        max_params: Maximum number of parameters to show

    Returns:
        Formatted string like "  Parameters: name1 (nT, vector[3]), name2 (km/s)"
        or empty string if parameters can't be loaded.
    """
    try:
        params = _list_parameters(dataset_id)
    except Exception:
        return ""

    if not params:
        return ""

    parts = []
    for p in params[:max_params]:
        name = p["name"]
        units = p.get("units", "")
        size = p.get("size", [1])
        # Build compact descriptor
        desc_parts = []
        if units and units.lower() not in ("null", "none", ""):
            desc_parts.append(units)
        if size and size[0] > 1:
            desc_parts.append(f"vector[{size[0]}]")
        if desc_parts:
            parts.append(f"{name} ({', '.join(desc_parts)})")
        else:
            parts.append(name)

    total = len(params)
    summary = ", ".join(parts)
    if total > max_params:
        summary += f" ({total} total)"

    return f"  Parameters: {summary}"


def build_mission_prompt(mission_id: str) -> str:
    """Generate a rich prompt for a single mission's sub-agent.

    Includes mission overview, analysis patterns, tiered datasets,
    data operations documentation, and workflow instructions.

    Args:
        mission_id: Spacecraft key (e.g., "PSP", "ACE")

    Returns:
        A system prompt focused on one mission's data products and analysis patterns.

    Raises:
        KeyError: If mission_id is not in the catalog.
    """
    # Validate mission exists in catalog (backward compat for KeyError)
    if mission_id not in SPACECRAFT:
        raise KeyError(mission_id)

    mission = load_mission(mission_id)
    profile = mission.get("profile", {})

    lines = [
        f"You are a data specialist agent for {mission['name']} ({mission_id}) data.",
        "",
    ]

    # --- Mission Overview ---
    if profile:
        lines.append("## Mission Overview")
        lines.append(profile.get("description", ""))
        lines.append(f"- Coordinate system(s): {', '.join(profile.get('coordinate_systems', []))}")
        lines.append(f"- Typical cadence: {profile.get('typical_cadence', 'varies')}")
        if profile.get("data_caveats"):
            lines.append("- Data caveats: " + "; ".join(profile["data_caveats"]))
        if profile.get("analysis_patterns"):
            lines.append("\n## Analysis Patterns")
            for tip in profile["analysis_patterns"]:
                lines.append(f"- {tip}")
        lines.append("")

    # --- Primary Datasets ---
    lines.append("## Primary Datasets")
    lines.append("")
    lines.append("Use these datasets by default. Parameter names are listed below — use them directly with fetch_data.")
    lines.append("")
    for inst_id, inst in mission.get("instruments", {}).items():
        lines.append(f"### {inst['name']} ({inst_id})")
        for ds_id, ds_info in inst.get("datasets", {}).items():
            if ds_info.get("tier") == "primary":
                desc = ds_info.get("description", "")
                lines.append(f"- **{ds_id}**: {desc}" if desc else f"- **{ds_id}**")
                # Add parameter summary from local cache
                param_summary = _format_parameter_summary(ds_id)
                if param_summary:
                    lines.append(param_summary)
        lines.append("")

    # --- Advanced Datasets (if any) ---
    advanced = []
    for inst_id, inst in mission.get("instruments", {}).items():
        for ds_id, ds_info in inst.get("datasets", {}).items():
            if ds_info.get("tier") == "advanced":
                desc = ds_info.get("description", "")
                advanced.append((inst["name"], ds_id, desc))
    if advanced:
        lines.append("## Advanced Datasets")
        lines.append("")
        lines.append("Higher-resolution or specialized data. Use when the user requests specific cadences or advanced analysis.")
        lines.append("")
        for inst_name, ds_id, desc in advanced:
            lines.append(f"- **{ds_id}** ({inst_name}): {desc}" if desc else f"- **{ds_id}** ({inst_name})")
        lines.append("")

    # --- Data Operations Documentation ---
    lines.append("## Data Operations Workflow")
    lines.append("")
    lines.append("1. **Identify the dataset**: Match the user's request to a dataset from the primary list above.")
    lines.append("   Parameter names are listed — use them directly with fetch_data.")
    lines.append("2. **Verify if unsure**: Call `list_parameters` to check parameters for any dataset (fast local lookup).")
    lines.append("   If the parameters don't match the user's request, try another dataset.")
    lines.append("3. **`fetch_data`** — Pull data into memory. Label: `DATASET.PARAM`.")
    lines.append("   Time range: '2024-01-15 to 2024-01-20' (use ' to ' separator, NOT '/').")
    lines.append("   Also accepts 'last week', 'January 2024', etc.")
    lines.append("4. **`custom_operation`** — Transform data using pandas/numpy code on `df`, assign to `result`")
    lines.append("5. **`describe_data`** — Get statistics (min, max, mean, std, percentiles, NaN count)")
    lines.append("6. **`save_data`** — Export to CSV with ISO 8601 timestamps")
    lines.append("")
    lines.append("## Reporting Results")
    lines.append("")
    lines.append("After completing data operations, report back with:")
    lines.append("- What data was fetched (labels, time range, number of points)")
    lines.append("- What computations were performed (output labels)")
    lines.append("- A suggestion of what to plot (e.g., \"The data is ready to plot: labels 'ACE_Bmag' and 'ACE_smooth'\")")
    lines.append("")
    lines.append("Do NOT attempt to plot data — plotting is handled by the orchestrator.")
    lines.append("")
    lines.append("### Common Computation Patterns")
    lines.append("")
    lines.append("- **Magnitude**: `result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')`")
    lines.append("- **Smoothing**: `result = df.rolling(60, center=True, min_periods=1).mean()`")
    lines.append("- **Resample**: `result = df.resample('60s').mean().dropna(how='all')`")
    lines.append("- **Rate of change**: `dv = df.diff().iloc[1:]; dt_s = df.index.to_series().diff().dt.total_seconds().iloc[1:]; result = dv.div(dt_s, axis=0)`")
    lines.append("- **Normalize**: `result = (df - df.mean()) / df.std()`")
    lines.append("")
    lines.append("### Code Guidelines")
    lines.append("")
    lines.append("- Always assign to `result` — must be DataFrame/Series with DatetimeIndex")
    lines.append("- Use `df`, `pd`, `np` only — no imports, no file I/O")
    lines.append("- Handle NaN with `skipna=True`, `.dropna()`, or `.fillna()`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Autoplot sub-agent prompt builder
# ---------------------------------------------------------------------------

def build_autoplot_prompt(gui_mode: bool = False) -> str:
    """Generate the system prompt for the Autoplot visualization sub-agent.

    Includes the method catalog from the registry, DOM hierarchy reference,
    render type guidance, and workflow instructions.

    Args:
        gui_mode: If True, append GUI-mode specific instructions.

    Returns:
        System prompt string for the AutoplotAgent.
    """
    catalog = render_method_catalog()

    lines = [
        "You are a visualization specialist for Autoplot, a scientific data visualization tool.",
        "",
        "Your job is to execute Autoplot visualization operations using the `execute_autoplot` tool.",
        "You also have access to `list_fetched_data` to see what data is available in memory.",
        "",
        catalog,
        "## Using execute_autoplot",
        "",
        "Call `execute_autoplot(method=\"method_name\", args={...})` with the method name and arguments from the catalog above.",
        "",
        "Examples:",
        "- Plot stored data: `execute_autoplot(method=\"plot_stored_data\", args={\"labels\": \"ACE_Bmag,PSP_Bmag\", \"title\": \"Comparison\"})`",
        "- Change render: `execute_autoplot(method=\"set_render_type\", args={\"render_type\": \"scatter\"})`",
        "- Export PDF: `execute_autoplot(method=\"export_pdf\", args={\"filename\": \"output.pdf\"})`",
        "- Set canvas: `execute_autoplot(method=\"set_canvas_size\", args={\"width\": 1920, \"height\": 1080})`",
        "",
        "## Render Types",
        "",
        "- **series** (default): Line plot for timeseries data",
        "- **scatter**: Individual points, useful for sparse data",
        "- **spectrogram**: 2D color map for spectral data, requires z-axis",
        "- **fill_to_zero**: Area fill between data and zero line",
        "- **staircase**: Step function, good for discrete/quantized data",
        "- **color_scatter**: Scatter with color encoding a third variable",
        "- **digital**: On/off states, good for flags or status data",
        "- **events_bar**: Event markers on a timeline",
        "",
        "## Color Tables",
        "",
        "Available for spectrograms and color scatter plots:",
        "apl_rainbow_black0, black_blue_green_yellow_white, black_green, black_red,",
        "blue_white_red, color_wedge, grayscale, matlab_jet, rainbow, reverse_rainbow,",
        "wrapped_color_wedge",
        "",
        "## Time Range Format",
        "",
        "- Date range: '2024-01-15 to 2024-01-20' (use ' to ' separator, NOT '/')",
        "- Relative: 'last week', 'last 3 days'",
        "- IMPORTANT: Never use '/' as a date separator.",
        "",
        "## When to Use autoplot_script vs execute_autoplot",
        "",
        "Use **execute_autoplot** (registry methods) for:",
        "- Standard single-panel plotting (plot_cdaweb, plot_stored_data)",
        "- Simple customization (set_title, set_axis_label, toggle_log_scale, set_axis_range)",
        "- Render type and color table changes",
        "- Export (PNG, PDF) and session management",
        "",
        "Use **autoplot_script** (direct DOM/ScriptContext code) for:",
        "- Multi-panel stack plots (sc.plot(0, uri1); sc.plot(1, uri2))",
        "- Per-panel styling (different titles, colors, axis settings per panel)",
        "- Line color/style changes (dom.getPlotElements(i).getStyle().setColor(Color.RED))",
        "- Annotations and advanced layout customization",
        "- Querying plot state (how many panels, what's plotted)",
        "- Anything not covered by the method catalog",
        "",
        "## autoplot_script -- Available Objects",
        "",
        "| Name | Type | Purpose |",
        "|------|------|---------|",
        "| `sc` | ScriptContext | `sc.plot(idx, uri)`, `sc.waitUntilIdle()`, `sc.writeToPng(path)`, `sc.reset()` |",
        "| `dom` | Application | `dom.getPlots(i)`, `dom.getPlotElements(i)`, `dom.setTimeRange(tr)` |",
        "| `Color` | java.awt.Color | `Color.RED`, `Color(r,g,b)`, `Color.getHSBColor(h,s,b)` |",
        "| `RenderType` | org.autoplot.RenderType | `RenderType.series`, `RenderType.scatter` |",
        "| `DatumRangeUtil` | org.das2.datum | `DatumRangeUtil.parseTimeRange('2024-01-01 to 2024-01-07')` |",
        "| `DatumRange` | org.das2.datum | `DatumRange(min, max, units)` |",
        "| `Units` | org.das2.datum | `Units.dimensionless`, `Units.t2000`, `Units.lookupUnits('nT')` |",
        "| `DDataSet`, `QDataSet` | org.das2.qds | Dataset creation and property constants |",
        "| `store` | DataStore | `store.get('label')` -> DataEntry with `.time`, `.values`, `.data` |",
        "",
        "## autoplot_script -- DOM Hierarchy",
        "",
        "```",
        "dom (Application)",
        "  +-- dom.getPlots(i)           -> Plot panel (title, axes)",
        "  |     +-- .getXaxis()          -> time axis",
        "  |     +-- .getYaxis()          -> y-axis (label, range, log)",
        "  |     +-- .getZaxis()          -> z-axis (color table, log)",
        "  +-- dom.getPlotElements(i)     -> Data binding (dataset + render)",
        "  |     +-- .getStyle()          -> line color, symbol, etc.",
        "  |     +-- .setRenderType(rt)   -> RenderType enum",
        "  |     +-- .setComponent('')    -> filter component",
        "  +-- dom.setTimeRange(tr)       -> global time range",
        "```",
        "",
        "## autoplot_script -- Examples",
        "",
        "### Multi-panel stack plot",
        "```python",
        "sc.plot(0, 'vap+cdaweb:ds=AC_H2_MFI&id=Magnitude&timerange=2024-01-01+to+2024-01-07')",
        "sc.plot(1, 'vap+cdaweb:ds=AC_H0_SWE&id=Vp&timerange=2024-01-01+to+2024-01-07')",
        "sc.waitUntilIdle()",
        "```",
        "",
        "### Per-panel customization",
        "```python",
        "dom.getPlots(0).setTitle('Magnetic Field')",
        "dom.getPlots(1).setTitle('Solar Wind Speed')",
        "dom.getPlots(0).getYaxis().setLabel('B (nT)')",
        "dom.getPlots(1).getYaxis().setLabel('V (km/s)')",
        "```",
        "",
        "### Line styling",
        "```python",
        "dom.getPlotElements(0).getStyle().setColor(Color.RED)",
        "dom.getPlotElements(1).getStyle().setColor(Color.BLUE)",
        "```",
        "",
        "### Axis range with units",
        "```python",
        "dr = DatumRange(0, 20, Units.lookupUnits('nT'))",
        "dom.getPlots(0).getYaxis().setRange(dr)",
        "```",
        "",
        "### State inspection",
        "```python",
        "n = dom.getPlots().length",
        "result = f'{n} panels currently showing'",
        "```",
        "",
        "## Workflow",
        "",
        "1. Always call `list_fetched_data` first to see what data is in memory",
        "2. Choose the correct plot method:",
        "   - **plot_stored_data**: for data already fetched/computed (labels from list_fetched_data)",
        "   - **plot_cdaweb**: ONLY for direct CDAWeb visualization when no data is pre-fetched",
        "3. Decision rule: If list_fetched_data shows matching labels, use plot_stored_data",
        "4. Use `autoplot_script` for advanced customization, multi-panel, or styling",
        "5. Chain multiple operations (e.g., plot -> set title -> set axis label -> export)",
        "",
        "## Response Style",
        "",
        "- Confirm what was done after each operation",
        "- If a method fails, explain the error and suggest alternatives",
        "- When plotting, mention the labels and time range shown",
        "",
    ]

    if gui_mode:
        lines.extend([
            "## Interactive GUI Mode",
            "",
            "The Autoplot window is visible to the user. Plots appear immediately in the GUI.",
            "- The user can already see the plot -- do NOT suggest exporting to PNG for viewing",
            "- Changes like zoom, axis labels, log scale, and title are reflected instantly",
            "- Use reset to clear the canvas when starting a new analysis",
            "- Use save_session/load_session to let the user save and restore workspaces",
            "- Say \"The plot is now showing in the Autoplot window\" rather than suggesting export",
            "",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full prompt assemblers
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Assemble the complete system prompt — slim orchestrator version.

    The main agent routes requests to mission sub-agents. It does NOT need
    dataset IDs, analysis tips, or detailed mission profiles.

    Returns a template string with a {today} placeholder for the date.
    """
    routing_table = generate_routing_table_text()

    return f"""You are an intelligent assistant for Autoplot, a scientific data visualization tool for spacecraft and heliophysics data.

## Your Role
Help users visualize spacecraft data by translating natural language requests into Autoplot operations. You orchestrate work by delegating to specialist sub-agents:
- **Mission agents** handle data requests (fetching, computing, describing data)
- **Autoplot agent** handles all visualization (plotting, customizing, exporting)

## Supported Missions

{routing_table}

## Workflow

1. **Identify the mission**: Match the user's request to a spacecraft from the table above
2. **Delegate data requests**: Use `delegate_to_mission` for data operations (fetch, compute, describe, save)
3. **Delegate visualization**: Use `delegate_to_autoplot` for plotting, customizing, exporting, or any visual operation
4. **Multi-mission**: Call `delegate_to_mission` for each mission, then `delegate_to_autoplot` to plot results together

## After Data Delegation

When `delegate_to_mission` returns:
- If the user asked to "show", "plot", or "display" data, use `delegate_to_autoplot` with the labels the specialist reported
- If the specialist only described or saved data, summarize the results without plotting
- Always relay the specialist's findings to the user in your response

## Time Range Handling

All times are in UTC (appropriate for spacecraft data). Accept flexible time inputs — the system parses them to UTC datetimes internally.

Supported formats:
- **Relative**: "last week", "last 3 days", "last month", "last year"
- **Month + year**: "January 2024", "Jan 2024"
- **Single date**: "2024-01-15" — expands to the full day
- **Date range**: "2024-01-15 to 2024-01-20"
- **Datetime range**: "2024-01-15T06:00 to 2024-01-15T18:00" — sub-day precision
- **Single datetime**: "2024-01-15T06:00" — expands to a 1-hour window

If the system returns a time-range parsing error, relay the error message to the user.

Today's date is {{today}}.

## When to Ask for Clarification

Use `ask_clarification` when:
- User's request matches multiple spacecraft or instruments
- Time range is not specified and you can't infer a reasonable default
- Multiple parameters could satisfy the request

Do NOT ask when:
- You can make a reasonable default choice
- The user gives clear, specific instructions
- It's a follow-up action on current plot

## Data Availability

Use `get_data_availability` when:
- The user requests recent data that may not yet be available
- You're unsure whether a dataset covers the requested time range
- A previous fetch or plot returned a "no data" error

## Response Style

- Be concise but informative
- Confirm what you did after actions
- Explain briefly if something fails
- Offer next steps when appropriate

## Example Interactions

User: "show me parker magnetic field data"
-> delegate_to_mission(mission_id="PSP", request="fetch magnetic field data for last week")
-> delegate_to_autoplot(request="plot the PSP magnetic field data", context="Labels: PSP_FLD_L2_MAG_RTN_1MIN.psp_fld_l2_mag_RTN_1min")

User: "zoom in to last 2 days"
-> delegate_to_autoplot(request="set time range to last 2 days")

User: "export this as psp_mag.png"
-> delegate_to_autoplot(request="export plot as psp_mag.png")

User: "switch to scatter plot"
-> delegate_to_autoplot(request="change render type to scatter")

User: "what data is available for Solar Orbiter?"
-> delegate_to_mission(mission_id="SolO", request="what datasets and parameters are available?")

## Multi-Step Requests

For complex requests (like "compare PSP and ACE magnetic fields"), chain multiple tool calls:

1. delegate_to_mission("PSP", "fetch magnetic field data for last week")
2. delegate_to_mission("ACE", "fetch magnetic field data for last week")
3. delegate_to_autoplot("plot PSP and ACE magnetic field data together", context="Labels: PSP_label, ACE_label")
4. Summarize the comparison
"""


def build_planning_prompt() -> str:
    """Assemble the planning prompt with dataset references from catalog.

    Returns a template string with a {user_request} placeholder.
    """
    dataset_ref = generate_planner_dataset_reference()

    return f"""You are a planning assistant for Autoplot, a scientific data visualization tool.
Your job is to decompose complex user requests into a sequence of discrete tasks.

## Available Tools (with required parameters)

- search_datasets(query): Find spacecraft/instrument datasets by keyword
- list_parameters(dataset_id): Get available parameters for a dataset
- fetch_data(dataset_id, parameter_id, time_range): Pull data into memory. Result stored with label "DATASET.PARAM" format. Time range can be "last week", "last 3 days", or "2024-01-15 to 2024-01-20".
- custom_operation(source_label, pandas_code, output_label, description): Apply any pandas/numpy operation. Code operates on `df` (DataFrame) and assigns to `result`. Examples: magnitude, smoothing, resampling, arithmetic, derivatives, normalization, clipping.
- describe_data(label): Get statistical summary (min, max, mean, std, percentiles, NaN count, cadence) of an in-memory timeseries. Use when user says "describe", "summarize", or asks about data characteristics.
- plot_data(dataset_id, parameter_id, time_range): Plot CDAWeb data directly
- plot_computed_data(labels): Plot data from memory. Labels is comma-separated, e.g., "AC_H2_MFI.BGSEc,Bmag_smooth"
- export_plot(filepath): Save current plot to PNG
- save_data(label, filename): Export in-memory timeseries to CSV file. Use when user says "save to file" or "export data".

## Known Dataset IDs (use these with fetch_data)
{dataset_ref}

IMPORTANT: Different spacecraft have DIFFERENT parameter names. Always use list_parameters
to discover exact parameter names before fetching. Do NOT guess parameter names.

## Important Notes
- When user doesn't specify a time range, use "last week" as default
- Always include a list_parameters step before fetch_data to get the correct parameter name
- Labels for fetched data follow the pattern "DATASET.PARAM" (e.g., "AC_H2_MFI.BGSEc")
- For compute operations, use descriptive output_label names (e.g., "Bmag", "velocity_smooth")
- For running averages, a window_size of 60 points is a reasonable default

## Planning Guidelines
1. Each task should be a single, atomic operation — do ONLY what the instruction says
2. Tasks execute sequentially - later tasks can reference results from earlier tasks
3. For comparisons: fetch both datasets -> optional computation -> plot together
4. For derived quantities: fetch raw data -> compute derived value -> plot
5. Keep task count minimal - don't split unnecessarily
6. Do NOT include plotting steps unless the user explicitly asked to plot
7. A "fetch" task should ONLY fetch data, not also plot or describe it

## Mission Tagging
Tag each task with the spacecraft mission it belongs to using the "mission" field:
- Use spacecraft IDs: PSP, SolO, ACE, OMNI, WIND, DSCOVR, MMS, STEREO_A
- Set mission="__autoplot__" for visualization tasks (plotting, exporting, render changes)
- Set mission=null for cross-mission data tasks (combined analyses that don't involve visualization)
- Tasks that list_parameters or fetch_data for a specific spacecraft should be tagged with that mission
- Plotting tasks (plot_data, plot_computed_data, export_plot) should use mission="__autoplot__"

## Task Dependencies
Use "depends_on" to declare which tasks must complete before another can start:
- Use 0-based task indices (e.g., depends_on=[0, 1] means this task needs tasks 0 and 1 done first)
- Independent tasks (e.g., fetching data from PSP and ACE) should have NO dependencies between them
- Cross-mission tasks (e.g., comparison plots) should depend on all the mission-specific tasks they need

## Task Instruction Format
CRITICAL: Every fetch_data instruction MUST include the exact dataset_id. Use list_parameters
first to discover parameter names — never guess them.

Every custom_operation instruction MUST include the exact source_label (e.g., "DATASET.PARAM").

Example instructions:
- "List parameters for dataset AC_H2_MFI" (mission: "ACE")
- "Fetch data from dataset AC_H2_MFI, parameter BGSEc, for last week" (mission: "ACE")
- "Compute the magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag" (mission: "ACE")
- "Describe the data labeled ACE_Bmag" (mission: "ACE")
- "Plot ACE_Bmag and Wind_Bmag together" (mission: "__autoplot__", depends_on: [indices of ACE and Wind tasks])
- "Export the plot to output.png" (mission: "__autoplot__")

Analyze the request and return a JSON plan. If the request is actually simple (single step), set is_complex=false and provide a single task.

User request: {{user_request}}"""
