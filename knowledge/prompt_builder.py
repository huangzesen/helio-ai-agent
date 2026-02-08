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
from rendering.registry import render_method_catalog


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

    Lists all instrument-level datasets from JSON files.
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

    Includes mission overview, analysis patterns, recommended datasets,
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

    # --- Recommended Datasets ---
    lines.append("## Recommended Datasets")
    lines.append("")
    lines.append("These are the most commonly used datasets. **Many more are available** — use `browse_datasets` to see the full catalog.")
    lines.append("")
    for inst_id, inst in mission.get("instruments", {}).items():
        lines.append(f"### {inst['name']} ({inst_id})")
        for ds_id, ds_info in inst.get("datasets", {}).items():
            desc = ds_info.get("description", "")
            lines.append(f"- **{ds_id}**: {desc}" if desc else f"- **{ds_id}**")
            # Add parameter summary from local cache
            param_summary = _format_parameter_summary(ds_id)
            if param_summary:
                lines.append(param_summary)
        lines.append("")

    # --- Dataset Discovery Rule ---
    lines.append("## IMPORTANT: Dataset Discovery Rule")
    lines.append("")
    lines.append("If the user asks for a dataset, cadence, coordinate system, or instrument you don't see in the recommended list above, you MUST call `browse_datasets` to search the full catalog BEFORE telling the user it's unavailable. Never say \"I'm not familiar with that\" without checking first.")
    lines.append("")

    # --- Dataset Documentation ---
    lines.append("## Dataset Documentation")
    lines.append("")
    lines.append("Use `get_dataset_docs` when the user asks about:")
    lines.append("- Coordinate systems (GSE, GSM, RTN, etc.)")
    lines.append("- Principal investigator or data contact")
    lines.append("- Data quality issues, calibration, or known caveats")
    lines.append("- What specific parameters measure")
    lines.append("- Instrument details or references")
    lines.append("")
    lines.append("This fetches documentation from CDAWeb at runtime.")
    lines.append("")

    # --- CDAWeb Dataset ID Conventions ---
    lines.append("## CDAWeb Dataset ID Conventions")
    lines.append("")
    lines.append("- Some CDAWeb datasets use `@N` suffixes (e.g., `PSP_FLD_L2_RFS_LFR@2`, `WI_H0_MFI@0`).")
    lines.append("  These are **valid sub-datasets** that split large datasets into manageable parts.")
    lines.append("  Treat them exactly like regular dataset IDs — pass them to `fetch_data` and `list_parameters` as-is.")
    lines.append("- Attitude datasets (`_AT_`), orbit datasets (`_ORBIT_`, `_OR_`), and key-parameter")
    lines.append("  datasets (`_K0_`, `_K1_`, `_K2_`) are all valid CDAWeb datasets that can be fetched normally.")
    lines.append("- Cross-mission datasets like `OMNI_COHO1HR_MERGED_MAG_PLASMA` or `SOLO_HELIO1HR_POSITION`")
    lines.append("  are merged products from COHOWeb/HelioWeb — also valid for fetch_data.")
    lines.append("")

    # --- Data Operations Documentation ---
    lines.append("## Data Operations Workflow")
    lines.append("")
    lines.append("1. **When given an exact dataset ID and parameter**: Call `fetch_data` DIRECTLY.")
    lines.append("   Do NOT call browse_datasets or search_datasets first — go straight to fetch_data.")
    lines.append("   The caller has already identified the dataset and parameter for you.")
    lines.append("2. **When given a vague request** (e.g., \"get magnetic field data\"): Use your recommended")
    lines.append("   datasets above, or call `browse_datasets` to find the right dataset, then `fetch_data`.")
    lines.append("3. **Parameter verification**: Only call `list_parameters` if you're unsure of the")
    lines.append("   parameter name. If the parameter name is provided, trust it and call fetch_data directly.")
    lines.append("4. **Time range format**: '2024-01-15 to 2024-01-20' (use ' to ' separator, NOT '/').")
    lines.append("   Also accepts 'last week', 'January 2024', etc.")
    lines.append("5. **Labels**: fetch_data stores data with label `DATASET.PARAM`.")
    lines.append("")
    lines.append("## Reporting Results")
    lines.append("")
    lines.append("After completing data operations, report back with:")
    lines.append("- The **exact stored label(s)** for fetched data, e.g., 'Stored labels: DATASET.Param1, DATASET.Param2'")
    lines.append("- What time range was fetched and how many data points")
    lines.append("- A suggestion of what to do next (e.g., \"The data is ready to plot or compute on\")")
    lines.append("")
    lines.append("IMPORTANT: Always state the exact stored label(s) so downstream agents can reference them correctly.")
    lines.append("")
    lines.append("Do NOT attempt data transformations (magnitude, smoothing, etc.) — those are handled by the DataOps agent.")
    lines.append("Do NOT attempt to plot data — plotting is handled by the orchestrator.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataOps sub-agent prompt builder
# ---------------------------------------------------------------------------

def build_data_ops_prompt() -> str:
    """Generate the system prompt for the DataOps sub-agent.

    Includes computation patterns, code guidelines, and workflow instructions
    for data transformation, analysis, and export.

    Returns:
        System prompt string for the DataOpsAgent.
    """
    lines = [
        "You are a data transformation and analysis specialist for scientific spacecraft data.",
        "",
        "Your job is to transform, analyze, describe, and export in-memory timeseries data.",
        "You have access to `list_fetched_data`, `custom_operation`, `describe_data`, and `save_data` tools.",
        "",
        "## Workflow",
        "",
        "1. **Discover data**: Call `list_fetched_data` to see what timeseries are in memory",
        "2. **Transform**: Use `custom_operation` to compute derived quantities",
        "3. **Analyze**: Use `describe_data` to get statistical summaries",
        "4. **Export**: Use `save_data` to write data to CSV files",
        "",
        "## Common Computation Patterns",
        "",
        "Use `custom_operation` with pandas/numpy code. The code operates on `df` (a DataFrame",
        "with DatetimeIndex) and must assign the result to `result`.",
        "",
        "- **Magnitude**: `result = df.pow(2).sum(axis=1, skipna=False).pow(0.5).to_frame('magnitude')`",
        "- **Smoothing**: `result = df.rolling(60, center=True, min_periods=1).mean()`",
        "- **Resample**: `result = df.resample('60s').mean().dropna(how='all')`",
        "- **Rate of change**: `dv = df.diff().iloc[1:]; dt_s = df.index.to_series().diff().dt.total_seconds().iloc[1:]; result = dv.div(dt_s, axis=0)`",
        "- **Normalize**: `result = (df - df.mean()) / df.std()`",
        "- **Clip values**: `result = df.clip(lower=-50, upper=50)`",
        "- **Log transform**: `result = np.log10(df.abs().replace(0, np.nan))`",
        "- **Interpolate gaps**: `result = df.interpolate(method='linear')`",
        "- **Select columns**: `result = df[['x', 'z']]`",
        "- **Detrend**: `result = df - df.rolling(100, center=True, min_periods=1).mean()`",
        "- **Absolute value**: `result = df.abs()`",
        "- **Cumulative sum**: `result = df.cumsum()`",
        "- **Z-score filter**: `z = (df - df.mean()) / df.std(); result = df[z.abs() < 3].reindex(df.index)`",
        "",
        "## Code Guidelines",
        "",
        "- Always assign to `result` — must be DataFrame/Series with DatetimeIndex",
        "- Use `df`, `pd` (pandas), `np` (numpy) only — no imports, no file I/O",
        "- Handle NaN with `skipna=True`, `.dropna()`, or `.fillna()`",
        "- Use descriptive output_label names (e.g., 'ACE_Bmag', 'velocity_smooth')",
        "",
        "## Reporting Results",
        "",
        "After completing operations, report back with:",
        "- The **exact output label(s)** for computed data",
        "- How many data points in the result",
        "- A brief description of what was computed",
        "- A suggestion of what to do next (e.g., \"Ready to plot: label 'ACE_Bmag'\")",
        "",
        "IMPORTANT: Always state the exact label(s) so downstream agents can reference them.",
        "",
        "Do NOT attempt to fetch new data — fetching is handled by mission agents.",
        "Do NOT attempt to plot data — plotting is handled by the visualization agent.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization sub-agent prompt builder
# ---------------------------------------------------------------------------

def build_visualization_prompt(gui_mode: bool = False) -> str:
    """Generate the system prompt for the visualization sub-agent.

    Includes the method catalog from the registry, render type guidance,
    and workflow instructions.

    Args:
        gui_mode: If True, append GUI-mode specific instructions.

    Returns:
        System prompt string for the VisualizationAgent.
    """
    catalog = render_method_catalog()

    lines = [
        "You are a visualization specialist for a scientific data visualization tool.",
        "",
        "Your job is to execute visualization operations using the `execute_visualization` tool.",
        "You also have access to `list_fetched_data` to see what data is available in memory.",
        "",
        catalog,
        "## Using execute_visualization",
        "",
        "Call `execute_visualization(method=\"method_name\", args={...})` with the method name and arguments from the catalog above.",
        "",
        "Examples:",
        "- Plot stored data: `execute_visualization(method=\"plot_stored_data\", args={\"labels\": \"ACE_Bmag,PSP_Bmag\", \"title\": \"Comparison\"})`",
        "- Plot in specific panel: `execute_visualization(method=\"plot_stored_data\", args={\"labels\": \"Bmag\", \"index\": 1})`",
        "- Change render: `execute_visualization(method=\"set_render_type\", args={\"render_type\": \"scatter\"})`",
        "- Export PDF: `execute_visualization(method=\"export_pdf\", args={\"filename\": \"output.pdf\"})`",
        "- Set canvas: `execute_visualization(method=\"set_canvas_size\", args={\"width\": 1920, \"height\": 1080})`",
        "",
        "## Render Types",
        "",
        "- **series** (default): Line plot for timeseries data",
        "- **scatter**: Individual points, useful for sparse data",
        "- **fill_to_zero**: Area fill between data and zero line",
        "- **staircase**: Step function, good for discrete/quantized data",
        "- **digital**: On/off states, good for flags or status data",
        "",
        "## Time Range Format",
        "",
        "- Date range: '2024-01-15 to 2024-01-20' (use ' to ' separator, NOT '/')",
        "- Relative: 'last week', 'last 3 days'",
        "- IMPORTANT: Never use '/' as a date separator.",
        "",
        "## Notes",
        "",
        "- plot_cdaweb is not supported -- always use fetch_data first, then plot_stored_data",
        "- Session save/load (.vap files) is not available",
        "- Vector data (e.g., magnetic field Bx/By/Bz) is automatically decomposed into x/y/z components",
        "",
        "## Workflow",
        "",
        "1. Always call `list_fetched_data` first to see what data is in memory",
        "2. Use **plot_stored_data** for data already fetched/computed (labels from list_fetched_data)",
        "3. Chain multiple operations (e.g., plot -> set title -> set axis label -> export)",
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
            "## Interactive Mode",
            "",
            "Plots are rendered as interactive Plotly figures visible in the UI.",
            "- The user can already see the plot -- do NOT suggest exporting to PNG for viewing",
            "- Changes like zoom, axis labels, log scale, and title are reflected instantly",
            "- Use reset to clear the canvas when starting a new analysis",
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
- **Mission agents** handle data fetching (mission-specific knowledge of datasets and parameters)
- **DataOps agent** handles data transformations, analysis, and export (compute, describe, save)
- **Visualization agent** handles all visualization (plotting, customizing, exporting)

## Supported Missions

{routing_table}

## Workflow

1. **Identify the mission**: Match the user's request to a spacecraft from the table above
2. **Delegate data fetching**: Use `delegate_to_mission` for fetching data (requires mission-specific knowledge of datasets and parameters)
3. **Delegate data operations**: Use `delegate_to_data_ops` for computations (magnitude, smoothing, etc.), statistical summaries, and data export
4. **Delegate visualization**: Use `delegate_to_visualization` for plotting, customizing, exporting, or any visual operation
5. **Multi-mission**: Call `delegate_to_mission` for each mission, then `delegate_to_data_ops` if needed, then `delegate_to_visualization` to plot results
6. **Memory check**: Use `list_fetched_data` to see what data is currently in memory

## After Data Delegation

When `delegate_to_mission` returns:
- If the user asked to "show", "plot", or "display" data, use `delegate_to_visualization` with the labels the specialist reported
- If the user asked to compute something (magnitude, smoothing, etc.), use `delegate_to_data_ops`
- Always relay the specialist's findings to the user in your response

When `delegate_to_data_ops` returns:
- If the user asked to plot the result, use `delegate_to_visualization` with the output labels
- If the specialist only described or saved data, summarize the results without plotting

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
- The user provides a specific dataset_id AND parameter_id — use them directly without asking
- The user names a spacecraft + data type (e.g., "ACE magnetic field") — delegate to the mission agent immediately
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
-> delegate_to_visualization(request="plot the PSP magnetic field data", context="Labels: PSP_FLD_L2_MAG_RTN_1MIN.psp_fld_l2_mag_RTN_1min")

User: "zoom in to last 2 days"
-> delegate_to_visualization(request="set time range to last 2 days")

User: "export this as psp_mag.png"
-> delegate_to_visualization(request="export plot as psp_mag.png")

User: "switch to scatter plot"
-> delegate_to_visualization(request="change render type to scatter")

User: "what data is available for Solar Orbiter?"
-> delegate_to_mission(mission_id="SolO", request="what datasets and parameters are available?")

## Multi-Step Requests

For complex requests (like "compare PSP and ACE magnetic fields"), chain multiple tool calls:

1. delegate_to_mission("PSP", "fetch magnetic field data for last week")
2. delegate_to_mission("ACE", "fetch magnetic field data for last week")
3. delegate_to_visualization("plot PSP and ACE magnetic field data together", context="Labels: PSP_label, ACE_label")
4. Summarize the comparison

For "fetch ACE mag, compute magnitude, and plot":

1. delegate_to_mission("ACE", "fetch magnetic field vector data for last week")
2. delegate_to_data_ops("compute magnitude of AC_H2_MFI.BGSEc", context="Labels: AC_H2_MFI.BGSEc")
3. delegate_to_visualization("plot the magnitude", context="Labels: ACE_Bmag")
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
- browse_datasets(mission_id): Browse all available science datasets for a mission (filtered, no calibration/housekeeping)
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
- Set mission="__visualization__" for visualization tasks (plotting, exporting, render changes)
- Set mission="__data_ops__" for data transformation/analysis/export tasks (custom_operation, describe_data, save_data)
- Set mission=null for cross-mission data tasks (combined analyses that don't involve visualization or computation)
- Tasks that list_parameters or fetch_data for a specific spacecraft should be tagged with that mission
- Plotting tasks (plot_data, plot_computed_data, export_plot) should use mission="__visualization__"
- Compute tasks (magnitude, smoothing, resampling, describe, save to CSV) should use mission="__data_ops__"

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
- "Compute the magnitude of AC_H2_MFI.BGSEc, save as ACE_Bmag" (mission: "__data_ops__", depends_on: [index of fetch task])
- "Describe the data labeled ACE_Bmag" (mission: "__data_ops__")
- "Save ACE_Bmag to CSV" (mission: "__data_ops__")
- "Plot ACE_Bmag and Wind_Bmag together" (mission: "__visualization__", depends_on: [indices of ACE and Wind tasks])
- "Export the plot to output.png" (mission: "__visualization__")

Analyze the request and return a JSON plan. If the request is actually simple (single step), set is_complex=false and provide a single task.

User request: {{user_request}}"""
