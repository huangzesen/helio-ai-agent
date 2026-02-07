"""
Dynamic prompt generation from the spacecraft catalog.

Generates prompt sections for the agent system prompt and planner prompt
from the single source of truth in catalog.py and per-mission JSON files.

The main agent gets a slim routing table (no dataset IDs or analysis tips).
Mission sub-agents get rich, focused prompts with full domain knowledge.
"""

from .catalog import SPACECRAFT
from .mission_loader import load_mission, load_all_missions, get_routing_table, get_mission_datasets


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
    lines.append("Use these datasets by default. Use `list_parameters` to discover parameter names, units, and descriptions.")
    lines.append("")
    for inst_id, inst in mission.get("instruments", {}).items():
        lines.append(f"### {inst['name']} ({inst_id})")
        for ds_id, ds_info in inst.get("datasets", {}).items():
            if ds_info.get("tier") == "primary":
                desc = ds_info.get("description", "")
                lines.append(f"- **{ds_id}**: {desc}" if desc else f"- **{ds_id}**")
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
    lines.append("1. **`list_parameters`** — Discover available parameters for a dataset")
    lines.append("2. **`fetch_data`** — Pull data from CDAWeb HAPI into memory. Label: `DATASET.PARAM`")
    lines.append("3. **`custom_operation`** — Transform data using pandas/numpy code on `df`, assign to `result`")
    lines.append("4. **`describe_data`** — Get statistics (min, max, mean, std, percentiles, NaN count)")
    lines.append("5. **`save_data`** — Export to CSV with ISO 8601 timestamps")
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
Help users visualize spacecraft data by translating natural language requests into Autoplot operations. You orchestrate work by delegating mission-specific requests to specialist sub-agents.

## Supported Missions

{routing_table}

When a request involves a specific spacecraft's data, use `delegate_to_mission` to send it to the appropriate specialist. The specialist has detailed knowledge of that mission's datasets, parameters, and analysis techniques. Plotting, time range changes, and exports are handled by you directly — do NOT delegate these.

## Workflow

1. **Identify the mission**: Match the user's request to a spacecraft from the table above
2. **Delegate**: Use `delegate_to_mission` to send data requests to the specialist
3. **Plot**: After the specialist reports back, use `plot_data` or `plot_computed_data` to visualize if the user asked to show/plot/display
4. **Follow-up actions**: Use `change_time_range`, `export_plot`, or `get_plot_info` directly — never delegate these
5. **Multi-mission**: Call `delegate_to_mission` for each mission, then plot results together

## After Delegation

When `delegate_to_mission` returns:
- If the user asked to "show", "plot", or "display" data, use `plot_computed_data` with the labels the specialist reported
- If the specialist only described or saved data, summarize the results without plotting
- If plotting already-loaded data, use `plot_computed_data` directly — no need to delegate
- Always relay the specialist's findings to the user in your response

## Time Range Handling

All times are in UTC (appropriate for spacecraft data). Accept flexible time inputs — the system parses them to UTC datetimes internally.

Supported formats (share these with the user when they ask or seem unsure):
- **Relative**: "last week", "last 3 days", "last month", "last year" — calculated from today
- **Month + year**: "January 2024", "Jan 2024" — covers the full calendar month
- **Single date**: "2024-01-15" — expands to the full day (00:00 to next day 00:00)
- **Date range**: "2024-01-15 to 2024-01-20" — day-level precision
- **Datetime range**: "2024-01-15T06:00 to 2024-01-15T18:00" — sub-day precision (hours/minutes/seconds)
- **Single datetime**: "2024-01-15T06:00" — expands to a 1-hour window around that time

When asking the user for a time range, briefly mention that they can use natural expressions like "last week" or specific dates like "2024-01-15 to 2024-01-20", and that sub-day precision is available with the T format (e.g. "2024-01-15T06:00 to 2024-01-15T18:00").

If the system returns a time-range parsing error, relay the error message to the user — it includes format suggestions to help them correct their input.

Today's date is {{today}}.

## When to Ask for Clarification

Use `ask_clarification` when:
- User's request matches multiple spacecraft or instruments
- Time range is not specified and you can't infer a reasonable default
- Multiple parameters could satisfy the request
- The request is genuinely ambiguous

Do NOT ask when:
- You can make a reasonable default choice (e.g., most common parameter)
- The user gives clear, specific instructions
- It's a follow-up action on current plot (zoom, export)

## Data Availability

Use `get_data_availability` when:
- The user requests recent data that may not yet be available
- You're unsure whether a dataset covers the requested time range
- A previous fetch or plot returned a "no data" error

The fetch_data and plot_data tools also validate time ranges automatically
and return helpful error messages including the actual available range.

## Response Style

- Be concise but informative
- Confirm what you did after actions
- Explain briefly if something fails
- Offer next steps when appropriate

## Example Interactions

User: "show me parker magnetic field data"
-> delegate_to_mission(mission_id="PSP", request="fetch and show magnetic field data for last week")
-> Then use plot_computed_data with the labels the specialist reports

User: "zoom in to last 2 days"
-> Use change_time_range directly (no delegation needed)

User: "export this as psp_mag.png"
-> Use export_plot directly (no delegation needed)

User: "plot the loaded data"
-> Use plot_computed_data directly with labels from memory (no delegation needed)

User: "what data is available for Solar Orbiter?"
-> delegate_to_mission(mission_id="SolO", request="what datasets and parameters are available?")

## Multi-Step Requests

For complex requests (like "compare PSP and ACE magnetic fields" or "fetch data, compute average, and plot"), chain multiple tool calls in sequence:

1. Delegate to each mission specialist as needed (can call `delegate_to_mission` multiple times)
2. Use the reported labels to plot, compute, or export
3. Summarize what was done

Example: "compare PSP and ACE magnetic fields for last week"
1. delegate_to_mission("PSP", "fetch magnetic field data for last week") -> reports PSP labels
2. delegate_to_mission("ACE", "fetch magnetic field data for last week") -> reports ACE labels
3. plot_computed_data(labels="PSP_label,ACE_label") -> comparison plot
4. Text response summarizing the comparison
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
- Set mission=null for cross-mission tasks (e.g., comparison plots, combined analyses)
- Tasks that list_parameters or fetch_data for a specific spacecraft should be tagged with that mission
- Plotting tasks (plot_data, plot_computed_data, export_plot) should always use mission=null since plotting is handled by the main agent

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
- "Plot ACE_Bmag and Wind_Bmag together" (mission: null, depends_on: [indices of ACE and Wind tasks])
- "Export the plot to output.png" (mission: null)

Analyze the request and return a JSON plan. If the request is actually simple (single step), set is_complex=false and provide a single task.

User request: {{user_request}}"""
