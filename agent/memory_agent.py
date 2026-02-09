"""
Passive MemoryAgent — monitors log growth and conversation turns, then runs
a thorough analysis to extract pitfalls, preferences, summaries, and error
patterns.  Triggered automatically at the end of process_message() when
enough new material has accumulated.

Not a BaseSubAgent — it doesn't use Gemini tools or participate in routing.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from google.genai import types

from .logging import get_logger, get_current_log_path, get_log_size, LOG_DIR
from .memory import Memory, MemoryStore
from .model_fallback import get_active_model

logger = get_logger()

# Thresholds for triggering analysis
LOG_GROWTH_THRESHOLD = 10 * 1024  # 10 KB
ERROR_COUNT_THRESHOLD = 5
TURN_COUNT_THRESHOLD = 10

# Cap on log content sent to Gemini
MAX_LOG_BYTES = 50 * 1024  # 50 KB

# Directories
STATE_DIR = Path.home() / ".helio-agent"
REPORTS_DIR = STATE_DIR / "reports"
STATE_FILE = STATE_DIR / "memory_agent_state.json"


@dataclass
class AnalysisResult:
    """Result from a MemoryAgent analysis."""
    preferences: list[str] = field(default_factory=list)
    summary: str = ""
    pitfalls: list[str] = field(default_factory=list)
    error_patterns: list[dict] = field(default_factory=list)
    report_path: Optional[str] = None


class MemoryAgent:
    """Passive agent that monitors log growth and extracts operational knowledge."""

    def __init__(
        self,
        client,
        model_name: str,
        memory_store: MemoryStore,
        verbose: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.memory_store = memory_store
        self.verbose = verbose

    # ---- State persistence ----

    def _load_state(self) -> dict:
        """Read the state file, returning defaults if missing or corrupt."""
        if not STATE_FILE.exists():
            return {}
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_state(self, state: dict) -> None:
        """Write the state file atomically."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp.replace(STATE_FILE)

    # ---- Trigger checks ----

    def should_analyze(self, turn_count: int) -> bool:
        """Lightweight check — no LLM call.  Returns True when any threshold is met."""
        state = self._load_state()

        # Check turn count
        last_turn = state.get("last_turn_count", 0)
        if turn_count - last_turn >= TURN_COUNT_THRESHOLD:
            return True

        # Check log growth
        log_path = get_current_log_path()
        log_name = log_path.name
        current_size = get_log_size(log_path)

        last_log_name = state.get("last_log_file", "")
        last_offset = state.get("last_log_byte_offset", 0)

        if log_name != last_log_name:
            # New day / log file rotated — treat all content as new
            if current_size >= LOG_GROWTH_THRESHOLD:
                return True
        else:
            if current_size - last_offset >= LOG_GROWTH_THRESHOLD:
                return True

        # Check error count
        new_errors = self._count_new_errors(state)
        if new_errors >= ERROR_COUNT_THRESHOLD:
            return True

        return False

    # ---- Log reading ----

    def _read_new_log_content(self, state: dict) -> str:
        """Read log bytes from last offset to current EOF (capped at MAX_LOG_BYTES)."""
        log_path = get_current_log_path()
        if not log_path.exists():
            return ""

        last_log_name = state.get("last_log_file", "")
        last_offset = state.get("last_log_byte_offset", 0)

        # If log file rotated, start from beginning
        if log_path.name != last_log_name:
            last_offset = 0

        current_size = get_log_size(log_path)
        if current_size <= last_offset:
            return ""

        # Read new bytes, capped
        read_start = max(last_offset, current_size - MAX_LOG_BYTES)
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(read_start)
                return f.read()
        except OSError:
            return ""

    def _count_new_errors(self, state: dict) -> int:
        """Count ERROR/WARNING lines in new log bytes since last analysis."""
        content = self._read_new_log_content(state)
        if not content:
            return 0
        count = 0
        for line in content.split("\n"):
            if "| ERROR" in line or "| WARNING" in line:
                count += 1
        return count

    # ---- Analysis ----

    def analyze(
        self,
        conversation_text: str,
        turn_count: int,
        session_id: str = "",
    ) -> AnalysisResult:
        """Run a single-shot Gemini call to analyze conversation + logs.

        Returns an AnalysisResult with extracted memories and report path.
        """
        state = self._load_state()
        log_content = self._read_new_log_content(state)

        # Build existing memories for dedup
        existing = self.memory_store.get_all()
        existing_prefs = [m.content for m in existing if m.type == "preference"]
        existing_pitfalls = [m.content for m in existing if m.type == "pitfall"]
        existing_sums = [m.content for m in existing if m.type == "summary"]

        prompt = self._build_analysis_prompt(
            conversation_text, log_content,
            existing_prefs, existing_pitfalls, existing_sums,
        )

        try:
            response = self.client.models.generate_content(
                model=get_active_model(self.model_name),
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
        except Exception as e:
            logger.debug(f"[MemoryAgent] Gemini call failed: {e}")
            return AnalysisResult()

        text = (response.text or "").strip()
        data = self._parse_analysis_response(text)
        if data is None:
            return AnalysisResult()

        result = AnalysisResult(
            preferences=data.get("preferences", []),
            summary=data.get("summary", ""),
            pitfalls=data.get("pitfalls", []),
            error_patterns=data.get("error_patterns", []),
        )

        # Save memories
        self._save_preferences_and_summary(
            result.preferences, result.summary, session_id,
            existing_prefs, existing_sums,
        )
        self._save_pitfalls(result.pitfalls, session_id, existing_pitfalls)

        # Save report
        if result.error_patterns or result.pitfalls:
            log_path = get_current_log_path()
            last_offset = state.get("last_log_byte_offset", 0)
            if log_path.name != state.get("last_log_file", ""):
                last_offset = 0
            result.report_path = self._save_report(
                result, session_id, log_path.name,
                last_offset, get_log_size(log_path), turn_count,
            )

        # Consolidate if memory count exceeds target
        enabled_count = len(self.memory_store.get_enabled())
        if enabled_count > 30:
            removed = self.consolidate(max_total=30)
            if removed > 0:
                logger.info(f"[MemoryAgent] Consolidated memories: removed {removed}, kept {enabled_count - removed}")

        # Update state
        log_path = get_current_log_path()
        new_state = {
            "last_log_file": log_path.name,
            "last_log_byte_offset": get_log_size(log_path),
            "last_analysis_timestamp": datetime.now().isoformat(),
            "last_turn_count": turn_count,
        }
        self._save_state(new_state)

        return result

    # ---- Consolidation ----

    def consolidate(self, max_total: int = 10) -> int:
        """Merge and prune memories down to max_total using an LLM call.

        Returns the number of memories removed.
        """
        enabled = self.memory_store.get_enabled()
        if len(enabled) <= max_total:
            return 0

        before_count = len(enabled)

        # Build memory listing for the prompt
        memory_lines = []
        for m in enabled:
            memory_lines.append(
                f'  {{"id": "{m.id}", "type": "{m.type}", '
                f'"content": "{m.content}", "created_at": "{m.created_at}"}}'
            )
        memories_json = "[\n" + ",\n".join(memory_lines) + "\n]"

        prompt = f"""You are managing a memory system for an AI heliophysics assistant.
Below are all current memories. Consolidate them to at most {max_total} entries.

Rules:
- Merge duplicates/overlapping entries into one concise entry
- Keep the most actionable and generalizable entries
- Prefer recent entries over old ones for summaries
- Keep at most 10-15 preferences, 5-10 summaries, 10-15 pitfalls
- For kept-as-is entries, preserve the original id
- For merged entries, use a new id (any short string)
- Return JSON array only, no markdown fencing

Each entry must have: {{"id": "...", "type": "preference|summary|pitfall", "content": "..."}}

Current memories:
{memories_json}"""

        try:
            response = self.client.models.generate_content(
                model=get_active_model(self.model_name),
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
        except Exception as e:
            logger.debug(f"[MemoryAgent] Consolidation LLM call failed: {e}")
            return 0

        text = (response.text or "").strip()

        # Strip markdown fencing if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            entries = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"[MemoryAgent] Consolidation parse failed: {e}")
            return 0

        if not isinstance(entries, list):
            logger.debug("[MemoryAgent] Consolidation returned non-list")
            return 0

        # Build new Memory objects
        new_memories = []
        for entry in entries[:max_total]:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content", "").strip()
            mem_type = entry.get("type", "preference")
            if not content:
                continue
            if mem_type not in ("preference", "summary", "pitfall"):
                mem_type = "preference"
            new_memories.append(Memory(
                id=str(entry.get("id", ""))[:12] or None,
                type=mem_type,
                content=content,
                created_at=datetime.now().isoformat(),
            ))

        if not new_memories:
            logger.debug("[MemoryAgent] Consolidation produced no memories, skipping")
            return 0

        # Archive evicted memories to cold storage
        kept_ids = {m.id for m in new_memories}
        evicted = [m for m in enabled if m.id not in kept_ids]
        if evicted:
            self.memory_store.archive_to_cold(evicted)

        # Also preserve any disabled memories (consolidation only touches enabled)
        disabled = [m for m in self.memory_store.get_all() if not m.enabled]
        self.memory_store.replace_all(disabled + new_memories)

        removed = before_count - len(new_memories)
        return max(removed, 0)

    # ---- Prompt building ----

    def _build_analysis_prompt(
        self,
        conversation_text: str,
        log_content: str,
        existing_prefs: list[str],
        existing_pitfalls: list[str],
        existing_sums: list[str],
    ) -> str:
        existing_section = ""
        if existing_prefs:
            existing_section += "\nExisting preferences (do NOT duplicate):\n"
            existing_section += "\n".join(f"- {p}" for p in existing_prefs)
        if existing_pitfalls:
            existing_section += "\nExisting pitfalls (do NOT duplicate):\n"
            existing_section += "\n".join(f"- {p}" for p in existing_pitfalls)
        if existing_sums:
            existing_section += "\nExisting summaries (do NOT duplicate):\n"
            existing_section += "\n".join(f"- {s}" for s in existing_sums)

        log_section = ""
        if log_content:
            log_section = f"\n\nAgent log (recent entries):\n{log_content[:MAX_LOG_BYTES]}"

        return f"""Analyze this conversation and agent log to extract useful information for future sessions.
{existing_section}

Conversation:
{conversation_text}
{log_section}

Respond with JSON only (no markdown fencing):
{{
  "preferences": ["list of user preferences, habits, or styles observed (empty list if none)"],
  "summary": "one-sentence summary of what was analyzed in this session (empty string if nothing notable)",
  "pitfalls": ["generalizable operational lessons learned, e.g. 'OMNI data for recent dates may have empty CSV strings — treat as NaN' (empty list if none)"],
  "error_patterns": [
    {{
      "component": "file or module where error occurred",
      "occurrences": 1,
      "pattern": "description of the error pattern",
      "suggested_fix": "how to fix or work around it"
    }}
  ]
}}

Rules:
- Preferences: plot style choices, spacecraft of interest, workflow habits, display preferences
- Summary: what data was fetched, what analysis was done, key findings
- Pitfalls: generalizable lessons from errors or unexpected behavior (NOT user preferences). Frame as actionable advice for the agent. Only include if there's real evidence in the log or conversation.
- Error patterns: recurring errors from the log with component, count, description, and fix suggestion. Only include if there are actual ERROR or WARNING entries.
- Do NOT repeat existing memories listed above
- Keep each entry concise (one sentence max)
- Return empty lists/strings if nothing new to extract"""

    # ---- Response parsing ----

    def _parse_analysis_response(self, text: str) -> Optional[dict]:
        """Parse JSON from Gemini response, stripping markdown fencing if present."""
        if not text:
            return None
        # Strip markdown fencing
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3].strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"[MemoryAgent] Failed to parse response: {e}")
            return None

    # ---- Saving memories ----

    def _save_preferences_and_summary(
        self,
        preferences: list,
        summary: str,
        session_id: str,
        existing_prefs: list[str],
        existing_sums: list[str],
    ) -> int:
        """Add preference and summary memories with dedup. Returns count added."""
        count = 0
        for pref in preferences:
            if not pref or not isinstance(pref, str):
                continue
            if any(pref.lower() in ep.lower() or ep.lower() in pref.lower()
                   for ep in existing_prefs):
                continue
            self.memory_store.add(Memory(
                type="preference", content=pref, source_session=session_id,
            ))
            count += 1

        if summary and isinstance(summary, str):
            if not any(summary.lower() in es.lower() or es.lower() in summary.lower()
                       for es in existing_sums):
                self.memory_store.add(Memory(
                    type="summary", content=summary, source_session=session_id,
                ))
                count += 1
        return count

    def _save_pitfalls(
        self,
        pitfalls: list,
        session_id: str,
        existing_pitfalls: list[str],
    ) -> int:
        """Add pitfall memories with substring dedup. Returns count added."""
        count = 0
        for pitfall in pitfalls:
            if not pitfall or not isinstance(pitfall, str):
                continue
            # Skip if substring of existing pitfall or vice versa
            if any(pitfall.lower() in ep.lower() or ep.lower() in pitfall.lower()
                   for ep in existing_pitfalls):
                continue
            self.memory_store.add(Memory(
                type="pitfall", content=pitfall, source_session=session_id,
            ))
            existing_pitfalls.append(pitfall)  # track within batch
            count += 1
        return count

    # ---- Report generation ----

    def _save_report(
        self,
        result: AnalysisResult,
        session_id: str,
        log_file_name: str,
        start_offset: int,
        end_offset: int,
        turn_count: int,
    ) -> str:
        """Write a markdown report to ~/.helio-agent/reports/. Returns the path."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"report_{ts}.md"

        lines = [
            f"# Analysis Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Session Context",
            f"- Session: {session_id or 'unknown'}",
            f"- Log analyzed: {log_file_name} (bytes {start_offset}–{end_offset})",
            f"- Conversation turns: {turn_count}",
        ]

        if result.error_patterns:
            lines.append("")
            lines.append("## Error Patterns")
            for i, ep in enumerate(result.error_patterns, 1):
                if not isinstance(ep, dict):
                    continue
                lines.append("")
                lines.append(f"### {i}. {ep.get('pattern', 'Unknown')}")
                lines.append(f"- **Component**: {ep.get('component', 'unknown')}")
                lines.append(f"- **Occurrences**: {ep.get('occurrences', '?')}")
                lines.append(f"- **Pattern**: {ep.get('pattern', '')}")
                lines.append(f"- **Suggested fix**: {ep.get('suggested_fix', '')}")

        if result.pitfalls:
            lines.append("")
            lines.append("## Pitfalls Extracted")
            for p in result.pitfalls:
                lines.append(f"- {p}")

        if result.preferences or result.summary:
            lines.append("")
            lines.append("## Memories Extracted")
            for p in result.preferences:
                lines.append(f"- Preference: {p}")
            if result.summary:
                lines.append(f"- Summary: {result.summary}")

        lines.append("")  # trailing newline
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[MemoryAgent] Report saved: {report_path}")
        return str(report_path)
