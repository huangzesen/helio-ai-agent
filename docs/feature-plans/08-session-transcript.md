# Feature 08: Session Transcript Export

## Summary

Add a `transcript` CLI command that saves the current conversation to a markdown file. Users can review, share, or reproduce their analysis sessions.

## Motivation

Scientists often need to document what they did: "I plotted X, computed Y, and found Z." Currently the conversation vanishes when the session ends. Saving a transcript lets users:
- Share their analysis workflow with collaborators
- Reproduce the same steps later
- Include in lab notebooks or reports

## Files to Modify

### 1. `agent/core.py` — Track conversation for transcript

Add a conversation log list to `AutoplotAgent.__init__`:

```python
self._transcript: list[dict] = []  # {"role": "user"|"agent"|"tool", "content": str, "timestamp": str}
```

Record entries in `process_message` and `_process_single_message`:

```python
from datetime import datetime

# At the start of process_message:
self._transcript.append({
    "role": "user",
    "content": user_message,
    "timestamp": datetime.now().isoformat(),
})

# Before returning the response:
self._transcript.append({
    "role": "agent",
    "content": response_text,
    "timestamp": datetime.now().isoformat(),
})
```

Optionally, in verbose mode, also log tool calls:

```python
# In _execute_tool_safe, after execution:
if self.verbose:
    self._transcript.append({
        "role": "tool",
        "content": f"{tool_name}({tool_args}) -> {result.get('status', 'ok')}",
        "timestamp": datetime.now().isoformat(),
    })
```

Add a method to export:

```python
def export_transcript(self, filename: str = "") -> str:
    """Export conversation transcript to a markdown file.

    Args:
        filename: Output path. Auto-generated if empty.

    Returns:
        Absolute path to the saved file.
    """
    from pathlib import Path
    from datetime import datetime

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.md"
    if not filename.endswith(".md"):
        filename += ".md"

    lines = [
        f"# Autoplot Session Transcript",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Messages**: {sum(1 for t in self._transcript if t['role'] in ('user', 'agent'))}",
        f"",
        f"---",
        f"",
    ]

    for entry in self._transcript:
        ts = entry["timestamp"][:19]  # Trim microseconds
        if entry["role"] == "user":
            lines.append(f"### You ({ts})")
            lines.append(f"")
            lines.append(entry["content"])
            lines.append(f"")
        elif entry["role"] == "agent":
            lines.append(f"### Agent ({ts})")
            lines.append(f"")
            lines.append(entry["content"])
            lines.append(f"")
        elif entry["role"] == "tool":
            lines.append(f"> `{entry['content']}`")
            lines.append(f"")

    # Token usage summary
    usage = self.get_token_usage()
    if usage["api_calls"] > 0:
        lines.extend([
            "---",
            "",
            "## Session Statistics",
            "",
            f"- **API calls**: {usage['api_calls']}",
            f"- **Input tokens**: {usage['input_tokens']:,}",
            f"- **Output tokens**: {usage['output_tokens']:,}",
            f"- **Total tokens**: {usage['total_tokens']:,}",
        ])

    content = "\n".join(lines)
    Path(filename).write_text(content, encoding="utf-8")

    return str(Path(filename).resolve())
```

### 2. `main.py` — Add `transcript` command

Add to the command handling section (after the `errors` command):

```python
if user_input.lower().startswith("transcript"):
    # Optional filename: "transcript myfile.md"
    parts = user_input.split(maxsplit=1)
    filename = parts[1] if len(parts) > 1 else ""

    if not agent._transcript:
        print("No conversation to save yet.")
    else:
        filepath = agent.export_transcript(filename)
        print(f"Transcript saved to: {filepath}")
    print()
    continue
```

Also add to the help text:

```python
print("Commands: quit, reset, status, retry, cancel, errors, transcript, help")
```

### 3. `docs/capability-summary.md` — Update CLI commands table

Add row:
```
| `transcript` | Save conversation history to a markdown file |
```

## Output Format Example

```markdown
# Autoplot Session Transcript

**Date**: 2026-02-06 14:30:00
**Messages**: 6

---

### You (2026-02-06T14:30:12)

Show me ACE magnetic field data for last week

> `search_datasets({"query": "ACE magnetic"}) -> success`
> `plot_data({"dataset_id": "AC_H2_MFI", ...}) -> success`

### Agent (2026-02-06T14:30:45)

I've plotted the ACE magnetic field magnitude for the past week (2026-01-30 to 2026-02-06). The data is displayed in the Autoplot window.

### You (2026-02-06T14:31:02)

Export this as ace_mag.png

> `export_plot({"filename": "ace_mag.png"}) -> success`

### Agent (2026-02-06T14:31:08)

Exported the plot to C:\Users\...\ace_mag.png (245 KB).

---

## Session Statistics

- **API calls**: 4
- **Input tokens**: 3,842
- **Output tokens**: 287
- **Total tokens**: 4,129
```

## Testing

```python
def test_transcript_empty():
    agent = AutoplotAgent.__new__(AutoplotAgent)
    agent._transcript = []
    # Should handle gracefully

def test_transcript_records_messages():
    agent = AutoplotAgent.__new__(AutoplotAgent)
    agent._transcript = []
    agent._transcript.append({"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00"})
    agent._transcript.append({"role": "agent", "content": "Hi!", "timestamp": "2026-01-01T00:00:01"})
    # Export and verify content

def test_transcript_auto_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Export with empty filename
    # Verify file created with session_*.md pattern
```

## Notes

- Tool calls are only recorded in transcript when `--verbose` mode is active. This keeps the transcript clean for normal use but detailed for debugging.
- The transcript is in-memory only — it doesn't persist across `reset` commands.
- Future enhancement: auto-save transcript on exit (opt-in via config).
