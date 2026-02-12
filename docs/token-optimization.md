# Token Usage Optimization

Analysis of token consumption patterns and optimization strategies for helio-agent sessions.

## Current Token Usage Profile

Based on a representative session (Solar Cycle 25 comparison, Feb 2026):

| Metric | Value |
|---|---|
| Duration | 8.9 minutes |
| API calls | 109 |
| Total tokens | 876,613 |
| Input tokens | 845,798 (96.5%) |
| Output tokens | 12,503 (1.4%) |
| Thinking tokens | 18,312 (2.1%) |
| Input:Output ratio | 68:1 |
| Estimated cost (Gemini 2.5 Flash) | ~$0.20 |

Typical sessions range from 300K to 5.5M total tokens ($0.35-$1.90/session).

## Per-Agent Token Distribution

| Agent | Calls | Input Tokens | % of Total Input | Avg Input/Call |
|---|---|---|---|---|
| OrchestratorAgent | 19 | 304,505 | 36.0% | 16,027 |
| DataOps Agent | 38 | 207,514 | 24.5% | 5,461 |
| Mission Agent (OMNI) | 28 | 178,871 | 21.1% | 6,388 |
| Visualization Agent | 16 | 101,391 | 12.0% | 6,337 |
| PlannerAgent | 8 | 53,517 | 6.3% | 6,690 |

## System Prompt Sizes

Each agent has a system prompt re-sent on every API call:

| Agent | Approx Tokens | Static? |
|---|---|---|
| Orchestrator | 12K | Mostly (has `{today}` placeholder) |
| Mission Agent (per mission) | 15K | Yes, per mission |
| Planner | 8K | Dynamic (routing table) |
| Visualization | 5K | Partial (gui_mode flag) |
| DataOps | 3K | Static |
| DataExtraction | 2.5K | Static |
| Discovery | 1.5K | Static |

In the sample session, system prompts alone account for roughly:
- Orchestrator: 12K x 19 calls = 228K tokens
- OMNI Agent: 15K x 28 calls = 420K tokens
- All prompts are stable within a session.

## Identified Waste Patterns

### 1. Context re-send overhead (~36% of input)
Every API call re-sends system prompt + full conversation history. The orchestrator's per-call input grows from 8.8K to 34K tokens as history accumulates. Estimated overhead above minimal baseline: ~304K tokens.

### 2. Low-output calls (~25% of input budget)
30 of 109 calls (27.5%) produced <=30 output tokens while consuming 3,000+ input. These include:
- `initial_message` handoffs to sub-agents (25 tokens out)
- Failed `custom_operation` calls returning error responses (12 tokens out)
- Total: 213K input tokens for minimal output.

### 3. Tool call retries
`custom_operation` was called 15 times (80K input) due to repeated `.rolling()` errors. Fixing the underlying bug would eliminate 5-6 retry calls per computation (~30K tokens saved).

### 4. Redundant dataset browsing
`list_parameters` was called 10 times (70K input) across multiple delegation rounds to the same mission agent, re-discovering parameters already found earlier in the session.

## Optimization Strategies

### Near-term (no architecture changes)

1. **Fix known bugs** (e.g., `custom_operation` rolling average) to eliminate retry loops. Estimated saving: 30-50K tokens/session.

2. **Summarize completed task results** in orchestrator history instead of keeping full tool outputs. The orchestrator context grows from 9K to 34K per call; truncating old tool results could cap growth.

3. **Persist sub-agent discovery** across delegation rounds so the same mission agent doesn't re-call `list_parameters` for parameters it already found.

### Medium-term (API-level changes)

4. **Gemini context caching** — cache system prompts server-side for 90% input discount on cached portion. All system prompts are stable within a session. Implementation:
   - Create `CachedContent` on agent init with system prompt
   - Reference cache ID in `GenerateContentConfig` for subsequent calls
   - Set TTL to session duration (e.g., 1 hour)
   - Minimum 4,096 tokens required (all agent prompts qualify except Discovery)
   - Only the prefix (system prompt) can be cached, not mid-conversation content

5. **Batch sub-agent handoffs** to reduce the number of orchestrator round-trips. Each orchestrator call carries the full conversation; fewer calls = less re-sending.

### Long-term (architecture changes)

6. **Conversation summarization** — periodically compress old turns into a summary, reducing the growing conversation context that dominates input tokens.

7. **Prompt compression** — audit system prompts for redundancy. Mission prompts (~15K tokens) include full parameter summaries that may be retrievable on-demand instead.

8. **Shared context across sub-agents** — currently each sub-agent maintains its own conversation history. A shared context store could avoid re-discovering the same information.

## Measurement

Token usage is logged per API call in `~/.helio-agent/logs/token_*.log` with format:
```
timestamp | agent | tool_context | in:N out:N think:N | cum_in:N cum_out:N cum_think:N | calls:N
```

Session-level totals are in `~/.helio-agent/sessions/*/metadata.json` under `token_usage`.
