# LLM Backend Abstraction Plan

Design document for decoupling helio-ai-agent from the `google-genai` SDK so that
multiple LLM providers (OpenAI, Anthropic, DeepSeek, Qwen, Kimi, Mistral, etc.)
can be used as interchangeable backends.

**Status:** Phase 1 complete (February 2026). Phases 2-4 pending.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Current Gemini Coupling — Inventory](#2-current-gemini-coupling--inventory)
3. [Provider Landscape](#3-provider-landscape)
4. [Design Approach — Thin Custom Adapter](#4-design-approach--thin-custom-adapter)
5. [Core Abstractions](#5-core-abstractions)
6. [Adapter Implementations](#6-adapter-implementations)
7. [Integration Points — File-by-File Changes](#7-integration-points--file-by-file-changes)
8. [Configuration Changes](#8-configuration-changes)
9. [Session Persistence Changes](#9-session-persistence-changes)
10. [Provider-Specific Features — Escape Hatches](#10-provider-specific-features--escape-hatches)
11. [Migration Strategy](#11-migration-strategy)
12. [What NOT to Abstract](#12-what-not-to-abstract)
13. [Testing Strategy](#13-testing-strategy)
14. [Implementation Phases](#14-implementation-phases)
15. [Risk Assessment](#15-risk-assessment)

---

## 1. Motivation

The project is currently hardwired to Google Gemini via the `google-genai` SDK
(`google-genai==1.62.0`). This creates three problems:

1. **Vendor lock-in** — if Gemini pricing changes, models degrade, or Google
   imposes new restrictions, we have no alternative.
2. **Missed opportunities** — Anthropic Claude excels at agentic tool management,
   DeepSeek offers 10-30x cheaper inference, open-weight models (Qwen3, Llama 4,
   Mistral Large 3) can be self-hosted for zero per-token cost.
3. **Research flexibility** — comparing model performance on the same agentic task
   requires running the same tool schemas against different backends.

The goal is to introduce a thin adapter layer (`agent/llm/`) that normalizes the
LLM interaction surface — tool schemas, message sending, response parsing, and
session management — without adding heavyweight framework dependencies.

---

## 2. Current Gemini Coupling — Inventory

Every `from google import genai` / `from google.genai import types` import
represents a coupling point. Here is the complete list:

### 2.1 SDK Types Used

| Gemini Type | Where Used | Purpose |
|---|---|---|
| `genai.Client` | `core.py`, `base_agent.py`, `planner.py`, `memory_agent.py` | API client initialization |
| `types.HttpOptions` | `core.py` | HTTP timeout/retry config |
| `types.FunctionDeclaration` | `core.py`, `base_agent.py`, `planner.py` | Tool schema wrapping |
| `types.Tool` | `core.py`, `base_agent.py`, `planner.py`, `memory_agent.py` | Tool container |
| `types.ToolConfig` / `types.FunctionCallingConfig` | `base_agent.py` | Forced function calling (`mode="ANY"`) |
| `types.GenerateContentConfig` | `core.py`, `base_agent.py`, `planner.py`, `memory_agent.py` | Chat config (system prompt, tools, thinking) |
| `types.ThinkingConfig` | `core.py`, `base_agent.py`, `planner.py` | Reasoning effort control |
| `types.Part.from_function_response` | `core.py`, `base_agent.py`, `tool_loop.py` | Sending tool results back |
| `types.GoogleSearch` | `core.py` | Google Search grounding |
| `response.candidates[0].content.parts` | `core.py`, `base_agent.py`, `tool_loop.py`, `thinking.py` | Response parsing |
| `part.function_call` / `part.function_call.name` / `part.function_call.args` | `core.py`, `base_agent.py`, `tool_loop.py` | Extracting tool calls |
| `part.thought` | `thinking.py` | Extracting thinking/reasoning |
| `response.usage_metadata` | `core.py`, `base_agent.py`, `planner.py` | Token counting |
| `Content.model_dump()` | `core.py` (via `chat.get_history()`) | History serialization |
| `client.chats.create(model, config, history)` | `core.py`, `base_agent.py`, `planner.py` | Chat session lifecycle |
| `chat.send_message(message)` | `core.py`, `base_agent.py`, `planner.py`, `tool_loop.py` | Message sending |
| `client.models.generate_content()` | `core.py` (Google Search), `memory_agent.py` | One-shot generation |
| `response_mime_type` / `response_schema` | `planner.py` | JSON schema enforcement |
| `genai_errors.ClientError` | `model_fallback.py` | Error detection |

### 2.2 Files That Import Gemini

| File | Import | Lines |
|---|---|---|
| `agent/core.py` | `from google import genai; from google.genai import types` | 15-16 |
| `agent/base_agent.py` | `from google import genai; from google.genai import types` | 17-18 |
| `agent/planner.py` | `from google import genai; from google.genai import types` | 16-17 |
| `agent/memory_agent.py` | `from google.genai import types` | 16 |
| `agent/tool_loop.py` | `from google.genai import types` | 11 |
| `agent/thinking.py` | (none, but coupled to Gemini response shape) | — |
| `agent/model_fallback.py` | `from google.genai import errors as genai_errors` | 8 |
| `agent/session.py` | (none, but handles Gemini `Content.model_dump()` dicts) | — |
| `config.py` | (none, but `GOOGLE_API_KEY` is Gemini-specific) | 10 |

### 2.3 Interaction Patterns

There are 4 distinct Gemini interaction patterns in the codebase:

1. **Persistent chat with tools** — `OrchestratorAgent` keeps one `self.chat`
   across the entire conversation. Messages and tool results are appended to
   this session. History is saved/restored via `chat.get_history()`.

2. **Fresh chat per request with tools** — `BaseSubAgent.process_request()` and
   `BaseSubAgent.execute_task()` create a new `client.chats.create()` for each
   invocation. No history carryover.

3. **One-shot generation (no tools)** — `memory_agent.py` and the Google Search
   handler in `core.py` use `client.models.generate_content()` for single-turn
   calls without tool calling.

4. **JSON-schema-enforced generation** — `PlannerAgent` planning phase uses
   `response_mime_type="application/json"` + `response_schema` for guaranteed
   structured output.

---

## 3. Provider Landscape

### 3.1 Target Providers (Priority Order)

| Priority | Provider | Model | API Style | Why |
|---|---|---|---|---|
| **P0** (current) | Google Gemini | 2.5 Flash, 3 Flash/Pro | `google-genai` SDK | Already works |
| **P1** (high value) | OpenAI-compatible | GPT-4.1, GPT-5.2, DeepSeek V3.2, Qwen3, Kimi K2.5, MiniMax M2, Mistral Large 3, xAI Grok, local models via Ollama/vLLM | OpenAI `chat/completions` | One adapter covers ~80% of all providers |
| **P2** (high quality) | Anthropic | Sonnet 4.5, Opus 4.6 | Anthropic Messages API | Best agentic tool management |

### 3.2 Why Three Adapters Cover Everything

- **GeminiAdapter** — `google-genai` SDK (preserves current behavior, including ThinkingConfig, Google Search grounding, chat sessions)
- **OpenAIAdapter** — OpenAI SDK pointed at any compatible endpoint. Covers: OpenAI, DeepSeek, Qwen (Alibaba), Kimi (Moonshot), MiniMax, Mistral, xAI Grok, Together AI, Groq, Fireworks, Ollama, vLLM
- **AnthropicAdapter** — Anthropic SDK. Covers: Claude models, ByteDance Doubao (Anthropic-compatible)

### 3.3 Provider Capabilities Matrix

| Feature | Gemini | OpenAI | Anthropic |
|---|---|---|---|
| Tool calling | Native | Native | Native |
| Parallel tool calls | Yes | Yes | Yes (newer models) |
| Forced function calling | `mode="ANY"` | `tool_choice="required"` | `tool_choice={"type":"any"}` |
| JSON schema output | `response_mime_type` + `response_schema` | `response_format: json_schema` | Via tool-use or prefill |
| System prompt | `system_instruction` in config | `role: "system"` message | `system` parameter |
| Thinking/reasoning | `ThinkingConfig(thinking_level)` | Model-specific (o1/o3) | `thinking: {type: "enabled", budget_tokens}` |
| Message ordering | Strict user/model alternation | Flexible | Strict user/assistant alternation |
| Chat sessions | SDK-managed (`chats.create`) | Client-managed (message list) | Client-managed (message list) |
| Tool result format | `Part.from_function_response(name, response)` | `{"role":"tool", "tool_call_id":..., "content":...}` | `{"type":"tool_result", "tool_use_id":..., "content":...}` in user message |
| Tool call args format | Parsed dict (`fc.args`) | JSON string (`fc.function.arguments`) | Parsed dict (`tc.input`) |
| Tool call ID | Not explicit | `call_xxxxx` | `toolu_xxxxx` |
| Token counting | `prompt_token_count`, `candidates_token_count`, `thoughts_token_count` | `prompt_tokens`, `completion_tokens` | `input_tokens`, `output_tokens` |

---

## 4. Design Approach — Thin Custom Adapter

### 4.1 Why Not LiteLLM / LangChain / Pydantic AI

| Option | Verdict | Reason |
|---|---|---|
| **LiteLLM** | Considered, rejected | Normalizes to OpenAI format — would lose Gemini ThinkingConfig, Google Search grounding, SDK-managed chat sessions. Adds opaque translation layer. |
| **LangChain** | No | Too heavy; 45% of developers who tried it never deployed to production; declining adoption. |
| **Pydantic AI** | No | Requires restructuring entire agent layer; our 26 hand-written tool schemas would need conversion to decorated functions. |
| **Custom thin adapter** | **Yes** | Minimal code, no new framework dependencies, escape hatch for provider-specific features. Industry consensus for 2025-2026. |

### 4.2 Architecture

```
agent/
├── llm/                         # NEW — LLM abstraction layer
│   ├── __init__.py              # Factory: create_adapter(provider, ...) -> LLMAdapter
│   ├── base.py                  # Abstract base: LLMAdapter, LLMResponse, ToolCall, ChatSession
│   ├── gemini_adapter.py        # GeminiAdapter (wraps google-genai SDK)
│   ├── openai_adapter.py        # OpenAIAdapter (wraps openai SDK)
│   └── anthropic_adapter.py     # AnthropicAdapter (wraps anthropic SDK)
├── core.py                      # Modified — uses LLMAdapter instead of genai.Client
├── base_agent.py                # Modified — uses LLMAdapter instead of genai.Client
├── planner.py                   # Modified — uses LLMAdapter
├── memory_agent.py              # Modified — uses LLMAdapter
├── tool_loop.py                 # Modified — provider-agnostic response handling
├── thinking.py                  # Modified — provider-agnostic thought extraction
├── model_fallback.py            # Modified — provider-agnostic error detection
└── session.py                   # Modified — provider-agnostic history format
```

### 4.3 Design Principles

1. **Tool schemas stay as-is** — the `{name, description, parameters}` dicts in
   `agent/tools.py` are already provider-agnostic. Each adapter converts them to
   the provider's format internally.

2. **Adapters own the translation** — each adapter handles converting tool schemas,
   parsing responses, formatting tool results, and managing sessions.

3. **Escape hatch via `raw`** — `LLMResponse` includes a `raw` field with the
   original provider response for features that can't be normalized (Google
   Search grounding, Anthropic cache_control, etc.).

4. **No streaming abstraction** — streaming event formats differ too fundamentally.
   Abstract at the "complete response" level. Add streaming later per-provider if
   needed.

5. **Thinking = model selection, not a parameter** — the project uses exactly two
   model tiers: a "smart" model for orchestrator/planner (deep reasoning) and a
   "fast" model for all sub-agents (speed + tool calling). Instead of abstracting
   thinking levels across providers (each has a wildly different API — see below),
   **the user simply picks the right model pair in config**. The adapter doesn't
   need a `thinking_level` parameter at all.

   Provider thinking/reasoning control is extremely fragmented:
   | Provider | How thinking is controlled |
   |---|---|
   | Gemini | `ThinkingConfig(thinking_level="HIGH"/"LOW")` on same model |
   | OpenAI | `reasoning_effort` param (none/low/medium/high/xhigh); or pick o3 vs gpt-4.1 |
   | Anthropic | `thinking: {type: "enabled", budget_tokens: N}` on same model |
   | DeepSeek | Model name: `deepseek-reasoner` vs `deepseek-chat` (same V3.2 underneath) |
   | Qwen3 | `enable_thinking: true/false` + `thinking_budget` param; or `/think` `/no_think` in prompt |
   | Mistral | Model choice: Magistral (reasoning) vs Mistral Large (non-reasoning) |
   | Kimi K2.5 | Model variant: thinking mode vs instant mode |

   Trying to unify these behind a single `thinking_level` kwarg is brittle and
   provider-specific. The two-model-tier approach sidesteps this entirely:
   ```json
   {"model": "deepseek-reasoner", "sub_agent_model": "deepseek-chat"}
   {"model": "o3", "sub_agent_model": "gpt-4.1-mini"}
   {"model": "magistral-medium-latest", "sub_agent_model": "mistral-large-latest"}
   {"model": "qwen3-235b-a22b-thinking-2507", "sub_agent_model": "qwen3-32b"}
   ```
   The GeminiAdapter is the one exception — since Gemini controls thinking via a
   config parameter on the same model, the adapter internally applies
   `ThinkingConfig(thinking_level="HIGH")` for `model` and `"LOW"` for
   `sub_agent_model`. This is an adapter-internal detail, not exposed in the
   abstract interface.

---

## 5. Core Abstractions

### 5.1 Data Classes (`agent/llm/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolSchema:
    """Provider-agnostic tool definition.

    This is what agent/tools.py already produces — just formalized.
    """
    name: str
    description: str
    parameters: dict  # JSON Schema {"type": "object", "properties": {...}, ...}


@dataclass
class ToolCall:
    """Normalized tool invocation extracted from an LLM response."""
    id: str           # Provider-assigned ID (for sending results back)
    name: str         # Tool name
    arguments: dict   # Parsed arguments


@dataclass
class TokenUsage:
    """Normalized token counts."""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0  # 0 for providers without thinking


@dataclass
class LLMResponse:
    """Normalized LLM response — the core return type from all adapters."""
    text: Optional[str]           # Concatenated text output (None if only tool calls)
    tool_calls: list[ToolCall]    # Tool calls to execute (empty if text-only)
    thinking: Optional[str]       # Reasoning/thinking content (None if not available)
    usage: TokenUsage             # Token counts
    raw: Any                      # Original provider response (escape hatch)
    finish_reason: str = ""       # "stop", "tool_use", "length", etc.


@dataclass
class ToolResult:
    """Tool execution result to send back to the LLM."""
    tool_call_id: str   # Must match ToolCall.id
    name: str           # Tool name
    result: dict        # The result payload (will be serialized per-provider)
```

### 5.2 Abstract Adapter (`agent/llm/base.py`)

```python
class ChatSession:
    """Opaque session handle. Each adapter stores its own state here."""
    pass


class LLMAdapter(ABC):
    """Abstract interface for LLM provider adapters.

    Adapters handle:
    - Converting ToolSchema to provider format
    - Creating and managing chat sessions
    - Sending messages and tool results
    - Parsing responses into LLMResponse
    - One-shot generation (no tools, no session)
    """

    @abstractmethod
    def create_session(
        self,
        system_prompt: str,
        tools: list[ToolSchema],
        *,
        force_tool_calling: bool = False,
        json_schema: Optional[dict] = None,  # For structured output
        history: Optional[list] = None,      # For session resumption
        **kwargs,
    ) -> ChatSession:
        """Create a new chat session with the given configuration.

        Args:
            system_prompt: System instructions for the LLM.
            tools: Tool definitions available in this session.
            force_tool_calling: If True, the LLM must call a tool (no text-only).
            json_schema: If set, enforce structured JSON output matching this schema.
            history: Serialized history from a previous session for resumption.
            **kwargs: Provider-specific options (passed through to adapter).

        Note: Thinking/reasoning level is NOT a session parameter. It is controlled
        by model selection — the user configures a "smart" model (orchestrator/planner)
        and a "fast" model (sub-agents) in config.json. Each adapter applies
        provider-specific thinking config internally based on the model name.
        """
        ...

    @abstractmethod
    def send_message(
        self,
        session: ChatSession,
        message: str,
    ) -> LLMResponse:
        """Send a user text message to the LLM."""
        ...

    @abstractmethod
    def send_tool_results(
        self,
        session: ChatSession,
        results: list[ToolResult],
    ) -> LLMResponse:
        """Send tool execution results back to the LLM."""
        ...

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        json_schema: Optional[dict] = None,
        **kwargs,
    ) -> LLMResponse:
        """One-shot generation without tools or session state.

        Used by MemoryAgent, Google Search handler, etc.
        """
        ...

    @abstractmethod
    def get_history(self, session: ChatSession) -> list:
        """Serialize session history for persistence.

        Returns a provider-agnostic list of dicts that can be JSON-serialized
        and later passed to create_session(history=...).
        """
        ...

    @abstractmethod
    def is_quota_error(self, exc: Exception) -> bool:
        """Check if an exception is a rate-limit / quota error."""
        ...
```

### 5.3 Factory (`agent/llm/__init__.py`)

```python
def create_adapter(
    provider: str,
    api_key: str,
    model: str,
    *,
    base_url: Optional[str] = None,
    timeout_ms: int = 300_000,
    **kwargs,
) -> LLMAdapter:
    """Create an LLM adapter for the given provider and model.

    Args:
        provider: "gemini", "openai", or "anthropic"
        api_key: API key for the provider
        model: Model identifier (e.g. "gemini-2.5-flash", "gpt-4.1", "claude-sonnet-4-5-20250929")
        base_url: Override base URL (for OpenAI-compatible endpoints like DeepSeek, Ollama)
        timeout_ms: HTTP timeout in milliseconds

    The application creates TWO adapter instances — one per model tier:
        smart_adapter = create_adapter(provider, api_key, model=LLM_MODEL, ...)
        fast_adapter  = create_adapter(provider, api_key, model=LLM_SUB_AGENT_MODEL, ...)

    The orchestrator and planner use smart_adapter. All sub-agents use fast_adapter.
    For providers where thinking is controlled by model name (DeepSeek, Qwen, Mistral),
    this naturally gives the right behavior. For providers where thinking is a config
    parameter (Gemini, Anthropic), the adapter can detect the model and apply internally.
    """
    if provider == "gemini":
        from .gemini_adapter import GeminiAdapter
        return GeminiAdapter(api_key=api_key, model=model, timeout_ms=timeout_ms, **kwargs)
    elif provider == "openai":
        from .openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key=api_key, model=model, base_url=base_url, timeout_ms=timeout_ms, **kwargs)
    elif provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key=api_key, model=model, timeout_ms=timeout_ms, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

---

## 6. Adapter Implementations

### 6.1 GeminiAdapter (`agent/llm/gemini_adapter.py`)

Wraps the existing `google-genai` SDK. This is largely a reorganization of the
current code in `core.py` and `base_agent.py`.

**Key translation points:**

| Adapter method | Gemini SDK call |
|---|---|
| `create_session()` | `client.chats.create(model, config)` where config = `GenerateContentConfig(system_instruction, tools, thinking_config, tool_config)` |
| `send_message()` | `chat.send_message(text)` → parse `response.candidates[0].content.parts` |
| `send_tool_results()` | `chat.send_message([Part.from_function_response(...)])` |
| `generate()` | `client.models.generate_content(model, contents, config)` |
| `get_history()` | `chat.get_history()` → `[c.model_dump(exclude_none=True) for c in history]` |
| `is_quota_error()` | Check `genai_errors.ClientError` with code 429 |

**ToolSchema → Gemini:**
```python
types.FunctionDeclaration(
    name=schema.name,
    description=schema.description,
    parameters=schema.parameters,
)
```

**Response → LLMResponse:**
```python
# Extract tool calls
tool_calls = []
for part in response.candidates[0].content.parts:
    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
        tool_calls.append(ToolCall(
            id=part.function_call.name,  # Gemini has no explicit tool call ID
            name=part.function_call.name,
            arguments=dict(part.function_call.args) if part.function_call.args else {},
        ))

# Extract thinking
thoughts = [p.text for p in parts if getattr(p, "thought", False) and p.text]

# Extract text
texts = [p.text for p in parts if hasattr(p, "text") and p.text and not getattr(p, "thought", False)]
```

**ToolResult → Gemini:**
```python
types.Part.from_function_response(
    name=result.name,
    response={"result": result.result},
)
```

**Special: ThinkingConfig** — applied internally based on whether the model is
the orchestrator/planner model or a sub-agent model. Not exposed in the abstract
interface. The adapter constructor receives `is_smart_model: bool` (or infers it
from model name) and applies accordingly:
```python
# Applied internally in create_session(), not controlled by caller
if self._is_smart_model:
    thinking_config = types.ThinkingConfig(include_thoughts=True, thinking_level="HIGH")
else:
    thinking_config = types.ThinkingConfig(include_thoughts=True, thinking_level="LOW")
```

**Special: JSON schema enforcement:**
```python
if json_schema:
    config.response_mime_type = "application/json"
    config.response_schema = json_schema
```

**Special: Google Search grounding** — exposed as an adapter-specific method
(not part of the abstract interface), accessible via the `raw` escape hatch or
a Gemini-specific method:
```python
class GeminiAdapter(LLMAdapter):
    def google_search(self, query: str, system_prompt: str) -> LLMResponse:
        """Gemini-only: perform Google Search grounding."""
        ...
```

### 6.2 OpenAIAdapter (`agent/llm/openai_adapter.py`)

Uses the `openai` Python SDK. A single adapter covers OpenAI, DeepSeek, Qwen,
Kimi, MiniMax, Mistral, xAI Grok, Ollama, vLLM, Together, Groq, Fireworks.

**Key translation points:**

| Adapter method | OpenAI SDK call |
|---|---|
| `create_session()` | Store `messages: list[dict]` in session (client-managed history) |
| `send_message()` | `client.chat.completions.create(model, messages, tools)` |
| `send_tool_results()` | Append `{"role":"tool", "tool_call_id":..., "content":...}` then call `completions.create()` |
| `generate()` | `client.chat.completions.create(model, messages)` (single-turn) |
| `get_history()` | Return the `messages` list directly |
| `is_quota_error()` | Check `openai.RateLimitError` |

**ToolSchema → OpenAI:**
```python
{
    "type": "function",
    "function": {
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters,
    }
}
```

**Response → LLMResponse:**
```python
choice = response.choices[0]
message = choice.message

tool_calls = []
if message.tool_calls:
    for tc in message.tool_calls:
        tool_calls.append(ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments),
        ))

text = message.content  # str or None
usage = TokenUsage(
    input_tokens=response.usage.prompt_tokens,
    output_tokens=response.usage.completion_tokens,
)
```

**ToolResult → OpenAI:**
```python
{
    "role": "tool",
    "tool_call_id": result.tool_call_id,
    "content": json.dumps(result.result),
}
```

**Special: Forced function calling:**
```python
if force_tool_calling:
    tool_choice = "required"
```

**Special: JSON schema enforcement:**
```python
if json_schema:
    response_format = {"type": "json_schema", "json_schema": {...}}
```

**Special: System prompt** — prepended as `{"role": "system", "content": system_prompt}`.

**Special: Base URL override** — for OpenAI-compatible providers:
```python
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url or "https://api.openai.com/v1",
)
```

### 6.3 AnthropicAdapter (`agent/llm/anthropic_adapter.py`)

Uses the `anthropic` Python SDK.

**Key translation points:**

| Adapter method | Anthropic SDK call |
|---|---|
| `create_session()` | Store `messages: list[dict]` + system prompt in session |
| `send_message()` | `client.messages.create(model, system, messages, tools)` |
| `send_tool_results()` | Append user message with `tool_result` blocks, then call `messages.create()` |
| `generate()` | `client.messages.create(model, system, messages)` |
| `get_history()` | Return the `messages` list |
| `is_quota_error()` | Check `anthropic.RateLimitError` |

**ToolSchema → Anthropic:**
```python
{
    "name": schema.name,
    "description": schema.description,
    "input_schema": schema.parameters,  # Note: input_schema, not parameters
}
```

**Response → LLMResponse:**
```python
tool_calls = []
texts = []
for block in response.content:
    if block.type == "tool_use":
        tool_calls.append(ToolCall(
            id=block.id,
            name=block.name,
            arguments=block.input,
        ))
    elif block.type == "text":
        texts.append(block.text)
    elif block.type == "thinking":
        thinking_parts.append(block.thinking)

usage = TokenUsage(
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens,
)
```

**ToolResult → Anthropic:**
```python
# Tool results must be in a user message
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": json.dumps(result.result),
        }
        for result in results
    ]
}
```

**Special: Forced function calling:**
```python
if force_tool_calling:
    tool_choice = {"type": "any"}
```

**Special: Thinking** — like Gemini, Anthropic controls thinking via API parameter
on the same model. The adapter decides internally whether to enable thinking based
on `is_smart_model`. For the "smart" model (orchestrator/planner), enable extended
thinking with a generous budget; for "fast" models (sub-agents), either disable
or use a small budget:
```python
if self._is_smart_model:
    thinking = {"type": "enabled", "budget_tokens": 16384}
else:
    thinking = {"type": "enabled", "budget_tokens": 2048}  # or disabled
```

**Special: Message alternation** — Anthropic requires strict user/assistant alternation.
The adapter must merge consecutive same-role messages.

---

## 7. Integration Points — File-by-File Changes

### 7.1 `config.py`

**Current:**
```python
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = get("model", "gemini-3-flash-preview")
GEMINI_SUB_AGENT_MODEL = get("sub_agent_model", "gemini-3-flash-preview")
```

**New:**
```python
# Provider selection
LLM_PROVIDER = get("provider", "gemini")          # "gemini", "openai", "anthropic"
LLM_API_KEY = os.getenv(f"{LLM_PROVIDER.upper()}_API_KEY") or os.getenv("GOOGLE_API_KEY")  # backward compat
LLM_BASE_URL = get("base_url", None)               # For OpenAI-compatible providers
LLM_MODEL = get("model", "gemini-3-flash-preview")
LLM_SUB_AGENT_MODEL = get("sub_agent_model", LLM_MODEL)
LLM_PLANNER_MODEL = get("planner_model", LLM_MODEL)
LLM_FALLBACK_MODEL = get("fallback_model", None)

# Backward compat aliases (deprecate later)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = LLM_MODEL
GEMINI_SUB_AGENT_MODEL = LLM_SUB_AGENT_MODEL
```

### 7.2 `agent/core.py` — OrchestratorAgent

**Current (lines 108-147):**
```python
self.client = genai.Client(api_key=GOOGLE_API_KEY, ...)
# ... build types.FunctionDeclaration ...
self.config = types.GenerateContentConfig(...)
self.chat = self.client.chats.create(model=..., config=self.config)
```

**New:**
```python
from agent.llm import create_adapter
from agent.llm.base import ToolSchema

# Two adapters: "smart" for orchestrator/planner, "fast" for sub-agents
self.adapter = create_adapter(
    provider=LLM_PROVIDER,
    api_key=LLM_API_KEY,
    model=self.model_name,          # e.g. "o3", "deepseek-reasoner", "gemini-3-pro-preview"
    base_url=LLM_BASE_URL,
)
self.sub_adapter = create_adapter(
    provider=LLM_PROVIDER,
    api_key=LLM_API_KEY,
    model=SUB_AGENT_MODEL,          # e.g. "gpt-4.1-mini", "deepseek-chat", "gemini-3-flash-preview"
    base_url=LLM_BASE_URL,
)
# sub_adapter is passed to MissionAgent, VisualizationAgent, DataOpsAgent, etc.

tool_schemas = [
    ToolSchema(name=t["name"], description=t["description"], parameters=t["parameters"])
    for t in get_tool_schemas(categories=ORCHESTRATOR_CATEGORIES, extra_names=ORCHESTRATOR_EXTRA_TOOLS)
]
self.session = self.adapter.create_session(
    system_prompt=get_system_prompt(gui_mode=gui_mode),
    tools=tool_schemas,
)
```

**Message sending (was `chat.send_message`):**
```python
response = self.adapter.send_message(self.session, user_message)
```

**Tool results (was `types.Part.from_function_response`):**
```python
from agent.llm.base import ToolResult
results = [
    ToolResult(tool_call_id=tc.id, name=tc.name, result=sanitized_result)
    for tc, (_, _, sanitized_result) in zip(response.tool_calls, tool_results)
]
response = self.adapter.send_tool_results(self.session, results)
```

**Response parsing (was `response.candidates[0].content.parts`):**
```python
# Now normalized:
if response.tool_calls:
    for tc in response.tool_calls:
        tool_name = tc.name
        tool_args = tc.arguments
        ...
if response.text:
    final_text = response.text
```

### 7.3 `agent/base_agent.py` — BaseSubAgent

Replace `self.client: genai.Client` with `self.adapter: LLMAdapter`. The
`process_request()` and `execute_task()` methods use adapter methods instead
of raw Gemini calls.

**Constructor change:**
```python
# Old
def __init__(self, client: genai.Client, model_name: str, ...):
    self.client = client
    self._function_declarations = [types.FunctionDeclaration(...) for ...]

# New
def __init__(self, adapter: LLMAdapter, model_name: str, ...):
    self.adapter = adapter
    self._tool_schemas = [ToolSchema(...) for ...]
```

**process_request() change:**
```python
# Old
conv_config = types.GenerateContentConfig(...)
chat = self.client.chats.create(model=..., config=conv_config)
response = self._send_with_timeout(chat, user_message)

# New
session = self.adapter.create_session(
    system_prompt=self.system_prompt,
    tools=self._tool_schemas,
)
response = self._send_with_timeout(self.adapter, session, user_message)
```

Note: No `thinking_level` — the sub-agent adapter was already constructed with the
`sub_agent_model` (fast model), so thinking is inherently controlled by model choice.
For Gemini, the adapter internally applies `ThinkingConfig(thinking_level="LOW")`
based on knowing this is the sub-agent model.

### 7.4 `agent/planner.py` — PlannerAgent

Two phases need different adapter configurations. The planner uses the "smart" model
(same as orchestrator), so thinking is controlled by model choice:

1. **Discovery phase**: `adapter.create_session(tools=discovery_tools)`
2. **Planning phase**: `adapter.create_session(tools=[], json_schema=PLANNER_RESPONSE_SCHEMA)`

### 7.5 `agent/memory_agent.py` — MemoryAgent

Currently uses `client.models.generate_content()` for one-shot calls.

**New:** `adapter.generate(system_prompt=..., user_message=...)` — no tools, no session.

### 7.6 `agent/tool_loop.py`

Currently tightly coupled to `types.Part.from_function_response`. The loop needs
to work with `LLMAdapter.send_tool_results()` instead.

**New signature:**
```python
def run_tool_loop(
    adapter: LLMAdapter,
    session: ChatSession,
    response: LLMResponse,
    tool_executor,
    ...
) -> LLMResponse:
```

The loop body changes from Gemini-specific part inspection to using the
normalized `response.tool_calls` list.

### 7.7 `agent/thinking.py`

**Current:** Inspects `response.candidates[0].content.parts` for `part.thought`.
**New:** Simply reads `response.thinking` from `LLMResponse`.

```python
def extract_thoughts(response: LLMResponse) -> list[str]:
    if response.thinking:
        return [response.thinking]
    return []
```

### 7.8 `agent/model_fallback.py`

**Current:** Checks `genai_errors.ClientError`.
**New:** Delegates to `adapter.is_quota_error(exc)`.

```python
def is_quota_error(adapter: LLMAdapter, exc: Exception) -> bool:
    return adapter.is_quota_error(exc)
```

### 7.9 `agent/session.py`

**Current:** Handles Gemini `Content.model_dump()` dicts with `thought_signature` bytes.
**New:** Uses `adapter.get_history(session)` which returns a provider-agnostic format.

The serialization format changes from Gemini Content dicts to a normalized format:
```python
[
    {"role": "user", "content": "Show me ACE data"},
    {"role": "assistant", "content": "I'll fetch that for you.", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."},
    ...
]
```

**Migration:** Old Gemini-format sessions detected by presence of `"parts"` key;
new format uses `"role"`/`"content"`. The `load_session()` method handles both.

---

## 8. Configuration Changes

### 8.1 `~/.helio-agent/config.json`

```json
{
    "provider": "gemini",
    "model": "gemini-3-flash-preview",
    "sub_agent_model": "gemini-3-flash-preview",
    "planner_model": "gemini-3-flash-preview",
    "fallback_model": "gemini-2.5-flash",

    "base_url": null,

    "data_backend": "cdf"
}
```

**Example: DeepSeek with reasoning for orchestrator, non-reasoning for sub-agents:**
```json
{
    "provider": "openai",
    "model": "deepseek-reasoner",
    "sub_agent_model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1"
}
```
With `.env`:
```
OPENAI_API_KEY=sk-...
```

**Example: OpenAI with o3 for orchestrator, gpt-4.1-mini for sub-agents:**
```json
{
    "provider": "openai",
    "model": "o3",
    "sub_agent_model": "gpt-4.1-mini"
}
```
With `.env`:
```
OPENAI_API_KEY=sk-...
```

**Example: Anthropic Claude Opus for orchestrator, Haiku for sub-agents:**
```json
{
    "provider": "anthropic",
    "model": "claude-opus-4-6",
    "sub_agent_model": "claude-haiku-4-5-20251001"
}
```
With `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

**Example: Qwen3 thinking for orchestrator, fast for sub-agents:**
```json
{
    "provider": "openai",
    "model": "qwen3-235b-a22b-thinking-2507",
    "sub_agent_model": "qwen3-32b",
    "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
}
```

**Example: Mistral Magistral for orchestrator, Large for sub-agents:**
```json
{
    "provider": "openai",
    "model": "magistral-medium-latest",
    "sub_agent_model": "mistral-large-latest",
    "base_url": "https://api.mistral.ai/v1"
}
```

**Example: Local Ollama:**
```json
{
    "provider": "openai",
    "model": "llama4-scout",
    "sub_agent_model": "llama4-scout",
    "base_url": "http://localhost:11434/v1"
}
```
With `.env`:
```
OPENAI_API_KEY=ollama
```

### 8.2 `.env` — API Keys

```bash
# One key per provider (only the active provider's key is needed)
GOOGLE_API_KEY=...      # For provider=gemini
OPENAI_API_KEY=...      # For provider=openai (also DeepSeek, Qwen, Kimi, etc.)
ANTHROPIC_API_KEY=...   # For provider=anthropic
```

### 8.3 CLI Flag

```bash
python main.py --provider openai --model gpt-4.1 --base-url https://api.deepseek.com/v1
```

---

## 9. Session Persistence Changes

### 9.1 Normalized History Format

Sessions currently store Gemini `Content.model_dump()` dicts. After abstraction,
we store a provider-agnostic format:

```json
[
    {
        "role": "user",
        "content": "Show me ACE magnetic field data for last week"
    },
    {
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {
                "id": "tc_001",
                "name": "delegate_to_mission",
                "arguments": {"mission": "ACE", "request": "..."}
            }
        ]
    },
    {
        "role": "tool_result",
        "results": [
            {
                "tool_call_id": "tc_001",
                "name": "delegate_to_mission",
                "content": "{\"status\": \"success\", ...}"
            }
        ]
    },
    {
        "role": "assistant",
        "content": "Here's the ACE magnetic field data..."
    }
]
```

### 9.2 Backward Compatibility

`session.py:load_session()` detects old Gemini format (presence of `"parts"` key)
and converts on-the-fly to the normalized format. Old sessions can only be
resumed with the Gemini provider; cross-provider session resumption is not
supported (different models have different context anyway).

### 9.3 Provider Tag in Metadata

```json
{
    "id": "20260211_143022_a1b2c3d4",
    "provider": "gemini",
    "model": "gemini-3-flash-preview",
    "history_format": "v2"
}
```

---

## 10. Provider-Specific Features — Escape Hatches

Some features are inherently provider-specific and should NOT be abstracted:

| Feature | Provider | Approach |
|---|---|---|
| Google Search grounding | Gemini only | `GeminiAdapter.google_search()` method; core.py checks `isinstance(adapter, GeminiAdapter)` |
| ThinkingConfig levels | Gemini | Mapped from `thinking_level` kwarg; other providers map to their equivalents or ignore |
| Extended thinking with budget | Anthropic | AnthropicAdapter interprets `thinking_level` as budget sizes |
| Context caching | Gemini, Anthropic (different APIs) | Future optimization, adapter-specific methods |
| `cache_control` on messages | Anthropic | Pass-through via `**kwargs` on `create_session` |
| Prompt caching | OpenAI (automatic) | No action needed |
| `response.grounding_metadata` | Gemini | Access via `response.raw` |

### Fallback for Missing Features

When `google_search` tool is called but provider is not Gemini:
- Option A: Return error "Google Search grounding is only available with the Gemini provider"
- Option B: Fall back to a web search API (e.g., SerpAPI, Tavily) — separate from LLM
- **Recommended:** Option A initially, Option B as a future enhancement

When `json_schema` is requested but provider doesn't support it:
- OpenAI: Native support via `response_format`
- Anthropic: Use tool-based structured output (define a tool with the schema, force the LLM to call it)
- Gemini: Native support via `response_mime_type`

---

## 11. Migration Strategy

### Phase 1: Extract (non-breaking)
- Create `agent/llm/` package with base abstractions
- Implement `GeminiAdapter` wrapping all existing Gemini code
- All existing code continues to work unchanged

### Phase 2: Refactor (non-breaking)
- Modify `core.py`, `base_agent.py`, `planner.py`, `memory_agent.py`, `tool_loop.py`
  to use `LLMAdapter` instead of direct `google-genai` imports
- `config.py` gets `provider` setting (default: `"gemini"` — backward compatible)
- All tests pass with Gemini adapter (no behavior change)

### Phase 3: Add providers
- Implement `OpenAIAdapter`
- Implement `AnthropicAdapter`
- Add `openai` and `anthropic` to `requirements.txt` as optional deps
- Test with DeepSeek, Qwen, Ollama, Claude

### Phase 4: Polish
- Session format migration
- CLI `--provider` flag
- Config template update
- Documentation update
- Gradio UI provider selector

---

## 12. What NOT to Abstract

Per Armin Ronacher's analysis ("LLM APIs are a Synchronization Problem"), these
should remain provider-specific:

1. **Streaming event formats** — SSE delta shapes differ fundamentally. Abstract
   at "complete response" level; add streaming per-provider later if needed.

2. **Token counting semantics** — thinking tokens, cached tokens, reasoning tokens
   have different meanings per provider. Track them but don't pretend they're
   comparable.

3. **Prompt caching** — Gemini's context caching, Anthropic's `cache_control`,
   OpenAI's automatic caching are all different mechanisms. Expose as adapter-
   specific methods.

4. **Multimodal content** — Image/audio/video handling differs significantly.
   Currently only used for `read_document` tool (Gemini vision). Keep as
   adapter-specific.

5. **Hidden state** — Providers inject reasoning tokens, encrypted blobs
   (`thought_signature`), and other state that can't be normalized.

---

## 13. Testing Strategy

### 13.1 Unit Tests (No API Key)

- **Mock adapter tests** — verify that `core.py`, `base_agent.py`, etc. correctly
  use the adapter interface by mocking `LLMAdapter` methods.
- **Adapter translation tests** — verify that each adapter correctly converts
  `ToolSchema` → provider format and provider response → `LLMResponse`.
- **Session roundtrip tests** — verify history serialization/deserialization.

### 13.2 Integration Tests (API Key Required)

- **Per-provider smoke test** — send a simple tool-calling request through each
  adapter and verify the round-trip works.
- **Tool schema compatibility** — verify that all 26 tool schemas are accepted
  by each provider without errors.
- **JSON schema enforcement** — verify structured output works with each provider.

### 13.3 Regression Tests

- Run existing `run_agent_tests.py` suite with Gemini adapter (must pass as-is).
- Run the same suite with OpenAI adapter (DeepSeek) and Anthropic adapter to
  verify behavioral compatibility.

---

## 14. Implementation Phases

### Phase 1: Core Abstractions + Gemini Adapter ✅ COMPLETE

**Branch:** `feature/llm-abstraction`
**928 tests passing, 0 failures.**

Files created:
- `agent/llm/__init__.py` — package init, re-exports public API
- `agent/llm/base.py` — `ToolCall`, `UsageMetadata`, `LLMResponse`, `FunctionSchema`, `ChatSession` ABC, `LLMAdapter` ABC
- `agent/llm/gemini_adapter.py` — `GeminiAdapter`, `GeminiChatSession`, Gemini-specific escape hatches (`google_search`, `generate_multimodal`, `make_bytes_part`)
- `tests/test_llm_adapter.py` — 17 tests for adapter types and GeminiAdapter (mocked SDK)

Files modified:
- `config.py` — added `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `get_api_key()`
- `config.template.json` — added `llm_provider`, `llm_api_key`, `llm_base_url` fields
- `agent/model_fallback.py` — removed `google.genai` import, `is_quota_error()` accepts optional adapter
- `agent/thinking.py` — works with `LLMResponse` (backward-compat fallback to raw Gemini)
- `agent/tool_loop.py` — added `adapter` param, uses `adapter.make_tool_result_message()`
- `agent/base_agent.py` — `client: genai.Client` → `adapter: LLMAdapter` (~30 touchpoints)
- `agent/planner.py` — same pattern (~15 touchpoints)
- `agent/memory_agent.py` — `client` → `adapter`, `generate_content` → `adapter.generate()`
- `agent/mission_agent.py`, `visualization_agent.py`, `data_ops_agent.py`, `data_extraction_agent.py` — constructor parameter updates
- `agent/core.py` — the big one (~50 touchpoints): `genai.Client` → `GeminiAdapter`, all response parsing via `LLMResponse`
- `tests/test_agent.py`, `tests/test_planner.py`, `tests/test_memory_agent.py` — updated for new parameter names

**Key design decisions in implementation (diverging from original plan):**
- `ToolCall.args` is a plain `dict` (no `.id` field — Gemini doesn't have explicit tool call IDs; handled by `make_tool_result_message` per-provider)
- `ChatSession.send()` replaces both `send_message()` and `send_tool_results()` — tool results are passed as provider-specific objects built by `make_tool_result_message()`
- `create_chat()` has a `thinking` param ("low"/"high"/"default") even though the design doc said thinking=model selection. This is needed because Gemini uses the same model with different ThinkingConfig levels. Other adapters can ignore it.
- `tool_loop.py` has a lazy-import fallback for `google.genai.types` when no adapter is provided (safety net during migration)
- `_extract_grounding_sources()` and `_log_grounding_queries()` access `response.raw` for Gemini-specific grounding metadata
- `generate()` on LLMAdapter accepts `contents: str | list` to support multimodal via `generate_multimodal()` escape hatch

### Phase 2: OpenAI Adapter (estimated ~350 lines new code) — PENDING

Create `agent/llm/openai_adapter.py` implementing `LLMAdapter`:
- Implement `create_chat()` → client-managed message list (no SDK chat sessions)
- Implement `send()` → `client.chat.completions.create(model, messages, tools)`
- Implement `make_tool_result_message()` → `{"role": "tool", "tool_call_id": ..., "content": ...}`
- Handle `ToolCall.id` — OpenAI assigns `call_xxxxx` IDs that must round-trip via tool results
- `force_tool_call=True` → `tool_choice="required"`
- `json_schema` → `response_format: {"type": "json_schema", ...}`
- `is_quota_error()` → check `openai.RateLimitError`
- `base_url` param for OpenAI-compatible providers (DeepSeek, Qwen, Ollama, etc.)
- `thinking` param: OpenAI has `reasoning_effort` for o3/o4-mini; ignore for other models
- `get_history()` → return the message list directly

Implementation notes from Phase 1:
- `make_tool_result_message()` must include `tool_call_id` — Phase 1 ToolCall has no `.id` field since Gemini doesn't use one. Options: (a) add `id: str | None` to ToolCall, populate from provider; (b) have the adapter track IDs internally in the ChatSession. Option (a) is cleaner — add optional `id` field to `ToolCall` in `base.py`, populate in OpenAI/Anthropic adapters, leave None for Gemini.
- `ChatSession.send()` receives either a string or a list of tool-result objects. For OpenAI, tool results are dicts; the session appends them to the message list and calls completions.create.
- The `tool_loop.py` fallback import of `google.genai.types` can be removed once Phase 2 confirms all callers pass an adapter.

Files to create:
- `agent/llm/openai_adapter.py` (~300 lines)
- `tests/test_openai_adapter.py`

Files to modify:
- `agent/llm/base.py` — add `id: str | None = None` to `ToolCall`
- `requirements.txt` — add `openai>=1.0.0` (optional dep)

### Phase 3: Anthropic Adapter (estimated ~400 lines new code) — PENDING

Create `agent/llm/anthropic_adapter.py` implementing `LLMAdapter`:
- Implement client-managed message list (like OpenAI adapter)
- Handle strict user/assistant alternation (merge consecutive same-role messages)
- `make_tool_result_message()` → `{"type": "tool_result", "tool_use_id": ..., "content": ...}` (must be in user message)
- Tool results from multiple calls must be batched in a single user message
- `force_tool_call=True` → `tool_choice={"type": "any"}`
- `json_schema` → use tool-based structured output (define schema as a tool, force call)
- `thinking` param → `thinking: {"type": "enabled", "budget_tokens": N}` — map "high"→16384, "low"→2048
- `is_quota_error()` → check `anthropic.RateLimitError`
- Extended thinking content → `LLMResponse.thoughts`

Implementation notes from Phase 1:
- `ToolCall.id` (added in Phase 2) will carry `toolu_xxxxx` IDs from Anthropic
- `system` parameter is separate from messages in Anthropic API (not a system message)
- `response.content` blocks: `text`, `tool_use`, `thinking` — map to our LLMResponse fields

Files to create:
- `agent/llm/anthropic_adapter.py` (~350 lines)
- `tests/test_anthropic_adapter.py`

Files to modify:
- `requirements.txt` — add `anthropic>=0.40.0` (optional dep)

### Phase 4: Session Persistence + Config + UI (estimated ~200 lines modified) — PENDING

Normalize session history format so sessions saved with one provider can at least be detected:
- Add `provider` tag to session metadata
- Old Gemini-format sessions (with `"parts"` key) continue to work with Gemini adapter
- New sessions use a provider-agnostic format (see design doc Section 9)
- Cross-provider session resumption NOT supported (tag mismatch → fresh chat)

Add CLI and UI support:
- `main.py` — `--provider` CLI flag
- `gradio_app.py` — provider selector dropdown
- `agent/llm/__init__.py` — `create_adapter(provider, ...)` factory function
- `agent/core.py` — read `LLM_PROVIDER` from config to select adapter at startup
- Remove `tool_loop.py` backward-compat lazy import of `google.genai.types`
- Update `docs/capability-summary.md`

Optional Phase 5 (cost optimization):
- Context caching for repeated system prompts (Gemini: 90% discount, Anthropic: cache_control)
- Add `cache_system_prompt()` as adapter-specific method
- OpenAI: automatic prompt caching, no action needed

---

## 15. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gemini-specific features break when switching providers | High | Medium | Escape hatches (`raw` field), feature flags, graceful degradation |
| Session history incompatibility across providers | High | Low | Don't support cross-provider session resumption; tag sessions with provider |
| Tool schema edge cases per provider | Medium | Medium | Extensive test suite with all 26 schemas per provider |
| Token counting differences cause confusion | Medium | Low | Clearly label token types in UI; don't compare across providers |
| Performance regression from abstraction layer | Low | Low | Adapter is a thin wrapper; no extra network calls |
| Google Search grounding unavailable on non-Gemini | High | Medium | Return clear error message; future: alternative search APIs |
| Sub-agents behave differently across providers | Medium | High | Run `run_agent_tests.py` per-provider; document known differences |
| Chinese provider (DeepSeek, Qwen) API instability | Medium | Low | Use as alternatives, not primary; timeout/retry in adapter |

---

## Appendix: Quick Reference — Provider API Keys + Base URLs

| Provider | API Key Env Var | Base URL | Model Examples |
|---|---|---|---|
| Google Gemini | `GOOGLE_API_KEY` | (SDK default) | `gemini-2.5-flash`, `gemini-3-flash-preview`, `gemini-3-pro-preview` |
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` | `gpt-4.1`, `gpt-4.1-mini`, `gpt-5.2`, `o3` |
| Anthropic | `ANTHROPIC_API_KEY` | (SDK default) | `claude-sonnet-4-5-20250929`, `claude-opus-4-6`, `claude-haiku-4-5-20251001` |
| DeepSeek | `OPENAI_API_KEY` | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-reasoner` |
| Alibaba Qwen | `OPENAI_API_KEY` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen3-235b-a22b`, `qwen3-32b` |
| Moonshot Kimi | `OPENAI_API_KEY` | `https://api.moonshot.cn/v1` | `kimi-k2.5` |
| MiniMax | `OPENAI_API_KEY` | `https://api.minimax.chat/v1` | `MiniMax-M2.1` |
| Mistral | `OPENAI_API_KEY` | `https://api.mistral.ai/v1` | `mistral-large-latest` |
| xAI Grok | `OPENAI_API_KEY` | `https://api.x.ai/v1` | `grok-4.1-fast` |
| Ollama (local) | `OPENAI_API_KEY=ollama` | `http://localhost:11434/v1` | `llama4-scout`, `qwen3:32b` |
| Together AI | `OPENAI_API_KEY` | `https://api.together.xyz/v1` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| Groq | `OPENAI_API_KEY` | `https://api.groq.com/openai/v1` | `llama-4-scout-17b-16e-instruct` |
