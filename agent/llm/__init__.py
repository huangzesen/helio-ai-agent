"""LLM abstraction layer — provider-agnostic interface for LLM interactions.

Re-exports the public API so consumers can write:
    from agent.llm import LLMAdapter, GeminiAdapter, LLMResponse, ...
"""

from .base import LLMAdapter, LLMResponse, ToolCall, UsageMetadata, ChatSession, FunctionSchema
from .gemini_adapter import GeminiAdapter
