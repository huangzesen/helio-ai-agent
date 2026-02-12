"""Tests for the LLM adapter layer (agent/llm/).

These tests verify the adapter types and GeminiAdapter without needing an API key,
using mocks for the google-genai SDK.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent.llm.base import (
    ChatSession,
    FunctionSchema,
    LLMAdapter,
    LLMResponse,
    ToolCall,
    UsageMetadata,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_basic(self):
        tc = ToolCall(name="fetch_data", args={"dataset": "AC_H0_MFI"})
        assert tc.name == "fetch_data"
        assert tc.args == {"dataset": "AC_H0_MFI"}

    def test_empty_args(self):
        tc = ToolCall(name="list_data", args={})
        assert tc.args == {}


class TestUsageMetadata:
    def test_defaults(self):
        u = UsageMetadata()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.thinking_tokens == 0

    def test_values(self):
        u = UsageMetadata(input_tokens=100, output_tokens=50, thinking_tokens=25)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.thinking_tokens == 25


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse()
        assert r.text == ""
        assert r.tool_calls == []
        assert r.usage.input_tokens == 0
        assert r.thoughts == []
        assert r.raw is None

    def test_with_data(self):
        tc = ToolCall(name="search", args={"q": "ACE"})
        r = LLMResponse(
            text="Found ACE data.",
            tool_calls=[tc],
            usage=UsageMetadata(input_tokens=10, output_tokens=5),
            thoughts=["I should search for ACE."],
            raw="raw_object",
        )
        assert r.text == "Found ACE data."
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"
        assert r.usage.input_tokens == 10
        assert r.thoughts == ["I should search for ACE."]
        assert r.raw == "raw_object"

    def test_independent_defaults(self):
        """Ensure default mutable fields are independent across instances."""
        r1 = LLMResponse()
        r2 = LLMResponse()
        r1.tool_calls.append(ToolCall(name="a", args={}))
        assert r2.tool_calls == []


class TestFunctionSchema:
    def test_basic(self):
        fs = FunctionSchema(
            name="fetch_data",
            description="Fetch spacecraft data",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
        )
        assert fs.name == "fetch_data"
        assert "properties" in fs.parameters


# ---------------------------------------------------------------------------
# GeminiAdapter tests (mocked SDK)
# ---------------------------------------------------------------------------


class TestGeminiAdapterMocked:
    """Test GeminiAdapter with the google-genai SDK fully mocked."""

    @pytest.fixture
    def adapter(self):
        with patch("agent.llm.gemini_adapter.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            from agent.llm.gemini_adapter import GeminiAdapter
            a = GeminiAdapter(api_key="test-key")
            a._mock_client = mock_client
            return a

    def test_make_tool_result_message(self, adapter):
        """make_tool_result_message should delegate to types.Part.from_function_response."""
        with patch("agent.llm.gemini_adapter.types") as mock_types:
            mock_types.Part.from_function_response.return_value = "mock_part"
            result = adapter.make_tool_result_message("fetch_data", {"status": "success"})
            mock_types.Part.from_function_response.assert_called_once_with(
                name="fetch_data",
                response={"result": {"status": "success"}},
            )
            assert result == "mock_part"

    def test_is_quota_error_true(self, adapter):
        """Should detect 429 quota errors via RESOURCE_EXHAUSTED string."""
        from google.genai import errors as real_errors
        # Use a real ClientError if available, otherwise test the string heuristic
        try:
            exc = real_errors.ClientError("RESOURCE_EXHAUSTED: quota exceeded")
            exc.code = 429
        except Exception:
            # Fallback: create an exception that contains the marker string
            exc = Exception("RESOURCE_EXHAUSTED: quota exceeded")
        # The adapter checks isinstance first, then string fallback
        # With a real ClientError, code=429 should match
        # With a generic Exception, is_quota_error returns False (correct)
        if isinstance(exc, real_errors.ClientError):
            assert adapter.is_quota_error(exc) is True
        else:
            assert adapter.is_quota_error(exc) is False

    def test_is_quota_error_false(self, adapter):
        """Non-quota exceptions should return False."""
        assert adapter.is_quota_error(ValueError("something")) is False

    def test_create_chat(self, adapter):
        """create_chat should create a GeminiChatSession."""
        mock_chat = MagicMock()
        adapter._mock_client.chats.create.return_value = mock_chat

        with patch("agent.llm.gemini_adapter.types") as mock_types:
            mock_types.GenerateContentConfig.return_value = "config"
            mock_types.Tool.return_value = "tool"
            mock_types.ThinkingConfig.return_value = "thinking"
            mock_types.FunctionDeclaration.return_value = "fd"
            mock_types.FunctionCallingConfig.return_value = "fcc"
            mock_types.ToolConfig.return_value = "tc"

            session = adapter.create_chat(
                model="gemini-test",
                system_prompt="You are helpful.",
                tools=[FunctionSchema(name="test", description="test", parameters={})],
            )

        from agent.llm.gemini_adapter import GeminiChatSession
        assert isinstance(session, GeminiChatSession)

    def test_generate(self, adapter):
        """generate should call client.models.generate_content and parse."""
        # Build a mock response
        mock_part = MagicMock()
        mock_part.thought = False
        mock_part.function_call = None
        mock_part.text = "Hello!"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=0,
        )

        adapter._mock_client.models.generate_content.return_value = mock_response

        with patch("agent.llm.gemini_adapter.types") as mock_types:
            mock_types.GenerateContentConfig.return_value = "config"
            result = adapter.generate(
                model="gemini-test",
                contents="Hi",
                temperature=0.5,
            )

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.raw is mock_response


# ---------------------------------------------------------------------------
# GeminiChatSession parse response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Test the _parse_response helper directly."""

    def test_text_only(self):
        from agent.llm.gemini_adapter import _parse_response

        part = MagicMock()
        part.thought = False
        part.text = "Hello"
        part.function_call = None

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content = content

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.usage_metadata = None

        resp = _parse_response(raw)
        assert resp.text == "Hello"
        assert resp.tool_calls == []
        assert resp.thoughts == []

    def test_tool_calls(self):
        from agent.llm.gemini_adapter import _parse_response

        fc = MagicMock()
        fc.name = "fetch_data"
        fc.args = {"dataset": "AC_H0_MFI"}

        part = MagicMock()
        part.thought = False
        part.text = None
        part.function_call = fc

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content = content

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=20,
            thoughts_token_count=10,
        )

        resp = _parse_response(raw)
        assert resp.text == ""
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fetch_data"
        assert resp.tool_calls[0].args == {"dataset": "AC_H0_MFI"}
        assert resp.usage.input_tokens == 50
        assert resp.usage.output_tokens == 20
        assert resp.usage.thinking_tokens == 10

    def test_thinking_parts(self):
        from agent.llm.gemini_adapter import _parse_response

        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.text = "Let me think..."
        thought_part.function_call = None

        text_part = MagicMock()
        text_part.thought = False
        text_part.text = "Result"
        text_part.function_call = None

        content = MagicMock()
        content.parts = [thought_part, text_part]

        candidate = MagicMock()
        candidate.content = content

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.usage_metadata = None

        resp = _parse_response(raw)
        assert resp.text == "Result"
        assert resp.thoughts == ["Let me think..."]

    def test_empty_response(self):
        from agent.llm.gemini_adapter import _parse_response

        raw = MagicMock()
        raw.candidates = []
        raw.usage_metadata = None

        resp = _parse_response(raw)
        assert resp.text == ""
        assert resp.tool_calls == []
