"""
Unit tests for LLM client abstraction, provider factory, clean_json_string,
retry backoff, and budget enforcement.
"""
import os
import json
import pytest
from unittest.mock import MagicMock, patch

from datapilot.llm.client import (
    LLMClient,
    LLMMessage,
    LLMBudgetExceeded,
    MockLLMClient,
    AnthropicClient,
    NvidiaClient,
    clean_json_string,
    get_llm_client,
)


def test_clean_json_string_variants():
    # 1. Clean JSON
    assert clean_json_string('{"a": 1}') == '{"a": 1}'
    
    # 2. Markdown fenced JSON
    fenced = '```json\n{"a": 1}\n```'
    assert clean_json_string(fenced) == '{"a": 1}'
    
    # 3. Bare fence
    bare = '```\n{"a": 1}\n```'
    assert clean_json_string(bare) == '{"a": 1}'
    
    # 4. JSON with preamble
    preamble = 'Here is the JSON output:\n```json\n{"a": 1}\n```\nHope that helps!'
    assert clean_json_string(preamble) == '{"a": 1}'
    
    # 5. Raw text with array braces
    arr_preamble = 'Result list:\n[1, 2, 3]'
    assert clean_json_string(arr_preamble) == '[1, 2, 3]'


def test_get_llm_client_factory():
    # Mock provider
    client_mock = get_llm_client(provider="mock")
    assert isinstance(client_mock, MockLLMClient)
    
    # Unknown provider raises ValueError
    with pytest.raises(ValueError, match="Unknown DATAPILOT_PROVIDER"):
        get_llm_client(provider="invalid_provider")
        
    # Anthropic without key raises ValueError
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            get_llm_client(provider="anthropic")

    # Nvidia without key raises ValueError
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="NVIDIA_API_KEY not set"):
            get_llm_client(provider="nvidia")


def test_budget_exceeded():
    client = MockLLMClient(max_calls=2)
    msg = [LLMMessage("user", "hi")]
    client.complete("system", msg)
    client.complete("system", msg)
    
    with pytest.raises(LLMBudgetExceeded, match="LLM call budget exceeded"):
        client.complete("system", msg)


def test_complete_json_retry_on_parse_error():
    client = MockLLMClient()
    # First call returns invalid JSON, second returns valid JSON
    client.register("system", "invalid json preamble")
    
    # Mocking complete to return invalid then valid
    calls = 0
    def mock_complete(sys, msgs, max_tokens=4096, temperature=0.0):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "This is not json at all!"
        return '{"status": "fixed"}'

    client.complete = mock_complete
    
    result = client.complete_json("system", [LLMMessage("user", "test")])
    assert result == {"status": "fixed"}
    assert calls == 2


def test_nvidia_client_mocked():
    with patch("openai.OpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        
        mock_choice = MagicMock()
        mock_choice.message.content = '{"key": "value"}'
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 15
        mock_resp.usage.completion_tokens = 25
        mock_instance.chat.completions.create.return_value = mock_resp

        client = NvidiaClient(api_key="fake-nvapi-key")
        res = client.complete("system", [LLMMessage("user", "hello")])
        
        assert res == '{"key": "value"}'
        assert client.call_count == 1
        assert client.input_tokens == 15
        assert client.output_tokens == 25


def test_anthropic_client_mocked():
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        
        mock_content = MagicMock()
        mock_content.text = '{"anthropic": "ok"}'
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 20
        mock_instance.messages.create.return_value = mock_resp

        client = AnthropicClient(api_key="fake-sk-key")
        res = client.complete("system", [LLMMessage("user", "hello")])
        
        assert res == '{"anthropic": "ok"}'
        assert client.call_count == 1
        assert client.input_tokens == 10
        assert client.output_tokens == 20
