"""
LLM client abstraction — supports Anthropic Claude and a deterministic mock for testing.
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        ...

    @abstractmethod
    def complete_json(
        self,
        system: str,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        ...


class AnthropicClient(LLMClient):
    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

    def complete(self, system: str, messages: List[LLMMessage], max_tokens: int = 4096, temperature: float = 0.0) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[m.to_dict() for m in messages],
        )
        return resp.content[0].text

    def complete_json(self, system: str, messages: List[LLMMessage], max_tokens: int = 4096) -> Dict[str, Any]:
        # Add JSON instruction to system prompt
        sys_json = system + "\n\nRespond with valid JSON only — no markdown fences, no commentary."
        raw = self.complete(sys_json, messages, max_tokens=max_tokens, temperature=0.0)
        # Strip any accidental fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[-1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw)


class MockLLMClient(LLMClient):
    """Deterministic mock for unit tests — returns pre-registered responses."""

    def __init__(self):
        self._responses: Dict[str, str] = {}
        self._calls: List[Dict] = []

    def register(self, keyword: str, response: str):
        self._responses[keyword] = response

    def complete(self, system: str, messages: List[LLMMessage], max_tokens: int = 4096, temperature: float = 0.0) -> str:
        combined = " ".join(m.content for m in messages)
        self._calls.append({"system": system, "messages": combined})
        for kw, resp in self._responses.items():
            if kw.lower() in combined.lower() or kw.lower() in system.lower():
                return resp
        return '{"hypotheses": [], "plan": []}'

    def complete_json(self, system: str, messages: List[LLMMessage], max_tokens: int = 4096) -> Dict[str, Any]:
        return json.loads(self.complete(system, messages, max_tokens))
