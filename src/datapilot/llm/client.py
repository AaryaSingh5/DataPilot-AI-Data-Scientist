"""
LLM client abstraction — supports Anthropic Claude, NVIDIA AI Foundation Endpoints (OpenAI-compatible),
and a deterministic mock for testing.
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBudgetExceeded(Exception):
    """Raised when the maximum number of LLM calls for a run is reached."""
    pass


class LLMMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


def clean_json_string(raw: str) -> str:
    """Robustly extract JSON string from raw model response by stripping markdown fences and preambles."""
    raw = raw.strip()
    
    # 1. Look for markdown code fences ```json ... ``` or ``` ... ```
    if "```" in raw:
        first_idx = raw.find("```")
        last_idx = raw.rfind("```")
        if first_idx != -1 and last_idx != -1 and first_idx < last_idx:
            block = raw[first_idx + 3:last_idx].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            return block

    # 2. Look for outermost JSON object {...} or array [...]
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    first_bracket = raw.find("[")
    last_bracket = raw.rfind("]")

    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        if first_bracket == -1 or first_brace < first_bracket:
            return raw[first_brace:last_brace + 1]
            
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        return raw[first_bracket:last_bracket + 1]

    return raw


class LLMClient(ABC):
    def __init__(self, max_calls: Optional[int] = None):
        if max_calls is None:
            max_calls = int(os.environ.get("DATAPILOT_MAX_LLM_CALLS", "40"))
        self.max_calls = max_calls
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _check_budget(self):
        if self.call_count >= self.max_calls:
            raise LLMBudgetExceeded(
                f"LLM call budget exceeded: max {self.max_calls} calls allowed per run."
            )

    def _track_usage(self, input_tokens: int = 0, output_tokens: int = 0):
        self.call_count += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        ...

    def complete_json(
        self,
        system: str,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        sys_json = system + "\n\nRespond with valid JSON only — no preamble, no markdown fences."
        raw = self.complete(sys_json, messages, max_tokens=max_tokens, temperature=0.0)
        cleaned = clean_json_string(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Retry once with error feedback
            error_msg = LLMMessage(
                "user",
                f"Your output failed JSON parsing with error: {exc}. Please respond with ONLY valid JSON."
            )
            retry_messages = list(messages) + [LLMMessage("assistant", raw), error_msg]
            raw_retry = self.complete(sys_json, retry_messages, max_tokens=max_tokens, temperature=0.0)
            cleaned_retry = clean_json_string(raw_retry)
            return json.loads(cleaned_retry)


class AnthropicClient(LLMClient):
    DEFAULT_MODEL = "claude-sonnet-5-5"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_calls: Optional[int] = None,
    ):
        super().__init__(max_calls=max_calls)
        try:
            import anthropic
            self._anthropic_module = anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._client = self._anthropic_module.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("DATAPILOT_MODEL") or self.DEFAULT_MODEL

    def complete(
        self,
        system: str,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        self._check_budget()
        
        anthropic_module = self._anthropic_module
        catchable = (
            anthropic_module.RateLimitError,
            anthropic_module.APIConnectionError,
            anthropic_module.APITimeoutError,
        )

        last_exc = None
        for attempt in range(1, 4):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[m.to_dict() for m in messages],
                )
                in_tok = getattr(resp.usage, "input_tokens", 0)
                out_tok = getattr(resp.usage, "output_tokens", 0)
                self._track_usage(input_tokens=in_tok, output_tokens=out_tok)
                return resp.content[0].text
            except catchable as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(1.0 * (2 ** (attempt - 1)))
                else:
                    raise exc
            except Exception as exc:
                raise exc

        if last_exc:
            raise last_exc
        raise RuntimeError("LLM request failed")


class NvidiaClient(LLMClient):
    DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_calls: Optional[int] = None,
    ):
        super().__init__(max_calls=max_calls)
        try:
            import openai
            self._openai_module = openai
        except ImportError:
            raise ImportError("pip install openai")
        api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not set")
        self._client = self._openai_module.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        self.model = model or os.environ.get("DATAPILOT_MODEL") or self.DEFAULT_MODEL

    def complete(
        self,
        system: str,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        self._check_budget()

        openai_module = self._openai_module
        catchable = (
            openai_module.RateLimitError,
            openai_module.APIConnectionError,
            openai_module.APITimeoutError,
        )

        api_messages = [{"role": "system", "content": system}]
        api_messages.extend([m.to_dict() for m in messages])

        last_exc = None
        for attempt in range(1, 4):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                in_tok = 0
                out_tok = 0
                if hasattr(resp, "usage") and resp.usage:
                    in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
                    out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
                self._track_usage(input_tokens=in_tok, output_tokens=out_tok)
                return resp.choices[0].message.content or ""
            except catchable as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(1.5 * (2 ** (attempt - 1)))
                else:
                    raise exc
            except Exception as exc:
                raise exc

        if last_exc:
            raise last_exc
        raise RuntimeError("LLM request failed")


class MockLLMClient(LLMClient):
    """Deterministic mock for unit tests — returns pre-registered responses."""

    def __init__(self, max_calls: Optional[int] = None):
        super().__init__(max_calls=max_calls)
        self._responses: Dict[str, str] = {}
        self._calls: List[Dict] = []

    def register(self, keyword: str, response: str):
        self._responses[keyword] = response

    def complete(
        self,
        system: str,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        self._check_budget()
        combined = " ".join(m.content for m in messages)
        self._calls.append({"system": system, "messages": combined})
        self._track_usage(input_tokens=10, output_tokens=10)
        for kw, resp in self._responses.items():
            if kw.lower() in combined.lower() or kw.lower() in system.lower():
                return resp
        return '{"hypotheses": [], "plan": []}'


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMClient:
    """Factory helper to instantiate LLMClient based on provider selection or env var."""
    prov = (provider or os.environ.get("DATAPILOT_PROVIDER") or "mock").lower().strip()
    if prov in ("mock", "deterministic"):
        return MockLLMClient()
    elif prov == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    elif prov == "nvidia":
        return NvidiaClient(api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Unknown DATAPILOT_PROVIDER '{prov}'. Supported providers: 'mock', 'anthropic', 'nvidia'."
        )
