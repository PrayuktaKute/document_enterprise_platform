"""OpenAI-compatible LLM client.

One interface for every backend we use:
  * Colab-Ollama (via tunnel)  -- build + eval
  * laptop Ollama              -- "fully local" story
  * OpenRouter free variant    -- fallback

Only three env vars switch between them: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from dip.config import get_pipeline_config, get_settings


@dataclass
class TokenLogprob:
    token: str
    logprob: float


@dataclass
class LLMResponse:
    text: str
    tokens: list[TokenLogprob] = field(default_factory=list)
    model: str = ""
    finish_reason: str | None = None
    raw: Any = None

    @property
    def has_logprobs(self) -> bool:
        return len(self.tokens) > 0


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self._max_retries = max_retries
        # Disable the SDK's own retry loop; tenacity wraps calls below.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    @classmethod
    def from_config(cls) -> "LLMClient":
        s = get_settings()
        llm = get_pipeline_config().llm
        return cls(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            model=s.llm_model,
            timeout=llm.request_timeout_s,
            max_retries=llm.max_retries,
        )

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_object: bool = False,
        json_schema: dict[str, Any] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": json_schema, "strict": False},
            }
        elif json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        if extra_body:
            kwargs["extra_body"] = extra_body

        completion = self._call(**kwargs)
        choice = completion.choices[0]
        tokens: list[TokenLogprob] = []
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if content:
            for item in content:
                tokens.append(TokenLogprob(token=item.token, logprob=float(item.logprob)))

        return LLMResponse(
            text=choice.message.content or "",
            tokens=tokens,
            model=completion.model,
            finish_reason=choice.finish_reason,
            raw=completion,
        )

    # ------------------------------------------------------------------ #
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=20))
    def _call(self, **kwargs: Any):
        return self._client.chat.completions.create(**kwargs)
