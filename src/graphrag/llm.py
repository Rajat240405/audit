"""
LLM provider abstraction for GraphRAG entity/relationship extraction.

Providers (selectable via ``GRAPHRAG_LLM_PROVIDER``):
  - ``ollama``             local Ollama (default; existing behavior)
  - ``groq``               Groq cloud
  - ``openai_compatible``  any OpenAI-compatible ``/chat/completions`` endpoint

MODEL + KEY FAILOVER (Groq / OpenAI-compatible / multi-model Ollama):

  ``GROQ_API_KEYS=key1,key2,...``   ``GRAPHRAG_GROQ_MODELS=m1,m2,m3``
  (or ``OPENAI_API_KEYS`` / ``GRAPHRAG_OPENAI_MODELS``)

  Failover order:
    1. try Model 1 with every API key (in order)
    2. per key: exponential backoff on 429 / quota / 5xx / network errors
    3. when a key's backoff budget is exhausted -> switch to the next key
    4. when every key for a model fails -> switch to the next model
    5. when every model AND every key fails -> ``LLMBackendExhaustedError``
       (the pipeline saves the checkpoint and exits cleanly; the build can be
       resumed later from the same document)

Observability:
  - every request is logged with provider + model + MASKED key (last 4 chars)
  - every key/model switch is recorded as an event (``events`` list) so the
    pipeline can print human-readable "switching..." messages
  - per-(provider, model, masked-key) request counts in ``usage``
  - total key/model switches in ``switch_counts``
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import httpx

from src.graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """A provider-level failure (network, HTTP, quota, ...)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class LLMBackendExhaustedError(Exception):
    """All configured providers/models/keys failed — hard stop; resume later."""


@dataclass
class LLMResult:
    """A successful model response plus audit info."""

    text: str
    provider: str
    model: str
    key_label: str  # masked, e.g. "...abcd"


def mask_key(key: str) -> str:
    """Mask an API key, showing only the last 4 characters."""
    key = (key or "").strip()
    if len(key) <= 4:
        return "****"
    return f"...{key[-4:]}"


def parse_key_list(raw: str) -> list[str]:
    """Parse a comma-separated API-key list."""
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


def parse_model_list(raw: str, fallback: str) -> list[str]:
    """Parse a comma-separated model list; fall back to a single model."""
    models = [m.strip() for m in (raw or "").split(",") if m.strip()]
    return models or [fallback]


def _jitter(delay: float) -> float:
    return delay * (1 + random.uniform(0, 0.25))


class LLMProvider(ABC):
    """Common interface implemented by every backend."""

    name: str = "base"

    def __init__(self, models: list[str]) -> None:
        self.models = list(models)
        self.model = models[0] if models else None
        # Failover audit trail
        self.events: list[dict] = []
        self.usage: dict[str, dict[str, dict[str, int]]] = {}
        self.switch_counts: dict[str, int] = {
            "key_switches": 0,
            "model_switches": 0,
        }

    # ── audit helpers ──────────────────────────────────────────────────

    def _emit(self, event_type: str, **kw) -> None:
        self.events.append({"type": event_type, **kw})

    def _record_usage(self, model: str, key_label: str) -> None:
        self.usage.setdefault(self.name, {})
        self.usage[self.name].setdefault(model, {})
        self.usage[self.name][model][key_label] = (
            self.usage[self.name][model].get(key_label, 0) + 1
        )

    def drain_events(self) -> list[dict]:
        events = self.events
        self.events = []
        return events

    def usage_summary(self) -> dict:
        """Flatten usage into a report-friendly structure."""
        return self.usage

    # ── interface ──────────────────────────────────────────────────────

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 120,
        context: Optional[str] = None,
    ) -> LLMResult:
        """Return the raw model text for ``prompt``.

        ``context`` (e.g. the document id) is only used for logging.
        """


class OllamaProvider(LLMProvider):
    """Local Ollama backend (original behavior for a single model).

    With multiple models configured, Ollama tries each model in priority order
    and raises ``LLMBackendExhaustedError`` when all models fail. With the
    default single model, the original mark-failed-and-continue behaviour is
    preserved (a provider error, not an exhausted stop).
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        models: list[str],
        backoff_base: float = 1.0,
        max_attempts: int = 3,
        max_retryable_backoff: float = 30.0,
    ) -> None:
        super().__init__(models)
        self.base_url = base_url.rstrip("/")
        self.backoff_base = backoff_base
        self.max_attempts = max_attempts
        self.max_retryable_backoff = max_retryable_backoff

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 120,
        context: Optional[str] = None,
    ) -> LLMResult:
        last_error: Optional[Exception] = None
        for mi, model in enumerate(self.models):
            try:
                return self._generate_model(
                    model, prompt, temperature, max_tokens, timeout_seconds, context
                )
            except LLMProviderError as e:
                last_error = e
                if mi < len(self.models) - 1:
                    self.switch_counts["model_switches"] += 1
                    self._emit(
                        "model_switch",
                        from_model=self.models[mi],
                        to_model=self.models[mi + 1],
                        reason=str(e)[:120],
                        context=context,
                    )
        if len(self.models) > 1:
            raise LLMBackendExhaustedError(
                f"ollama: all {len(self.models)} model(s) failed. Last error: {last_error}"
            )
        raise LLMProviderError(f"ollama failed after retries: {last_error}")

    def _generate_model(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        context: Optional[str],
    ) -> LLMResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        last_error: Optional[Exception] = None
        delay = self.backoff_base
        for attempt in range(self.max_attempts):
            try:
                with httpx.Client(timeout=timeout_seconds) as client:
                    resp = client.post(f"{self.base_url}/api/generate", json=payload)
                if resp.status_code == 429:
                    raise LLMProviderError(
                        "rate limited (429)", status_code=429, retryable=True
                    )
                if resp.status_code != 200:
                    raise LLMProviderError(
                        f"ollama HTTP {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                        retryable=resp.status_code >= 500,
                    )
                text = resp.json().get("response", "")
                if not text.strip():
                    raise LLMProviderError("ollama empty response", retryable=True)
                self._record_usage(model, "local")
                logger.info(
                    "LLM request handled: provider=%s model=%s key=local context=%s",
                    self.name, model, context,
                )
                return LLMResult(text=text, provider=self.name, model=model, key_label="local")
            except LLMProviderError as e:
                last_error = e
                if not e.retryable:
                    break
                if attempt < self.max_attempts - 1:
                    time.sleep(_jitter(min(delay, self.max_retryable_backoff)))
                    delay *= 2
            except Exception as e:  # noqa: BLE001 - network errors
                last_error = e
                if attempt < self.max_attempts - 1:
                    time.sleep(_jitter(min(delay, self.max_retryable_backoff)))
                    delay *= 2
        raise LLMProviderError(
            f"ollama model {model} failed after {self.max_attempts} attempts: {last_error}"
        )


class ChatCompletionsProvider(LLMProvider):
    """
    OpenAI-compatible ``/chat/completions`` backend with MODEL + KEY failover.

    Used for Groq and any OpenAI-compatible endpoint (future). Failover:
      - models tried in priority order, keys tried in order within each model
      - per (model, key): exponential backoff on 429 / quota / 5xx / network
      - non-retryable errors (401/403) switch keys immediately
      - all keys for a model exhausted -> next model
      - all models and all keys exhausted -> ``LLMBackendExhaustedError``
    """

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        models: list[str],
        api_keys: list[str],
        backoff_base: float = 1.0,
        max_attempts_per_key: int = 3,
        max_retryable_backoff: float = 30.0,
    ) -> None:
        super().__init__(models)
        self.name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_keys = api_keys
        self.backoff_base = backoff_base
        self.max_attempts_per_key = max_attempts_per_key
        self.max_retryable_backoff = max_retryable_backoff

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 120,
        context: Optional[str] = None,
    ) -> LLMResult:
        if not self.api_keys:
            raise LLMBackendExhaustedError(
                f"{self.name}: no API keys configured. Set "
                "GROQ_API_KEYS=key1,key2,... (or OPENAI_API_KEYS for "
                "openai_compatible)."
            )
        last_error: Optional[Exception] = None
        for mi, model in enumerate(self.models):
            for ki, key in enumerate(self.api_keys):
                masked = mask_key(key)
                try:
                    return self._generate_with_key(
                        model, key, masked, prompt, temperature,
                        max_tokens, timeout_seconds, context,
                    )
                except LLMProviderError as e:
                    last_error = e
                    if ki < len(self.api_keys) - 1:
                        self.switch_counts["key_switches"] += 1
                        self._emit(
                            "key_switch",
                            from_key=masked,
                            to_key=mask_key(self.api_keys[ki + 1]),
                            model=model,
                            reason=str(e)[:120],
                            context=context,
                        )
                    elif mi < len(self.models) - 1:
                        self.switch_counts["model_switches"] += 1
                        self._emit(
                            "model_switch",
                            from_model=model,
                            to_model=self.models[mi + 1],
                            reason=f"all {len(self.api_keys)} key(s) exhausted for model",
                            context=context,
                        )
        raise LLMBackendExhaustedError(
            f"{self.name}: all {len(self.models)} model(s) and "
            f"{len(self.api_keys)} key(s) failed. Last error: {last_error}"
        )

    def _generate_with_key(
        self,
        model: str,
        key: str,
        masked: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        context: Optional[str],
    ) -> LLMResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[Exception] = None
        delay = self.backoff_base
        for attempt in range(self.max_attempts_per_key):
            try:
                with httpx.Client(timeout=timeout_seconds) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                if resp.status_code == 429:
                    raise LLMProviderError(
                        "rate limited (429)", status_code=429, retryable=True
                    )
                if resp.status_code in (401, 403):
                    raise LLMProviderError(
                        f"auth failed ({resp.status_code})",
                        status_code=resp.status_code,
                        retryable=False,
                    )
                if resp.status_code >= 500:
                    raise LLMProviderError(
                        f"server error ({resp.status_code})",
                        status_code=resp.status_code,
                        retryable=True,
                    )
                if resp.status_code != 200:
                    raise LLMProviderError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                        retryable=False,
                    )
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if not text.strip():
                    raise LLMProviderError("empty completion", retryable=True)
                self._record_usage(model, masked)
                logger.info(
                    "LLM request handled: provider=%s model=%s key=%s context=%s",
                    self.name, model, masked, context,
                )
                return LLMResult(text=text, provider=self.name, model=model, key_label=masked)
            except LLMProviderError as e:
                last_error = e
                if not e.retryable:
                    raise  # move to the next key immediately
                if attempt < self.max_attempts_per_key - 1:
                    time.sleep(_jitter(min(delay, self.max_retryable_backoff)))
                    delay *= 2
            except Exception as e:  # noqa: BLE001 - network errors
                last_error = LLMProviderError(
                    f"network failure: {type(e).__name__}: {str(e)[:120]}",
                    retryable=True,
                )
                if attempt < self.max_attempts_per_key - 1:
                    time.sleep(_jitter(min(delay, self.max_retryable_backoff)))
                    delay *= 2
        raise LLMProviderError(
            f"{self.name} model {model} key {masked} failed after "
            f"{self.max_attempts_per_key} attempts: {last_error}"
        )


def build_llm_provider(config: GraphRAGConfig) -> LLMProvider:
    """Construct the LLM provider selected by ``GRAPHRAG_LLM_PROVIDER``."""
    provider = (config.llm_provider or "ollama").strip().lower()
    # A generic --llm-models CLI override takes precedence for any provider.
    generic_models = config.llm_models

    if provider == "ollama":
        models = generic_models or config.ollama_models
        return OllamaProvider(
            base_url=config.ollama_base_url,
            models=models,
            backoff_base=config.llm_backoff_base,
            max_attempts=config.llm_max_attempts_per_key,
        )
    if provider in ("groq", "openai_compatible", "openai"):
        if provider == "groq":
            keys = config.groq_api_keys
            base_url = config.groq_base_url
            models = generic_models or config.groq_models
        else:
            keys = config.openai_api_keys
            base_url = config.openai_base_url
            models = generic_models or config.openai_models
        return ChatCompletionsProvider(
            provider_name=provider,
            base_url=base_url,
            models=models,
            api_keys=keys,
            backoff_base=config.llm_backoff_base,
            max_attempts_per_key=config.llm_max_attempts_per_key,
        )
    raise ValueError(
        f"unknown GRAPHRAG_LLM_PROVIDER {provider!r}; "
        "expected one of: ollama, groq, openai_compatible"
    )
