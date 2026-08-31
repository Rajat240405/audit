"""
LLM provider abstraction for GraphRAG entity/relationship extraction.

Providers (selectable via ``GRAPHRAG_LLM_PROVIDER``):
  - ``ollama``  local Ollama (default)

MODEL FAILOVER (multi-model Ollama):

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

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
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


class DocumentExtractionError(Exception):
    """
    A per-document extraction failure caused by the document/prompt content,
    NOT by the backend (e.g. HTTP 400 json_validate_failed, content-filter,
    schema violations).

    This is a document-level failure: the pipeline marks ONLY this document as
    failed, saves the checkpoint, and continues. It must never trigger
    key/model failover or backend exhaustion.
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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


def _print_debug_block(
    *,
    provider: str,
    model: str,
    key_label: str,
    context: Optional[str],
    payload: dict,
    resp: httpx.Response,
    kind: str,
) -> None:
    """Print the full per-request debug block for a single document.

    Prints:
      1. the exact JSON payload (API key excluded — it lives in headers only),
      2. estimated input token count (chars/4 heuristic, no tokenizer dep),
      3. the raw HTTP response body before any parsing,
      4. content analysis: was the ``content`` field absent / null / a real
         string, and did the client convert a non-string to ""?
    ``kind`` is "chat" (messages/choices) or "ollama" (prompt/response).
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    raw_body = resp.text
    data = None
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001 - body may be a non-JSON error page
        pass

    lines: list[str] = []
    lines.append(
        f"provider: {provider} | model: {model} | key: {key_label} | "
        f"document: {context or '(none)'}"
    )
    lines.append(
        f"max_tokens: {payload.get('max_tokens')} | "
        f"response_format: {json.dumps(payload.get('response_format'))} | "
        f"reasoning_format: {payload.get('reasoning_format')}"
    )
    lines.append("")
    lines.append(
        "[bold]1) EXACT JSON PAYLOAD sent (API key excluded — the key is in "
        "the Authorization header, never in the body):[/bold]"
    )
    lines.append(json.dumps(payload, indent=2))
    lines.append("")

    # 2) estimated input tokens (chars/4 heuristic — exact counts come from
    #    the provider's usage field, printed in section 4 when available)
    if kind == "chat":
        msgs = payload.get("messages", [])
        system_chars = sum(len(m.get("content", "")) for m in msgs if m.get("role") == "system")
        user_chars = sum(len(m.get("content", "")) for m in msgs if m.get("role") == "user")
    else:
        system_chars = len(payload.get("system", "") or "")
        user_chars = len(payload.get("prompt", "") or "")
    total_chars = system_chars + user_chars
    lines.append(
        "[bold]2) ESTIMATED INPUT TOKENS:[/bold] "
        f"~{total_chars // 4:,}  "
        f"({total_chars:,} chars / 4 heuristic; "
        f"system={system_chars:,} chars, user={user_chars:,} chars)"
    )
    lines.append("")
    lines.append(
        f"[bold]3) RAW HTTP RESPONSE BODY (HTTP {resp.status_code}, "
        "before any parsing):[/bold]"
    )
    lines.append(raw_body if raw_body else "(empty body)")
    lines.append("")
    lines.append("[bold]4) CONTENT / EMPTY-STRING ANALYSIS:[/bold]")

    if kind == "chat":
        choices = (data or {}).get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        raw_content = message.get("content")
        present = "content" in message
        converted = not isinstance(raw_content, str)
        lines.append(f"  content field present in message: {'yes' if present else 'no'}")
        lines.append(f"  content raw type: {type(raw_content).__name__}")
        value_repr = repr(raw_content)
        lines.append(
            f"  content raw value: {value_repr[:400]}{'... (truncated)' if len(value_repr) > 400 else ''}"
        )
        lines.append(
            f"  client converted to empty string: {'yes' if converted else 'no'}  "
            "(non-str content -> '' before .strip() check)"
        )
        if choices:
            lines.append(f"  finish_reason: {choices[0].get('finish_reason')}")
        usage = (data or {}).get("usage") or {}
        if usage:
            lines.append(
                f"  usage (exact, from provider): prompt_tokens="
                f"{usage.get('prompt_tokens')}, completion_tokens="
                f"{usage.get('completion_tokens')}, total={usage.get('total_tokens')}"
            )
    else:
        raw_resp = (data or {}).get("response")
        lines.append(f"  response field raw type: {type(raw_resp).__name__}")
        value_repr = repr(raw_resp)
        lines.append(
            f"  response field value: {value_repr[:400]}{'... (truncated)' if len(value_repr) > 400 else ''}"
        )
        lines.append(
            f"  client converted to empty string: "
            f"{'yes' if raw_resp is not None and not isinstance(raw_resp, str) else 'no'}"
        )
        lines.append(f"  done_reason: {(data or {}).get('done_reason')}")
        ev = (data or {}).get("eval_count")
        if ev is not None:
            lines.append(f"  eval_count (exact, from provider): {ev}")

    console.print(Panel.fit(
        "\n".join(lines),
        title=f"[bold cyan]{provider.upper()} per-document debug — {context or model}[/bold cyan]",
        border_style="cyan",
    ))




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
            "schema_downgrades": 0,
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
        system: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> LLMResult:
        """Return the raw model text for ``prompt``.

        ``context`` (e.g. the document id) is only used for logging.
        ``system`` is an optional system message (chat providers use the
        ``system`` role; Ollama uses its native ``system`` field).
        ``json_schema`` is the extraction JSON Schema used when the provider
        is configured for Structured Outputs (``json_schema`` response format).
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
        debug: bool = False,
        max_tokens_cap: int = 8192,
        debug_one: Optional[str] = None,
    ) -> None:
        super().__init__(models)
        self.base_url = base_url.rstrip("/")
        self.backoff_base = backoff_base
        self.max_attempts = max_attempts
        self.max_retryable_backoff = max_retryable_backoff
        self.debug = debug
        self.max_tokens_cap = max_tokens_cap
        self.debug_one = debug_one

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 120,
        context: Optional[str] = None,
        system: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> LLMResult:
        last_error: Optional[Exception] = None
        for mi, model in enumerate(self.models):
            try:
                return self._generate_model(
                    model, prompt, temperature, max_tokens, timeout_seconds,
                    context, system,
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
        system: Optional[str] = None,
    ) -> LLMResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        current_max = max_tokens
        if self.debug:
            logger.info(
                "OLLAMA payload (%s): %s", model, json.dumps(payload, indent=2)[:4000]
            )
        is_debug = self.debug or (
            self.debug_one is not None and context == self.debug_one
        )
        last_error: Optional[Exception] = None
        delay = self.backoff_base
        for attempt in range(self.max_attempts):
            try:
                with httpx.Client(timeout=timeout_seconds) as client:
                    resp = client.post(f"{self.base_url}/api/generate", json=payload)
                if self.debug:
                    logger.info(
                        "OLLAMA raw response (HTTP %s): %s", resp.status_code, resp.text[:8000]
                    )
                if is_debug:
                    _print_debug_block(
                        provider=self.name, model=model, key_label="local",
                        context=context, payload=payload, resp=resp, kind="ollama",
                    )
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
                data = resp.json()
                text = data.get("response", "")
                # Ollama signals truncation with done_reason == "length".
                # Escalate the output budget and retry the same model.
                if data.get("done_reason") == "length":
                    next_max = min(current_max * 2, self.max_tokens_cap)
                    if next_max > current_max:
                        logger.warning(
                            "ollama: output truncated at %d tokens — escalating "
                            "num_predict to %d (model=%s)",
                            current_max, next_max, model,
                        )
                        current_max = next_max
                        payload["options"]["num_predict"] = current_max
                        continue
                    raise LLMProviderError(
                        f"ollama output truncated even at {current_max} tokens",
                        retryable=False,
                    )
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
            except DocumentExtractionError:
                raise  # per-document failure — propagate unchanged
            except Exception as e:  # noqa: BLE001 - network errors
                last_error = e
                if attempt < self.max_attempts - 1:
                    time.sleep(_jitter(min(delay, self.max_retryable_backoff)))
                    delay *= 2
        raise LLMProviderError(
            f"ollama model {model} failed after {self.max_attempts} attempts: {last_error}"
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
            debug=config.llm_debug,
            max_tokens_cap=config.extract_max_tokens_cap,
            debug_one=config.debug_one,
        )
    raise ValueError(
        f"unknown GRAPHRAG_LLM_PROVIDER {provider!r}; "
        "expected: ollama"
    )
