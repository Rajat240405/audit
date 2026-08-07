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

import json
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


def _extract_failed_generation(raw_body: str) -> Optional[str]:
    """Pull ``error.failed_generation`` out of a Groq 400 error body.

    Groq returns the model's (invalid) raw output here — the ONLY place the
    raw content is visible when json_validate_failed fires. Returns None when
    absent (e.g. ``""`` or the body is not the expected shape).
    """
    try:
        body = json.loads(raw_body)
        gen = body.get("error", {}).get("failed_generation")
        return gen if isinstance(gen, str) else None
    except Exception:  # noqa: BLE001
        return None


def _looks_truncated(generation: str) -> bool:
    """True when the model's output looks like cut-off JSON.

    Heuristic used to decide whether to escalate the output-token budget:
    the generation starts with ``{`` (JSON-shaped) but does not parse —
    i.e. it was cut mid-emission. Prose refusals ("No entities found...")
    and empty generations are NOT truncation and do NOT trigger escalation.
    """
    s = (generation or "").strip()
    if not s.startswith("{"):
        return False
    try:
        json.loads(s)
    except json.JSONDecodeError:
        return True
    return False


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
        response_format: str = "json_object",
        reasoning_format: Optional[str] = None,
        debug: bool = False,
        max_tokens_cap: int = 8192,
        debug_one: Optional[str] = None,
    ) -> None:
        super().__init__(models)
        self.name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_keys = api_keys
        self.backoff_base = backoff_base
        self.max_attempts_per_key = max_attempts_per_key
        self.max_retryable_backoff = max_retryable_backoff
        self.max_tokens_cap = max_tokens_cap
        self.debug_one = debug_one
        if response_format not in ("json_object", "json_schema"):
            raise ValueError(
                f"response_format must be 'json_object' or 'json_schema', got "
                f"{response_format!r}"
            )
        self.response_format = response_format
        self.reasoning_format = reasoning_format
        self.debug = debug
        # Per-model Structured-Outputs capability, learned at runtime:
        #   "strict"      -> attempt json_schema + strict:true (start here)
        #   "best_effort" -> json_schema + strict:false (after a strict 400)
        #   "json_object" -> legacy JSON mode (after any other json_schema 400)
        # A model is only ever downgraded (never re-upgraded within a run), so
        # documents after the first never re-attempt a rejected format.
        self._schema_level: dict[str, str] = {}

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
                        max_tokens, timeout_seconds, context, system, json_schema,
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
        system: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        # Structured Outputs capability for this model, defaulting to the
        # configured format. Runtime 400s downgrade it per model (see
        # _schema_level). json_object is the floor — it needs no server-side
        # schema support and works on every model.
        level = self._schema_level.get(
            model,
            "strict" if (self.response_format == "json_schema" and json_schema) else "json_object",
        )
        # Output-token budget. Starts at the SMALL configured default (1536) —
        # never reserving 8192 per request — and is escalated (doubled, up to
        # max_tokens_cap) ONLY when the model actually truncates the JSON.
        current_max = max_tokens

        def _build_payload() -> dict:
            p: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": current_max,
            }
            if level == "strict":
                p["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_result",
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            elif level == "best_effort":
                p["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_result",
                        "strict": False,
                        "schema": json_schema,
                    },
                }
            else:
                p["response_format"] = {"type": "json_object"}
            if self.reasoning_format:
                # Groq: when JSON mode is used with reasoning models only
                # "parsed" or "hidden" are supported. "hidden" returns only the
                # final answer — keeps reasoning from leaking into content.
                p["reasoning_format"] = self.reasoning_format
            return p

        payload = _build_payload()
        if self.debug:
            _debug = json.loads(json.dumps(payload))
            _debug["messages"] = [
                {"role": m["role"], "content": m["content"][:2000]} for m in messages
            ]
            logger.info(
                "%s payload (%s): %s",
                self.name, model, json.dumps(_debug, indent=2)[:4000],
            )
        # Per-document debug: enabled for the single document matching
        # config.debug_one (or globally when llm_debug is on). Prints the
        # full payload, estimated tokens, raw body and content analysis.
        is_debug = self.debug or (
            self.debug_one is not None and context == self.debug_one
        )
        last_error: Optional[Exception] = None
        delay = self.backoff_base
        attempts = 0
        while True:
            try:
                with httpx.Client(timeout=timeout_seconds) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                if self.debug:
                    logger.info(
                        "%s raw response (HTTP %s): %s",
                        self.name, resp.status_code, resp.text[:8000],
                    )
                if is_debug:
                    # Covers 200s AND 400s (e.g. json_validate_failed) — the
                    # raw body is printed before any parsing/classification.
                    _print_debug_block(
                        provider=self.name, model=model, key_label=masked,
                        context=context, payload=payload, resp=resp, kind="chat",
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
                if resp.status_code == 400:
                    # 400 = the provider rejected the REQUEST. A per-document
                    # content problem (json_validate_failed — the model's
                    # output was not valid JSON; or a content-filter hit) is a
                    # DOCUMENT failure, not a backend failure.
                    body = resp.text[:2000]
                    if (
                        "json_validate_failed" in body
                        or "content_filter" in body
                        or "responsible_genai" in body.lower()
                        or "policy violation" in body.lower()
                    ):
                        failed_gen = _extract_failed_generation(resp.text)
                        # Truncated JSON (JSON-shaped but cut off) is the
                        # recoverable case: escalate the output budget and
                        # retry the SAME model/key. Empty generations and
                        # prose refusals are NOT truncation — those stay
                        # per-document failures (never reserve more tokens
                        # for them).
                        if (
                            failed_gen is not None
                            and _looks_truncated(failed_gen)
                            and "content_filter" not in body
                        ):
                            next_max = min(current_max * 2, self.max_tokens_cap)
                            if next_max > current_max:
                                logger.warning(
                                    "%s: model %s output truncated in JSON "
                                    "mode (HTTP 400) — escalating max_tokens "
                                    "%d -> %d and retrying same model/key.",
                                    self.name, model, current_max, next_max,
                                )
                                current_max = next_max
                                payload = _build_payload()
                                continue
                        detail = body
                        if failed_gen is not None:
                            detail = (
                                f"model output before validation: "
                                f"{failed_gen[:2000]!r}"
                            )
                        raise DocumentExtractionError(
                            f"{self.name} rejected document as invalid JSON "
                            f"(HTTP 400): {detail}",
                            status_code=400,
                        )
                    # Any OTHER 400 while still on a json_schema level usually
                    # means THIS model does not support the configured format
                    # (e.g. "model does not support response_format
                    # json_schema", "strict not supported"). Automatically
                    # downgrade the model — strict -> best-effort ->
                    # json_object — and retry the SAME model+key instead of
                    # failing the document or failing over. Genuine backend
                    # errors (unknown model, malformed request) survive the
                    # downgrade and still trigger key/model failover.
                    if level != "json_object":
                        # "strict"-specific rejection -> try best-effort
                        # (strict:false) first; any other json_schema rejection
                        # (e.g. "model does not support response_format
                        # json_schema") -> straight to json_object.
                        next_level = (
                            "best_effort"
                            if (level == "strict" and "strict" in body.lower())
                            else "json_object"
                        )
                        logger.warning(
                            "%s: model %s rejected response_format "
                            "(HTTP 400: %s). Downgrading %s -> %s and retrying "
                            "same model/key.",
                            self.name, model, body[:300], level, next_level,
                        )
                        level = next_level
                        self._schema_level[model] = level
                        self.switch_counts["schema_downgrades"] += 1
                        self._emit(
                            "schema_downgrade",
                            model=model,
                            from_level=payload.get("response_format", {}).get("type"),
                            to_level=level,
                            reason=body[:200],
                            context=context,
                        )
                        payload = _build_payload()
                        if self.debug:
                            logger.info(
                                "%s downgraded payload (%s): %s",
                                self.name, model,
                                json.dumps(payload, indent=2)[:4000],
                            )
                        continue  # retry the same key — not a retryable failure
                    raise LLMProviderError(
                        f"HTTP 400: {body}",
                        status_code=400,
                        retryable=False,
                    )
                if resp.status_code != 200:
                    raise LLMProviderError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                        retryable=False,
                    )
                data = resp.json()
                choices = data.get("choices") or []
                # finish_reason "length" = the model ran out of output tokens
                # mid-JSON. Escalate the budget and retry the same model/key;
                # only give up (failover) once we're at the cap.
                if choices and choices[0].get("finish_reason") == "length":
                    next_max = min(current_max * 2, self.max_tokens_cap)
                    if next_max > current_max:
                        logger.warning(
                            "%s: model %s output truncated "
                            "(finish_reason=length at %d tokens) — escalating "
                            "max_tokens to %d and retrying same model/key.",
                            self.name, model, current_max, next_max,
                        )
                        current_max = next_max
                        payload = _build_payload()
                        continue
                    raise LLMProviderError(
                        f"output truncated even at {current_max} tokens",
                        retryable=False,
                    )
                # Robust empty-content handling: a reasoning model may return
                # content: null / missing. We convert ANY non-str to "" — this
                # is exactly what section 4 of the per-document debug reports.
                message = choices[0].get("message", {}) if choices else {}
                raw_content = message.get("content")
                text = raw_content if isinstance(raw_content, str) else ""
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
                attempts += 1
                if attempts >= self.max_attempts_per_key:
                    break
                time.sleep(_jitter(min(delay, self.max_retryable_backoff)))
                delay *= 2
            except DocumentExtractionError:
                # Per-document content failure (e.g. HTTP 400
                # json_validate_failed). Propagate unchanged — it must NOT be
                # treated as a network error, must NOT trigger key/model
                # failover, and must NOT become backend exhaustion.
                raise
            except Exception as e:  # noqa: BLE001 - network errors
                last_error = LLMProviderError(
                    f"network failure: {type(e).__name__}: {str(e)[:120]}",
                    retryable=True,
                )
                attempts += 1
                if attempts >= self.max_attempts_per_key:
                    break
                time.sleep(_jitter(min(delay, self.max_retryable_backoff)))
                delay *= 2
        raise LLMProviderError(
            f"{self.name} model {model} key {masked} failed after "
            f"{attempts} attempts: {last_error}"
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
            response_format=config.chat_response_format,
            reasoning_format=config.chat_reasoning_format,
            debug=config.llm_debug,
            max_tokens_cap=config.extract_max_tokens_cap,
            debug_one=config.debug_one,
        )
    raise ValueError(
        f"unknown GRAPHRAG_LLM_PROVIDER {provider!r}; "
        "expected one of: ollama, groq, openai_compatible"
    )
