"""
LLM client using Ollama, OpenAI, and Groq for provider-agnostic inference.

Design Decisions
----------------
1. PROVIDER ABSTRACTION: Exposes a unified interface for both local (Ollama)
   and cloud (OpenAI, Groq) LLM backends.
2. ZERO INFRASTRUCTURE OVERHEAD: Groq is compatible with the standard OpenAI REST contract,
   allowing us to route Groq requests through a shared endpoint while changing only the
   target endpoint URL and API Key in memory.
3. HEALTH CHECKS: Proactively queries endpoints to detect if a service is online
   or if authentication credentials are valid.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from src.generation.defaults import default_model_name


@dataclass
class LLMResponse:
    """Response from an LLM API call."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str = "stop"
    raw_response: dict | None = None


class LLMClient:
    """
    Provider-agnostic LLM client for parliamentary grounded generation.
    Supports Ollama (local), OpenAI, and Groq (cloud).
    Delegates implementation to the ProviderRegistry in Phase 10.
    """

    PROVIDERS = ["ollama", "openai", "groq", "litellm", "openrouter"]

    def __init__(
        self,
        provider: str = "ollama",
        model: str | None = None,  # None -> default_model_name() (single source)
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_seconds: int | None = None,
        num_ctx: int = 16384,
    ) -> None:
        """
        Parameters
        ----------
        provider : str
            LLM provider: "ollama", "groq", "openai".
        model : str
            Model name. For Ollama: "qwen2.5:7b", etc. For Groq: "llama-3.3-70b-versatile", etc.
        timeout_seconds : int, optional
            Request timeout for generation, in seconds. Local models (e.g. Qwen 3 8B
            on CPU via Ollama) can take several minutes to generate a full answer, so
            the default is 300 s (5 min). Resolution order:
              1. explicit ``timeout_seconds`` argument
              2. ``LLM_TIMEOUT_SECONDS`` environment variable (project config via .env)
              3. 300 (default)
            Health checks keep a short 5 s probe timeout regardless of this value.
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. Options: {self.PROVIDERS}"
            )

        self.provider = provider
        self.model = model if model else default_model_name()
        self.temperature = temperature
        self.max_tokens = max_tokens
        if timeout_seconds is None:
            timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.api_key: str | None = None  # In-memory runtime session key storage

        # Remember whether the caller pinned a base_url. If not, we re-resolve
        # it from the current provider on every network call — so a runtime
        # provider switch (ollama -> openrouter) never leaves a stale URL like
        # http://localhost:11434 behind.
        self._base_url_explicit = base_url is not None
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = self._default_base_url(provider)

    @staticmethod
    def _default_base_url(provider: str) -> str:
        """Canonical API base URL for a provider (used unless the caller
        pinned an explicit base_url, e.g. a custom Ollama host)."""
        p = provider.lower().strip()
        if p == "ollama":
            return "http://localhost:11434"
        if p == "openai":
            return "https://api.openai.com/v1"
        if p == "groq":
            return "https://api.groq.com/openai/v1"
        if p == "openrouter":
            return "https://openrouter.ai/api/v1"
        return "http://localhost:8000"

    def _ensure_base_url(self, provider: str | None = None) -> None:
        """Recompute self.base_url from the current provider unless the user
        explicitly supplied one at construction. Call before any network op
        so runtime provider switches stay correct."""
        if not self._base_url_explicit:
            self.base_url = self._default_base_url(provider or self.provider)

    def _effective_max_tokens(self) -> int:
        """Raise a small max_tokens for reasoning-capable cloud models.

        Qwen3.6 (OpenRouter) spends tokens on chain-of-thought BEFORE writing
        the answer; a tight budget leaves content:null. Reasoning tokens are
        billed either way, so truncating them just wastes the call — give the
        model headroom to actually finish the answer.
        """
        if self.provider == "openrouter" and self.max_tokens < 8192:
            return 8192
        return self.max_tokens

    # ── Ollama Implementation ────────────────────────────────────────────

    def _generate_ollama(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate via Ollama — delegates to the registry's OllamaProvider.

        R4: this used to be a SECOND full Ollama implementation (raw httpx +
        /api/chat) that drifted from OllamaProvider. Now it is a thin
        delegation so there is exactly ONE Ollama generate path.
        """
        from src.generation.registry import provider_registry

        prov = provider_registry.get("ollama")
        if prov is None:
            raise ValueError("Ollama provider not registered")
        return prov.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_ctx=self.num_ctx,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            **kwargs,
        )

    # ── OpenAI / Groq Compatible Implementation ───────────────────────────

    def _generate_openai_compatible(
        self,
        prompt: str,
        system: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using OpenAI-compatible Chat Completions endpoints (kept for backward-compatibility)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self._effective_max_tokens(),
        }

        # Resolve API Key
        resolved_key = api_key or self._get_api_key()

        start_time = time.monotonic()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.monotonic() - start_time) * 1000

        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            text=text.strip(),
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    def _get_api_key(self) -> str:
        """Get API key from environment (kept for backward-compatibility)."""
        env_var = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(self.provider, "OPENAI_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                f"{self.provider.upper()} API key not found in environment. "
                f"Set the {env_var} environment variable."
            )
        return key

    # ── Public Interface ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the currently selected provider using the Provider Registry.
        """
        self._ensure_base_url()
        from src.generation.registry import provider_registry

        resolved_key = api_key or self.api_key
        prov_inst = provider_registry.get(self.provider)
        if prov_inst:
            return prov_inst.generate(
                model=self.model,
                prompt=prompt,
                system=system,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                num_ctx=self.num_ctx,
                api_key=resolved_key,
                timeout_seconds=self.timeout_seconds,
                think=getattr(self, "think", None),
                **kwargs
            )

        # Fallback to direct inline check if provider was not in registry
        # (base_url already re-resolved by _ensure_base_url() at the top).
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system, **kwargs)
        elif self.provider in ("openai", "groq", "openrouter"):
            return self._generate_openai_compatible(prompt, system, api_key=resolved_key, **kwargs)
        else:
            raise ValueError(f"Provider {self.provider!r} not implemented yet.")

    # ── Streaming (token-by-token) ───────────────────────────────────────

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """Yield structured events as the model generates (streaming).

        Each item is a dict:
          {"type": "reasoning", "text": str}  — model's hidden chain-of-thought
                                               (qwen3's reasoning_content), live.
          {"type": "tokens", "text": str}     — visible answer text.
          {"type": "answer_start"}            — first visible token emitted.
          {"type": "done"}                    — stream finished.

        Used by the frontend SSE endpoint so the workspace can show BOTH the
        model's thinking (Model Activity panel) and the answer live, instead
        of silently waiting while the model reasons. Ollama uses its NDJSON
        stream; Groq/OpenAI-compatible use SSE ``chat/completions`` streaming.
        """
        self._ensure_base_url()
        resolved_key = api_key or self.api_key
        if self.provider == "ollama":
            yield from self._stream_ollama(prompt, system, think=getattr(self, "think", None), **kwargs)
        elif self.provider == "huggingface":
            prov = provider_registry.get("huggingface")
            if prov:
                yield from prov.generate_stream(
                    model=self.model, prompt=prompt, system=system,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    num_ctx=self.num_ctx, think=getattr(self, "think", None), **kwargs
                )
            else:
                raise ValueError("huggingface provider not registered")
        elif self.provider in ("openai", "groq", "openrouter"):
            yield from self._stream_openai_compatible(
                prompt, system, api_key=resolved_key, **kwargs
            )
        else:
            raise ValueError(f"Provider {self.provider!r} does not support streaming.")

    def _stream_ollama(
        self,
        prompt: str,
        system: str | None = None,
        think: bool | None = None,
        **kwargs,
    ):
        """Stream from Ollama's /api/chat (messages + stream: true, NDJSON).

        Uses `messages` (not raw `prompt`) so Ollama applies the model's chat
        template — required for Qwen3-based fine-tuned models (incois-qa).
        Yields dict events; reasoning-capable models (qwen3) emit their
        thinking in one of several shapes depending on the Ollama build:
          1. a separate ``message.reasoning_content`` field (most builds),
          2. a separate ``message.reasoning`` field (some builds / the
             OpenAI-compatible route),
          3. inline ``<think>…</think>`` tags inside ``message.content``
             (default for several builds).
        We handle all three so the Model Activity panel always shows the live
        thinking instead of silently waiting.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        # qwen3-family models: explicitly enable reasoning so Ollama streams
        # the thinking (the Model Activity panel shows it live). Ollama accepts
        # `think` at the TOP LEVEL of the request (not inside options) —
        # unknown option keys are silently ignored, so put it where it counts.
        if think is not None:
            payload["think"] = bool(think)  # Fast=False (instant), Deep=True
        elif "qwen3" in self.model.lower():
            payload["think"] = True

        # Inline <think> blocks may span several NDJSON chunks; buffer content
        # until a closing tag arrives so we never split a thought mid-way.
        buf = ""
        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                answered = False
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    msg = data.get("message") or {}

                    # 1) Structured reasoning fields (builds 1 & 2 & your Ollama)
                    # Ollama streams qwen3 thinking under DIFFERENT field names
                    # depending on version: `reasoning_content` (most),
                    # `reasoning` (some), `thinking` (newer builds — confirmed
                    # on the user's machine via raw /api/chat capture).
                    reasoning = (
                        msg.get("reasoning_content")
                        or msg.get("reasoning")
                        or msg.get("thinking")
                        or ""
                    )
                    if reasoning:
                        yield {"type": "reasoning", "text": reasoning}

                    # 2) Inline <think> tags inside content (build 3)
                    chunk = msg.get("content") or ""
                    if chunk:
                        buf += chunk
                        # flush every fully-closed <think>…</think> block
                        while True:
                            m = re.search(r"<think>(.*?)</think>", buf, re.DOTALL)
                            if not m:
                                break
                            yield {"type": "reasoning", "text": m.group(1)}
                            buf = buf[: m.start()] + buf[m.end():]
                        # if the buffer ends in an unclosed <think>, defer it
                        open_idx = buf.rfind("<think>")
                        if open_idx >= 0 and "</think>" not in buf[open_idx:]:
                            pre = buf[:open_idx]
                            if pre:
                                if not answered:
                                    yield {"type": "answer_start"}
                                    answered = True
                                yield {"type": "tokens", "text": pre}
                            buf = buf[open_idx:]
                            continue
                        if buf:
                            if not answered:
                                yield {"type": "answer_start"}
                                answered = True
                            yield {"type": "tokens", "text": buf}
                            buf = ""

                    if data.get("done"):
                        # flush any unclosed tail as content (defensive)
                        if buf:
                            if not answered:
                                yield {"type": "answer_start"}
                                answered = True
                            yield {"type": "tokens", "text": buf}
                            buf = ""
                        yield {"type": "done"}
                        break

    def _stream_openai_compatible(
        self,
        prompt: str,
        system: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """Stream from Groq / OpenRouter / OpenAI-compatible /chat/completions (SSE)."""
        # Explicit arg > in-memory session key > env var
        resolved_key = api_key or self.api_key
        if not resolved_key:
            resolved_key = self._get_api_key()
        if not resolved_key:
            raise ValueError(
                f"{self.provider.upper()} API key not configured for streaming."
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self._effective_max_tokens(),
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {resolved_key}"}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST", f"{self.base_url}/chat/completions",
                json=payload, headers=headers,
            ) as resp:
                resp.raise_for_status()
                answered = False
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except Exception:  # noqa: BLE001
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    # Some providers stream reasoning in delta.reasoning_content
                    # (DeepSeek), delta.reasoning, or delta.thinking (Ollama /v1)
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thinking")
                        or ""
                    )
                    if reasoning:
                        yield {"type": "reasoning", "text": reasoning}
                    chunk = delta.get("content") or ""
                    if chunk:
                        if not answered:
                            yield {"type": "answer_start"}
                            answered = True
                        yield {"type": "tokens", "text": chunk}

    def check_health(self, api_key: str | None = None) -> bool:
        """Check if the selected LLM service is reachable/authorized using Provider Registry."""
        self._ensure_base_url()
        from src.generation.registry import provider_registry

        resolved_key = api_key or self.api_key

        prov_inst = provider_registry.get(self.provider)
        if prov_inst:
            return prov_inst.health(api_key=resolved_key)

        try:
            with httpx.Client(timeout=5) as client:
                if self.provider == "ollama":
                    response = client.get(f"{self.base_url}/api/tags")
                    return response.status_code == 200
                elif self.provider == "openai":
                    resolved_key = resolved_key or os.environ.get("OPENAI_API_KEY", "")
                    response = client.get(
                        f"{self.base_url}/models",
                        headers={"Authorization": f"Bearer {resolved_key}"},
                    )
                    return response.status_code == 200
                elif self.provider == "groq":
                    resolved_key = resolved_key or os.environ.get("GROQ_API_KEY", "")
                    if not resolved_key:
                        return False
                    response = client.get(
                        f"{self.base_url}/models",
                        headers={"Authorization": f"Bearer {resolved_key}"},
                    )
                    return response.status_code == 200
                elif self.provider == "openrouter":
                    resolved_key = resolved_key or os.environ.get("OPENROUTER_API_KEY", "")
                    if not resolved_key:
                        return False
                    response = client.get(
                        f"{self.base_url}/models",
                        headers={"Authorization": f"Bearer {resolved_key}"},
                    )
                    return response.status_code == 200
            return False
        except Exception:
            return False

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count from text.
        Simple heuristic: ~4 characters per token for English.
        """
        return max(1, len(text) // 4)
