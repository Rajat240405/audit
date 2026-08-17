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
from dataclasses import dataclass
from typing import Any

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


def ollama_base_url() -> str:
    """THE configured Ollama endpoint — one resolver for every code path
    (client base URL, provider health, generation, streaming, /api/models
    discovery). Resolution order:

      1. ``OLLAMA_BASE_URL`` — full URL, e.g. ``http://127.0.0.1:12000``
      2. ``OLLAMA_HOST``     — upstream-style ``host[:port]`` (scheme added;
                               a wildcard bind like ``0.0.0.0:12000`` becomes
                               ``127.0.0.1:12000`` because clients cannot dial
                               the wildcard address)
      3. ``http://localhost:11434`` (upstream default)

    Windows note: if the default port fails with ``bind: An attempt was made
    to access a socket in a way forbidden by its access permissions`` while
    nothing listens there, the port is inside a WinHTTP excluded range
    (``netsh interface ipv4 show excludedportrange protocol=tcp``). Serve
    Ollama on a free port (e.g. ``OLLAMA_HOST=127.0.0.1:12000 ollama serve``)
    and set the same value here — no code change, no hardcoded port.
    """
    raw = (os.environ.get("OLLAMA_BASE_URL") or "").strip()
    if not raw:
        host = (os.environ.get("OLLAMA_HOST") or "").strip()
        if host:
            raw = host if "://" in host else f"http://{host}"
    if not raw:
        return "http://localhost:11434"
    raw = raw.rstrip("/")
    # Clients cannot connect to a wildcard bind address — dial loopback instead.
    raw = re.sub(r"^(https?://)0\.0\.0\.0(?=[:/]|$)", r"\g<1>127.0.0.1", raw)
    raw = raw.replace("://[::]", "://127.0.0.1").replace("://[::1]", "://127.0.0.1")
    return raw


class LLMClient:
    """
    Provider-agnostic LLM client for parliamentary grounded generation.
    Supports Ollama (local), HuggingFace (in-process HPC), and any
    OpenAI-compatible server (vLLM HPC / Ollama /v1 — provider "vllm").
    Delegates implementation to the ProviderRegistry.
    """

    PROVIDERS = ["ollama", "huggingface", "vllm", "openai_compatible"]

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
            LLM provider: "ollama" (local), "huggingface" (in-process HPC), or "vllm" (HPC server).
        model : str
            Model name. For Ollama: "qwen3:8b", "qwen2.5:7b", etc.
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
        # provider switch never leaves a stale URL like http://localhost:11434
        # behind.
        self._base_url_explicit = base_url is not None
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = self._default_base_url(provider)

    def _family_think_mode(self) -> str:
        """think_mode for the active (provider, model) family from the catalog.

        The model catalog (config/models.yaml) decides how this provider/model
        signals think/nothink:
          - vLLM families: "template" -> chat_template_kwargs.enable_thinking
            per request; the model NAME is never mangled with /think|/nothink
            (not a vLLM mechanism — the server 404s unknown model ids).
          - Ollama / other servers: "none" -> model name untouched (server default)
        Safe default when the family isn't found: "none" (never mangle a name)."""
        try:
            from src.generation.registry import model_registry
        except Exception:  # noqa: BLE001
            return "none"
        # 1) exact provider+model match
        for f in model_registry.list_all():
            if f.provider == self.provider and (f.model_name == self.model or f.id == self.model):
                return f.think_mode
        # 2) fallback: match by model identity across any provider — covers
        #    dev-parity where provider stays "vllm" but VLLM_BASE_URL points
        #    at Ollama /v1 with an ollama model (qwen3:8b -> think_mode none).
        for f in model_registry.list_all():
            if f.model_name == self.model or f.id == self.model:
                return f.think_mode
        return "none"

    @staticmethod
    def _default_base_url(provider: str) -> str:
        """Canonical API base URL for a provider (used unless the caller
        pinned an explicit base_url, e.g. a custom Ollama host)."""
        p = provider.lower().strip()
        if p == "ollama":
            return ollama_base_url()
        if p == "vllm":
            return (os.environ.get("VLLM_BASE_URL") or "http://localhost:8001").rstrip("/")
        return "http://localhost:8000"

    def _ensure_base_url(self, provider: str | None = None) -> None:
        """Recompute self.base_url from the current provider unless the user
        explicitly supplied one at construction. Call before any network op
        so runtime provider switches stay correct."""
        if not self._base_url_explicit:
            self.base_url = self._default_base_url(provider or self.provider)

    def _effective_max_tokens(self) -> int:
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
                think_mode=self._family_think_mode(),
                base_url=self.base_url,
                **kwargs
            )

        # Fallback to direct inline check if provider was not in registry
        # (base_url already re-resolved by _ensure_base_url() at the top).
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system, **kwargs)
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
        stream.
        """
        self._ensure_base_url()
        resolved_key = api_key or self.api_key
        if self.provider == "ollama":
            yield from self._stream_ollama(prompt, system, think=getattr(self, "think", None), **kwargs)
        elif self.provider in ("huggingface", "vllm", "openai_compatible"):
            from src.generation.registry import provider_registry
            prov = provider_registry.get(self.provider)
            if prov:
                yield from prov.generate_stream(
                    model=self.model, prompt=prompt, system=system,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    num_ctx=self.num_ctx, think=getattr(self, "think", None),
                    base_url=self.base_url,
                    think_mode=self._family_think_mode(),
                    **kwargs
                )
            else:
                raise ValueError(f"{self.provider} provider not registered")
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

    def check_health(self, api_key: str | None = None) -> bool:
        """Check if the selected LLM service is reachable/authorized using Provider Registry."""
        self._ensure_base_url()
        from src.generation.registry import provider_registry

        resolved_key = api_key or self.api_key

        prov_inst = provider_registry.get(self.provider)
        if prov_inst:
            return prov_inst.health(
                api_key=resolved_key,
                base_url=self.base_url if self._base_url_explicit else None,
            )

        try:
            with httpx.Client(timeout=5) as client:
                if self.provider == "ollama":
                    response = client.get(f"{self.base_url}/api/tags")
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
