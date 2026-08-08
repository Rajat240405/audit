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
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


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

    PROVIDERS = ["ollama", "openai", "groq", "litellm"]

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_seconds: int | None = None,
        num_ctx: int = 8192,
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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if timeout_seconds is None:
            timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", "300"))
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.api_key: str | None = None  # In-memory runtime session key storage

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif provider == "ollama":
            self.base_url = "http://localhost:11434"
        elif provider == "openai":
            self.base_url = "https://api.openai.com/v1"
        elif provider == "groq":
            self.base_url = "https://api.groq.com/openai/v1"
        else:
            self.base_url = "http://localhost:8000"

    # ── Ollama Implementation ────────────────────────────────────────────

    def _generate_ollama(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Ollama's REST API (kept for backward-compatibility)."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        start_time = time.monotonic()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.monotonic() - start_time) * 1000

        # Estimate tokens (Ollama doesn't always return token counts)
        prompt_tokens = self._estimate_tokens(full_prompt)
        completion_tokens = self._estimate_tokens(data.get("response", ""))
        total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            text=data.get("response", "").strip(),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason", "stop"),
            raw_response=data,
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
            "max_tokens": self.max_tokens,
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
        env_var = "OPENAI_API_KEY" if self.provider == "openai" else "GROQ_API_KEY"
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
                **kwargs
            )

        # Fallback to direct inline check if provider was not in registry
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system, **kwargs)
        elif self.provider in ("openai", "groq"):
            if self.provider == "openai":
                self.base_url = "https://api.openai.com/v1"
            else:
                self.base_url = "https://api.groq.com/openai/v1"
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
        """Yield text chunks as the model generates them (streaming).

        Provider-agnostic streaming used by the frontend SSE endpoint so the
        drafting workspace can render tokens live (ChatGPT-style) instead of
        waiting for the full generation. Ollama uses its NDJSON stream; Groq /
        OpenAI-compatible use SSE ``chat/completions`` streaming.
        """
        resolved_key = api_key or self.api_key
        if self.provider == "ollama":
            yield from self._stream_ollama(prompt, system, **kwargs)
        elif self.provider in ("openai", "groq"):
            yield from self._stream_openai_compatible(
                prompt, system, api_key=resolved_key, **kwargs
            )
        else:
            raise ValueError(f"Provider {self.provider!r} does not support streaming.")

    def _stream_ollama(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ):
        """Stream from Ollama's /api/generate (stream: true, NDJSON lines)."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    chunk = (data.get("response") or "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break

    def _stream_openai_compatible(
        self,
        prompt: str,
        system: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """Stream from Groq / OpenAI-compatible /chat/completions (SSE)."""
        resolved_key = api_key or self.api_key
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
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {resolved_key}"}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST", f"{self.base_url}/chat/completions",
                json=payload, headers=headers,
            ) as resp:
                resp.raise_for_status()
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
                    chunk = delta.get("content") or ""
                    if chunk:
                        yield chunk

    def check_health(self, api_key: str | None = None) -> bool:
        """Check if the selected LLM service is reachable/authorized using Provider Registry."""
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
