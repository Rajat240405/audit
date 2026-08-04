"""
LLM client using Ollama for local inference.

Design Decisions
----------------
1. We use a thin wrapper around Ollama's REST API via httpx.
   This avoids the Ollama Python SDK dependency and gives us
   full control over the API contract.

2. Ollama is preferred for this project because:
   - All inference stays on-premise (important for government data)
   - No API costs or rate limits
   - Full control over model selection
   - The RTX 3050 Mobile can run Qwen2.5:7B-Q4_K_M

3. We implement structured JSON output parsing as a fallback
   when JSON mode is not available.

4. Streaming is supported but not required for Phase 2.
   The generate() method returns the full response.

5. Token counting is estimated using a simple heuristic:
   ~4 characters per token for English text. This is approximate
   but sufficient for cost estimation and latency analysis.

6. We use the `liteLLM` package as an alternative if the user
   wants to swap between Ollama and OpenAI API easily.
   The interface is designed to be LiteLLM-compatible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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
    LLM client for parliamentary Q&A answer generation.

    Supports Ollama (local) as the primary backend.
    Can be extended to support OpenAI, Anthropic, etc.

    Usage
    -----
    ```python
    client = LLMClient(provider="ollama", model="qwen2.5:7b")
    response = client.generate("What is malaria?")
    print(response.text)
    ```
    """

    PROVIDERS = ["ollama", "openai", "litellm"]

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_seconds: int = 300,
        num_ctx: int = 8192,  # Configurable context window to avoid truncation
    ) -> None:
        """
        Parameters
        ----------
        provider : str
            LLM provider: "ollama" (local) or "openai" (cloud).
        model : str
            Model name. For Ollama: "qwen2.5:7b", "llama3.2:3b", etc.
            For OpenAI: "gpt-4o-mini", "gpt-4o", etc.
        base_url : str, optional
            API base URL. Default for Ollama: http://localhost:11434
        temperature : float
            Sampling temperature. 0.0 = deterministic (best for RAG).
            Range: 0.0–2.0. Default 0.1 is near-deterministic.
        max_tokens : int
            Maximum completion tokens. Controls response length.
        timeout_seconds : int
            Request timeout. Ollama can be slow on CPU-only systems.
        num_ctx : int
            Size of the LLM context window to prevent silent truncation of long contexts.
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. Options: {self.PROVIDERS}"
            )

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif provider == "ollama":
            self.base_url = "http://localhost:11434"
        elif provider == "openai":
            self.base_url = "https://api.openai.com/v1"
        else:
            self.base_url = "http://localhost:8000"

    # ── Ollama Implementation ────────────────────────────────────────────

    def _generate_ollama(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Ollama's REST API."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,  # Explicitly pass the budget context limit to avoid silent truncations
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

    # ── OpenAI Implementation ─────────────────────────────────────────────

    def _generate_openai(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using OpenAI's Chat Completions API."""
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

        api_key = kwargs.get("api_key") or self._get_api_key()

        start_time = time.monotonic()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
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
        """Get API key from environment."""
        import os
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable."
            )
        return key

    # ── Public Interface ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Parameters
        ----------
        prompt : str
            The user prompt (query + context).
        system : str, optional
            System prompt for instructions/context.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        LLMResponse
            Contains the generated text and metadata.
        """
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system, **kwargs)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system, **kwargs)
        else:
            raise ValueError(f"Provider {self.provider!r} not implemented yet.")

    def check_health(self) -> bool:
        """Check if the LLM service is reachable."""
        try:
            with httpx.Client(timeout=5) as client:
                if self.provider == "ollama":
                    response = client.get(f"{self.base_url}/api/tags")
                    return response.status_code == 200
                elif self.provider == "openai":
                    response = client.get(
                        f"{self.base_url}/models",
                        headers={"Authorization": f"Bearer {self._get_api_key()}"},
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
        More accurate with tiktoken, but we don't require it.
        """
        return max(1, len(text) // 4)
