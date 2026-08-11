"""
Provider and Model Registry System for Intelligent LLM Management (Phase 10).
Defines standard provider interfaces, model families, and auto-resolution policies.
Fast/Deep are execution profiles that adjust standard parameters (temperature, tokens).
Refactored to apply symmetric quality-optimized boundaries for all models.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from src.generation.client import LLMResponse


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelFamily:
    """
    Metadata representation of a family of models.
    """
    id: str
    display_name: str
    provider: str  # "ollama", "groq", "openai"
    model_name: str  # Single concrete model name on the provider
    context_window: int
    thinking_capable: bool
    recommended_execution_mode: str = "GPU"

    def get_execution_params(self, mode: str) -> Dict[str, Any]:
        """
        Adjust execution parameters based on Fast/Deep profiles.
        Identical across all models (independent of thinking capabilities):
        - Fast Profile: temperature = 0.0, max_tokens = 512, top-k docs = 3, max characters = 1000.
        - Deep Profile: temperature = 0.2, max_tokens = 2048, top-k docs = 5, max characters = 3000.
        This provides a symmetric speed vs quality boundary across all endpoints.
        """
        mode = mode.lower().strip()
        if mode == "deep":
            return {
                "temperature": 0.2,
                "max_tokens": 2048,
                "max_context_docs": 5,
                "max_doc_chars": 3000,
            }
        else:
            return {
                "temperature": 0.0,
                "max_tokens": 512,
                "max_context_docs": 3,
                "max_doc_chars": 1000,
            }


class ModelRegistry:
    """
    Registry storing all configured model families as the single source of truth.
    """

    def __init__(self) -> None:
        self._families: Dict[str, ModelFamily] = {}

    def register(self, family: ModelFamily) -> None:
        """Register a new model family."""
        self._families[family.id] = family

    def get(self, family_id: str) -> Optional[ModelFamily]:
        """Retrieve a model family by its ID, with robust fallback check on display_name or model_name."""
        if family_id in self._families:
            return self._families[family_id]
        
        # Fallback substring/match checks
        fid_lower = family_id.lower()
        for f in self._families.values():
            if f.id.lower() == fid_lower or f.display_name.lower() == fid_lower or f.model_name.lower() == fid_lower:
                return f
        return None

    def list_by_provider(self, provider: str) -> List[ModelFamily]:
        """List all model families belonging to a specific provider."""
        return [f for f in self._families.values() if f.provider.lower() == provider.lower()]

    def list_all(self) -> List[ModelFamily]:
        """List all model families registered."""
        return list(self._families.values())


# Create global pre-populated Model Registry
model_registry = ModelRegistry()

# Ollama Families (Fast/Deep are execution profiles over a single concrete model name)
model_registry.register(ModelFamily(
    id="qwen3",
    display_name="Qwen 3",
    provider="ollama",
    model_name="qwen3:8b",
    context_window=32768,
    thinking_capable=True,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="qwen2.5",
    display_name="Qwen 2.5",
    provider="ollama",
    model_name="qwen2.5:7b",
    context_window=8192,
    thinking_capable=False,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="llama3.2",
    display_name="Llama 3.2",
    provider="ollama",
    model_name="llama3.2:3b",
    context_window=8192,
    thinking_capable=False,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="gemma2",
    display_name="Gemma 2",
    provider="ollama",
    model_name="gemma2:9b",
    context_window=8192,
    thinking_capable=False,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="qwen2.5_1.5b",
    display_name="Qwen 2.5 (1.5B)",
    provider="ollama",
    model_name="qwen2.5:1.5b",
    context_window=8192,
    thinking_capable=False,
    recommended_execution_mode="CPU"
))

# Groq Families
model_registry.register(ModelFamily(
    id="llama3.3_70b",
    display_name="Llama 3.3 70B",
    provider="groq",
    model_name="llama-3.3-70b-versatile",
    context_window=128000,
    thinking_capable=False,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="llama3.1_8b",
    display_name="Llama 3.1 8B",
    provider="groq",
    model_name="llama-3.1-8b-instant",
    context_window=128000,
    thinking_capable=False,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="mixtral_8x7b",
    display_name="Mixtral 8x7B",
    provider="groq",
    model_name="mixtral-8x7b-32768",
    context_window=32768,
    thinking_capable=False,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="gemma2_9b",
    display_name="Gemma 2 9B",
    provider="groq",
    model_name="gemma2-9b-it",
    context_window=8192,
    thinking_capable=False,
    recommended_execution_mode="GPU"
))

model_registry.register(ModelFamily(
    id="deepseek_r1_70b",
    display_name="DeepSeek R1 (70B)",
    provider="groq",
    model_name="deepseek-r1-distill-llama-70b",
    context_window=128000,
    thinking_capable=True,
    recommended_execution_mode="GPU"
))

# OpenRouter Families (OpenAI-compatible aggregation of many models)
model_registry.register(ModelFamily(
    id="qwen3.6_27b",
    display_name="Qwen 3.6 27B",
    provider="openrouter",
    model_name="qwen/qwen3.6-27b",
    context_window=262144,
    thinking_capable=True,
    recommended_execution_mode="GPU"
))


# ─────────────────────────────────────────────────────────────────────────────
# Provider Interface and Implementations
# ─────────────────────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        num_ctx: int = 16384,
        api_key: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        """Execute text generation against the provider endpoint."""
        pass

    @abstractmethod
    def models(self) -> List[str]:
        """Fetch list of available models supported by this provider."""
        pass

    @abstractmethod
    def health(self, api_key: Optional[str] = None) -> bool:
        """Perform a quick connectivity check to the provider's API."""
        pass

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Return provider-level capacities/metadata."""
        pass

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Simple token count heuristic (char length // 4)."""
        return max(1, len(text) // 4)


class OllamaProvider(BaseProvider):
    """
    Provider implementation for local Ollama service.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        num_ctx: int = 16384,
        api_key: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        # CRITICAL: use /api/chat + messages so Ollama applies the model's own
        # chat template (Qwen3's <|im_start|> format). The old /api/generate +
        # raw "prompt" string confused chat-templated models like the
        # fine-tuned incois-qa — they refused or emitted <think> blocks.
        # This matches the already-correct _generate_ollama() in client.py.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }

        start_time = time.monotonic()

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.monotonic() - start_time) * 1000

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        prompt_tokens = self._estimate_tokens(full_prompt)
        completion_tokens = self._estimate_tokens(data.get("message", {}).get("content", ""))
        total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            text=data.get("message", {}).get("content", "").strip(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason", "stop"),
            raw_response=data,
        )

    def models(self) -> List[str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        # Fallback matching the requested UI defaults
        return ["qwen2.5:7b", "llama3.2:3b", "gemma2:9b", "qwen2.5:1.5b"]

    def health(self, api_key: Optional[str] = None) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "execution_environment": "Local (CPU/GPU)",
            "default_context": 8192,
            "latency_profile": "Low overhead, variable execution speed"
        }


class GroqProvider(BaseProvider):
    """
    Provider implementation for Groq Cloud service.
    """

    def __init__(self, base_url: str = "https://api.groq.com/openai/v1") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        num_ctx: int = 128000,
        api_key: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not resolved_key:
            raise ValueError("Groq API key not found. Please set the GROQ_API_KEY env or supply it.")

        start_time = time.monotonic()

        with httpx.Client(timeout=timeout_seconds) as client:
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
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    def models(self) -> List[str]:
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b"
        ]

    def health(self, api_key: Optional[str] = None) -> bool:
        resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not resolved_key:
            return False
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {resolved_key}"},
                )
                return response.status_code == 200
        except Exception:
            return False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "execution_environment": "Cloud (LPU)",
            "default_context": 128000,
            "latency_profile": "Ultra-low token latency, network-dependent"
        }


class OpenAIProvider(BaseProvider):
    """
    Provider implementation for OpenAI Service.
    """

    def __init__(self, base_url: str = "https://api.openai.com/v1") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        num_ctx: int = 8192,
        api_key: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY env or supply it.")

        start_time = time.monotonic()

        with httpx.Client(timeout=timeout_seconds) as client:
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
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    def models(self) -> List[str]:
        return ["gpt-4o-mini", "gpt-4o"]

    def health(self, api_key: Optional[str] = None) -> bool:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            return False
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {resolved_key}"},
                )
                return response.status_code == 200
        except Exception:
            return False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "execution_environment": "Cloud (OpenAI)",
            "default_context": 128000,
            "latency_profile": "Standard API latency"
        }


class OpenRouterProvider(BaseProvider):
    """
    Provider implementation for OpenRouter (aggregates many open models under
    one OpenAI-compatible API). Base URL https://openrouter.ai/api/v1.
    Key resolution: explicit ``api_key`` arg, else ``OPENROUTER_API_KEY`` env.
    """

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        num_ctx: int = 262144,
        api_key: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY env or supply it."
            )

        start_time = time.monotonic()

        with httpx.Client(timeout=timeout_seconds) as client:
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
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    def models(self) -> List[str]:
        return [
            "qwen/qwen3.6-27b",
            "qwen/qwen3-32b",
        ]

    def health(self, api_key: Optional[str] = None) -> bool:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not resolved_key:
            return False
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {resolved_key}"},
                )
                return response.status_code == 200
        except Exception:
            return False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "execution_environment": "Cloud (OpenRouter aggregation)",
            "default_context": 262144,
            "latency_profile": "Provider-dependent, network overhead"
        }


# ─────────────────────────────────────────────────────────────────────────────
# Provider Registry
# ─────────────────────────────────────────────────────────────────────────────

class ProviderRegistry:
    """
    Registry managing LLM Provider instances.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}

    def register(self, name: str, provider: BaseProvider) -> None:
        self._providers[name.lower()] = provider

    def get(self, name: str) -> Optional[BaseProvider]:
        return self._providers.get(name.lower())


# Create global pre-populated Provider Registry
provider_registry = ProviderRegistry()
provider_registry.register("ollama", OllamaProvider())
provider_registry.register("groq", GroqProvider())
provider_registry.register("openai", OpenAIProvider())
provider_registry.register("openrouter", OpenRouterProvider())
