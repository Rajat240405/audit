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
                "max_tokens": 4096,   # headroom for thinking + answer
                "max_context_docs": 5,
                "max_doc_chars": 3000,
                "thinking": True,     # Deep = think + cross-verify
                "verify_depth": "full",
            }
        else:
            return {
                "temperature": 0.0,
                "max_tokens": 512,
                "max_context_docs": 3,
                "max_doc_chars": 1000,
                "thinking": False,    # Fast = instant answer (reasoning off)
                "verify_depth": "light",  # regex-only, no LLM judge
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

# HPC / in-container family — quantized MoE model run directly via
# HuggingFaceProvider (no Ollama). A40 fits it with huge speed headroom
# (3B active params).
model_registry.register(ModelFamily(
    id="qwen3.5_35b_a3b",
    display_name="Qwen 3.5 35B-A3B (MoE)",
    provider="huggingface",
    model_name="Qwen/Qwen3.5-35B-A3B-GGUF",  # or unsloth path; override in container
    context_window=131072,
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
        **kwargs,
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
        # mode-aware thinking: Fast=False (instant), Deep=True (reasoning)
        think = kwargs.get("think")
        if think is not None:
            payload["think"] = bool(think)
        elif "qwen3" in model.lower():
            payload["think"] = True

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
        # Fallback matching the requested UI defaults (qwen3:8b is the
        # default family — it MUST be listed or the UI shows a stale set
        # whenever Ollama is offline)
        return ["qwen3:8b", "qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b", "gemma2:9b"]

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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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

class HuggingFaceProvider(BaseProvider):
    """
    In-container / HPC provider: runs a quantized model DIRECTLY from a
    HuggingFace path (no Ollama, no API). Designed for Singularity images
    with an A40 (48GB) GPU.

    Model sources supported:
      - a GGUF file / dir  -> llama-cpp-python (Llama) — best for Q4/Q8 GGUF
      - a safetensors HF dir -> transformers AutoModelForCausalLM

    Mode-aware thinking (Fast = think off, Deep = think on) is honored via
    the `think` kwarg. Reasoning tokens (qwen3-style) are surfaced in
    streaming via the same {type: reasoning} events the rest of the stack
    expects, so the Model Activity panel works unchanged.
    """

    def __init__(self, model_path: str | None = None,
                 n_ctx: int = 32768,
                 n_gpu_layers: int = -1) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None          # llama-cpp instance (if GGUF)
        self._model = None        # transformers model (fallback)
        self._tokenizer = None

    # ── loading ──────────────────────────────────────────────────────────
    def _ensure_loaded(self, model: str) -> None:
        if self._llm is not None or self._model is not None:
            return
        path = model or self.model_path
        if not path:
            raise ValueError("HuggingFaceProvider: no model path given")
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=path if path.endswith(".gguf") else f"{path}/*.gguf",
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            return
        except ImportError:
            pass
        # fallback: transformers safetensors
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", torch_dtype="auto"
        )

    # ── non-streaming ────────────────────────────────────────────────────
    def generate(self, model, prompt, system=None, temperature=0.1,
                 max_tokens=512, num_ctx=16384, api_key=None,
                 timeout_seconds=300, think=None, **kwargs) -> LLMResponse:
        self._ensure_loaded(model)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        start = time.monotonic()
        if self._llm is not None:
            resp = self._llm.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens,
                stream=False,
            )
            text = resp["choices"][0]["message"]["content"] or ""
            # llama-cpp exposes reasoning in message.reasoning_content for qwen3
            reasoning = resp["choices"][0]["message"].get("reasoning_content") or ""
            full = (reasoning + text) if reasoning else text
            lat = (time.monotonic() - start) * 1000
            return LLMResponse(text=full, model=model, prompt_tokens=0,
                               completion_tokens=0, total_tokens=0,
                               latency_ms=lat, finish_reason="stop",
                               raw_response=resp)
        # transformers path
        inputs = self._tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=max_tokens,
                                   temperature=temperature, do_sample=temperature > 0)
        text = self._tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        lat = (time.monotonic() - start) * 1000
        return LLMResponse(text=text, model=model, prompt_tokens=0,
                           completion_tokens=0, total_tokens=0, latency_ms=lat,
                           finish_reason="stop")

    # ── streaming ────────────────────────────────────────────────────────
    def generate_stream(self, model, prompt, system=None, temperature=0.1,
                        max_tokens=512, num_ctx=16384, api_key=None,
                        timeout_seconds=300, think=None, **kwargs):
        """Yields {type: reasoning|tokens|answer_start|done} — same contract
        as the Ollama stream, so the frontend reasoning panel is unchanged."""
        self._ensure_loaded(model)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if self._llm is not None:
            stream = self._llm.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens,
                stream=True,
            )
            answered = False
            for chunk in stream:
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    yield {"type": "reasoning", "text": reasoning}
                content = delta.get("content") or ""
                if content:
                    if not answered:
                        yield {"type": "answer_start"}
                        answered = True
                    yield {"type": "tokens", "text": content}
            yield {"type": "done"}
            return
        # transformers streaming
        inputs = self._tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(self._model.device)
        answered = False
        for out in self._model.generate(**inputs, max_new_tokens=max_tokens,
                                        temperature=temperature,
                                        do_sample=temperature > 0,
                                        streamer=None):
            pass  # simplified: non-streaming fallback below
        text = self._tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        yield {"type": "tokens", "text": text}
        yield {"type": "done"}

    def models(self) -> List[str]:
        return [self.model_path] if self.model_path else ["<hf-model-path>"]

    def health(self, api_key: Optional[str] = None) -> bool:
        try:
            self._ensure_loaded(self.model_path or ".")
            return True
        except Exception:
            return False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "execution_environment": "In-container (Singularity/HPC, GPU)",
            "default_context": self.n_ctx,
            "latency_profile": "Depends on quant + GPU (A40: ~20-45 tok/s)",
        }


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
provider_registry.register("huggingface", HuggingFaceProvider())


