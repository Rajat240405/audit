"""
Provider and Model Registry System for Intelligent LLM Management (Phase 10).
Defines standard provider interfaces, model families, and auto-resolution policies.
Fast/Deep are execution profiles that adjust standard parameters (temperature, tokens).
Refactored to apply symmetric quality-optimized boundaries for all models.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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
    provider: str  # "ollama" | "huggingface"
    model_name: str  # Single concrete model name on the provider
    context_window: int
    thinking_capable: bool
    recommended_execution_mode: str = "GPU"
    # How this provider's model signals think/nothink per request:
    #   "template" -> send chat_template_kwargs.enable_thinking (vLLM + Qwen3.x;
    #                 model name untouched, thinking controlled per-request)
    #   "key"      -> top-level request flag (Ollama `think`)
    #   "none"     -> provider decides (default)
    # The old "suffix" value (append /think or /nothink to the model NAME) was
    # removed: it is not a vLLM mechanism — the server 404s any model id it
    # does not serve. populate_model_registry migrates legacy "suffix" entries
    # to "template" with a warning. Model names are NEVER mangled.
    think_mode: str = "none"

    @classmethod
    def from_dict(cls, d: dict) -> "ModelFamily":
        """Build from a catalog entry (config/models.yaml). Unknown keys ignored."""
        allowed = {
            "id", "display_name", "provider", "model_name", "context_window",
            "thinking_capable", "recommended_execution_mode", "think_mode",
        }
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def get_execution_params(self, mode: str) -> Dict[str, Any]:
        """
        Adjust execution parameters based on Fast/Deep profiles.
        Identical across all models (independent of thinking capabilities):
        - Fast Profile: temperature = 0.0, max_tokens = 4096, top-k docs = 3, max characters = 1000.
        - Deep Profile: temperature = 0.2, max_tokens = 12288, top-k docs = 5, max characters = 3000.

        Deep mode uses 12288 max_tokens because thinking-capable models (Qwen3.6)
        consume max_tokens for BOTH reasoning + answer. With 4096 the model would
        reason for ~3500 tokens and produce a truncated or empty answer.
        """
        mode = mode.lower().strip()
        if mode == "deep":
            return {
                "temperature": 0.2,
                "max_tokens": 12288,  # reasoning (~4-8k) + answer (~4k)
                "max_context_docs": 5,
                "max_doc_chars": 3000,
                "thinking": True,     # Deep = think + cross-verify
                "verify_depth": "full",
            }
        else:
            return {
                "temperature": 0.0,
                "max_tokens": 4096,   # no reasoning overhead in Standard
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


# ─────────────────────────────────────────────────────────────────────────────
# Model catalog loader — the "plugin" system.
# config/models.yaml is the single source of truth for model families.
# Add a model there (any provider) -> it appears in the UI/API with no code
# change. Missing/unreadable catalog falls back to DEFAULT_CATALOG below so
# the app never dies on a config typo.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CATALOG: dict = {
    "providers": {
        "ollama": {"families": [
            {"id": "qwen3", "display_name": "Qwen 3", "model_name": "qwen3:8b",
             "context_window": 32768, "thinking_capable": True, "recommended_execution_mode": "GPU"},
            {"id": "qwen2.5", "display_name": "Qwen 2.5", "model_name": "qwen2.5:7b",
             "context_window": 8192, "thinking_capable": False, "recommended_execution_mode": "GPU"},
            {"id": "qwen2.5_1.5b", "display_name": "Qwen 2.5 (1.5B)", "model_name": "qwen2.5:1.5b",
             "context_window": 8192, "thinking_capable": False, "recommended_execution_mode": "CPU"},
        ]},
        "huggingface": {"families": [
            {"id": "qwen3.5_35b_a3b", "display_name": "Qwen 3.5 35B-A3B (MoE)",
             "model_name": "Qwen/Qwen3.5-35B-A3B-GGUF", "context_window": 131072,
             "thinking_capable": True, "recommended_execution_mode": "GPU"},
        ]},
    }
}

_MODEL_CATALOG_PATH = os.environ.get(
    "MODEL_CATALOG", str(Path(__file__).resolve().parents[2] / "config" / "models.yaml")
)


def load_model_catalog(path: Optional[str] = None) -> dict:
    """Load the model catalog file. Falls back to DEFAULT_CATALOG on any error."""
    p = Path(path or _MODEL_CATALOG_PATH)
    if not p.exists():
        print(f"[models] catalog not found ({p}) — using built-in defaults")
        return DEFAULT_CATALOG
    try:
        import yaml  # local import — optional dep
        with p.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data.get("providers"), dict):
            raise ValueError("catalog missing 'providers'")
        return data
    except Exception as e:  # noqa: BLE001
        print(f"[models] catalog load failed ({e}) — using built-in defaults")
        return DEFAULT_CATALOG


def populate_model_registry(reg: ModelRegistry, catalog: Optional[dict] = None) -> int:
    """Register every family in the catalog into the registry. Returns count."""
    data = catalog or load_model_catalog()
    count = 0
    for provider, cfg in (data.get("providers") or {}).items():
        for entry in cfg.get("families") or []:
            entry = dict(entry)
            entry.setdefault("provider", provider)
            entry.setdefault("recommended_execution_mode", "GPU")
            # vLLM controls Qwen3.x thinking per-request via
            # chat_template_kwargs.enable_thinking ("template") — never via a
            # /think|/nothink model-name suffix (not a vLLM mechanism).
            entry.setdefault("think_mode", "template" if provider == "vllm" else "none")
            if entry["think_mode"] == "suffix":
                print(
                    f"[models] family {entry.get('id')!r}: think_mode 'suffix' was removed "
                    "(vLLM does not serve '<model>/think' ids) — migrated to 'template' "
                    "(chat_template_kwargs.enable_thinking, model name untouched)"
                )
                entry["think_mode"] = "template"
            reg.register(ModelFamily.from_dict(entry))
            count += 1
    return count


def resolve_family_for_provider(
    reg: ModelRegistry,
    provider: str,
    family_id: str,
    preferred_model: Optional[str] = None,
) -> Optional[ModelFamily]:
    """Resolve the boot-time model family for an env-selected provider.

    The global default family is the PC/Ollama one ("qwen3"). When the
    environment selects a different provider (HPC: APP_DEFAULT_PROVIDER=vllm),
    the boot model must be one that provider actually serves — otherwise the
    first request 404s against vLLM until someone manually switches. Order:

      1. ``family_id`` itself, if it belongs to the active provider (PC path —
         unchanged behavior);
      2. the provider family whose model_name/id equals ``preferred_model``
         (HPC: ``VLLM_MODEL``);
      3. the first family registered for the active provider;
      4. the original ``family_id`` family as a last resort.

    Pure lookup — no side effects.
    """
    fam = reg.get(family_id)
    if fam is None or fam.provider == provider:
        return fam
    candidates = reg.list_by_provider(provider)
    wanted = (preferred_model or "").strip()
    if wanted:
        for f in candidates:
            if f.model_name == wanted or f.id == wanted:
                return f
    if candidates:
        return candidates[0]
    return fam


# Create global pre-populated Model Registry (from the catalog file)
model_registry = ModelRegistry()
populate_model_registry(model_registry)

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

    def health(self, api_key: Optional[str] = None, base_url: str | None = None) -> bool:
        """Same signature as OpenAICompatibleProvider.health (LLMClient always passes base_url)."""
        url = (base_url or self.base_url).rstrip("/")
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "execution_environment": "Local (CPU/GPU)",
            "default_context": 8192,
            "latency_profile": "Low overhead, variable execution speed"
        }


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
                 timeout_seconds=300, think=None, base_url=None, **kwargs) -> LLMResponse:
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
                        timeout_seconds=300, think=None, base_url=None, **kwargs):
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

    def health(self, api_key: Optional[str] = None, base_url: str | None = None) -> bool:
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


class OpenAICompatibleProvider(BaseProvider):
    """Inference adapter for ANY OpenAI-compatible server (vLLM, Ollama /v1,
    llama.cpp, LM Studio, TGI, LiteLLM…).

    No API keys — this is for in-cluster / local servers only. Multi-user
    batching is handled by the server itself (vLLM continuous batching), so
    FastAPI can run several stateless workers pointing at one base URL.

    Model-agnostic contract: chat completion + streaming with
    ``reasoning_content`` handling (shared frontend reasoning panel). No
    RAG/chat logic lives here — only the transport adapter differs.

    Base URL resolution order: explicit constructor arg -> VLLM_BASE_URL env
    -> http://localhost:8001. Windows dev can point VLLM_BASE_URL at Ollama's
    OpenAI-compatible endpoint (http://localhost:11434/v1) to exercise this
    exact code path locally.

    Think/nothink is PROVIDER-AWARE via ``think_mode`` (from the model
    catalog, config/models.yaml — per model/provider, not hardcoded):
      - "template": send ``chat_template_kwargs.enable_thinking`` per request
        (vLLM + Qwen3.x — the HPC Qwen3.6 control). The model NAME is never
        modified: /think|/nothink name suffixes are not a vLLM mechanism
        (the server 404s any model id it does not serve).
      - "none":     the server uses its own default thinking behaviour
        (Ollama /v1, llama.cpp, TGI, LM Studio…).
      - ``None`` think (unspecified) -> no chat_template_kwargs either; the
        server default applies.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url_explicit = base_url is not None
        self.base_url = (
            (base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8001")
            .rstrip("/")
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _url(self, base_url: str | None = None) -> str:
        if base_url:
            return base_url.rstrip("/")
        if not self._base_url_explicit:
            self.base_url = (os.environ.get("VLLM_BASE_URL") or "http://localhost:8001").rstrip("/")
        return self.base_url

    @staticmethod
    def chat_completions_url(base: str) -> str:
        from src.generation.openai_url import chat_completions_url as _join

        return _join(base)

    def _completions_url(self, base_url: str | None = None) -> str:
        return self.chat_completions_url(self._url(base_url))

    def _payload(
        self, model: str, messages: list, temperature: float, max_tokens: int,
        num_ctx: int, stream: bool, think: bool | None, think_mode: str = "none",
    ) -> dict:
        body: dict = {
            "model": model,  # sent verbatim — never mangled with /think|/nothink
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # chat_template_kwargs.enable_thinking is THE vLLM Qwen3.x thinking
        # control (Standard=False / Deep=True, per request, model id untouched).
        # Send it ONLY for "template" families — for "none" the server decides
        # (Ollama /v1, llama.cpp, etc. would just ignore the extra kwarg, but
        # not sending keeps the contract explicit).
        if think is not None and think_mode == "template":
            body["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        return body

    # ── non-streaming ────────────────────────────────────────────────────
    def generate(self, model, prompt, system=None, temperature=0.1,
                 max_tokens=512, num_ctx=16384, api_key=None,
                 timeout_seconds=300, think=None, base_url=None,
                 think_mode: str = "none", **kwargs) -> LLMResponse:
        messages: list = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = self._payload(model, messages, temperature, max_tokens,
                             num_ctx, stream=False, think=think, think_mode=think_mode)
        start = time.monotonic()
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                resp = client.post(self._completions_url(base_url), json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  # noqa: BLE001 - surface to caller as LLM error
            raise RuntimeError(f"OpenAI-compatible generate failed: {e}") from e
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        # Parity with OllamaProvider: ``text`` is the VISIBLE answer only.
        # The chain-of-thought (qwen3 ``reasoning_content``) stays available in
        # ``raw_response``; the UI surfaces reasoning via the streaming path.
        # Gluing CoT onto the answer here would corrupt downstream citation
        # verification and diverge from the Ollama response shape.
        text = msg.get("content") or ""
        usage = data.get("usage") or {}
        lat = (time.monotonic() - start) * 1000
        return LLMResponse(
            text=text, model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            latency_ms=lat, finish_reason=choice.get("finish_reason", "stop"),
            raw_response=data,
        )

    # ── streaming ────────────────────────────────────────────────────────
    def generate_stream(self, model, prompt, system=None, temperature=0.1,
                        max_tokens=512, num_ctx=16384, api_key=None,
                        timeout_seconds=300, think=None, base_url=None,
                        think_mode: str = "none", **kwargs):
        """Yields {type: reasoning|tokens|answer_start|done} — same contract
        as the Ollama/HF streams, so the frontend reasoning panel is shared."""
        messages: list = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = self._payload(model, messages, temperature, max_tokens,
                             num_ctx, stream=True, think=think, think_mode=think_mode)
        with httpx.Client(timeout=timeout_seconds) as client:
            with client.stream("POST", self._completions_url(base_url), json=body) as resp:
                resp.raise_for_status()
                answered = False
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except Exception:  # noqa: BLE001
                        continue
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
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

    def models(self) -> List[str]:
        return [os.environ.get("VLLM_MODEL") or "Qwen3.6-35B-A3B-FP8"]

    def health(self, api_key: Optional[str] = None, base_url: str | None = None) -> bool:
        """Readiness for any OpenAI-compatible server.

        Tries, in order, and returns healthy if ANY succeeds:
          1. GET /health            (vLLM, llama.cpp server)
          2. GET /v1/models         (standard OpenAI-compatible models list)
          3. GET /models            (some servers mount models at root)
        Never reports unhealthy just because /health is missing."""
        url = self._url(base_url)
        v1 = url if url.endswith("/v1") else f"{url}/v1"
        host = url[:-3] if url.endswith("/v1") else url
        probes = (f"{host}/health", f"{v1}/models", f"{host}/models")
        try:
            with httpx.Client(timeout=5) as client:
                for probe in probes:
                    try:
                        r = client.get(probe)
                        if r.status_code == 200:
                            return True
                    except Exception:  # noqa: BLE001 - try next probe
                        continue
        except Exception:  # noqa: BLE001
            return False
        return False

    def capabilities(self) -> Dict[str, Any]:
        return {
            "execution_environment": "Any OpenAI-compatible server (vLLM HPC / Ollama /v1 local)",
            "default_context": 32768,
            "latency_profile": "Server-dependent (vLLM A40 27B ~15-25 tok/s)",
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
# "vllm" is the app-level name (models.yaml / env). "openai_compatible" is an
# explicit alias for the same adapter — any OpenAI-compatible server works.
provider_registry = ProviderRegistry()
provider_registry.register("ollama", OllamaProvider())
provider_registry.register("huggingface", HuggingFaceProvider())
_openai_compat = OpenAICompatibleProvider()
provider_registry.register("vllm", _openai_compat)
provider_registry.register("openai_compatible", _openai_compat)


