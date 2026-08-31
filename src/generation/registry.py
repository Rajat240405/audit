"""
Provider and Model Registry System for Intelligent LLM Management.
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
# Model capability model
#
# Layer separation (Model Capability Architecture, Task 1):
#   ModelCapabilities (ModelFamily) — WHAT the model can do. Metadata only.
#   ServingSpec                     — HOW the server must be launched (never
#                                     consulted by execution policy).
#   Provider adapters (below)       — HOW requests are encoded on the wire.
#   src.generation.policy           — WHICH parameters an execution mode means,
#                                     derived FROM capabilities (not hardcoded).
# ModelFamily stays import-compatible (same fields, plus optional new ones);
# legacy catalog keys (think_mode / thinking_capable) are dual-read into the
# capability specs, and the legacy fields are kept consistent for old readers.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical thinking-control mechanisms (capability vocabulary), mapped onto
# the legacy adapter "wire" strings used by the provider implementations
# (adapters are intentionally untouched — wire behavior is preserved exactly):
#   "chat_template_kwargs" -> "template"  (vLLM: chat_template_kwargs.enable_thinking)
#   "request_flag"         -> "key"       (Ollama: top-level `think`)
#   "server_default"       -> "none"      (send nothing; server decides)
_CONTROL_TO_WIRE: Dict[str, str] = {
    "chat_template_kwargs": "template",
    "request_flag": "key",
    "server_default": "none",
}
_CONTROL_ALIASES: Dict[str, str] = {
    # canonical values
    "chat_template_kwargs": "chat_template_kwargs",
    "request_flag": "request_flag",
    "server_default": "server_default",
    # legacy wire aliases accepted from catalogs
    "template": "chat_template_kwargs",
    "key": "request_flag",
    "none": "server_default",
}


def _normalize_control(value: Any) -> Optional[str]:
    """Normalize any accepted thinking-control spelling to the canonical value."""
    if value is None:
        return None
    norm = _CONTROL_ALIASES.get(str(value).strip().lower())
    if norm is None:
        print(f"[models] unrecognized thinking control {value!r} — ignored")
    return norm


def _wire_for_control(control: Optional[str], fallback: str = "none") -> str:
    """Legacy adapter wire string for a canonical control value."""
    if control is None:
        return fallback
    return _CONTROL_TO_WIRE.get(control, fallback)


def provider_transport_default_control(provider: str) -> str:
    """PROVIDER transport default (not model metadata): how this provider
    encodes thinking on the wire when a family says nothing. vLLM /
    OpenAI-compatible servers use chat_template_kwargs (inert for templates
    that don't read it); everything else defers to the server."""
    if provider in ("vllm", "openai_compatible"):
        return "chat_template_kwargs"
    return "server_default"


def provider_transport_default_think_mode(provider: str) -> str:
    """Legacy-string view of provider_transport_default_control (used by
    served-model discovery when registering dynamic families)."""
    return _wire_for_control(provider_transport_default_control(provider))


@dataclass
class ThinkingSpec:
    """Model thinking CAPABILITY. `supported`: None = unknown (never claimed).
    `control`: canonical mechanism or None (provider transport default)."""
    supported: Optional[bool] = None
    control: Optional[str] = None  # "request_flag" | "chat_template_kwargs" | "server_default" | None


@dataclass
class ServingSpec:
    """Server-launch requirements for this model (deployment metadata ONLY —
    never read by execution policy, never sent in a request).

    `reasoning_parser` is also read by _payload() to decide whether to send
    `thinking_token_budget` — models served with a reasoning parser have
    thinking ON by default and need a budget cap to avoid token loops."""
    reasoning_parser: Optional[str] = None
    max_model_len: Optional[int] = None
    notes: Optional[str] = None
    # Per-request thinking token cap (sent when reasoning_parser is set).
    # Prevents thinking loops on models whose reasoning is always-ON
    # (e.g. gpt-oss with openai_gptoss parser). 0 = no cap (server default).
    default_thinking_budget: int = 0


@dataclass
class GenerationDefaults:
    """Model-specific generation defaults (used only where the execution
    profile does not pin the value; all optional, unset = unchanged wire)."""
    temperature: Optional[float] = None
    top_p: Optional[float] = None


@dataclass
class ModelFamily:
    """
    Metadata representation of a family of models (a capability record).
    """
    id: str
    display_name: str
    provider: str  # "ollama" | "vllm" | "openai_compatible" | "huggingface"
    model_name: str  # Single concrete model name on the provider (never mangled)
    context_window: int
    thinking_capable: bool
    recommended_execution_mode: str = "GPU"
    # Model NATIVE context (capability the model itself supports, e.g. from
    # the HF card). Distinct from context_window, which is the DEPLOYED /
    # application-blessed window for this catalogue entry ("how much of the
    # context this deployment intends to use"). None = not documented in the
    # catalog — capability math falls back to context_window (never invented).
    # effective runtime context = min(native_or_context_window,
    # vllm_serving_limit_if_known, application_ceiling) — resolved by policy.
    native_context_window: Optional[int] = None
    # LEGACY (kept consistent; legacy adapter wire string):
    #   "template" -> send chat_template_kwargs.enable_thinking (vLLM + Qwen3.x;
    #                 model name untouched, thinking controlled per-request)
    #   "key"      -> top-level request flag (Ollama `think`)
    #   "none"     -> provider decides (default)
    # The old "suffix" value (append /think or /nothink to the model NAME) was
    # removed: it is not a vLLM mechanism — the server 404s any model id it
    # does not serve. populate_model_registry migrates legacy "suffix" entries
    # to "template" with a warning. Model names are NEVER mangled.
    think_mode: str = "none"
    # Capability specs (optional; dual-read with the legacy keys above).
    thinking: ThinkingSpec = None  # type: ignore[assignment]  (set in __post_init__)
    serving: ServingSpec = None    # type: ignore[assignment]
    defaults: GenerationDefaults = None  # type: ignore[assignment]
    max_output_tokens: Optional[int] = None
    metadata_source: str = "catalog"  # catalog | server | fallback (discovery)

    def __post_init__(self) -> None:
        if self.thinking is None:
            self.thinking = ThinkingSpec(
                supported=bool(self.thinking_capable),
                control=_normalize_control(self.think_mode),
            )
        if self.serving is None:
            self.serving = ServingSpec()
        if self.defaults is None:
            self.defaults = GenerationDefaults()

    @classmethod
    def from_dict(cls, d: dict) -> "ModelFamily":
        """Build from a catalog entry (config/models.yaml).

        Dual-read: the new capability blocks (`thinking`, `serving`,
        `defaults`) and the legacy flat keys (`think_mode`,
        `thinking_capable`) are both accepted. When both are present and
        disagree, the capability block wins with a loud warning. The legacy
        fields on the resulting family are kept consistent either way, so
        existing readers see identical values."""
        d = dict(d)
        provider = str(d.get("provider", ""))
        provider_default = provider_transport_default_control(provider)

        t_block = d.get("thinking") or {}
        raw_control = t_block.get("control", d.get("think_mode"))
        control = _normalize_control(raw_control)
        if control is None and raw_control is None:
            control = provider_default
        wire = _wire_for_control(control, fallback=provider_transport_default_think_mode(provider))

        legacy_capable = d.get("thinking_capable")
        supported = t_block.get("supported", legacy_capable)
        if (
            control is not None
            and "think_mode" in d
            and _normalize_control(d.get("think_mode")) is not None
            and "thinking" in d
            and _normalize_control(d["think_mode"]) != control
        ):
            print(
                f"[models] family {d.get('id')!r}: think_mode={d['think_mode']!r} conflicts "
                f"with thinking.control={control!r} — capability block wins"
            )

        allowed = {
            "id", "display_name", "provider", "model_name", "context_window",
            "recommended_execution_mode", "metadata_source",
            "native_context_window",
        }
        base = {k: v for k, v in d.items() if k in allowed}
        serving = d.get("serving") or {}
        defaults = d.get("defaults") or {}
        return cls(
            **base,
            thinking_capable=bool(supported) if supported is not None else False,
            think_mode=wire,
            thinking=ThinkingSpec(
                supported=(bool(supported) if supported is not None else None),
                control=control,
            ),
            serving=ServingSpec(
                reasoning_parser=serving.get("reasoning_parser"),
                max_model_len=(
                    int(serving["max_model_len"])
                    if isinstance(serving.get("max_model_len"), (int, float))
                    else None
                ),
                notes=serving.get("notes"),
                default_thinking_budget=int(serving.get("default_thinking_budget") or 0),
            ),
            defaults=GenerationDefaults(
                temperature=defaults.get("temperature"),
                top_p=defaults.get("top_p"),
            ),
            max_output_tokens=(
                int(d["max_output_tokens"])
                if isinstance(d.get("max_output_tokens"), (int, float))
                else None
            ),
        )

    def get_execution_params(self, mode: str) -> Dict[str, Any]:
        """COMPATIBILITY SHIM — legacy fast/deep table, now resolved through
        the execution policy (src.generation.policy.resolve_execution) so
        every consumer sees the same capability-derived values. Prefer
        resolve_execution() in new code.

        Fast profile: temperature 0.0 · max_tokens 4096 · docs 3 · chars 1000 ·
        thinking OFF · verify light.
        Deep profile: temperature 0.2 · max_tokens 12288 · docs 5 · chars 3000 ·
        thinking ON · verify full. (Deep's 12288 = reasoning 8192 + output 4096
        because thinking-capable models spend max_tokens on reasoning AND answer.)"""
        from src.generation.policy import resolve_execution

        plan = resolve_execution(self, mode, self.provider)
        return {
            "temperature": plan.temperature,
            "max_tokens": plan.max_tokens,
            "max_context_docs": plan.max_context_docs,
            "max_doc_chars": plan.max_doc_chars,
            "thinking": plan.thinking,
            "verify_depth": plan.verify_depth,
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
            # Legacy default only when the entry carries NO thinking metadata
            # at all. vLLM controls Qwen3.x thinking per-request via
            # chat_template_kwargs.enable_thinking ("template") — never via a
            # /think|/nothink model-name suffix (not a vLLM mechanism).
            if "think_mode" not in entry and "thinking" not in entry:
                entry["think_mode"] = "template" if provider == "vllm" else "none"
            if entry.get("think_mode") == "suffix":
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


def _resolve_family_for_model(model: str) -> "Optional[ModelFamily]":
    """Return the ModelFamily for the given model_name/id, or None."""
    for f in model_registry.list_all():
        if f.model_name == model or f.id == model:
            return f
    return None


def resolve_think_mode(provider: str, model: str) -> str:
    """Resolve the legacy adapter wire string ("template" | "key" | "none")
    for (provider, model) from capability data — the single resolution point
    behind LLMClient._family_think_mode.

    Order: exact provider+model family -> model identity across any provider
    (dev-parity: provider vllm pointed at Ollama /v1 with an ollama model) ->
    "none" (never send thinking control to an unknown model).
    """
    for f in model_registry.list_all():
        if f.provider == provider and (f.model_name == model or f.id == model):
            return f.think_mode
    for f in model_registry.list_all():
        if f.model_name == model or f.id == model:
            return f.think_mode
    return "none"

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


def _resolved_ollama_url() -> str:
    """Current configured Ollama endpoint (env-driven; see client.ollama_base_url)."""
    from src.generation.client import ollama_base_url

    return ollama_base_url()


class OllamaProvider(BaseProvider):
    """
    Provider implementation for local Ollama service.

    The endpoint is NEVER hardcoded to 11434: it defaults to
    :func:`src.generation.client.ollama_base_url` (OLLAMA_BASE_URL /
    OLLAMA_HOST / localhost:11434), so health checks, generation, streaming
    and model discovery all hit the same configured endpoint even when the
    default port is unusable (e.g. a Windows excluded port range).
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url_explicit = base_url is not None
        self.base_url = (base_url or _resolved_ollama_url()).rstrip("/")

    def _url(self, base_url: str | None = None) -> str:
        """Resolve the endpoint for THIS call: explicit arg wins; a provider
        built without an explicit URL re-resolves the env every call (same
        laziness as OpenAICompatibleProvider), so health/generate/discovery
        can never drift to a stale 11434 after an env change."""
        if base_url:
            return base_url.rstrip("/")
        if not self._base_url_explicit:
            self.base_url = _resolved_ollama_url()
        return self.base_url

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
        base_url: Optional[str] = None,
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
        # mode-aware thinking: Fast=False (instant), Deep=True (reasoning).
        # Sent only when resolved upstream — no model-name heuristics here;
        # capability metadata + the execution plan decide when to force it.
        think = kwargs.get("think")
        if think is not None:
            payload["think"] = bool(think)

        start_time = time.monotonic()

        # LLMClient always passes its (resolved) base_url — explicit arg
        # wins; otherwise the env is re-resolved (never a stale default).
        url = self._url(base_url)

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(
                f"{url}/api/chat",
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
            return self.list_tags()
        except Exception:
            pass
        # Fallback matching the requested UI defaults (qwen3:8b is the
        # default family — it MUST be listed or the UI shows a stale set
        # whenever Ollama is offline)
        return ["qwen3:8b", "qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b", "gemma2:9b"]

    def list_tags(self, base_url: str | None = None) -> List[str]:
        """Models actually installed in THIS Ollama (``GET /api/tags``).

        Raises on connection/HTTP failure — callers that need an
        offline-tolerant path use :meth:`models`; the discovery endpoint
        turns the exception into HTTP 503 instead of pretending the catalog
        is installed."""
        url = self._url(base_url)
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in (r.json().get("models") or []) if m.get("name")]

    def show_context_length(self, model: str, base_url: str | None = None) -> int | None:
        """Real ``context_length`` from ``POST /api/show`` (server-reported,
        not guessed). None when Ollama does not report it."""
        url = self._url(base_url)
        try:
            with httpx.Client(timeout=3.0) as client:
                r = client.post(f"{url}/api/show", json={"name": model})
                if r.status_code != 200:
                    return None
                info = r.json().get("model_info") or {}
                for k, v in info.items():
                    if str(k).endswith("context_length") and isinstance(v, int):
                        return v
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return None
        return None

    def health(self, api_key: Optional[str] = None, base_url: str | None = None) -> bool:
        """Same signature as OpenAICompatibleProvider.health (LLMClient always passes base_url)."""
        url = self._url(base_url)
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


# ── Inline <think>…</think> extraction (OpenAI-compatible streaming) ────────
# A server launched WITHOUT a reasoning parser embeds the model's thinking
# inside ``delta.content`` as ``<think>…</think>`` text. The qwen3/deepseek_r1
# parsers split it server-side instead. This splitter handles the no-parser
# wire shape so thinking still surfaces separately from the visible answer.
# Buffered across SSE deltas — a think block (or even the tag itself) can be
# split over several chunks. Exact parity with LLMClient._stream_ollama
# (build 3), plus partial-tag hold-back.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _drain_inline_thinking(buf: str) -> tuple[List[Dict[str, str]], str]:
    """Split buffered content into reasoning/tokens events.

    Returns ``(events, remainder)`` where ``remainder`` holds an unterminated
    think-block or a partial ``<think>`` tag tail (deferred until more deltas
    arrive). Closed ``<think>…</think>`` blocks become ``reasoning`` events;
    everything else becomes ``tokens`` events in original order.
    """
    events: List[Dict[str, str]] = []
    out = buf
    while True:
        open_i = out.find(_THINK_OPEN)
        if open_i < 0:
            # No complete open tag. Hold back any tail that could be the
            # START of a "<think>" tag split across deltas ("<", "<t", …).
            hold = 0
            for n in range(min(len(out), len(_THINK_OPEN) - 1), 0, -1):
                if _THINK_OPEN.startswith(out[-n:]):
                    hold = n
                    break
            emit = out[:-hold] if hold else out
            if emit:
                events.append({"type": "tokens", "text": emit})
            return events, (out[-hold:] if hold else "")
        close_i = out.find(_THINK_CLOSE, open_i + len(_THINK_OPEN))
        if close_i < 0:
            # Think block still open — emit the answer text before it, defer
            # the in-progress thinking until its close tag arrives.
            if open_i:
                events.append({"type": "tokens", "text": out[:open_i]})
            return events, out[open_i:]
        if open_i:
            events.append({"type": "tokens", "text": out[:open_i]})
        events.append({"type": "reasoning", "text": out[open_i + len(_THINK_OPEN):close_i]})
        out = out[close_i + len(_THINK_CLOSE):]


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
            # Cap reasoning tokens to prevent thinking loops on large contexts.
            # Without this, Qwen3 defaults to xhigh effort and exhausts its
            # entire token budget in <think> before writing any answer.
            # Only applied in thinking-ON (Deep) mode — Fast mode skips this.
            if bool(think):
                body["thinking_token_budget"] = 4096

        # Models served with a reasoning_parser (e.g. openai_gptoss) have
        # thinking always-ON regardless of think_mode/enable_thinking.
        # Send thinking_token_budget unconditionally so they never loop.
        # Resolved from the catalog ServingSpec; 0 means no cap (server default).
        _fam = _resolve_family_for_model(model)
        if _fam is not None:
            budget = getattr(getattr(_fam, "serving", None), "default_thinking_budget", 0)
            if budget and budget > 0 and "thinking_token_budget" not in body:
                body["thinking_token_budget"] = budget

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
        as the Ollama/HF streams, so the frontend reasoning panel is shared.

        Thinking deltas are separated from answer deltas for EVERY wire shape
        an OpenAI-compatible server produces:
          1. ``delta.reasoning_content`` — older vLLM reasoning parsers
             (deepseek_r1), DeepSeek-style servers.
          2. ``delta.reasoning`` — newer vLLM ``Qwen3ReasoningParser`` (what
             the HPC catalog declares: serving.reasoning_parser: qwen3).
             Newer vLLM renamed the streaming field; reading ONLY
             ``reasoning_content`` silently drops all thinking here — the
             answer renders fine, the Thinking panel never fills.
          3. ``delta.thinking`` — a few builds use this name.
          4. No reasoning parser at all — thinking arrives INLINE inside
             ``delta.content`` as ``<think>…</think>``; extracted exactly
             like the Ollama adapter (build 3) so the answer stays clean.
        """
        messages: list = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = self._payload(model, messages, temperature, max_tokens,
                             num_ctx, stream=True, think=think, think_mode=think_mode)
        buf = ""  # content-side buffer for inline <think> extraction (shape 4)
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
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thinking")
                        or ""
                    )
                    if reasoning:
                        yield {"type": "reasoning", "text": reasoning}
                    content = delta.get("content") or ""
                    if content:
                        buf += content
                        events, buf = _drain_inline_thinking(buf)
                        for ev in events:
                            if ev["type"] == "tokens":
                                if not answered:
                                    yield {"type": "answer_start"}
                                    answered = True
                            yield ev
        # Flush: the stream ended while a think block was still open (e.g.
        # the token budget ran out mid-thought) or with a partial tag tail.
        # Unterminated thinking was NEVER answer text — emit it as reasoning
        # so the visible answer stays empty and the server can surface its
        # "reasoning but no answer" notice instead of raw chain-of-thought.
        if buf:
            open_i = buf.find(_THINK_OPEN)
            if open_i >= 0:
                pre, tail = buf[:open_i], buf[open_i + len(_THINK_OPEN):]
                if pre:
                    if not answered:
                        yield {"type": "answer_start"}
                        answered = True
                    yield {"type": "tokens", "text": pre}
                if tail:
                    yield {"type": "reasoning", "text": tail}
            else:
                # dangling partial like "<t" — never meaningful, but don't lose text
                if not answered:
                    yield {"type": "answer_start"}
                    answered = True
                yield {"type": "tokens", "text": buf}
        yield {"type": "done"}

    def models(self) -> List[str]:
        """Model ids this provider may send. DISCOVERY-BACKED: the ids the
        connected server actually serves (TTL-cached — /v1/models is not
        re-polled per call). Falls back to the explicit ``VLLM_MODEL`` pin /
        the historical catalog default when the server is unreachable, so
        this legacy BaseProvider API keeps its non-raising contract."""
        try:
            from src.generation.vllm_discovery import discover_vllm_models

            return [m["id"] for m in discover_vllm_models()]
        except Exception:  # noqa: BLE001 - offline contract preserved
            return [os.environ.get("VLLM_MODEL") or "Qwen3.6-35B-A3B-FP8"]

    def served_models(self, base_url: str | None = None) -> List[Dict[str, Any]]:
        """Models actually SERVED by the connected server (``GET /v1/models``).

        This is the ONLY availability source for the vLLM path: a model file
        sitting in someone's directory is irrelevant — what the connected
        vLLM process serves is what Audit can use. Returns
        ``[{"id": <served id>, "max_model_len": <int|None>}, ...]``
        (vLLM reports ``max_model_len`` per model card when launched with a
        known context, so the context window is server-reported rather than
        invented). Raises on connection/HTTP failure — the endpoint surface
        turns that into HTTP 503 instead of silently falling back to the
        YAML catalog."""
        from src.generation.openai_url import models_url

        url = models_url(self._url(base_url))
        with httpx.Client(timeout=4.0) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        out: List[Dict[str, Any]] = []
        for m in (data.get("data") or []):
            mid = m.get("id")
            if not mid:
                continue
            max_len = m.get("max_model_len")
            out.append({
                "id": mid,
                "max_model_len": int(max_len) if isinstance(max_len, (int, float)) else None,
            })
        return out

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
            # Context is NOT a provider-level constant: effective context is
            # resolved per active model as min(native ∩ serving-limit ∩ app
            # ceiling) by the execution policy (src.generation.policy).
            "context_handling": "dynamic: min(model native, serving max_model_len, app ceiling)",
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


