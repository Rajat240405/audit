"""Active-generation-stack resolver (Phase 2 / WS2-C, additive).

ONE generation architecture (spec §9): the discovery → registry → policy →
client chain used by the server is the authoritative model-resolution path.
This module gives NON-serving components (the GraphRAG build CLI, tests,
future tools) the same resolution WITHOUT touching the server:

  enabled providers   : APP_PROVIDERS convention (same as the server)
  active provider     : APP_DEFAULT_PROVIDER → vllm when VLLM_BASE_URL is set
                        → ollama otherwise
  active model        : VLLM_MODEL pin, else live vLLM discovery
                        (discover_active_vllm_model + resolve_served_family)
                        for server providers, else the registry default family
  execution plan      : resolve_execution(family, mode, provider, serving_limit)
  client              : LLMClient(provider, model, num_ctx, plan params)

If the serving model changes (HPC swap), the next resolution picks it up —
no GraphRAG code change needed (spec §9). No model names, provider URLs or
provider-specific logic live here or in graphrag: everything is resolved from
the shared registry / discovery / policy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.generation.client import LLMClient
from src.generation.policy import ExecutionPlan, resolve_execution
from src.generation.registry import (
    _resolve_family_for_model,
    model_registry,
    provider_registry,
)

__all__ = ["ActiveStack", "resolve_active_stack"]

_SERVER_PROVIDERS = ("vllm", "openai_compatible")


@dataclass(frozen=True)
class ActiveStack:
    """The resolved (provider, model, family, plan, client) tuple."""

    provider: str
    model: str
    family_id: str
    plan: ExecutionPlan
    client: LLMClient
    source: str  # "pin" | "discovery" | "registry-default"


def _default_provider() -> str:
    """Same convention as the server: APP_DEFAULT_PROVIDER wins; else vllm
    when VLLM_BASE_URL is set; else ollama."""
    p = (os.environ.get("APP_DEFAULT_PROVIDER", "") or "").strip().lower()
    if p and provider_registry.get(p) is not None:
        return p
    return "vllm" if os.environ.get("VLLM_BASE_URL") else "ollama"


def _enabled_providers() -> list[str]:
    raw = (os.environ.get("APP_PROVIDERS", "") or "").strip()
    if raw:
        provs = [p.strip().lower() for p in raw.split(",") if p.strip()]
        valid = [p for p in provs if provider_registry.get(p) is not None]
        if valid:
            return valid
    return [_default_provider()]


def resolve_active_stack(mode: str = "fast") -> ActiveStack:
    """Resolve the currently active generation stack (see module docstring).

    ``mode`` selects the execution profile ("fast"|"deep"); extraction runs
    on "fast" (deterministic, low temperature). Raises RuntimeError only if
    the provider configuration is unusable (no enabled provider).
    """
    enabled = _enabled_providers()
    provider = _default_provider()
    if provider not in enabled:
        provider = enabled[0]

    pin = (os.environ.get("VLLM_MODEL", "") or "").strip()
    serving_limit: Optional[int] = None
    source = "registry-default"

    if provider in _SERVER_PROVIDERS:
        if pin:
            # explicit pin wins (override semantics, same as the server).
            # The pin names a model, but resolve_execution() needs the FAMILY
            # that carries its capabilities — resolve it through the shared
            # registry helper (exact model_name/id match, no heuristics) so no
            # model-specific mapping is introduced here.
            model = pin
            family = _resolve_family_for_model(model)
            if family is None:
                # A pin is an explicit operator assertion, so a name that is
                # not in the catalog is a configuration error, not something
                # to paper over with a synthesized fallback family.
                raise RuntimeError(
                    f"VLLM_MODEL={pin!r} is not a known model: no family in "
                    f"the registry has that model_name or id. "
                    f"Known models: {_known_model_names()}. "
                    f"Fix the VLLM_MODEL pin, add the model to "
                    f"config/models.yaml, or unset VLLM_MODEL to use live "
                    f"vLLM discovery."
                )
            source = "pin"
        else:
            from src.generation.vllm_discovery import (
                VLLMDiscoveryError,
                discover_active_vllm_model,
                resolve_served_family,
            )
            try:
                active = discover_active_vllm_model()
                fam, _meta_source = resolve_served_family(
                    active["id"], active.get("max_model_len"), provider=provider)
                model = fam.model_name
                family = fam
                serving_limit = active.get("max_model_len")
                source = "discovery"
            except VLLMDiscoveryError:
                # unreachable server at resolve time: fall back to the
                # registry default family (loud — the caller sees source)
                fam_id = (os.environ.get("APP_FAMILY_ID", "") or "").strip() \
                    or _default_family_id()
                family = model_registry.get(fam_id)
                model = family.model_name if family else _default_model_name()
    else:
        fam_id = (os.environ.get("APP_FAMILY_ID", "") or "").strip() \
            or _default_family_id()
        family = model_registry.get(fam_id)
        model = family.model_name if family else _default_model_name()

    plan = resolve_execution(family, mode, provider, serving_limit=serving_limit)
    client = LLMClient(
        provider=provider,
        model=model,
        num_ctx=plan.num_ctx,
        temperature=plan.temperature,
        max_tokens=plan.max_tokens,
    )
    client.think = plan.wire_think  # wire-level think control (server parity)
    return ActiveStack(
        provider=provider,
        model=model,
        family_id=getattr(family, "id", "") or "",
        plan=plan,
        client=client,
        source=source,
    )


def _known_model_names() -> str:
    """Sorted, de-duplicated catalog model names, for error messages."""
    names = sorted({f.model_name for f in model_registry.list_all() if f.model_name})
    return ", ".join(names) if names else "(registry is empty)"


def _default_family_id() -> str:
    from src.generation.defaults import default_family_id
    return default_family_id()


def _default_model_name() -> str:
    from src.generation.defaults import default_model_name
    return default_model_name()
