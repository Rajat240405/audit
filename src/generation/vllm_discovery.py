"""vLLM served-model discovery — the single place that answers
"what model is the configured OpenAI-compatible server CURRENTLY serving?".

Layer separation (kept strictly):

    A. DISCOVERY   → this module (server is authoritative for AVAILABILITY)
    B. CAPABILITIES→ ModelRegistry catalog (authoritative for KNOWN models);
                     dynamically-registered conservative families for unknown
                     ones (metadata_source="server" | "fallback", thinking
                     capability UNKNOWN — never claimed)
    C. POLICY      → src.generation.policy (Fast/Deep, capability-derived)

Selection policy (explicit and deterministic — see discover_active_vllm_model):

    1. ``VLLM_MODEL`` set (or an explicit ``override`` arg)  → that exact id
       is the pinned choice (local testing / intentional pinning). If the
       server is reachable and lists it, its server-reported ``max_model_len``
       enriches the entry; otherwise the pin still wins (loud note).
    2. Exactly one served model                             → that model.
    3. Multiple served models                               → the first id in
       sorted order (deterministic across calls; the app architecture expects
       exactly one active model), with a loud note. Re-pin with VLLM_MODEL to
       select another, or call refresh_vllm_discovery() after the server
       changes what it serves.
    4. Zero served models / server unreachable              → VLLMDiscoveryError.

Caching: ``GET /v1/models`` is NOT called per generation request — results are
cached per normalized base URL for ``VLLM_DISCOVERY_TTL_SECONDS`` (default
300). ``refresh_vllm_discovery()`` forces a re-read (HPC model swaps); the
POST /api/models/refresh endpoint calls exactly that.

Failure contract: discover_* raises VLLMDiscoveryError on any connection /
HTTP / payload failure (callers decide how to degrade); generation-path
consumers fall back to the last-known-good configuration with a loud note.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


class VLLMDiscoveryError(RuntimeError):
    """vLLM discovery failed (unreachable server / bad payload / no models)."""


# ─────────────────────────────────────────────────────────────────────────────
# TTL cache (per normalized base URL)
# ─────────────────────────────────────────────────────────────────────────────

def _ttl_seconds() -> float:
    raw = (os.environ.get("VLLM_DISCOVERY_TTL_SECONDS") or "").strip()
    try:
        return max(0.0, float(raw)) if raw else 300.0
    except ValueError:
        return 300.0


_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def _now() -> float:
    return time.monotonic()


def discover_vllm_models(
    base_url: Optional[str] = None,
    *,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Served models of the configured server: ``[{"id": str, "max_model_len":
    int|None}, ...]``, TTL-cached.

    ``base_url`` resolution matches the provider exactly (explicit arg →
    ``VLLM_BASE_URL`` env → http://localhost:8001); both ``http://host:8100``
    and ``http://host:8100/v1`` spellings are normalized by the provider's
    ``openai_url.models_url`` join. The provider's ``served_models`` is the
    single fetch+parse implementation (no duplicated wire logic here) and
    already raises on connection/HTTP failure — wrapped as VLLMDiscoveryError.
    """
    from src.generation.registry import OpenAICompatibleProvider  # lazy: no cycle

    probe = OpenAICompatibleProvider(base_url=base_url)  # explicit arg wins; else env
    key = probe.base_url  # normalized (rstrip "/") cache key
    now = _now()
    if not force:
        hit = _cache.get(key)
        if hit is not None and (now - hit[0]) < _ttl_seconds():
            return list(hit[1])
    try:
        served = probe.served_models()
    except Exception as e:  # noqa: BLE001
        raise VLLMDiscoveryError(
            f"vLLM discovery failed at {probe.base_url}: {e}"
        ) from e
    if not served:
        raise VLLMDiscoveryError(
            f"vLLM discovery at {probe.base_url}: server reports zero served models"
        )
    _cache[key] = (now, list(served))
    return served


def discover_active_vllm_model(
    base_url: Optional[str] = None,
    *,
    override: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """The ONE active served model, per the deterministic policy in the
    module docstring. Returns ``{"id": str, "max_model_len": int|None,
    "pinned": bool, "alternatives": [ids]}``.
    """
    pinned = (override if override is not None else os.environ.get("VLLM_MODEL") or "").strip()

    if pinned:
        # Explicit pin: the choice is made without the server. Enrich with
        # server-reported metadata when reachable (best-effort — a pin must
        # also work when the server is down or serves a different id, e.g.
        # local testing against Ollama's /v1).
        entry: Dict[str, Any] = {"id": pinned, "max_model_len": None,
                                 "pinned": True, "alternatives": []}
        try:
            served = discover_vllm_models(base_url, force=force)
        except VLLMDiscoveryError as e:
            print(f"[models] VLLM_MODEL={pinned!r} pinned; discovery unavailable ({e})")
            return entry
        for m in served:
            if m["id"] == pinned:
                entry["max_model_len"] = m.get("max_model_len")
                return entry
        print(
            f"[models] VLLM_MODEL={pinned!r} not in server list "
            f"{sorted(m['id'] for m in served)} — pin wins (override semantics)"
        )
        return entry

    served = discover_vllm_models(base_url, force=force)
    if len(served) == 1:
        m = served[0]
        return {"id": m["id"], "max_model_len": m.get("max_model_len"),
                "pinned": False, "alternatives": []}
    ordered = sorted(served, key=lambda m: m["id"])
    chosen = ordered[0]
    print(
        "[models] multiple served models "
        f"{[m['id'] for m in ordered]} — deterministically active={chosen['id']!r} "
        "(sorted-first; pin another via VLLM_MODEL)"
    )
    return {"id": chosen["id"], "max_model_len": chosen.get("max_model_len"),
            "pinned": False, "alternatives": [m["id"] for m in ordered[1:]]}


def refresh_vllm_discovery(base_url: Optional[str] = None) -> Dict[str, Any]:
    """Force re-discovery (bypasses the TTL cache) and re-derive the active
    model. This is the explicit refresh mechanism for HPC model swaps; the
    POST /api/models/refresh endpoint calls exactly this."""
    return discover_active_vllm_model(base_url, force=True)


def clear_discovery_cache() -> None:
    _cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Catalogue resolution (served id → ModelFamily capabilities)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_served_family(
    served_id: str,
    server_ctx: Optional[int] = None,
    *,
    provider: str = "vllm",
    registry=None,
):
    """Resolve a served model id to a ModelFamily: EXACT match wins; never a
    substring/heuristic match. Returns ``(family, metadata_source)`` where
    metadata_source is ``"catalog" | "server" | "fallback"``.

    Order (mirrors the /api/models endpoint — both share THIS implementation):

      1. exact ``model_name`` / ``id`` match within the provider's own catalog
         section          → the catalogued family (capabilities authoritative)
      2. exact ``model_name`` match in another provider's section
                            → capabilities reused (provenance still "catalog";
                              family registered dynamically for this provider,
                              transport follows the ACTIVE provider)
      3. server-reported ``max_model_len``
                            → dynamic family, metadata_source="server",
                              thinking capability UNKNOWN (supported=None —
                              never claimed, no invented reasoning parser)
      4. nothing            → dynamic family, metadata_source="fallback",
                              conservative 8192 context, thinking UNKNOWN

    Dynamic families are REGISTERED into the model registry (id stable:
    sanitized served id, ``__<provider>`` suffix on collision) so catalog
    resolution, think-mode resolution and /api/provider switching all see
    them through the existing machinery.
    """
    from src.generation.registry import model_registry as _default_registry

    reg = registry or _default_registry

    # 1. exact match in THIS provider's section (no registry.get substring path)
    for f in reg.list_by_provider(provider):
        if f.model_name == served_id or f.id == served_id:
            return f, "catalog"

    # 2. exact model_name match in another provider's section
    for f in reg.list_all():
        if f.provider == provider:
            continue
        if f.model_name == served_id:
            fam_id = _register_dynamic_family(
                reg, provider, served_id, f.context_window,
                display_name=f.display_name, thinking_supported=f.thinking.supported,
                metadata_source="catalog",
            )
            return reg.get(fam_id), "catalog"

    # 3/4. unknown to the catalog — server metadata or flagged fallback
    if isinstance(server_ctx, (int, float)) and server_ctx and int(server_ctx) > 0:
        fam_id = _register_dynamic_family(
            reg, provider, served_id, int(server_ctx),
            display_name=served_id, thinking_supported=None,
            metadata_source="server",
        )
        return reg.get(fam_id), "server"
    fam_id = _register_dynamic_family(
        reg, provider, served_id, default_unknown_context(),
        display_name=served_id, thinking_supported=None,
        metadata_source="fallback",
    )
    return reg.get(fam_id), "fallback"


def default_unknown_context() -> int:
    """Conservative last-resort context for unknown models not reporting one.
    Kept identical to the endpoint's historical fallback (8192) — small on
    purpose so evidence budgets err safe."""
    return 8192


def _register_dynamic_family(
    reg,
    provider: str,
    served_id: str,
    context_window: int,
    *,
    display_name: str,
    thinking_supported: Optional[bool],
    metadata_source: str,
) -> str:
    """Register (or reuse) a dynamic family for a served model.

    Capability honesty: ``thinking_capable`` (the legacy bool consumed by UI
    switches and smoke tests) stays False — nothing is claimed; the canonical
    ``thinking.supported`` is the tri-state (None = UNKNOWN, never a false
    claim of support). Control mechanism = the provider TRANSPORT default
    (metadata only): with supported=None the execution policy resolves
    ``wire_think=None`` and NOTHING thinking-related is ever sent on the wire
    for unknown models. No reasoning parser is invented (ServingSpec empty).
    Model names are used verbatim — never suffixed/mangled.
    """
    from src.generation.registry import (
        ModelFamily,
        ThinkingSpec,
        _normalize_control,
        provider_transport_default_control,
        provider_transport_default_think_mode,
    )

    base = re.sub(r"[^a-z0-9]+", "_", str(served_id).lower()).strip("_") or "served_model"
    fam_id = base
    existing = reg.get(fam_id)
    if existing is not None and (
        existing.provider != provider or existing.model_name != served_id
    ):
        fam_id = f"{base}__{provider}"
        existing = reg.get(fam_id)
    if existing is None:
        reg.register(ModelFamily(
            id=fam_id,
            display_name=display_name,
            provider=provider,
            model_name=served_id,  # verbatim — sent to the server as-is
            context_window=int(context_window),
            thinking_capable=bool(thinking_supported),  # legacy bool: False = unclaimed
            recommended_execution_mode="GPU",
            think_mode=provider_transport_default_think_mode(provider),
            thinking=ThinkingSpec(
                supported=thinking_supported,  # None → UNKNOWN (never claimed)
                control=_normalize_control(provider_transport_default_control(provider)),
            ),
            metadata_source=metadata_source,
        ))
    return fam_id
