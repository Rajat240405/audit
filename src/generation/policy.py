"""Execution policy — the single place where an execution MODE plus a model's
CAPABILITIES become concrete, wire-ready parameters.

    MODEL (ModelFamily capabilities, catalog/server metadata)
      ↓
    resolve_execution(family, mode, provider)
      ↓
    ExecutionPlan  ── provider adapters encode it on the wire ──> REQUEST

Separation rules (Model Capability Architecture, Task 1):
  * this module owns POLICY ONLY — the numbers Standard(=fast)/Deep mean and
    how they clamp against capabilities;
  * capabilities live on the family (registry); serving requirements live in
    ServingSpec and are never read here; provider adapters encode the wire.

Behavior-preserving refactor: for every family in the shipped catalogs the
resolved plan reproduces the legacy global fast/deep table EXACTLY:
  fast: temperature 0.0 · max_tokens 4096 · thinking OFF
  deep: temperature 0.2 · max_tokens 12288 · thinking ON
  prompt budget: int(context_window × 0.80)   (legacy report value)
Deep's 12288 is now DERIVED (reasoning budget 8192 + output budget 4096)
instead of a magic constant, so per-model clamps can apply without touching
the numbers for today's models.

Task 3 (dynamic evidence budgeting) is layered on top: the plan carries a
reserve-based evidence budget
  evidence_budget = num_ctx − max_tokens − scaffold − safety_margin
plus an initial retrieval candidate pool per profile (fast 5 / deep 10).
The pool is NOT a final document quota — src.generation.evidence admits as
many relevant candidates as the budget fits. Legacy max_context_docs /
max_doc_chars remain ONLY as fallbacks for plan-less consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# Legacy flat prompt-budget share. Preserved for compat reporting
# (plan.prompt_budget_tokens); Task-3 budgeting uses the reserve-based fields
# below (evidence_budget_tokens), not this ratio.
PROMPT_BUDGET_RATIO = 0.80

# Fixed prompt furniture reserve (Task 3): the non-evidence parts of the user
# prompt — intro line, RETRIEVED CONTEXT banner, USER QUESTION/ANSWER tail,
# separators. Per-source headers are accounted per candidate at admission
# time. Conservative (measured furniture is ~60 tokens).
PROMPT_SCAFFOLD_TOKENS = 120

# Safety margin absorbing chars//4 estimation error and provider-specific
# over-length behaviour (Ollama silently truncates over-num_ctx prompts;
# vLLM answers HTTP 400). The budget never spends this reserve.
SAFETY_MARGIN_RATIO = 0.05
SAFETY_MARGIN_MIN_TOKENS = 256

# ── Effective runtime context (dynamic context handling) ─────────────────────
# The budget calculator's context input is NOT a constant. Conceptually:
#
#   effective_context = min( MODEL NATIVE CONTEXT
#                          ∩ VLLM SERVING LIMIT (--max-model-len, if reported)
#                          ∩ APPLICATION SAFETY LIMIT )
#
# with only EXISTING fallback behavior allowed when an input is unknown:
#   * native unknown      → the family context_window (the catalogue's
#                           declared capability — never a fabricated number);
#   * serving unknown     → not clamped (the vLLM /v1/models endpoint does
#                           not always expose --max-model-len; nothing is
#                           invented in that case);
#   * no app ceiling set  → the family's catalogue context_window (i.e. the
#                           previous, de-facto application cap — preserving
#                           today's budgets exactly on current deployments).
#
# The application safety limit is deliberate and INDEPENDENT of model/server:
# RAG_MAX_CONTEXT_TOKENS (env). A 262K-native model served at 131K does not
# entitle every request to 131K; operators raise the ceiling explicitly —
# without a code change. Deep mode never means "fill the context": max_tokens
# stays profile-derived; only the AVAILABLE capacity seen by the evidence
# allocator grows.
RAG_MAX_CONTEXT_TOKENS_ENV = "RAG_MAX_CONTEXT_TOKENS"


def app_context_ceiling_from_env() -> Optional[int]:
    """Configured application-side context ceiling (RAG_MAX_CONTEXT_TOKENS),
    or None when unset/invalid (invalid values warn and fall back)."""
    import os

    raw = (os.environ.get(RAG_MAX_CONTEXT_TOKENS_ENV) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[exec] warning: {RAG_MAX_CONTEXT_TOKENS_ENV}={raw!r} is not an "
              "integer — ignoring (family context window applies)")
        return None
    if value <= 0:
        print(f"[exec] warning: {RAG_MAX_CONTEXT_TOKENS_ENV}={value} must be "
              "positive — ignoring (family context window applies)")
        return None
    return value


def describe_context(
    catalog_context: int,
    native_context: Optional[int],
    serving_limit: Optional[int],
    env_ceiling: Optional[int] = None,
) -> dict:
    """Resolve the effective runtime context AND expose every input (single
    source for resolve_execution and /api/status — no duplicated min logic).

    Returns keys: native_context_tokens, serving_context_tokens,
    app_context_limit_tokens, effective_context_tokens, clamped_by.
    Unknown inputs are None (nothing fabricated) and simply do not clamp.
    """
    if env_ceiling is None:
        env_ceiling = app_context_ceiling_from_env()
    inputs: dict = {
        "capability": native_context if isinstance(native_context, int) and native_context > 0 else int(catalog_context),
        "ceiling": env_ceiling if env_ceiling else int(catalog_context),
    }
    if isinstance(serving_limit, int) and serving_limit > 0:
        inputs["serving"] = serving_limit
    clamped_by = min(inputs, key=inputs.get)
    return {
        "native_context_tokens": (
            native_context if isinstance(native_context, int) and native_context > 0 else None
        ),
        "serving_context_tokens": inputs.get("serving"),
        "app_context_limit_tokens": inputs["ceiling"],
        "effective_context_tokens": min(inputs.values()),
        "clamped_by": clamped_by,
    }


# Standard(=fast)/Deep profile POLICY. These are policy choices applied ON TOP
# OF capabilities — not model metadata.
_PROFILES: dict = {
    "fast": {
        "temperature": 0.0,
        "output_budget": 4096,     # visible answer budget
        "reasoning_budget": 0,     # thinking OFF → no reasoning reserve
        "thinking": False,
        "retrieval_top_k": 5,      # INITIAL candidate pool (Task 3) — NOT a
                                   # final document quota; budget decides
        "max_context_docs": 3,     # LEGACY fallback only (no-plan consumers)
        "max_doc_chars": 1000,     # LEGACY fallback only (no-plan consumers)
        "verify_depth": "light",   # regex-only, no LLM judge
    },
    "deep": {
        "temperature": 0.2,
        "output_budget": 4096,     # answer still gets a full 4k
        "reasoning_budget": 8192,  # thinking spends max_tokens BEFORE answering
        "thinking": True,
        "retrieval_top_k": 10,     # INITIAL candidate pool (Task 3)
        "max_context_docs": 5,     # LEGACY fallback only
        "max_doc_chars": 3000,     # LEGACY fallback only
        "verify_depth": "full",
    },
}


def profile_for(mode: str) -> dict:
    """The policy row for an execution mode; unknown modes fall back to the
    Standard/fast profile (mirrors the legacy behavior of treating any
    non-'deep' mode as fast)."""
    key = "deep" if str(mode).lower().strip() == "deep" else "fast"
    return _PROFILES[key]


@dataclass(frozen=True)
class ExecutionPlan:
    """Fully-resolved parameters for ONE model + ONE mode + ONE provider.

    Everything the execution layers (server handlers, CLI, provider adapters
    via LLMClient) consume. Resolved once per request/switch from
    capabilities — no consumer may re-derive or hardcode these.
    """

    mode: str                     # normalized: "fast" | "deep"
    provider: str
    model: str
    # generation parameters
    temperature: float
    max_tokens: int               # output_budget + reasoning_budget (clamped)
    top_p: Optional[float]        # model default, if declared (unsent today)
    # thinking
    thinking: bool                # what the mode requests (OFF/ON)
    think_mode: str               # legacy adapter wire string for the mechanism
    # context
    num_ctx: int
    prompt_budget_tokens: int     # int(num_ctx × PROMPT_BUDGET_RATIO) — legacy report
    output_budget_tokens: int
    reasoning_budget_tokens: int
    # Task-3 dynamic evidence budgeting (reserve-based, capability-derived —
    # never model-named):
    retrieval_top_k: int          # initial candidate pool per profile
    prompt_scaffold_tokens: int   # fixed prompt furniture reserve
    safety_margin_tokens: int     # estimation/provider safety reserve
    evidence_budget_tokens: int   # num_ctx − max_tokens − scaffold − margin
                                  # (assembly additionally subtracts the ACTUAL
                                  # system-prompt and question tokens at runtime)
    # evidence/retrieval budgeting — LEGACY fallback values used only when no
    # plan is attached (standalone AnswerGenerator, interactive CLI)
    max_context_docs: int
    max_doc_chars: int
    # verification
    verify_depth: str
    # What may actually go on the wire for THIS model: the mode value when
    # thinking capability is KNOWN (supported true/false); None when the
    # capability is UNKNOWN (dynamically discovered model) — consumers must
    # send NOTHING thinking-related then (server default applies; no
    # model-specific control may be enabled without reliable metadata).
    # Optional trailing field: existing constructors stay valid; for every
    # catalogued family this equals `thinking` (capability is always known).
    wire_think: Optional[bool] = None
    # Context identity (dynamic effective-context resolution): every input of
    # the min() plus which one clamped — for status/debug surfaces.
    native_context_tokens: Optional[int] = None
    serving_context_tokens: Optional[int] = None
    app_context_limit_tokens: Optional[int] = None
    context_clamped_by: Optional[str] = None
    # resolution transparency — surfaced, never hidden
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def resolve_execution(
    family,
    mode: str,
    provider: str = "ollama",
    model_name: str = "",
    serving_limit: Optional[int] = None,
) -> ExecutionPlan:
    """Resolve (ModelFamily capabilities, mode, provider) → ExecutionPlan.

    `family` is a registry ModelFamily (or None for an unregistered model —
    a flagged conservative fallback family is synthesized). No model-name
    conditionals anywhere: every parameter comes from the profile policy row
    clamped/derived against family capabilities.

    `serving_limit` is the vLLM serving context (--max-model-len) when it was
    discovered from the running server; None when unknown — it then simply
    does not clamp (nothing is invented). The effective context is
    min(capability ∩ serving? ∩ application ceiling) — see describe_context —
    and feeds the SAME budget math downstream (num_ctx below is the
    effective value; the allocation logic itself is unchanged).
    """
    from src.generation.registry import ModelFamily, resolve_think_mode

    prof = profile_for(mode)
    mode_norm = "deep" if prof is _PROFILES["deep"] else "fast"
    warnings: list = []

    if family is None:
        # Unknown model (e.g. direct CLI use with a model not in the catalog):
        # conservative flagged fallback — capability claimed = nothing.
        from src.generation.defaults import default_num_ctx

        fam = ModelFamily(
            id="_unresolved",
            display_name=model_name or "(unregistered model)",
            provider=provider,
            model_name=model_name,
            context_window=default_num_ctx(),
            thinking_capable=False,
        )
        warnings.append(
            "model not in catalog — using flagged fallback capabilities "
            f"(context {fam.context_window}, thinking unsupported)"
        )
    else:
        fam = family

    model_name = getattr(fam, "model_name", "") or ""
    num_ctx = getattr(fam, "context_window", None)
    if not num_ctx or not isinstance(num_ctx, (int, float)):
        from src.generation.defaults import default_num_ctx

        num_ctx = default_num_ctx()
        warnings.append(
            f"context window unknown — fallback {num_ctx} used for budget math"
        )
    num_ctx = int(num_ctx)

    # ── Effective runtime context: capability ∩ serving? ∩ app ceiling ──
    # `num_ctx` above is the catalogue context (capability input + default
    # ceiling fallback). The effective value the budget math below consumes
    # is the dynamic minimum; every clamp is surfaced as a warning — never
    # silent. Context identity fields ride on the plan for debugging/status.
    ctx_info = describe_context(
        catalog_context=num_ctx,
        native_context=getattr(fam, "native_context_window", None),
        serving_limit=serving_limit,
    )
    effective_ctx = int(ctx_info["effective_context_tokens"])
    ctx_was_clamped = effective_ctx != num_ctx
    if ctx_was_clamped:
        warnings.append(
            # fires for restriction AND expansion alike (expansion = ceiling
            # above the catalogue window) — transparency, never silent.
            f"context resolved {num_ctx} → {effective_ctx} (limiting input: "
            f"{ctx_info['clamped_by']}; native="
            f"{ctx_info['native_context_tokens']}, serving="
            f"{ctx_info['serving_context_tokens']}, app ceiling="
            f"{ctx_info['app_context_limit_tokens']})"
        )
    num_ctx = effective_ctx

    # max_tokens = output budget + reasoning budget, clamped by the model's
    # declared output ceiling (if any). Clamp touches the REASONING share
    # first so the visible answer keeps its budget. No current catalog family
    # declares max_output_tokens → legacy 4096 / 12288 values are preserved.
    output_budget = int(prof["output_budget"])
    reasoning_budget = int(prof["reasoning_budget"])
    max_out = getattr(fam, "max_output_tokens", None)
    if isinstance(max_out, (int, float)) and max_out:
        room = max(0, int(max_out) - output_budget)
        if reasoning_budget > room:
            reasoning_budget = room
            warnings.append(
                f"reasoning budget clamped to {room} by model "
                f"max_output_tokens={int(max_out)}"
            )
        if output_budget > int(max_out):
            warnings.append(
                f"output budget clamped to model max_output_tokens={int(max_out)}"
            )
            output_budget = int(max_out)
    max_tokens = output_budget + reasoning_budget

    # Task-3 reserve-based evidence budget: context minus generation reserve
    # (max_tokens already = output + reasoning), fixed furniture and safety
    # margin. The runtime system-prompt/question cost is subtracted at
    # assembly time (they vary per request — e.g. appended TONE blocks).
    scaffold_tokens = PROMPT_SCAFFOLD_TOKENS
    margin_tokens = max(SAFETY_MARGIN_MIN_TOKENS, int(num_ctx * SAFETY_MARGIN_RATIO))
    evidence_budget = num_ctx - max_tokens - scaffold_tokens - margin_tokens
    if evidence_budget < 0:
        warnings.append(
            f"evidence budget clamped to 0 (num_ctx {num_ctx} − max_tokens "
            f"{max_tokens} − scaffold {scaffold_tokens} − margin {margin_tokens} "
            "< 0): generation reserve exceeds the context window"
        )
        evidence_budget = 0

    # Wire-level thinking: only models with a KNOWN capability verdict may be
    # sent a thinking control. ``supported`` is the tri-state capability
    # (True/False/None=unknown — None only for dynamically discovered,
    # uncatalogued models). For unknown models wire_think=None: adapters send
    # nothing (their `think is not None` guards), so the server default
    # applies and no model-specific control is ever invented for them.
    # Reasoning display gating keeps using plan.thinking (the MODE request).
    # (the synthesized fallback above is thinking_capable=False by
    # construction, so its resolved supported=False keeps legacy wire values).
    supported = getattr(getattr(fam, "thinking", None), "supported", None)
    wire_think: Optional[bool] = (
        None if supported is None else bool(prof["thinking"])
    )

    return ExecutionPlan(
        mode=mode_norm,
        provider=provider,
        model=model_name,
        temperature=float(prof["temperature"]),
        max_tokens=max_tokens,
        top_p=getattr(getattr(fam, "defaults", None), "top_p", None),
        thinking=bool(prof["thinking"]),
        think_mode=resolve_think_mode(provider, model_name) if model_name else "none",
        wire_think=wire_think,
        num_ctx=num_ctx,
        prompt_budget_tokens=int(num_ctx * PROMPT_BUDGET_RATIO),
        output_budget_tokens=output_budget,
        reasoning_budget_tokens=reasoning_budget,
        retrieval_top_k=int(prof["retrieval_top_k"]),
        prompt_scaffold_tokens=scaffold_tokens,
        safety_margin_tokens=margin_tokens,
        evidence_budget_tokens=evidence_budget,
        max_context_docs=int(prof["max_context_docs"]),
        max_doc_chars=int(prof["max_doc_chars"]),
        verify_depth=str(prof["verify_depth"]),
        native_context_tokens=ctx_info["native_context_tokens"],
        serving_context_tokens=ctx_info["serving_context_tokens"],
        app_context_limit_tokens=ctx_info["app_context_limit_tokens"],
        context_clamped_by=(ctx_info["clamped_by"] if ctx_was_clamped else None),
        warnings=tuple(warnings),
    )
