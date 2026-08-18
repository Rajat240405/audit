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
    # resolution transparency — surfaced, never hidden
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def resolve_execution(
    family,
    mode: str,
    provider: str = "ollama",
    model_name: str = "",
) -> ExecutionPlan:
    """Resolve (ModelFamily capabilities, mode, provider) → ExecutionPlan.

    `family` is a registry ModelFamily (or None for an unregistered model —
    a flagged conservative fallback family is synthesized). No model-name
    conditionals anywhere: every parameter comes from the profile policy row
    clamped/derived against family capabilities.
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

    return ExecutionPlan(
        mode=mode_norm,
        provider=provider,
        model=model_name,
        temperature=float(prof["temperature"]),
        max_tokens=max_tokens,
        top_p=getattr(getattr(fam, "defaults", None), "top_p", None),
        thinking=bool(prof["thinking"]),
        think_mode=resolve_think_mode(provider, model_name) if model_name else "none",
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
        warnings=tuple(warnings),
    )
