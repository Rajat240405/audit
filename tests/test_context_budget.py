"""Dynamic context handling — effective runtime context and budget coupling.

The budget calculator's context input is dynamic:

    effective_context = min( MODEL NATIVE CONTEXT
                           ∩ vLLM SERVING LIMIT (--max-model-len, if reported)
                           ∩ APPLICATION SAFETY LIMIT (RAG_MAX_CONTEXT_TOKENS) )

with only pre-existing fallbacks when an input is unknown (never fabricated):
native → catalogue context_window; serving → not clamped; ceiling → catalogue
context_window. The deep principle: larger context is AVAILABLE CAPACITY for
relevant verified evidence — Deep never means "fill the context".

Covers the required matrix (1–13): native context represented separately from
the 32768 deployment window; effective-context min() over the three inputs;
configurability of the app ceiling with zero code change; serving-limit
increases never silently lift the ceiling; budget allocation (system/evidence/
output/reasoning) intact; Fast stays efficient; Deep can use more capacity;
no hidden universal 32768 in the budget path.
"""

from __future__ import annotations

import pytest

from src.generation.policy import (
    PROMPT_SCAFFOLD_TOKENS,
    SAFETY_MARGIN_MIN_TOKENS,
    app_context_ceiling_from_env,
    describe_context,
    resolve_execution,
)
from src.generation.registry import ModelRegistry, populate_model_registry

Q38 = "Qwen3.8-27B-FP8"
Q36 = "Qwen3.6-35B-A3B-FP8"
PROMPT_RATIO = 0.80


@pytest.fixture(autouse=True)
def _no_env_ceiling(monkeypatch):
    monkeypatch.delenv("RAG_MAX_CONTEXT_TOKENS", raising=False)


@pytest.fixture()
def reg() -> ModelRegistry:
    r = ModelRegistry()
    populate_model_registry(r)
    return r


def _margin(ctx: int) -> int:
    return max(SAFETY_MARGIN_MIN_TOKENS, int(ctx * 0.05))


# ── 1. Native context is metadata, NOT collapsed onto 32768 ─────────────────

def test_qwen38_native_context_is_preserved_separately(reg):
    fam = reg.get("qwen3.8_27b_fp8")
    assert fam.native_context_window == 262144     # capability (HF card)
    assert fam.context_window == 32768             # deployed/app-blessed window
    assert fam.native_context_window != 32768      # NOT squashed by deployment


def test_qwen36_native_capped_at_repo_proven_window(reg):
    fam = reg.get("qwen3.6_35b_a3b_fp8")
    assert fam is not None and fam.model_name == Q36
    # The model card is not independently consulted; 131072 is the window the
    # sibling qwen3.6_30b_a3b entry is deployed at — repo-grounded, and the
    # comment in models.yaml says exactly that. Nothing higher is claimed.
    assert fam.native_context_window == 131072
    assert native_between_serving_and_ceiling_ok(fam)


def test_model_with_genuinely_unknown_native_reports_none(reg):
    fam = reg.get("qwen2.5")
    assert fam.native_context_window is None       # never fabricated


# ── 2/3/5. Effective context = min(capability, serving?, ceiling) ───────────

def test_effective_context_min_of_three_inputs(reg):
    fam = reg.get("qwen3.8_27b_fp8")
    # ceiling = catalogue default (32768), serving smaller -> serving clamps
    p = resolve_execution(fam, "deep", "vllm", serving_limit=8192)
    assert p.num_ctx == 8192 and p.context_clamped_by == "serving"


def test_task_example_native_262144_serving_131072_ceiling_65536(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "65536")
    fam = reg.get("qwen3.8_27b_fp8")
    p = resolve_execution(fam, "deep", "vllm", serving_limit=131072)
    assert p.num_ctx == 65536
    assert p.context_clamped_by == "ceiling"
    assert p.native_context_tokens == 262144
    assert p.serving_context_tokens == 131072
    assert p.app_context_limit_tokens == 65536


# ── 4/6. Unknown inputs never fabricated ─────────────────────────────────────

def test_unknown_native_falls_back_to_catalogue(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "65536")
    fam = reg.get("qwen2.5")                       # native undocumented (8192 ctx)
    p = resolve_execution(fam, "deep", "vllm", serving_limit=131072)
    # capability input = catalogue 8192 -> clamps; nothing invented upward
    assert p.num_ctx == 8192
    assert p.native_context_tokens is None


def native_between_serving_and_ceiling_ok(fam) -> bool:
    """Guard helper for catalogue sanity: native >= its deployed window."""
    return fam.native_context_window >= fam.context_window


def test_unknown_serving_does_not_clamp(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "65536")
    fam = reg.get("qwen3.8_27b_fp8")
    p = resolve_execution(fam, "deep", "vllm")     # no serving info at all
    assert p.serving_context_tokens is None
    assert p.num_ctx == 65536                      # min(native, ceiling) only


def test_unknown_model_keeps_flagged_fallback(reg):
    from src.generation.vllm_discovery import resolve_served_family

    fam, source = resolve_served_family("some-new-model", None,
                                        provider="vllm", registry=reg)
    assert source == "fallback"
    p = resolve_execution(fam, "deep", "vllm")
    assert p.num_ctx == 8192                       # conservative, flagged
    assert p.native_context_tokens is None


# ── 7/8/9. App ceiling: configurable, independent, code-free to raise ────────

def test_ceiling_configurable_from_env(monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "100000")
    assert app_context_ceiling_from_env() == 100000
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "bogus")
    assert app_context_ceiling_from_env() is None  # invalid -> safe fallback
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "-5")
    assert app_context_ceiling_from_env() is None


def test_bigger_serving_limit_does_not_lift_app_ceiling(reg):
    fam = reg.get("qwen3.8_27b_fp8")
    small = resolve_execution(fam, "deep", "vllm", serving_limit=131072)
    bigger = resolve_execution(fam, "deep", "vllm", serving_limit=262144)
    assert small.num_ctx == bigger.num_ctx == 32768  # ceiling is catalogue value
    # serving limit IS captured for transparency, it just doesn't clamp:
    assert bigger.serving_context_tokens == 262144


def test_raising_ceiling_unlocks_budget_without_code_changes(reg, monkeypatch):
    fam = reg.get("qwen3.8_27b_fp8")
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "65536")
    p64k = resolve_execution(fam, "deep", "vllm", serving_limit=262144)
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "131072")
    p128k = resolve_execution(fam, "deep", "vllm", serving_limit=262144)
    assert (p64k.num_ctx, p128k.num_ctx) == (65536, 131072)
    assert p128k.evidence_budget_tokens > p64k.evidence_budget_tokens


# ── 10. Budget allocation formula intact (all splits derive from effective) ──

def test_budget_allocation_intact_at_current_deployment(reg):
    fam = reg.get("qwen3.6_35b_a3b_fp8")
    p = resolve_execution(fam, "deep", "vllm", serving_limit=32768)
    eff = 32768
    assert p.num_ctx == eff
    assert p.max_tokens == 12288                        # output 4096 + reasoning 8192
    assert p.output_budget_tokens == 4096
    assert p.reasoning_budget_tokens == 8192
    assert p.prompt_scaffold_tokens == PROMPT_SCAFFOLD_TOKENS
    assert p.safety_margin_tokens == _margin(eff)
    assert p.evidence_budget_tokens == eff - 12288 - 120 - _margin(eff)
    assert p.prompt_budget_tokens == int(eff * PROMPT_RATIO)


def test_budget_scales_with_effective_context(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "65536")
    fam = reg.get("qwen3.8_27b_fp8")
    p = resolve_execution(fam, "deep", "vllm", serving_limit=131072)
    eff = 65536
    assert p.evidence_budget_tokens == eff - 12288 - 120 - _margin(eff)
    assert p.safety_margin_tokens == _margin(eff)


# ── 11/12. Fast stays efficient; Deep = more capacity, NOT fill-the-context ──

def test_fast_profile_unchanged_at_any_ceiling(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "131072")
    fam = reg.get("qwen3.8_27b_fp8")
    fast = resolve_execution(fam, "fast", "vllm", serving_limit=262144)
    assert (fast.temperature, fast.max_tokens, fast.thinking) == (0.0, 4096, False)
    assert fast.retrieval_top_k == 5                    # small candidate pool
    assert fast.num_ctx == 131072                       # capacity available
    assert fast.evidence_budget_tokens == 131072 - 4096 - 120 - _margin(131072)


def test_deep_uses_capacity_not_full_context(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "131072")
    fam = reg.get("qwen3.8_27b_fp8")
    deep = resolve_execution(fam, "deep", "vllm", serving_limit=262144)
    assert deep.max_tokens == 12288            # generation budget is NOT eff-sized
    assert deep.retrieval_top_k == 10          # initial pool unchanged
    assert deep.evidence_budget_tokens > 0     # bigger ARENA for relevant evidence
    assert deep.num_ctx == 131072


# ── 13. No hidden universal 32768 in the budget path ─────────────────────────

def test_no_hidden_32k_clamp(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "100000")
    fam = reg.get("qwen3.8_27b_fp8")
    p = resolve_execution(fam, "deep", "vllm", serving_limit=262144)
    assert p.num_ctx == 100000                 # would fail under any hidden 32K cap
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "40000")
    p2 = resolve_execution(fam, "deep", "vllm", serving_limit=262144)
    assert p2.num_ctx == 40000


def test_policy_module_has_no_32768_literal():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "generation" / "policy.py").read_text()
    assert "32768" not in src and "32_768" not in src


# ── describe_context helper (shared with /api/status) ────────────────────────

def test_describe_context_contract():
    info = describe_context(catalog_context=32768, native_context=262144,
                            serving_limit=131072, env_ceiling=65536)
    assert info == {
        "native_context_tokens": 262144,
        "serving_context_tokens": 131072,
        "app_context_limit_tokens": 65536,
        "effective_context_tokens": 65536,
        "clamped_by": "ceiling",
    }
    info2 = describe_context(catalog_context=8192, native_context=None,
                             serving_limit=None, env_ceiling=None)
    assert info2["effective_context_tokens"] == 8192
    assert info2["native_context_tokens"] is None
    assert info2["serving_context_tokens"] is None


# ── 12b. Clamp warnings are surfaced, never silent ────────────────────────────

def test_clamps_surface_warnings(reg, monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "65536")
    fam = reg.get("qwen2.5")                      # catalogue 8192 is effective
    p = resolve_execution(fam, "deep", "vllm", serving_limit=131072)
    assert p.num_ctx == 8192
    # effective == catalogue value -> no spurious note (today's deployments
    # resolve identically and silently):
    assert p.context_clamped_by is None
    assert not any("context resolved" in w for w in p.warnings)
    # a REAL change vs the catalogue default MUST be surfaced loudly:
    p2 = resolve_execution(reg.get("qwen3.6_35b_a3b_fp8"), "deep", "vllm",
                           serving_limit=16384)
    assert any("context resolved" in w for w in p2.warnings)
    assert p2.context_clamped_by == "serving"
    assert p2.evidence_budget_tokens == 16384 - 12288 - 120 - _margin(16384)
    # ...including EXPANSION beyond the catalogue window (env ceiling raised):
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "16384")
    p4 = resolve_execution(reg.get("qwen3.6_35b_a3b_fp8"), "deep", "vllm",
                           serving_limit=131072)
    assert p4.num_ctx == 16384 and p4.context_clamped_by == "ceiling"
    assert p4.num_ctx != 32768                     # no hidden 32K veto


# ── 14/15. Non-server providers see no serving clamp (Ollama unchanged) ───────

def test_ollama_path_unchanged(reg, monkeypatch):
    monkeypatch.delenv("RAG_MAX_CONTEXT_TOKENS", raising=False)
    fam = reg.get("qwen3")
    p = resolve_execution(fam, "deep", "ollama")
    assert p.num_ctx == 32768                    # catalogue capability as before
    assert p.serving_context_tokens is None
    assert p.evidence_budget_tokens == 32768 - 12288 - 120 - _margin(32768)
