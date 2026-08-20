"""Qwen3.8-27B-FP8 catalogue preparation (Task 2).

Pins the model's representation in the Model Capability Architecture:

  Qwen3.8-27B-FP8 → capability metadata (catalog) → ExecutionPlan → vLLM wire

Everything about the model lives in config/models.yaml (+ the docker overlay
mirror); NO Python knows the name "Qwen3.8". Specs in this file were verified
against the HF model card and the official vLLM recipe (see serving notes):
dense 27B hybrid (48 Gated-DeltaNet + 16 full-attention layers) with a native
vision tower, block-FP8 128×128 e4m3, native context 262,144 (served at 32768
on the A40), thinking via chat_template_kwargs.enable_thinking — the SAME
mechanism as Qwen3.6, plus optional reasoning_effort / preserve_thinking
kwargs that Task 2 deliberately leaves unwired (server defaults apply).

Proofs required by the task spec:
  1. catalog entry loads                       — test_entry_loads_from_production_catalog
  2. capabilities normalize correctly          — test_capability_views_normalize
  3. ExecutionPlan resolves correctly          — test_plan_fast / test_plan_deep
  4. thinking configuration resolves           — test_thinking_resolution_*
  5. model name stays verbatim on the wire     — test_wire_request_*
  6. serving metadata represented              — test_serving_metadata_*
  7. existing Qwen3.6 behavior unchanged       — test_qwen36_unchanged
  8. existing Ollama behavior unchanged        — test_ollama_unchanged
  9. no Qwen3.8-specific Python conditional    — test_no_model_name_conditionals_in_src
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from src.generation.client import LLMClient
from src.generation.registry import (
    ModelRegistry,
    OpenAICompatibleProvider,
    load_model_catalog,
    model_registry,
    populate_model_registry,
    resolve_family_for_provider,
    resolve_think_mode,
)
from src.generation.policy import resolve_execution
from src.utils.app_paths import config_path

Q38 = "Qwen3.8-27B-FP8"        # served-model-name — must stay verbatim everywhere
Q36 = "Qwen3.6-35B-A3B-FP8"    # pre-existing HPC deployment (regression pin)


def _fresh(catalog="models.yaml") -> ModelRegistry:
    reg = ModelRegistry()
    populate_model_registry(reg, load_model_catalog(str(config_path(catalog))))
    return reg


# ── 1. catalog entry loads ─────────────────────────────────────────────────

def test_entry_loads_from_production_catalog():
    fam = _fresh().get("qwen3.8_27b_fp8")
    assert fam is not None
    assert fam.provider == "vllm"
    assert fam.model_name == Q38
    assert fam.display_name == "Qwen 3.8 27B FP8"
    assert fam.metadata_source == "catalog"


def test_entry_loads_from_docker_overlay_mirror():
    fam = _fresh("models.docker.yaml").get("qwen3.8_27b_fp8")
    assert fam is not None
    assert fam.provider == "vllm"
    assert fam.model_name == Q38
    assert fam.think_mode == "template"


def test_boot_selection_is_env_driven_no_code():
    """Switching the HPC deployment to Qwen3.8 = VLLM_MODEL env, nothing else."""
    reg = _fresh()
    fam = resolve_family_for_provider(reg, "vllm", "qwen3", preferred_model=Q38)
    assert fam is not None and fam.id == "qwen3.8_27b_fp8"
    # … and the existing deployment still wins with its own env value.
    fam36 = resolve_family_for_provider(reg, "vllm", "qwen3", preferred_model=Q36)
    assert fam36 is not None and fam36.id == "qwen3.6_35b_a3b_fp8"


# ── 2. capabilities normalize correctly ────────────────────────────────────

def test_capability_views_normalize():
    fam = model_registry.get("qwen3.8_27b_fp8")
    # canonical capability view
    assert fam.thinking.supported is True
    assert fam.thinking.control == "chat_template_kwargs"
    # legacy views derived consistently (dual-read, both spellings in YAML)
    assert fam.thinking_capable is True
    assert fam.think_mode == "template"
    # deployed context window (native 262,144 is documented in the entry)
    assert fam.context_window == 32768
    # no output ceiling declared → policy budgets are not clamped
    assert fam.max_output_tokens is None


# ── 3. ExecutionPlan resolves correctly ────────────────────────────────────

def test_plan_fast_matches_standard_profile():
    plan = resolve_execution(model_registry.get("qwen3.8_27b_fp8"), "fast", "vllm")
    assert plan.mode == "fast"
    assert plan.model == Q38
    assert plan.temperature == 0.0
    assert plan.thinking is False
    assert plan.think_mode == "template"
    assert plan.max_tokens == 4096
    assert plan.output_budget_tokens == 4096
    assert plan.reasoning_budget_tokens == 0
    assert plan.max_context_docs == 3
    assert plan.max_doc_chars == 1000
    assert plan.verify_depth == "light"
    assert plan.num_ctx == 32768
    assert plan.prompt_budget_tokens == int(32768 * 0.80)
    assert plan.warnings == ()


def test_plan_deep_matches_standard_profile():
    plan = resolve_execution(model_registry.get("qwen3.8_27b_fp8"), "deep", "vllm")
    assert plan.mode == "deep"
    assert plan.model == Q38
    assert plan.temperature == 0.2
    assert plan.thinking is True
    assert plan.think_mode == "template"
    assert plan.max_tokens == 12288            # 4096 answer + 8192 reasoning
    assert plan.output_budget_tokens == 4096
    assert plan.reasoning_budget_tokens == 8192
    assert plan.max_context_docs == 5
    assert plan.max_doc_chars == 3000
    assert plan.verify_depth == "full"
    assert plan.num_ctx == 32768
    assert plan.warnings == ()


# ── 4. thinking configuration resolves correctly ───────────────────────────

def test_thinking_resolution_single_point():
    assert resolve_think_mode("vllm", Q38) == "template"
    # cross-provider identity (same model id reached via another adapter)
    assert resolve_think_mode("openai_compatible", Q38) == "template"


def test_thinking_resolution_through_llm_client():
    c = LLMClient(provider="vllm", model=Q38, base_url="http://hpc:8001")
    assert c._family_think_mode() == "template"


# ── 5. model name verbatim + correct request on the wire ───────────────────

class _OResp:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }


class _OCap:
    posts: list = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, **kw):
        type(self).posts.append(json)
        return _OResp()


def test_wire_request_uses_template_mechanism_with_verbatim_name(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _OCap)
    _OCap.posts = []
    fam = model_registry.get("qwen3.8_27b_fp8")
    p = OpenAICompatibleProvider(base_url="http://hpc:8001")

    fast = resolve_execution(fam, "fast", "vllm")
    p.generate(model=Q38, prompt="q", max_tokens=fast.max_tokens,
               think=fast.thinking, think_mode=fast.think_mode)
    body = _OCap.posts[-1]
    assert body["model"] == Q38                                  # never mangled
    assert "/think" not in body["model"] and "/nothink" not in body["model"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 4096

    deep = resolve_execution(fam, "deep", "vllm")
    p.generate(model=Q38, prompt="q", max_tokens=deep.max_tokens,
               think=deep.thinking, think_mode=deep.think_mode)
    body = _OCap.posts[-1]
    assert body["model"] == Q38
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["max_tokens"] == 12288


# ── 6. serving metadata represented correctly ──────────────────────────────

def test_serving_metadata_is_catalog_data_only():
    fam = model_registry.get("qwen3.8_27b_fp8")
    assert fam.serving.reasoning_parser == "qwen3"   # same parser as Qwen3.6
    # serving limit raised 32768 -> 131072 (2026-08-19 HPC change): 4× the old
    # 32K bottleneck; 2 fp8-KV sequences fit @0.90 A40 util — 262144 (native)
    # is deliberately NOT the serving limit at max-num-seqs 2.
    assert fam.serving.max_model_len == 131072
    assert fam.serving.max_model_len <= fam.native_context_window  # <= native
    assert fam.serving.notes and "transformers" in fam.serving.notes
    assert "0.17" in fam.serving.notes               # vLLM>=0.17.0 requirement
    # serving metadata NEVER leaks into the execution plan / wire
    plan = resolve_execution(fam, "deep", "vllm")
    assert not hasattr(plan, "reasoning_parser")
    assert plan.max_tokens == 12288                  # unaffected by serving spec


def test_hpc_serving_limits_lifted_above_32k_and_bounded_by_native():
    """The arbitrary 32K serving bottleneck is gone; nothing exceeds native."""
    f38 = model_registry.get("qwen3.8_27b_fp8")
    f36 = model_registry.get("qwen3.6_35b_a3b_fp8")
    assert f38.serving.max_model_len == 131072 > 32768
    assert f36.serving.max_model_len == 65536 > 32768
    assert f38.serving.max_model_len <= f38.native_context_window
    # app-side defaults unchanged: ceilings are per-model catalogue values,
    # and overrides never require code edits (RAG_MAX_CONTEXT_TOKENS)
    assert f38.context_window == 32768 and f36.context_window == 32768


# ── 7./8. existing models unchanged ────────────────────────────────────────

def test_qwen36_unchanged():
    fam = model_registry.get("qwen3.6_35b_a3b_fp8")
    assert fam.model_name == Q36
    assert fam.think_mode == "template"
    assert fam.thinking.control == "chat_template_kwargs"
    assert fam.serving.reasoning_parser == "qwen3"
    # serving limit raised 32768 -> 65536 (2026-08-19; A40 fp16-KV headroom)
    assert fam.serving.max_model_len == 65536
    assert fam.context_window == 32768
    fast = resolve_execution(fam, "fast", "vllm")
    deep = resolve_execution(fam, "deep", "vllm")
    assert (fast.max_tokens, fast.thinking, fast.temperature) == (4096, False, 0.0)
    assert (deep.max_tokens, deep.thinking, deep.temperature) == (12288, True, 0.2)
    assert resolve_think_mode("vllm", Q36) == "template"


def test_ollama_unchanged():
    fam = model_registry.get("qwen3")
    assert fam.model_name == "qwen3:8b"
    assert fam.think_mode == "key"                   # Ollama top-level think flag
    assert fam.thinking.control == "request_flag"
    fam25 = model_registry.get("qwen2.5")
    assert fam25.thinking_capable is False
    assert resolve_think_mode("ollama", "qwen3:8b") == "key"
    deep = resolve_execution(fam, "deep", "ollama")
    assert (deep.max_tokens, deep.thinking, deep.think_mode) == (12288, True, "key")


# ── 9. no model-name-specific Python conditional was introduced ────────────

def test_no_model_name_conditionals_in_src():
    src = Path(__file__).resolve().parents[1] / "src"
    offending = re.compile(r"qwen3[._\-]?8|27b[._\-]?fp8|3\.8-27b", re.IGNORECASE)
    hits = []
    for path in src.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if offending.search(line):
                hits.append(f"{path.relative_to(src)}:{lineno}: {line.strip()}")
    assert hits == [], "Qwen3.8-specific conditional leaked into Python:\n" + "\n".join(hits)
