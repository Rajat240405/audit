"""Model Capability Architecture — capability resolution (Task 1).

Locks the refactor's contract:
  * catalog dual-read: legacy keys (think_mode/thinking_capable) and the new
    capability blocks (thinking/serving/defaults) normalize to one model;
  * resolve_think_mode: capability → legacy adapter wire string (no name ifs);
  * resolve_execution: the single policy source — reproduces the legacy
    fast/deep effective values EXACTLY for today's catalog (behavior-preserving
    refactor), with capability clamps where metadata exists;
  * get_execution_params: compatibility shim delegating to the same plan;
  * wire preservation: /think|/nothink never appended; heuristics removed.
"""

from __future__ import annotations

import httpx

from src.generation.client import LLMClient
from src.generation.registry import (
    ModelFamily,
    ModelRegistry,
    load_model_catalog,
    populate_model_registry,
    provider_transport_default_think_mode,
    resolve_think_mode,
    model_registry,
)
from src.generation.policy import ExecutionPlan, resolve_execution
from src.utils.app_paths import config_path

FP8 = "Qwen3.6-35B-A3B-FP8"


def _fresh_reg(catalog="models.yaml") -> ModelRegistry:
    reg = ModelRegistry()
    populate_model_registry(reg, load_model_catalog(str(config_path(catalog))))
    return reg


# ── 1. catalog dual-read / normalization ───────────────────────────────────

def test_legacy_only_entry_normalizes_to_capability_spec():
    reg = ModelRegistry()
    populate_model_registry(reg, {"providers": {"vllm": {"families": [
        {"id": "x", "display_name": "x", "model_name": "X", "context_window": 1,
         "thinking_capable": True, "think_mode": "template"},
    ]}}})
    fam = reg.get("x")
    assert fam.think_mode == "template"                       # legacy kept
    assert fam.thinking.control == "chat_template_kwargs"     # canonical derived
    assert fam.thinking.supported is True
    assert fam.thinking_capable is True
    assert fam.serving.reasoning_parser is None               # serving separate


def test_capability_block_wins_and_derives_legacy_fields():
    reg = ModelRegistry()
    populate_model_registry(reg, {"providers": {"vllm": {"families": [
        {"id": "x", "display_name": "x", "model_name": "X", "context_window": 1,
         "thinking": {"supported": True, "control": "chat_template_kwargs"},
         "serving": {"reasoning_parser": "qwen3", "max_model_len": 32768}},
    ]}}})
    fam = reg.get("x")
    assert fam.think_mode == "template"                # derived legacy wire string
    assert fam.thinking_capable is True                # derived legacy bool
    assert fam.serving.reasoning_parser == "qwen3"
    assert fam.serving.max_model_len == 32768


def test_production_catalog_capability_views():
    reg = _fresh_reg()
    q = reg.get("qwen3")
    assert q.thinking.control == "request_flag"        # Ollama top-level think
    assert q.think_mode == "key"
    assert q.thinking.supported is True

    f = reg.get("qwen3.6_35b_a3b_fp8")
    assert f.thinking.control == "chat_template_kwargs"
    assert f.think_mode == "template"
    assert f.serving.reasoning_parser == "qwen3"
    # serving limit raised 32768 -> 65536 (2026-08-19 HPC deployment change;
    # A40 fp16-KV headroom). The pin exists to catch accidental metadata
    # edits, not to freeze the 32K bottleneck.
    assert f.serving.max_model_len == 65536

    # legacy-only HPC entries also normalize to the canonical control
    assert reg.get("qwen3.6_27b").thinking.control == "chat_template_kwargs"


def test_docker_catalog_legacy_entries_still_normalize():
    reg = _fresh_reg("models.docker.yaml")
    host = reg.get("ollama_qwen3_8b")
    assert host.think_mode == "none"
    assert host.thinking.control == "server_default"


def test_unknown_control_spelling_falls_back_safely():
    reg = ModelRegistry()
    populate_model_registry(reg, {"providers": {"vllm": {"families": [
        {"id": "odd", "display_name": "odd", "model_name": "O",
         "context_window": 1, "thinking_capable": True, "think_mode": "bogus"},
    ]}}})
    fam = reg.get("odd")
    assert fam.thinking.control is None                # unrecognized → ignored
    assert fam.think_mode == "template"                # provider transport default


def test_provider_transport_default_is_provider_level_not_name_based():
    assert provider_transport_default_think_mode("vllm") == "template"
    assert provider_transport_default_think_mode("openai_compatible") == "template"
    assert provider_transport_default_think_mode("ollama") == "none"


# ── 2. resolve_think_mode (single resolution point; no model-name ifs) ─────

def test_resolve_think_mode_paths():
    assert resolve_think_mode("vllm", FP8) == "template"
    assert resolve_think_mode("ollama", "qwen3:8b") == "key"
    # cross-provider identity (dev-parity): docker host entry is think none
    assert resolve_think_mode("ollama", "qwen2.5:7b") == "none"
    # unknown model → nothing sent
    assert resolve_think_mode("vllm", "Mystery-9B") == "none"
    assert all("/" not in resolve_think_mode("vllm", m) for m in (FP8, "Mystery-9B"))


def test_llmclient_think_resolution_matches_capability():
    c = LLMClient(provider="vllm", model=FP8, base_url="http://hpc:8001")
    assert c._family_think_mode() == "template"
    c = LLMClient(provider="ollama", model="qwen3:8b", base_url="http://pc:12000")
    assert c._family_think_mode() == "key"


# ── 3. resolve_execution — legacy effective values reproduced exactly ──────

def test_plan_fast_matches_legacy_effective_values():
    plan = resolve_execution(model_registry.get("qwen3.6_35b_a3b_fp8"), "fast", "vllm")
    assert plan.mode == "fast"
    assert plan.temperature == 0.0
    assert plan.thinking is False
    assert plan.max_tokens == 4096
    assert plan.output_budget_tokens == 4096
    assert plan.reasoning_budget_tokens == 0
    assert plan.max_context_docs == 3
    assert plan.max_doc_chars == 1000
    assert plan.verify_depth == "light"
    assert plan.num_ctx == 32768
    assert plan.prompt_budget_tokens == int(32768 * 0.80)
    assert plan.think_mode == "template"
    assert plan.warnings == ()


def test_plan_deep_matches_legacy_effective_values():
    plan = resolve_execution(model_registry.get("qwen3.6_35b_a3b_fp8"), "deep", "vllm")
    assert plan.mode == "deep"
    assert plan.temperature == 0.2
    assert plan.thinking is True
    assert plan.max_tokens == 12288                       # derived, not hardcoded
    assert plan.reasoning_budget_tokens == 8192
    assert plan.output_budget_tokens == 4096
    assert plan.reasoning_budget_tokens + plan.output_budget_tokens == plan.max_tokens
    assert plan.max_context_docs == 5
    assert plan.max_doc_chars == 3000
    assert plan.verify_depth == "full"
    assert plan.num_ctx == 32768
    assert plan.prompt_budget_tokens == int(32768 * 0.80)
    assert plan.warnings == ()


def test_plan_ollama_family_same_numbers_correct_mechanism():
    plan = resolve_execution(model_registry.get("qwen3"), "deep", "ollama")
    assert plan.max_tokens == 12288
    assert plan.thinking is True
    assert plan.think_mode == "key"                       # Ollama top-level think


def test_plan_clamps_reasoning_to_declared_max_output_tokens():
    fam = ModelFamily(
        id="clamp", display_name="clamp", provider="vllm", model_name="C",
        context_window=32768, thinking_capable=True, think_mode="template",
        max_output_tokens=6000,
    )
    reg = model_registry
    registered_before = reg.get("clamp")
    plan = resolve_execution(fam, "deep", "vllm")
    assert plan.max_tokens == 6000
    assert plan.output_budget_tokens == 4096               # answer budget preserved
    assert plan.reasoning_budget_tokens == 6000 - 4096     # clamp absorbed here
    assert any("clamped" in w for w in plan.warnings)
    assert reg.get("clamp") is registered_before           # resolver is read-only


def test_plan_unknown_model_flagged_fallback():
    plan = resolve_execution(None, "fast", "vllm", model_name="Mystery-9B")
    assert plan.thinking is False
    assert plan.think_mode == "none"                       # never send to unknowns
    assert plan.max_tokens == 4096
    assert plan.warnings                                   # fallback is surfaced


def test_plan_unknown_mode_falls_back_to_fast():
    plan = resolve_execution(model_registry.get("qwen3"), "turbo", "ollama")
    assert plan.mode == "fast"
    assert plan.thinking is False and plan.max_tokens == 4096


# ── 4. compatibility shim ─────────────────────────────────────────────────

def test_get_execution_params_shim_delegates_to_plan():
    fam = model_registry.get("qwen3.6_35b_a3b_fp8")
    for mode, mt, think in (("fast", 4096, False), ("deep", 12288, True)):
        legacy = fam.get_execution_params(mode)
        plan = resolve_execution(fam, mode, fam.provider)
        assert legacy["max_tokens"] == mt == plan.max_tokens
        assert legacy["thinking"] is think is plan.thinking
        assert legacy["temperature"] == plan.temperature
        assert legacy["max_context_docs"] == plan.max_context_docs
        assert legacy["max_doc_chars"] == plan.max_doc_chars
        assert legacy["verify_depth"] == plan.verify_depth


# ── 5. wire preservation + heuristic removal ──────────────────────────────

class _ChatResp:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "ok"}, "done_reason": "stop"}


class _Cap:
    posts: list = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, **kw):
        type(self).posts.append(json)
        return _ChatResp()


def test_ollama_think_unset_sends_no_think_key(monkeypatch):
    """The removed 'qwen3 in name' heuristic: unspecified thinking now sends
    NOTHING (server default) instead of forcing think=True by name."""
    monkeypatch.setattr(httpx, "Client", _Cap)
    _Cap.posts = []
    c = LLMClient(provider="ollama", model="qwen3:8b", base_url="http://pc:12000")
    c.generate("q")
    body = _Cap.posts[-1]
    assert "think" not in body                    # heuristic is gone
    assert body["model"] == "qwen3:8b"            # name never mangled
    c.think = False
    c.generate("q")
    assert _Cap.posts[-1]["think"] is False       # resolved plan still sent
    c.think = True
    c.generate("q")
    assert _Cap.posts[-1]["think"] is True


def test_vllm_body_uses_capability_mechanism_verbatim_name(monkeypatch):
    from src.generation.registry import OpenAICompatibleProvider

    monkeypatch.setattr(httpx, "Client", _Cap)
    _Cap.posts = []

    class _OResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    class _OCap(_Cap):
        def post(self, url, json=None, **kw):
            type(self).posts.append(json)
            return _OResp()

    monkeypatch.setattr(httpx, "Client", _OCap)
    p = OpenAICompatibleProvider(base_url="http://hpc:8001")
    plan = resolve_execution(model_registry.get("qwen3.6_35b_a3b_fp8"), "deep", "vllm")
    p.generate(model=FP8, prompt="q", max_tokens=plan.max_tokens,
               think=plan.thinking, think_mode=plan.think_mode)
    body = _Cap.posts[-1]
    assert body["model"] == FP8                                    # untouched
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["max_tokens"] == 12288


# ── 6. plan structure ─────────────────────────────────────────────────────

def test_execution_plan_is_frozen_and_typed():
    plan = resolve_execution(model_registry.get("qwen3"), "fast", "ollama")
    assert isinstance(plan, ExecutionPlan)
    try:
        plan.max_tokens = 1  # frozen dataclass
        raise AssertionError("plan must be immutable")
    except AttributeError:
        pass
