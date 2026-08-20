"""Dynamic vLLM served-model discovery + catalogue resolution.

Covers the required matrix:

 1.  /v1/models serves Qwen3.8-27B-FP8  -> existing qwen3.8_27b_fp8 catalog family
 2.  /v1/models serves Qwen3.6-35B-A3B-FP8 -> qwen3.6_35b_a3b_fp8
 3.  exact model_name matching (no substring resolution)
 4.  unknown model -> safe dynamic family (metadata_source=server/fallback,
     thinking UNKNOWN, no invented reasoning parser, no wire thinking control)
 5.  VLLM_MODEL set -> pinned override, usable without a reachable server
 6.  VLLM_MODEL absent -> discovery drives the active model
 7.  vLLM unavailable -> VLLMDiscoveryError (pin still works)
 8.  multiple served models -> deterministic documented policy
 9.  Fast/Deep policy unchanged for known Qwen models
 10. thinking control stays chat_template_kwargs for known models; the model
     NAME is never mangled; unknown models get NO thinking control on the wire
 11. TTL caching + explicit refresh (no /v1/models call per request)
 12. /api/status additive identity keys + POST /api/models/refresh endpoint

All discovery tests are network-free: ``OpenAICompatibleProvider.served_models``
is monkeypatched (same seam the existing test_provider_discovery.py uses).
"""

from __future__ import annotations

import pytest

from src.generation import vllm_discovery as vd
from src.generation.policy import resolve_execution
from src.generation.registry import (
    ModelRegistry,
    OpenAICompatibleProvider,
    model_registry,
    populate_model_registry,
)

Q38 = "Qwen3.8-27B-FP8"
Q36 = "Qwen3.6-35B-A3B-FP8"


@pytest.fixture(autouse=True)
def _clean_discovery_state(monkeypatch):
    """Every test starts with an empty discovery cache and a pristine global
    registry (dynamic families registered during a test are rolled back so
    other test files are unaffected)."""
    vd.clear_discovery_cache()
    saved = dict(model_registry._families)  # noqa: SLF001
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    yield
    vd.clear_discovery_cache()
    model_registry._families.clear()  # noqa: SLF001
    model_registry._families.update(saved)  # noqa: SLF001


def _patch_served(monkeypatch, served):
    monkeypatch.setattr(
        OpenAICompatibleProvider, "served_models",
        lambda self, base_url=None: served,
    )
    vd.clear_discovery_cache()


def _fresh_catalog_registry() -> ModelRegistry:
    reg = ModelRegistry()
    populate_model_registry(reg)
    return reg


# ── 1 & 2. known served models resolve to their catalog families ───────────

def test_served_qwen38_resolves_to_catalog_family(monkeypatch):
    _patch_served(monkeypatch, [{"id": Q38, "max_model_len": 32768}])
    active = vd.discover_active_vllm_model()
    assert active["id"] == Q38 and active["pinned"] is False
    fam, source = vd.resolve_served_family(
        active["id"], active["max_model_len"], provider="vllm",
        registry=_fresh_catalog_registry(),
    )
    assert source == "catalog"
    assert fam.id == "qwen3.8_27b_fp8"
    assert fam.model_name == Q38               # preserved metadata, verbatim id
    assert fam.context_window == 32768
    assert fam.metadata_source == "catalog"
    assert fam.thinking.supported is True
    assert fam.thinking.control == "chat_template_kwargs"
    assert fam.serving.reasoning_parser == "qwen3"


def test_served_qwen36_resolves_to_catalog_family(monkeypatch):
    _patch_served(monkeypatch, [{"id": Q36, "max_model_len": 32768}])
    active = vd.discover_active_vllm_model()
    fam, source = vd.resolve_served_family(
        active["id"], active["max_model_len"], provider="vllm",
        registry=_fresh_catalog_registry(),
    )
    assert source == "catalog"
    assert fam.id == "qwen3.6_35b_a3b_fp8"
    assert fam.thinking.supported is True
    assert fam.thinking.control == "chat_template_kwargs"


# ── 3. exact model_name matching — no substring resolution ─────────────────

def test_exact_matching_only_no_substring_resolution():
    reg = _fresh_catalog_registry()
    for served_id in ("Qwen3.8-27B", "Qwen3.6-35B-A3B-FP8-Extra", "qwen3.8-27b-fp8"):
        fam, source = vd.resolve_served_family(
            served_id, 65536, provider="vllm", registry=reg,
        )
        assert source == "server", f"{served_id} must NOT hit the catalog"
        assert fam.model_name == served_id
        assert fam.id not in ("qwen3.8_27b_fp8", "qwen3.6_35b_a3b_fp8")


# ── 4. unknown model — safe dynamic family ─────────────────────────────────

def test_unknown_model_dynamic_family_server_metadata():
    reg = _fresh_catalog_registry()
    fam, source = vd.resolve_served_family(
        "some-new-model", 131072, provider="vllm", registry=reg,
    )
    assert source == "server"
    assert fam.metadata_source == "server"
    assert fam.model_name == "some-new-model"      # verbatim — sent as-is
    assert fam.context_window == 131072            # server-reported, not invented
    assert fam.thinking.supported is None          # UNKNOWN — never claimed
    assert fam.thinking_capable is False           # legacy bool claims nothing
    assert fam.serving.reasoning_parser is None    # no invented parser
    assert fam.defaults.temperature is None        # no invented defaults


def test_unknown_model_without_ctx_gets_flagged_fallback():
    reg = _fresh_catalog_registry()
    fam, source = vd.resolve_served_family(
        "some-new-model", None, provider="vllm", registry=reg,
    )
    assert source == "fallback"
    assert fam.metadata_source == "fallback"
    assert fam.context_window == 8192              # conservative, flagged
    assert fam.thinking.supported is None


def test_unknown_model_wire_sends_no_thinking_control():
    reg = _fresh_catalog_registry()
    fam, _ = vd.resolve_served_family("some-new-model", 131072,
                                      provider="vllm", registry=reg)
    prov = OpenAICompatibleProvider(base_url="http://hpc:8001")
    for mode in ("fast", "deep"):
        plan = resolve_execution(fam, mode, "vllm")
        assert plan.wire_think is None             # consumers send NOTHING
        body = prov._payload(
            "some-new-model", [{"role": "user", "content": "x"}],
            plan.temperature, plan.max_tokens, plan.num_ctx,
            stream=False, think=plan.wire_think, think_mode=plan.think_mode,
        )
        assert "chat_template_kwargs" not in body
        assert body["model"] == "some-new-model"   # name never mangled


# ── 5. VLLM_MODEL pin is an intentional override ───────────────────────────

def test_vllm_model_pin_wins_over_discovery(monkeypatch):
    _patch_served(monkeypatch, [{"id": Q36, "max_model_len": 32768}])
    monkeypatch.setenv("VLLM_MODEL", Q38)
    active = vd.discover_active_vllm_model()
    assert active["id"] == Q38 and active["pinned"] is True


def test_vllm_model_pin_works_without_a_server(monkeypatch):
    monkeypatch.setattr(
        OpenAICompatibleProvider, "served_models",
        lambda self, base_url=None: (_ for _ in ()).throw(ConnectionError("down")),
    )
    monkeypatch.setenv("VLLM_MODEL", "qwen3:8b")   # e.g. local Ollama /v1 testing
    active = vd.discover_active_vllm_model()
    assert active["id"] == "qwen3:8b" and active["pinned"] is True


def test_vllm_model_pin_not_served_warns_but_wins(monkeypatch, capsys):
    _patch_served(monkeypatch, [{"id": Q36, "max_model_len": 32768}])
    monkeypatch.setenv("VLLM_MODEL", "my-test-model")
    active = vd.discover_active_vllm_model()
    assert active["id"] == "my-test-model"
    assert "pin wins" in capsys.readouterr().out


# ── 6. VLLM_MODEL absent — discovery drives the active model ───────────────

def test_single_served_model_is_selected(monkeypatch):
    _patch_served(monkeypatch, [{"id": Q38, "max_model_len": 32768}])
    active = vd.discover_active_vllm_model()
    assert active["id"] == Q38
    assert active["pinned"] is False
    assert active["alternatives"] == []


# ── 7. vLLM unavailable — clear failure ────────────────────────────────────

def test_unreachable_server_raises_discovery_error(monkeypatch):
    monkeypatch.setattr(
        OpenAICompatibleProvider, "served_models",
        lambda self, base_url=None: (_ for _ in ()).throw(ConnectionError("refused")),
    )
    vd.clear_discovery_cache()
    with pytest.raises(vd.VLLMDiscoveryError, match="discovery failed"):
        vd.discover_vllm_models()


def test_zero_served_models_raises_discovery_error(monkeypatch):
    _patch_served(monkeypatch, [])
    with pytest.raises(vd.VLLMDiscoveryError, match="zero served models"):
        vd.discover_vllm_models()


# ── 8. multiple models — deterministic documented policy ───────────────────

def test_multiple_models_sorted_first_with_note(monkeypatch, capsys):
    _patch_served(monkeypatch, [
        {"id": Q36, "max_model_len": 32768},
        {"id": Q38, "max_model_len": 32768},
    ])
    first = vd.discover_active_vllm_model()
    assert first["id"] == min(Q36, Q38)            # deterministic across calls
    assert set(first["alternatives"]) == {max(Q36, Q38)}
    assert "deterministically active" in capsys.readouterr().out
    vd.clear_discovery_cache()
    again = vd.discover_active_vllm_model()
    assert again["id"] == first["id"]              # stable, documented


def test_multiple_models_pin_selects(monkeypatch):
    _patch_served(monkeypatch, [
        {"id": Q36, "max_model_len": 32768},
        {"id": Q38, "max_model_len": 65536},
    ])
    monkeypatch.setenv("VLLM_MODEL", Q38)
    active = vd.discover_active_vllm_model()
    assert active["id"] == Q38
    assert active["max_model_len"] == 65536        # enriched from the server


# ── 9. Fast/Deep policy unchanged for known Qwen families ──────────────────

@pytest.mark.parametrize("model_id,fam_id", [(Q38, "qwen3.8_27b_fp8"), (Q36, "qwen3.6_35b_a3b_fp8")])
def test_fast_deep_plans_unchanged_for_catalogued_qwen(model_id, fam_id):
    reg = _fresh_catalog_registry()
    fam, source = vd.resolve_served_family(model_id, 32768, provider="vllm", registry=reg)
    assert fam.id == fam_id and source == "catalog"

    fast = resolve_execution(fam, "fast", "vllm")
    deep = resolve_execution(fam, "deep", "vllm")

    # legacy profile values, exactly (capability-derived — unchanged by discovery)
    assert (fast.temperature, fast.max_tokens, fast.thinking) == (0.0, 4096, False)
    assert (deep.temperature, deep.max_tokens, deep.thinking) == (0.2, 12288, True)
    assert fast.num_ctx == deep.num_ctx == 32768
    assert fast.think_mode == deep.think_mode == "template"
    # wire think mirrors the mode because capability is KNOWN:
    assert fast.wire_think is False and deep.wire_think is True
    assert deep.evidence_budget_tokens == 32768 - 12288 - 120 - max(256, int(32768 * 0.05))


# ── 10. thinking mechanism & verbatim model name for known models ──────────

@pytest.mark.parametrize("think", [True, False])
def test_known_qwen_wire_uses_chat_template_kwargs_verbatim_name(think):
    reg = _fresh_catalog_registry()
    fam, _ = vd.resolve_served_family(Q38, 32768, provider="vllm", registry=reg)
    plan = resolve_execution(fam, "deep" if think else "fast", "vllm")
    body = OpenAICompatibleProvider(base_url="http://hpc:8100")._payload(
        fam.model_name, [{"role": "user", "content": "x"}],
        plan.temperature, plan.max_tokens, plan.num_ctx,
        stream=False, think=plan.wire_think, think_mode=plan.think_mode,
    )
    assert body["model"] == Q38                    # never /think|/nothink
    assert body["chat_template_kwargs"] == {"enable_thinking": think}


def test_discovery_caching_and_refresh(monkeypatch):
    calls = {"n": 0}

    def _served(self, base_url=None):
        calls["n"] += 1
        return [{"id": Q38, "max_model_len": 32768}]

    monkeypatch.setattr(OpenAICompatibleProvider, "served_models", _served)
    monkeypatch.setenv("VLLM_DISCOVERY_TTL_SECONDS", "300")
    vd.clear_discovery_cache()

    vd.discover_vllm_models()
    vd.discover_vllm_models()
    vd.discover_active_vllm_model()
    assert calls["n"] == 1                          # cached — not per request

    vd.refresh_vllm_discovery()
    assert calls["n"] == 2                          # explicit refresh re-reads

    monkeypatch.setenv("VLLM_DISCOVERY_TTL_SECONDS", "0")
    vd.clear_discovery_cache()
    vd.discover_vllm_models()
    vd.discover_vllm_models()
    assert calls["n"] == 4                          # TTL 0 = always fresh


def test_provider_models_api_is_discovery_backed(monkeypatch):
    _patch_served(monkeypatch, [
        {"id": Q38, "max_model_len": 32768},
        {"id": Q36, "max_model_len": 32768},
    ])
    ids = OpenAICompatibleProvider(base_url="http://hpc:8001").models()
    assert ids == [Q38, Q36]                        # served ids, server order


def test_provider_models_api_offline_falls_back_to_pin(monkeypatch):
    monkeypatch.setattr(
        OpenAICompatibleProvider, "served_models",
        lambda self, base_url=None: (_ for _ in ()).throw(ConnectionError("down")),
    )
    vd.clear_discovery_cache()
    monkeypatch.setenv("VLLM_MODEL", Q36)
    assert OpenAICompatibleProvider(base_url="http://hpc:8001").models() == [Q36]


# ── 11/12. boot + API surface (endpoint level, via the real app) ───────────

@pytest.fixture()
def srv(monkeypatch):
    """Real FastAPI app module with runtime state restored after each test."""
    import src.retrieval.frontend.server as server

    saved_cfg = dict(server.ACTIVE_CONFIG)
    saved_llm = {k: getattr(server.llm_client, k, None)
                 for k in ("provider", "model", "num_ctx", "temperature",
                           "max_tokens", "think")}
    monkeypatch.delenv("APP_PROVIDERS", raising=False)
    yield server
    server.ACTIVE_CONFIG.clear()
    server.ACTIVE_CONFIG.update(saved_cfg)
    for k, v in saved_llm.items():
        setattr(server.llm_client, k, v)


def test_boot_resolves_discovered_model_for_vllm(monkeypatch, srv):
    _patch_served(monkeypatch, [{"id": Q38, "max_model_len": 32768}])
    fam = srv._boot_served_family("vllm")
    assert fam is not None and fam.id == "qwen3.8_27b_fp8"


def test_boot_falls_back_loudly_when_server_down(monkeypatch, srv, capsys):
    monkeypatch.setattr(
        OpenAICompatibleProvider, "served_models",
        lambda self, base_url=None: (_ for _ in ()).throw(ConnectionError("down")),
    )
    vd.clear_discovery_cache()
    fam = srv._boot_served_family("vllm")
    out = capsys.readouterr().out
    assert "discovery unavailable at boot" in out
    # legacy catalog behavior preserved: first vllm catalog family wins
    assert fam is not None and fam.provider == "vllm"


def test_boot_skipped_for_non_server_providers(srv):
    assert srv._boot_served_family("ollama") is None
    assert srv._boot_served_family("huggingface") is None


def test_refresh_hook_tracks_amodel_swap(monkeypatch, srv):
    monkeypatch.setattr(OpenAICompatibleProvider, "served_models",
                        lambda self, base_url=None: [{"id": Q36, "max_model_len": 32768}])
    vd.clear_discovery_cache()
    srv.ACTIVE_CONFIG["provider"] = "vllm"
    srv.ACTIVE_CONFIG["model"] = Q36
    srv.ACTIVE_CONFIG["model_family"] = "qwen3.6_35b_a3b_fp8"
    srv._refresh_active_served_family()
    assert srv.ACTIVE_CONFIG["model"] == Q36       # unchanged — same model

    # HPC swaps the served model -> the hook follows deterministically
    monkeypatch.setattr(OpenAICompatibleProvider, "served_models",
                        lambda self, base_url=None: [{"id": Q38, "max_model_len": 32768}])
    vd.clear_discovery_cache()
    srv._refresh_active_served_family()
    assert srv.ACTIVE_CONFIG["model"] == Q38
    assert srv.ACTIVE_CONFIG["model_family"] == "qwen3.8_27b_fp8"


def test_refresh_hook_is_a_noop_when_pinned(monkeypatch, srv):
    _patch_served(monkeypatch, [{"id": Q38, "max_model_len": 32768}])
    monkeypatch.setenv("VLLM_MODEL", Q36)
    srv.ACTIVE_CONFIG["provider"] = "vllm"
    srv.ACTIVE_CONFIG["model"] = Q36
    srv._refresh_active_served_family()
    assert srv.ACTIVE_CONFIG["model"] == Q36       # pin frozen


def test_status_reports_discovery_identity(monkeypatch, srv):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("APP_PROVIDERS", "ollama,vllm")
    srv.ACTIVE_CONFIG["provider"] = "vllm"
    srv.ACTIVE_CONFIG["model_family"] = "qwen3.6_35b_a3b_fp8"
    srv.ACTIVE_CONFIG["model"] = Q36
    data = TestClient(srv.app).get("/api/status").json()
    # existing keys untouched
    for k in ("provider", "model_family", "model", "mode", "retrieval_mode", "gpu"):
        assert k in data
    # additive discovery identity
    assert data["served_model"] == Q36
    assert data["model_display_name"] == "Qwen 3.6 35B-A3B FP8"
    assert data["model_metadata_source"] == "catalog"
    assert data["thinking_supported"] is True
    assert "provider_base_url" in data
    assert "@" not in data["provider_base_url"]      # sanitized


def test_status_reports_context_identity(monkeypatch, srv):
    """Additive context keys: native / serving / app ceiling / effective, all
    resolved by the same policy helper the budget calculator uses."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(OpenAICompatibleProvider, "served_models",
                        lambda self, base_url=None: [{"id": Q36, "max_model_len": 131072}])
    vd.clear_discovery_cache()
    monkeypatch.setenv("APP_PROVIDERS", "vllm")
    srv.ACTIVE_CONFIG["provider"] = "vllm"
    srv.ACTIVE_CONFIG["model_family"] = "qwen3.8_27b_fp8"
    srv.ACTIVE_CONFIG["model"] = Q38
    data = TestClient(srv.app).get("/api/status").json()
    assert data["native_context_tokens"] == 262144        # capability separate
    assert data["serving_context_tokens"] == 131072       # server-reported
    assert data["app_context_limit_tokens"] == 32768      # catalogue default ceiling
    assert data["effective_context_tokens"] == 32768      # min() — no hidden lift


def test_status_marks_unknown_capability(monkeypatch, srv):
    from fastapi.testclient import TestClient

    fam, _ = vd.resolve_served_family("some-new-model", 131072, provider="vllm")
    srv.ACTIVE_CONFIG["provider"] = "vllm"
    srv.ACTIVE_CONFIG["model_family"] = fam.id
    srv.ACTIVE_CONFIG["model"] = fam.model_name
    data = TestClient(srv.app).get("/api/status").json()
    assert data["model_metadata_source"] == "server"
    assert data["thinking_supported"] is None        # exposed as UNKNOWN


def test_refresh_endpoint_forces_rediscovery(monkeypatch, srv):
    from fastapi.testclient import TestClient

    calls = {"n": 0}

    def _served(self, base_url=None):
        calls["n"] += 1
        return [{"id": Q38, "max_model_len": 32768}]

    monkeypatch.setattr(OpenAICompatibleProvider, "served_models", _served)
    monkeypatch.setenv("APP_PROVIDERS", "vllm")
    vd.clear_discovery_cache()
    srv.ACTIVE_CONFIG["provider"] = "vllm"
    r = TestClient(srv.app).post("/api/models/refresh")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["active_model"] == Q38
    assert calls["n"] == 1


def test_refresh_endpoint_503_when_server_down(monkeypatch, srv):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        OpenAICompatibleProvider, "served_models",
        lambda self, base_url=None: (_ for _ in ()).throw(ConnectionError("down")),
    )
    monkeypatch.setenv("APP_PROVIDERS", "vllm")
    vd.clear_discovery_cache()
    srv.ACTIVE_CONFIG["provider"] = "vllm"
    r = TestClient(srv.app).post("/api/models/refresh")
    assert r.status_code == 503


def test_refresh_endpoint_skipped_for_ollama(monkeypatch, srv):
    from fastapi.testclient import TestClient

    srv.ACTIVE_CONFIG["provider"] = "ollama"
    r = TestClient(srv.app).post("/api/models/refresh")
    assert r.json()["status"] == "skipped"
