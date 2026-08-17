"""Ollama endpoint configuration + provider/model discovery and enforcement.

Covers:
1. The Ollama endpoint is env-driven (OLLAMA_BASE_URL / OLLAMA_HOST /
   fallback), and health checks, generation, and discovery all resolve the
   SAME configured endpoint — nothing assumes 11434.
2. /api/providers exposes only the deployment's enabled providers
   (APP_PROVIDERS), and /api/provider + /api/models enforce that boundary.
3. /api/models DISCOVERS models from the connected server
   (Ollama /api/tags + /api/show, vLLM /v1/models). The YAML catalog only
   supplies metadata for known ids; unknown served models get
   server-reported metadata or an explicitly flagged fallback — never
   invented values pretending to be real.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from src.generation.client import LLMClient, ollama_base_url
from src.generation.openai_url import models_url
from src.generation.registry import OllamaProvider, OpenAICompatibleProvider

FP8 = "Qwen3.6-35B-A3B-FP8"


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def api(monkeypatch):
    """TestClient against the real app, with global runtime state restored
    after each test (ACTIVE_CONFIG, dynamic families, llm_client fields)."""
    import src.retrieval.frontend.server as srv

    saved_cfg = dict(srv.ACTIVE_CONFIG)
    saved_fams = dict(srv.model_registry._families)  # noqa: SLF001
    saved_llm = {
        k: getattr(srv.llm_client, k, None)
        for k in ("provider", "model", "num_ctx", "temperature", "max_tokens", "think")
    }
    monkeypatch.delenv("APP_PROVIDERS", raising=False)
    monkeypatch.delenv("APP_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    client = TestClient(srv.app)
    yield client, srv
    srv.ACTIVE_CONFIG.clear()
    srv.ACTIVE_CONFIG.update(saved_cfg)
    srv.model_registry._families.clear()  # noqa: SLF001
    srv.model_registry._families.update(saved_fams)  # noqa: SLF001
    for k, v in saved_llm.items():
        setattr(srv.llm_client, k, v)


# ── 1. Ollama endpoint resolution ────────────────────────────────────────

def test_ollama_url_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_base_url() == "http://localhost:11434"


def test_ollama_base_url_env_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:19000")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:12000/")
    assert ollama_base_url() == "http://127.0.0.1:12000"


def test_ollama_host_normalization(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:12000")
    assert ollama_base_url() == "http://127.0.0.1:12000"  # scheme added
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:12000")
    assert ollama_base_url() == "http://127.0.0.1:12000"  # wildcard -> loopback
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.10:12000/")
    assert ollama_base_url() == "http://192.168.1.10:12000"


def test_client_and_provider_use_same_configured_endpoint(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:12000")
    c = LLMClient(provider="ollama")
    p = OllamaProvider()
    assert c.base_url == "http://127.0.0.1:12000"
    assert p.base_url == "http://127.0.0.1:12000"


class _OkTagsResp:
    status_code = 200

    def json(self):
        return {"models": [{"name": "qwen3:8b"}]}


class _CapClient:
    gets: list = []
    posts: list = []
    resp = None

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):
        type(self).gets.append(url)
        return type(self).resp or _OkTagsResp()

    def post(self, url, json=None, **kw):
        type(self).posts.append({"url": url, "json": json})
        return type(self).resp or _OkTagsResp()


def test_generation_hits_the_configured_endpoint(monkeypatch):
    """LLMClient's configured base_url must reach the actual POST — no
    silent fallback to a provider-singleton default."""
    monkeypatch.setattr(httpx, "Client", _CapClient)
    _CapClient.posts = []

    class _ChatResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "ok"}, "done_reason": "stop"}

    _CapClient.resp = _ChatResp()
    c = LLMClient(provider="ollama", model="qwen3:8b", base_url="http://127.0.0.1:12000")
    c.generate("hi")
    assert _CapClient.posts[-1]["url"] == "http://127.0.0.1:12000/api/chat"


def test_health_checks_configured_endpoint(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _CapClient)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:12000")
    _CapClient.gets = []
    _CapClient.resp = _OkTagsResp()
    assert LLMClient(provider="ollama").check_health() is True
    assert _CapClient.gets[-1] == "http://127.0.0.1:12000/api/tags"


# ── 2. /v1/models discovery plumbing (provider level) ────────────────────

def test_models_url_normalization():
    assert models_url("http://h:8001") == "http://h:8001/v1/models"
    assert models_url("http://h:8001/v1") == "http://h:8001/v1/models"
    assert models_url("http://h:8001/v1/") == "http://h:8001/v1/models"
    assert "/v1/v1/" not in models_url("http://h:8001/v1")


def test_served_models_parses_vllm_models_endpoint(monkeypatch):
    class _ModelsResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [
                {"id": FP8, "max_model_len": 32768, "object": "model"},
                {"id": "Qwen3.8-27B", "max_model_len": 65536.0},
                {"no_id": True},
                {"id": "no-len-model"},
            ]}

    _CapClient.resp = _ModelsResp()
    _CapClient.gets = []
    monkeypatch.setattr(httpx, "Client", _CapClient)
    p = OpenAICompatibleProvider(base_url="http://hpc:8001/v1")
    served = p.served_models()
    assert _CapClient.gets[-1] == "http://hpc:8001/v1/models"
    assert served == [
        {"id": FP8, "max_model_len": 32768},
        {"id": "Qwen3.8-27B", "max_model_len": 65536},
        {"id": "no-len-model", "max_model_len": None},
    ]


# ── 3. provider boundary (enabled providers) ─────────────────────────────

def test_providers_endpoint_returns_only_enabled(api):
    client, srv = api
    data = client.get("/api/providers").json()
    names = [p["name"] for p in data]
    assert names == [srv.ACTIVE_CONFIG["provider"]]  # single default provider
    assert data[0]["active"] is True


def test_providers_endpoint_multi_when_configured(api, monkeypatch):
    monkeypatch.setenv("APP_PROVIDERS", "ollama,vllm")
    client, _ = api
    names = [p["name"] for p in client.get("/api/providers").json()]
    assert names == ["ollama", "vllm"]


def test_switch_to_disabled_provider_is_rejected(api, monkeypatch):
    client, _ = api
    monkeypatch.setenv("APP_PROVIDERS", "ollama")  # vllm NOT enabled
    r = client.post("/api/provider", json={"provider": "vllm", "model": "qwen3.6_35b_a3b_fp8"})
    assert r.status_code == 403
    r = client.get("/api/models", params={"provider": "vllm"})
    assert r.status_code == 403


def test_status_reports_enabled_providers(api, monkeypatch):
    monkeypatch.setenv("APP_PROVIDERS", "vllm")
    client, _ = api
    assert client.get("/api/status").json()["enabled_providers"] == ["vllm"]


# ── 4. vLLM served-model discovery (endpoint level) ─────────────────────

def _patch_served(monkeypatch, served):
    monkeypatch.setattr(
        OpenAICompatibleProvider, "served_models",
        lambda self, base_url=None: served,
    )


def test_vllm_models_come_from_the_server_not_the_catalog(api, monkeypatch):
    monkeypatch.setenv("APP_PROVIDERS", "ollama,vllm")
    _patch_served(monkeypatch, [
        {"id": FP8, "max_model_len": 32768},          # known to catalog
        {"id": "Qwen3.8-27B", "max_model_len": 65536},  # NOT in catalog
    ])
    client, _ = api
    r = client.get("/api/models", params={"provider": "vllm"})
    assert r.status_code == 200
    entries = r.json()
    ids = [e["model_name"] for e in entries]
    # exactly the served ids — catalog-only models (27B/30B-A3B) are NOT offered
    assert ids == [FP8, "Qwen3.8-27B"]

    known = next(e for e in entries if e["model_name"] == FP8)
    assert known["metadata_source"] == "catalog"
    assert known["context_window"] == 32768
    assert known["thinking_capable"] is True
    assert known["think_mode"] == "template"

    new = next(e for e in entries if e["model_name"] == "Qwen3.8-27B")
    assert new["metadata_source"] == "server"      # real, server-reported ctx
    assert new["context_window"] == 65536
    assert new["thinking_capable"] is False        # unknown -> do not claim
    assert new["think_mode"] == "template"         # vLLM transport control


def test_vllm_unknown_served_model_without_ctx_gets_flagged_fallback(api, monkeypatch):
    monkeypatch.setenv("APP_PROVIDERS", "ollama,vllm")
    _patch_served(monkeypatch, [{"id": "SomeFuture-9B", "max_model_len": None}])
    client, _ = api
    (entry,) = client.get("/api/models", params={"provider": "vllm"}).json()
    assert entry["metadata_source"] == "fallback"  # honestly flagged, not "detected"
    assert entry["context_window"] == 8192
    assert entry["thinking_capable"] is False


def test_switch_to_newly_served_model_works(api, monkeypatch):
    monkeypatch.setenv("APP_PROVIDERS", "ollama,vllm")
    _patch_served(monkeypatch, [{"id": "Qwen3.8-27B", "max_model_len": 65536}])
    client, srv = api
    entries = client.get("/api/models", params={"provider": "vllm"}).json()
    fam_id = entries[0]["id"]
    monkeypatch.setattr(srv.llm_client, "check_health", lambda **k: False)
    r = client.post("/api/provider", json={"provider": "vllm", "model": fam_id})
    assert r.status_code == 200
    body = r.json()
    assert body["resolved_model"] == "Qwen3.8-27B"
    assert body["context_window"] == 65536
    assert srv.ACTIVE_CONFIG["provider"] == "vllm"
    assert srv.llm_client.think is False  # fast profile default


def test_vllm_models_503_when_server_unreachable(api, monkeypatch):
    monkeypatch.setenv("APP_PROVIDERS", "ollama,vllm")

    def _boom(self, base_url=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(OpenAICompatibleProvider, "served_models", _boom)
    client, _ = api
    assert client.get("/api/models", params={"provider": "vllm"}).status_code == 503


# ── 5. Ollama discovery (endpoint level) ────────────────────────────────

def _patch_ollama(monkeypatch, tags, ctx_map=None):
    monkeypatch.setattr(
        OllamaProvider, "list_tags", lambda self, base_url=None: tags
    )
    monkeypatch.setattr(
        OllamaProvider, "show_context_length",
        lambda self, model, base_url=None: (ctx_map or {}).get(model),
    )


def test_ollama_models_discovered_from_service(api, monkeypatch):
    _patch_ollama(monkeypatch, ["qwen3:8b", "qwen3:4b"], {"qwen3:4b": 40960})
    client, _ = api
    r = client.get("/api/models", params={"provider": "ollama"})
    assert r.status_code == 200
    entries = r.json()
    assert [e["model_name"] for e in entries] == ["qwen3:8b", "qwen3:4b"]

    known = next(e for e in entries if e["model_name"] == "qwen3:8b")
    assert known["id"] == "qwen3"                # catalog family
    assert known["context_window"] == 32768
    assert known["metadata_source"] == "catalog"
    assert known["thinking_capable"] is True

    new = next(e for e in entries if e["model_name"] == "qwen3:4b")
    assert new["metadata_source"] == "server"    # from /api/show, not guessed
    assert new["context_window"] == 40960


def test_ollama_models_503_when_offline(api, monkeypatch):
    def _boom(self, base_url=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(OllamaProvider, "list_tags", _boom)
    client, _ = api
    r = client.get("/api/models", params={"provider": "ollama"})
    assert r.status_code == 503


def test_catalog_is_not_treated_as_installed(api, monkeypatch):
    """Empty server -> empty model list (catalog must not fill in)."""
    api_client, srv = api

    class _Empty:
        def list_tags(self, base_url=None):
            return []

    monkeypatch.setattr(srv.provider_registry, "_providers", {**srv.provider_registry._providers, "ollama": _Empty()})  # noqa: SLF001
    r = api_client.get("/api/models", params={"provider": "ollama"})
    assert r.status_code == 200
    assert r.json() == []
