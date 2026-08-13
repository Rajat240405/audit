"""OllamaProvider.health must accept base_url (LLMClient always supplies it)."""

from __future__ import annotations


class _OkResp:
    status_code = 200


class _CapturingClient:
    last_get: str | None = None

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        type(self).last_get = url
        return _OkResp()


def test_ollama_health_accepts_base_url_kwarg(monkeypatch):
    import httpx
    from src.generation.registry import OllamaProvider

    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    p = OllamaProvider(base_url="http://localhost:11434")
    assert p.health(base_url="http://127.0.0.1:11434") is True
    assert _CapturingClient.last_get == "http://127.0.0.1:11434/api/tags"


def test_ollama_health_default_url_unchanged(monkeypatch):
    import httpx
    from src.generation.registry import OllamaProvider

    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    p = OllamaProvider()
    assert p.health() is True
    assert _CapturingClient.last_get == "http://localhost:11434/api/tags"


def test_llmclient_check_health_ollama_with_base_url(monkeypatch):
    import httpx
    from src.generation.client import LLMClient

    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    c = LLMClient(provider="ollama", model="qwen3:8b", base_url="http://localhost:11434")
    assert c.check_health() is True
    assert _CapturingClient.last_get == "http://localhost:11434/api/tags"
