"""Outgoing OpenAI-compatible URL must never contain /v1/v1/."""

from __future__ import annotations

from src.generation.openai_url import chat_completions_url
from src.generation.registry import OpenAICompatibleProvider


class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }

    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _CapturingClient:
    last_post: str | None = None
    last_stream: str | None = None

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None):
        type(self).last_post = url
        return _FakeResp()

    def stream(self, method, url, json=None):
        type(self).last_stream = url
        return _FakeResp()


def test_chat_completions_url_normalization():
    join = OpenAICompatibleProvider.chat_completions_url
    expected = "http://localhost:11434/v1/chat/completions"
    assert join("http://localhost:11434") == expected
    assert join("http://localhost:11434/v1") == expected
    assert join("http://localhost:11434/v1/") == expected
    assert "/v1/v1/" not in join("http://localhost:11434/v1")
    assert "/v1/v1/" not in join("http://host:8001/v1/")


def test_stream_and_generate_use_configured_base(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    p = OpenAICompatibleProvider(base_url="http://example.test:9/v1")

    p.generate(model="qwen3:8b", prompt="hi", timeout_seconds=5)
    assert _CapturingClient.last_post == "http://example.test:9/v1/chat/completions"
    assert "/v1/v1/" not in _CapturingClient.last_post

    list(p.generate_stream(model="qwen3:8b", prompt="hi", timeout_seconds=5))
    assert _CapturingClient.last_stream == "http://example.test:9/v1/chat/completions"
    assert "/v1/v1/" not in _CapturingClient.last_stream


def test_client_passes_configured_base_to_both_paths(monkeypatch):
    import httpx
    from src.generation.client import LLMClient

    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    c = LLMClient(
        provider="vllm",
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
    )
    c.generate("hello")
    assert _CapturingClient.last_post == "http://localhost:11434/v1/chat/completions"

    list(c.generate_stream("hello"))
    assert _CapturingClient.last_stream == "http://localhost:11434/v1/chat/completions"
