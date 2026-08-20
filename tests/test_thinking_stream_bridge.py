"""Thinking-token streaming bridge — the Deep-mode Canvas regression.

Bug (HPC validation, 2026-08-18): Deep mode resolved think=ON correctly and
the backend logged it, but NO thinking ever reached the Canvas "Model
thinking" section. Trace:

    vLLM (Qwen3ReasoningParser) ── streams thinking in ``delta.reasoning``
    OpenAICompatibleProvider      ── read ONLY ``delta.reasoning_content``
                                     ⇒ every thinking delta silently dropped
    generator / SSE / frontend    ── all intact (verified layer by layer)

The adapter now accepts all three server-side field names
(``reasoning_content`` / ``reasoning`` / ``thinking``) AND extracts inline
``<think>…</think>`` blocks from content deltas (servers launched without a
reasoning parser). Wire shapes under test:

  Shape 1  newer vLLM Qwen3ReasoningParser  -> delta.reasoning   (THE HPC case)
  Shape 2  older vLLM / deepseek_r1 parser  -> delta.reasoning_content
  Shape 3  misc builds                       -> delta.thinking
  Shape 4  no reasoning parser               -> inline <think>…</think> in content

Success criterion: thinking reaches the frontend stream as a "reasoning"
event AND the final answer stays separate.
"""

from __future__ import annotations

import json

import httpx

from src.generation.registry import (
    OpenAICompatibleProvider,
    _drain_inline_thinking,
)
from src.retrieval.result import RetrievedResult

FP8 = "Qwen3.6-35B-A3B-FP8"


# ── fake httpx wire ──────────────────────────────────────────────────────────

def _sse_lines(*deltas: dict) -> list[str]:
    """Serialize delta dicts the way a vLLM server's SSE stream delivers them."""
    lines = [f"data: {json.dumps({'choices': [{'delta': d}]})}" for d in deltas]
    lines.append("data: [DONE]")
    return lines


class _WireStream:
    status_code = 200

    def __init__(self, lines: list[str]):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeHTTPX:
    """Captures requests and replays scripted SSE lines (vllm-server double)."""

    lines: list[str] = []
    streams: list = []
    posts: list = []
    gets: list = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):  # health probes
        type(self).gets.append(url)

        class _R:
            status_code = 200

        return _R()

    def stream(self, method, url, json=None, **kw):
        type(self).streams.append({"url": url, "json": json})
        return _WireStream(type(self).lines)

    def post(self, url, json=None, **kw):  # non-streaming path (not used here)
        type(self).posts.append({"url": url, "json": json})
        return _WireStream(type(self).lines)

    @classmethod
    def reset(cls, lines: list[str]):
        cls.lines = lines
        cls.streams = []
        cls.posts = []
        cls.gets = []


def _stream(monkeypatch, lines: list[str]):
    monkeypatch.setattr(httpx, "Client", _FakeHTTPX)
    _FakeHTTPX.reset(lines)
    p = OpenAICompatibleProvider(base_url="http://hpc:8001")
    return list(p.generate_stream(model=FP8, prompt="q", think=True, think_mode="template"))


def _texts(events, kind):
    return "".join(e["text"] for e in events if e["type"] == kind)


# ── Shape 1: newer vLLM Qwen3ReasoningParser — delta.reasoning ───────────────

def test_newer_vllm_reasoning_field_surfaces_as_reasoning_events(monkeypatch):
    """THE HPC DROP: thinking under delta.reasoning was silently discarded."""
    events = _stream(monkeypatch, _sse_lines(
        {"reasoning": "The user asks for 2+2."},
        {"reasoning": " That is 4."},
        {"content": "The answer is 4."},
    ))
    kinds = [e["type"] for e in events]
    assert "reasoning" in kinds, "delta.reasoning must become reasoning events"
    assert _texts(events, "reasoning") == "The user asks for 2+2. That is 4."
    assert _texts(events, "tokens") == "The answer is 4."
    assert "think" not in _texts(events, "tokens")
    # thinking precedes the first visible token; stream terminates with done
    assert kinds.index("reasoning") < kinds.index("tokens")
    assert kinds[-1] == "done"
    assert "answer_start" in kinds


# ── Shape 3: builds that use delta.thinking ──────────────────────────────────

def test_thinking_field_fallback_surfaces(monkeypatch):
    events = _stream(monkeypatch, _sse_lines(
        {"thinking": "pondering…"},
        {"content": "42"},
    ))
    assert _texts(events, "reasoning") == "pondering…"
    assert _texts(events, "tokens") == "42"


# ── Shape 4: no reasoning parser — inline <think>…</think> in content ───────

def test_inline_think_single_delta_extracted_from_content(monkeypatch):
    events = _stream(monkeypatch, _sse_lines(
        {"content": "<think>why? because…</think>The answer."},
    ))
    assert _texts(events, "reasoning") == "why? because…"
    assert _texts(events, "tokens") == "The answer."
    assert "<think>" not in _texts(events, "tokens")
    assert "</think>" not in _texts(events, "tokens")


def test_inline_think_split_across_deltas_and_partial_tags(monkeypatch):
    """vLLM streams token-by-token: the <think> TAG ITSELF can be split."""
    events = _stream(monkeypatch, _sse_lines(
        {"content": "<th"},
        {"content": "ink>step 1"},
        {"content": " and step 2</"},
        {"content": "think>Ans"},
        {"content": "wer."},
    ))
    assert _texts(events, "reasoning") == "step 1 and step 2"
    assert _texts(events, "tokens") == "Answer."
    # no tag fragment may leak into the visible answer
    for frag in ("<th", "ink>", "</", "think>"):
        assert frag not in _texts(events, "tokens")
    kinds = [e["type"] for e in events]
    assert kinds.index("answer_start") < kinds.index("tokens")
    assert kinds.count("answer_start") == 1


def test_unclosed_think_tail_becomes_reasoning_never_answer(monkeypatch):
    """Token budget exhausted mid-thought (Deep 8192-hit): the tail was NEVER
    answer text — emit as reasoning so the visible answer stays empty and the
    server's 'reasoning but no answer' notice can fire instead."""
    events = _stream(monkeypatch, _sse_lines(
        {"content": "<think>a very long thought that never finis"},
    ))
    assert _texts(events, "reasoning") == "a very long thought that never finis"
    assert _texts(events, "tokens") == ""
    assert not any(e["type"] == "answer_start" for e in events)
    assert events[-1]["type"] == "done"


# ── Requirement 9: Standard / non-thinking behavior preserved ────────────────

def test_standard_plain_content_stream_unchanged(monkeypatch):
    """No reasoning fields and no tags: event sequence is EXACTLY the old one
    (answer_start + verbatim tokens + done), with zero reasoning events."""
    events = _stream(monkeypatch, _sse_lines({"content": "Hello "}, {"content": "world."}))
    assert [ (e["type"], e.get("text")) for e in events ] == [
        ("answer_start", None),
        ("tokens", "Hello "),
        ("tokens", "world."),
        ("done", None),
    ]


def test_drain_helper_reasoning_before_surrounding_answer_text():
    events, rem = _drain_inline_thinking("pre<think>coT</think>post")
    assert [e["type"] for e in events] == ["tokens", "reasoning", "tokens"]
    assert [e["text"] for e in events] == ["pre", "coT", "post"]
    assert rem == ""


# ── Requirement 8: provider thinking → frontend stream (full stack) ──────────

class _RetrievalStub:
    """Stands in for the lazy hybrid pipeline (no model/index load)."""

    def retrieve(self, query, top_k=5, on_stage=None, doc_types=None,
                 orgs=None, doc_categories=None):
        result = RetrievedResult(
            doc_id="17-7-2936",
            question="How many automatic weather stations are installed in West Bengal?",
            answer="(a) There are 33 Automatic Weather Stations installed.",
            score=1.0,
            retrieval_method="hybrid",
            metadata={"ministry": "EARTH SCIENCES", "subject": "Modern AWS"},
        )
        return [result], None


def test_provider_thinking_reaches_frontend_stream_as_reasoning_event(monkeypatch):
    """End to end: provider SSE (delta.reasoning) → LLMClient → generator →
    /api/chat/stream → SSE → a 'reasoning' event the Canvas renders — with the
    final answer separate and clean. This is the exact HPC stack: the module
    LLMClient is rebound to provider=vllm by _resolve_exec, then streams via
    OpenAICompatibleProvider against the faked wire."""
    from fastapi.testclient import TestClient

    import src.retrieval.frontend.server as srv
    from src.generation.client import LLMClient

    client = TestClient(srv.app)  # built BEFORE the httpx patch (ASGI subclass)

    monkeypatch.setattr(srv, "knowledge_lookup", lambda q: {"found": False})
    monkeypatch.setattr(srv, "pipeline", _RetrievalStub())
    monkeypatch.setattr(srv, "_admitted_ids",
                        lambda q, results: [r.doc_id for r in results])
    monkeypatch.setitem(srv.ACTIVE_CONFIG, "provider", "vllm")
    monkeypatch.setitem(srv.ACTIVE_CONFIG, "model_family", "qwen3.6_35b_a3b_fp8")
    # Shared module client/generator (mutated by the endpoint's _resolve_exec).
    monkeypatch.setattr(srv, "llm_client", LLMClient(provider="vllm", model=FP8))
    srv.generator.llm_client = srv.llm_client

    monkeypatch.setattr(httpx, "Client", _FakeHTTPX)
    _FakeHTTPX.reset(_sse_lines(
        {"reasoning": "The sources say 33 stations."},
        {"reasoning": " Answer with that figure."},
        {"content": "There are [Source 1] 33 automatic weather stations."},
    ))

    resp = client.post("/api/chat/stream", json={
        "message": "How many automatic weather stations are installed in West Bengal?",
        "mode": "deep",
        "retrieval_mode": "hybrid",
        "top_k": 5,
    })
    assert resp.status_code == 200

    events = [json.loads(line[5:]) for line in resp.text.splitlines()
              if line.startswith("data:")]
    by_type = {}
    for i, e in enumerate(events):
        by_type.setdefault(e.get("type"), []).append((i, e))

    # 1. thinking survived the whole pipeline as reasoning events
    reasoning = "".join(e["text"] for e in by_type.get("reasoning", []) for e in [e[1]])
    assert "The sources say 33 stations." in reasoning
    assert "Answer with that figure." in reasoning

    # 2. the request really asked the wire for thinking (Deep → template flag)
    body = _FakeHTTPX.streams[-1]["json"]
    assert body["model"] == FP8
    assert body["chat_template_kwargs"] == {"enable_thinking": True}

    # 3. final answer remains SEPARATE — no thinking, no tags anywhere visible
    final_text = next(e["text"] for _, e in by_type["final"])
    assert "33 automatic weather stations" in final_text
    assert "reasoning" not in final_text
    assert "The sources say" not in final_text
    assert "<think>" not in final_text
    streamed = "".join(e["text"] for _, e in by_type.get("tokens", []))
    assert "The sources say" not in streamed

    # 4. ordering: reasoning before phase=generating before first tokens;
    #    stream completes with done (never an error).
    first_reasoning = by_type["reasoning"][0][0]
    first_tokens = by_type["tokens"][0][0]
    assert first_reasoning < first_tokens
    assert by_type.get("error") is None
    assert events[-1]["type"] == "done"
    phases = [e.get("phase") for _, e in by_type.get("phase", [])]
    assert "thinking" in phases and "generating" in phases
