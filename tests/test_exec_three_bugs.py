"""Regression tests for the three HPC-validation bugs (2026-08-18).

Bug 1 — Standard mode forwarded provider reasoning to the Canvas.
        Invariant: reasoning may reach the frontend/consumers ONLY when the
        resolved ExecutionPlan has thinking=True. Enforced at BOTH the
        generator boundary and the /api/chat/stream SSE boundary — the
        provider's think=false is never trusted (some Ollama/qwen3 builds
        ignore it).

Bug 2 — "Standard only shows 3 documents". Diagnostics must distinguish
        CASE A (pool=5, admitted=3 → correct Task-3 budget admission),
        CASE B (pool itself <5 upstream), CASE C (legacy max_context_docs
        cap active on a planned request = alarm). Pools are plan-driven
        floors: Standard 5, Deep 10 (explicit larger top_k honored).

Bug 3 — Deep + qwen3.5:9b had no usable evidence. Root cause: model was
        absent from the catalog → served-model discovery registered it
        dynamically (FALLBACK ctx 8192, thinking unclaimed) → Deep evidence
        budget 8192 − 12288 − 120 − 409 = −4625 → clamped to 0 → ZERO
        candidates admitted. Fix: catalog registration with the deployed
        32768 window (plugin system working as designed); Deep budget 18722.
"""

from __future__ import annotations

import json

import httpx

from src.generation.generator import AnswerGenerator
from src.generation.policy import resolve_execution
from src.generation.registry import (
    ModelFamily, load_model_catalog, model_registry,
    populate_model_registry,
)
from src.retrieval.result import RetrievedResult
from src.utils.app_paths import config_path

Q35 = "qwen3.5:9b"


def _family():
    return model_registry.get("qwen3.5_9b")


# ────────────────────────────────────────────────────────────────────────────
# BUG 3 — metadata + plan math + admission
# ────────────────────────────────────────────────────────────────────────────

def test_qwen35_9b_catalogued_in_both_catalogs():
    for catalog in ("models.yaml", "models.docker.yaml"):
        data = load_model_catalog(str(config_path(catalog)))
        hits = [f for f in data["providers"]["ollama"]["families"]
                if f["model_name"] == Q35]
        assert hits, f"{Q35} must be catalogued in {catalog}"
        hit = hits[0]
        assert int(hit["context_window"]) == 32768  # deployed PC window
        assert hit["thinking_capable"] is True


def test_qwen35_9b_plans_standard_vs_deep():
    fam = _family()
    assert fam is not None and fam.model_name == Q35
    assert fam.thinking_capable is True

    fast = resolve_execution(fam, "fast", "ollama")
    assert fast.thinking is False
    assert fast.reasoning_budget_tokens == 0
    assert fast.max_tokens == 4096
    assert fast.retrieval_top_k == 5
    assert fast.max_context_docs == 3          # legacy fallback value, unused by planned path
    assert fast.evidence_budget_tokens == 26914   # 32768 − 4096 − 120 − 1638
    assert not fast.warnings

    deep = resolve_execution(fam, "deep", "ollama")
    assert deep.thinking is True
    assert deep.reasoning_budget_tokens == 8192
    assert deep.max_tokens == 12288
    assert deep.retrieval_top_k == 10
    assert deep.max_context_docs == 5          # legacy fallback value
    assert deep.evidence_budget_tokens == 18722   # 32768 − 12288 − 120 − 1638
    assert not deep.warnings


def test_prefix_fallback_family_deep_budget_clamped_to_zero_with_warning():
    """The failing requests' exact shape (8192 fallback, thinking unclaimed):
    Deep evidence budget clamps to 0 AND the plan loudly warns. Standard on
    the same family stayed healthy (3567) — matching 'Standard works, Deep
    has no usable evidence'. This pins the clamp+warn behavior (not weakened)
    while the catalog fix keeps qwen3.5:9b off this path."""
    fam = ModelFamily(
        id="_probe_fallback", display_name=Q35, provider="ollama",
        model_name=Q35, context_window=8192, thinking_capable=False,
    )
    fast = resolve_execution(fam, "fast", "ollama")
    assert fast.evidence_budget_tokens == 3567
    deep = resolve_execution(fam, "deep", "ollama")
    assert deep.evidence_budget_tokens == 0
    assert any("clamped to 0" in w for w in deep.warnings)


def test_qwen35_9b_deep_admits_evidence_end_to_end():
    """Bug-3 success criterion: usable evidence reaches generation in Deep."""
    from src.generation import evidence as ev

    fam = _family()
    plan = resolve_execution(fam, "deep", "ollama")
    pool = [
        RetrievedResult(doc_id=f"d{i}", question=f"Q{i}?",
                        answer=f"evidence for item {i} " * 150, score=1.0 - i * 0.01,
                        retrieval_method="hybrid",
                        metadata={"ministry": "EARTH SCIENCES", "subject": "x"})
        for i in range(10)
    ]
    alloc = ev.allocate_evidence(pool, "q?", plan.evidence_budget_tokens - 500)
    assert len(alloc.admitted_ids) == 10
    assert not alloc.skipped_doc_ids


def test_served_family_entry_resolves_qwen35_from_catalog_without_api_show():
    """Post-fix: discovery returns CATALOG metadata even when /api/show gives
    nothing (the pre-fix trigger). Server import is heavy but already used by
    the suite (test_pre_docker_fixes)."""
    import src.retrieval.frontend.server as srv

    # fresh prod-catalog state
    populate_model_registry(
        model_registry, load_model_catalog(str(config_path("models.yaml"))))
    entry = srv._served_family_entry("ollama", Q35, None)
    assert entry["metadata_source"] == "catalog"
    assert entry["context_window"] == 32768
    assert entry["thinking_capable"] is True


# ────────────────────────────────────────────────────────────────────────────
# BUG 2 — pool floors + admission diagnostics (CASE A/B/C discrimination)
# ────────────────────────────────────────────────────────────────────────────

def test_effective_top_k_plan_floors():
    import src.retrieval.frontend.server as srv

    fast = resolve_execution(_family(), "fast", "ollama")
    deep = resolve_execution(_family(), "deep", "ollama")
    assert srv._effective_top_k(3, fast) == 5    # plan floor beats small request
    assert srv._effective_top_k(5, fast) == 5
    assert srv._effective_top_k(5, deep) == 10
    assert srv._effective_top_k(12, deep) == 12  # explicit larger pool honored


def _make_results(n=5):
    return [
        RetrievedResult(doc_id=f"d{i}", question=f"Q{i}?",
                        answer=f"fact about item {i} " * 100, score=1.0 - i * 0.01,
                        retrieval_method="hybrid",
                        metadata={"ministry": "EARTH SCIENCES", "subject": "x"})
        for i in range(n)
    ]


def test_admission_diag_planned_path_flags_and_counts():
    import src.retrieval.frontend.server as srv

    srv.generator.plan = resolve_execution(_family(), "fast", "ollama")
    try:
        ids, diag = srv._admission_diag("q?", _make_results(5))
        assert diag["plan_attached"] is True
        assert diag["legacy_max_context_docs_fallback"] is False
        assert diag["retrieval_top_k"] == 5
        assert diag["retrieved_count"] == 5
        assert diag["admitted"] == len(ids) == 5  # healthy budget admits pool
        assert isinstance(diag["skipped_doc_ids"], list)
        assert diag["plan_evidence_budget_tokens"] == 26914
    finally:
        srv.generator.plan = None


def test_admission_diag_caseA_budget_admits_subset():
    """Pool 5 → admitted 3 under a tight budget: correct Task-3 behavior,
    and the diag proves WHY (skipped ids + used vs budget)."""
    import src.retrieval.frontend.server as srv

    srv.generator.plan = resolve_execution(_family(), "fast", "ollama")
    try:
        # shrink via prepare_context budget path used by _admission_diag:
        _, ids, d = srv.generator.prepare_context("q?", _make_results(5),
                                                  budget_override=450)
        assert d["pool"] == 5
        assert d["admitted"] < d["pool"]
        assert d["skipped_doc_ids"]                # named, not silent
        assert d["evidence_used_tokens"] <= d["evidence_budget_tokens"]
    finally:
        srv.generator.plan = None


def test_admission_diag_caseC_legacy_fallback_flagged():
    """Plan MISSING → legacy max_context_docs cap applies AND carries the
    CASE-C alarm flag (planned server requests must never show True)."""
    import src.retrieval.frontend.server as srv

    srv.generator.plan = None
    ids, diag = srv._admission_diag("q?", _make_results(7))
    assert diag["plan_attached"] is False
    assert diag["legacy_max_context_docs_fallback"] is True
    assert diag["legacy_cap"] == srv.generator.max_context_docs
    assert len(ids) == srv.generator.max_context_docs


# ────────────────────────────────────────────────────────────────────────────
# BUG 1 — application-boundary think-gate
# ────────────────────────────────────────────────────────────────────────────

class _DisobedientClient:
    """Provider that IGNORES think:false (observed Ollama/qwen3 builds) and
    streams reasoning regardless of the request."""

    provider = "ollama"
    model = Q35
    temperature = 0.0
    max_tokens = 4096
    num_ctx = 32768

    def generate_stream(self, prompt, system=None):
        yield {"type": "reasoning", "text": "CoT despite think=false"}
        yield {"type": "answer_start"}
        yield {"type": "tokens", "text": "Answer text."}
        yield {"type": "done"}


def _gen_with_plan(mode: str) -> AnswerGenerator:
    g = AnswerGenerator(llm_client=_DisobedientClient())
    g.plan = resolve_execution(_family(), mode, "ollama")
    return g


_CTX = _make_results(2)


def test_standard_stream_gates_provider_reasoning():
    events = list(_gen_with_plan("fast").generate_stream("q?", _CTX))
    kinds = [e["type"] for e in events]
    assert "reasoning" not in kinds, "Standard must NEVER forward reasoning"
    assert "tokens" in kinds  # answer still flows


def test_deep_stream_preserves_provider_reasoning():
    events = list(_gen_with_plan("deep").generate_stream("q?", _CTX))
    texts = [e["text"] for e in events if e["type"] == "reasoning"]
    assert texts and "CoT despite think=false" in texts[0]


def test_planless_legacy_stream_unchanged():
    """No plan attached → legacy behavior preserved (gate is a no-op)."""
    g = AnswerGenerator(llm_client=_DisobedientClient())
    g.plan = None
    kinds = [e["type"] for e in g.generate_stream("q?", _CTX)]
    assert "reasoning" in kinds


# ── Bug 1 + Bug 2 traced through the real SSE endpoint ───────────────────────

class _FakeHTTPX:
    """Ollama-wire double: NDJSON chunks whose message carries `thinking`
    even when the request sets think=false (the disobedient-build shape)."""

    lines: list[str] = []
    streams: list = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):
        class _R:
            status_code = 200
        return _R()

    def stream(self, method, url, json=None, **kw):
        type(self).streams.append(json)
        lines = list(type(self).lines)

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def iter_lines(self):
                return iter(lines)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    @classmethod
    def reset(cls, lines):
        cls.lines = lines
        cls.streams = []


class _RetrievalStub:
    def retrieve(self, query, top_k=5, on_stage=None, doc_types=None,
                 orgs=None, doc_categories=None):
        if on_stage:
            on_stage("dense", {"count": 7})
            on_stage("bm25", {"count": 6})
            on_stage("rrf", {"count": 6})
            on_stage("rerank", {"count": 1})
        return [RetrievedResult(
            doc_id="17-7-2936", question="AWS stations?",
            answer="(a) There are 33 Automatic Weather Stations.",
            score=1.0, retrieval_method="hybrid",
            metadata={"ministry": "EARTH SCIENCES", "subject": "Modern AWS"},
        )], None


def _sse_events(resp_text: str) -> list[dict]:
    return [json.loads(l[5:]) for l in resp_text.splitlines() if l.startswith("data:")]


def _run_chat(monkeypatch, mode: str, thought: str):
    from fastapi.testclient import TestClient
    import src.retrieval.frontend.server as srv
    from src.generation.client import LLMClient

    client = TestClient(srv.app)
    populate_model_registry(
        model_registry, load_model_catalog(str(config_path("models.yaml"))))
    monkeypatch.setattr(srv, "knowledge_lookup", lambda q: {"found": False})
    monkeypatch.setattr(srv, "pipeline", _RetrievalStub())
    monkeypatch.setitem(srv.ACTIVE_CONFIG, "provider", "ollama")
    monkeypatch.setitem(srv.ACTIVE_CONFIG, "model_family", "qwen3.5_9b")
    monkeypatch.setattr(srv, "llm_client", LLMClient(provider="ollama", model=Q35))
    srv.generator.llm_client = srv.llm_client

    lines = [
        json.dumps({"message": {"thinking": thought}, "done": False}),
        json.dumps({"message": {"content": "33 stations."}, "done": False}),
        json.dumps({"message": {}, "done": True}),
    ]
    monkeypatch.setattr(httpx, "Client", _FakeHTTPX)
    _FakeHTTPX.reset(lines)

    resp = client.post("/api/chat/stream", json={
        "message": "How many automatic weather stations are installed in West Bengal?",
        "mode": mode, "retrieval_mode": "hybrid", "top_k": 5,
    })
    assert resp.status_code == 200
    return _sse_events(resp.text)


def test_sse_standard_suppresses_reasoning_deep_keeps(monkeypatch):
    thought = "I should list the stations carefully."
    std = _run_chat(monkeypatch, "fast", thought)
    assert not [e for e in std if e.get("type") == "reasoning"], \
        "Standard boundary let provider reasoning through"
    assert "33" in next(e["text"] for e in std if e.get("type") == "final")
    assert std[-1]["type"] == "done"
    assert not [e for e in std if e.get("type") == "error"]
    # the request DID ask the wire for think=false (mode resolution intact)
    assert _FakeHTTPX.streams[-1]["think"] is False
    assert _FakeHTTPX.streams[-1]["options"]["num_ctx"] == 32768

    deep = _run_chat(monkeypatch, "deep", thought)
    texts = [e["text"] for e in deep if e.get("type") == "reasoning"]
    assert texts and texts[0] == thought
    assert _FakeHTTPX.streams[-1]["think"] is True


def test_sse_evidence_trace_line_diagnostics(monkeypatch, capsys):
    _run_chat(monkeypatch, "fast", "x")
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("[evidence-trace]"))
    assert "mode=fast" in line
    assert "provider=ollama" in line
    assert f"model={Q35}" in line
    assert "plan=ON" in line
    assert "retrieval_top_k=5" in line
    assert "effective_top_k=5" in line
    assert "dense=7" in line and "reranked=1" in line
    assert "retrieved=1" in line and "admitted=1" in line
    assert "legacy_max_context_docs_fallback=False" in line
    assert "evidence_budget=26914" in line
