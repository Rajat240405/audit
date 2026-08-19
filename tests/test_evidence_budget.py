"""Task 3 — Dynamic evidence budgeting & semantic-boundary preservation.

Locks the new evidence flow end to end:

  ExecutionPlan (reserve-based budget, initial pool)
    → evidence.segment_blocks / allocate_evidence (boundary-safe admission)
    → shared renderer / generator plan path (one assembly)
    → server helpers (pool, admitted sources)

Pins the DECLARED behavior changes:
  * no fixed final document count — Fast may admit >3, Deep >5 docs when the
    budget fits them (max_context_docs / max_doc_chars survive ONLY as
    legacy no-plan fallbacks);
  * pool widened per profile (request can still ask for more);
  * compression no-op gone — over-budget handling is now real (allocation),
    and the 413 heal is a budget-conform rebuild;
  * RetrievedResult.score leak fixed (group-max per parent).

And the INVARIANTS that must never regress:
  * never assemble a prompt over the evidence budget;
  * never cut mid-sentence — at most ONE marked sentence-boundary
    truncation, on the most relevant block;
  * headings never detach from their content; lists/tables stay atomic;
  * omissions are visible ([… N passage(s) omitted …]);
  * legacy no-plan generator behavior is byte-identical to before.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import numpy as np
import pytest

from src.generation.evidence import (
    AGGRESSIVE_HEAL_TOKENS,
    _OMISSION_TEMPLATE,
    allocate_evidence,
    assemble_budgeted_prompt,
    block_relevance,
    clean_parliament_text,
    enrich_deep_neighbors,
    estimate_tokens,
    query_keywords,
    render_user_prompt,
    segment_blocks,
)
from src.generation.generator import AnswerGenerator
from src.generation.policy import resolve_execution
from src.generation.registry import ModelFamily, model_registry
from src.generation.client import LLMResponse
from src.retrieval.result import RetrievedResult


def _plan(mode="deep", ctx=32768):
    fam = ModelFamily(
        id="t-plan", display_name="t", provider="ollama", model_name="t-model",
        context_window=ctx, thinking_capable=True,
    )
    return resolve_execution(fam, mode, "ollama")


def _res(i, answer, **meta):
    return RetrievedResult(
        doc_id=f"d{i}", question=f"Question {i}?", answer=answer,
        score=1.0 - i * 0.01, retrieval_method="rrf_fusion", metadata=meta,
    )


def _orig_lines(admission) -> set[str]:
    raw = clean_parliament_text(admission.result.answer or "")
    return {l.strip() for l in raw.split("\n") if l.strip()}


# ── 1. ExecutionPlan new fields (reserve-based; capability-derived) ────────

def test_plan_carries_pool_and_reserve_budget():
    fast = _plan("fast", 32768)
    deep = _plan("deep", 32768)
    assert (fast.retrieval_top_k, deep.retrieval_top_k) == (5, 10)
    assert fast.safety_margin_tokens == max(256, int(32768 * 0.05)) == 1638
    assert fast.prompt_scaffold_tokens == 120
    assert fast.evidence_budget_tokens == 32768 - 4096 - 120 - 1638
    assert deep.evidence_budget_tokens == 32768 - 12288 - 120 - 1638
    # legacy report value still present & unchanged
    assert fast.prompt_budget_tokens == int(32768 * 0.80)
    # legacy fallbacks unchanged
    assert (fast.max_context_docs, fast.max_doc_chars) == (3, 1000)
    assert (deep.max_context_docs, deep.max_doc_chars) == (5, 3000)
    assert fast.warnings == () and deep.warnings == ()


def test_plan_budget_clamps_when_generation_reserve_exceeds_context():
    deep = _plan("deep", 8192)  # 8192 − 12288 reserve ⇒ negative
    assert deep.evidence_budget_tokens == 0
    assert any("evidence budget clamped" in w for w in deep.warnings)
    fast = _plan("fast", 8192)   # 8192 − 4096 − 120 − 409 = 3567 → safe
    assert fast.evidence_budget_tokens == 8192 - 4096 - 120 - 409
    assert fast.warnings == ()


# ── 2. estimator / keywords ────────────────────────────────────────────────

def test_estimate_tokens_and_keywords():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100
    assert "incois" in query_keywords("What is INCOIS?")
    assert all(len(k) > 3 for k in query_keywords("a is an ok"))


# ── 3. semantic segmentation ───────────────────────────────────────────────

def test_segment_headings_bond_to_following_content():
    text = "ANNEXURE-I\nData coverage for the year 2024.\nNext paragraph here."
    blocks = segment_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].text.startswith("ANNEXURE-I\n")
    assert "Data coverage" in blocks[0].text          # heading bonded
    assert blocks[1].text == "Next paragraph here."  # heading consumed once


def test_segment_lists_stay_atomic():
    text = "(a) first item\n(b) second item\n(c) third item\nClosing paragraph."
    blocks = segment_blocks(text)
    kinds = [b.kind for b in blocks]
    assert kinds.count("list") == 1
    lst = blocks[kinds.index("list")]
    assert all(m in lst.text for m in ("(a)", "(b)", "(c)"))
    assert blocks[-1].kind == "paragraph"


def test_segment_tables_stay_atomic():
    text = ("TABLE 2: FLEET INVENTORY\n"
            "| Ship | Year | Cost  |\n"
            "| Sagar| 2021 | 12 cr |\n"
            "| Mani | 2022 | 15 cr |\n"
            "After the table.")
    blocks = segment_blocks(text)
    kinds = [b.kind for b in blocks]
    assert "table" in kinds
    tbl = blocks[kinds.index("table")]
    assert tbl.text.startswith("TABLE 2: FLEET INVENTORY")  # heading bonded
    assert "Sagar" in tbl.text and "Mani" in tbl.text        # rows together
    assert blocks[-1].kind == "paragraph"


def test_segment_oversized_line_stays_whole():
    big = "One huge line " + ("x" * 5000)
    blocks = segment_blocks("Intro line.\n" + big + "\nOutro line.")
    assert len(blocks) == 3
    assert big in blocks[1].text           # never force-split at segmentation


def test_block_relevance_counts_distinct_keywords():
    kws = ["incois", "tsunami"]
    assert block_relevance("INCOIS INCOIS INCOIS", kws) == 1
    assert block_relevance("INCOIS issues tsunami alerts", kws) == 2
    assert block_relevance("unrelated", kws) == 0


# ── 4./5. budget-driven admission ─────────────────────────────────────────

def test_example_all_five_docs_admitted_when_budget_allows():
    # The task-spec example: 2K / 1K / 3K / 2K / 1K tokens ≈ chars ×4
    sizes = [8000, 4000, 12000, 8000, 4000]  # chars ⇒ 2k/1k/3k/2k/1k tokens
    results = [_res(i, ("Content %d. " % i) * (sizes[i] // 12)) for i in range(5)]
    budget = 11000  # tokens — comfortably above Σ evidence + headers
    alloc = allocate_evidence(results, "content", budget)
    assert alloc.admitted_ids == ["d0", "d1", "d2", "d3", "d4"]
    assert all(a.whole for a in alloc.admissions)          # zero information loss
    assert alloc.used_tokens <= alloc.budget_tokens


def test_admission_is_budget_driven_not_count_driven():
    # 8 tiny docs on a big window → ALL admitted (legacy would cut at 3/5)
    results = [_res(i, f"Fact {i}: INCOIS operates buoy {i}.") for i in range(8)]
    alloc = allocate_evidence(results, "INCOIS buoys?", 50000)
    assert len(alloc.admitted_ids) == 8                    # > legacy 5-doc cap
    prompt = assemble_budgeted_prompt("INCOIS buoys?", alloc.admissions)
    assert "RETRIEVED CONTEXT (8 records)" in prompt


def test_tight_budget_drops_low_relevance_first_and_never_overflows():
    d0 = _res(0, "INCOIS answer " + "alpha " * 100)          # ~140 tok + header
    d1 = _res(1, "INCOIS beta " + "beta " * 100)
    big_answer = "INCOIS report.\n" + "\n".join(
        f"Section finding {j} mentions INCOIS." for j in range(40)
    )
    d2 = _res(2, big_answer)
    d3 = _res(3, "INCOIS gamma " + "gamma " * 100)
    d4 = _res(4, "INCOIS delta " + "delta " * 100)
    results = [d0, d1, d2, d3, d4]
    # measured: d0/d1 ≈ 165 tok each incl. header; d2 ≈ 385 incl. header.
    # budget 570 ⇒ d0,d1 whole; d2 can NOT go whole → block-select within the
    # remaining ~240; then d3/d4 necessarily dropped (lowest relevance first).
    budget = 570
    alloc = allocate_evidence(results, "INCOIS", budget)
    assert alloc.used_tokens <= alloc.budget_tokens        # hard invariant
    assert alloc.admitted_ids[:2] == ["d0", "d1"]
    assert all(a.whole for a in alloc.admissions[:2])
    d2adm = next((a for a in alloc.admissions if a.result.doc_id == "d2"), None)
    assert d2adm is not None and not d2adm.whole           # block-selected, not dropped
    assert "d3" in alloc.skipped_doc_ids and "d4" in alloc.skipped_doc_ids


def test_every_emitted_line_is_an_original_line_or_marker():
    # the anti-mid-content-cut invariant (boundary preservation proof)
    answer = "\n".join(f"Sentence number {j} about INCOIS ends here." for j in range(50))
    alloc = allocate_evidence([_res(0, answer)], "INCOIS", 300)
    assert alloc.admissions, "expected the degraded path to emit something"
    adm = alloc.admissions[0]
    originals = _orig_lines(adm)
    for line in adm.evidence_text.split("\n"):
        l = line.strip()
        if not l or "omitted" in l or l.endswith("[Truncated to fit context budget]"):
            continue
        l = l.replace(" ... [Truncated to fit context budget]", "").strip()
        assert any(l in o or o in l for o in originals), f"cut line leaked: {l!r}"


def test_block_selection_marks_internal_gaps_and_preserves_relevance():
    answer = "\n".join(
        [
            "Unrelated preamble about weather patterns.",
            "INCOIS deployed tsunami buoys across the Indian Ocean region.",
            "Filler paragraph about monsoon seasonality in general climate.",
            "INCOIS maintains the GNSS station network for seismology.",
            "Trailing unrelated epilogue about soil moisture.",
        ]
    )
    # whole answer ≈ 77 tok incl. header → 70 forces the block path: the two
    # keyword-bearing blocks fit (with an internal gap marker between them).
    # DECLARED (bridge task): the tail-omission marker now also marks the
    # dropped epilogue — markers cost budget, so the tuned budget rose 55→70;
    # the semantics under test (internal gap marked, relevance preserved) stand.
    alloc = allocate_evidence([_res(0, answer)], "INCOIS buoys GNSS", 70)
    adm = alloc.admissions[0]
    assert "tsunami buoys" in adm.evidence_text
    assert "GNSS station" in adm.evidence_text
    assert "omitted" in adm.evidence_text                  # gap marker present
    assert _OMISSION_TEMPLATE.format(n=1) in adm.evidence_text or "passage(s) omitted" in adm.evidence_text
    assert "soil moisture" not in adm.evidence_text        # epilogue still shed
    assert alloc.used_tokens <= alloc.budget_tokens


def test_huge_400k_parent_never_assembled_whole():
    huge = "INCOIS annual report.\n" + "\n".join(
        f"Paragraph {j} reports ocean activity and INCOIS work." for j in range(9000)
    )
    assert len(huge) > 400_000
    budget = 2000
    alloc = allocate_evidence([_res(0, huge)], "INCOIS", budget)
    adm = alloc.admissions[0]
    assert not adm.whole
    assert len(adm.evidence_text) < len(huge) // 10
    assert alloc.used_tokens <= alloc.budget_tokens
    prompt = assemble_budgeted_prompt("INCOIS?", alloc.admissions)
    assert estimate_tokens(prompt) < 5000                  # bounded wire


def test_degraded_tiny_budget_never_exceeds():
    answer = "INCOIS word " + "x " * 4000
    for budget in (0, 5, 119):
        alloc = allocate_evidence([_res(0, answer)], "INCOIS", budget)
        assert alloc.used_tokens <= alloc.budget_tokens
        if alloc.admissions:
            assert alloc.admissions[0].truncated           # single marked cut


def test_duplicate_evidence_admitted_once():
    same = "INCOIS operates the tsunami warning centre."
    alloc = allocate_evidence([_res(0, same), _res(1, same)], "tsunami", 5000)
    assert alloc.admitted_ids == ["d0"]
    assert "d1" in alloc.skipped_doc_ids


def test_header_cost_is_charged_per_block_selected_admission():
    # regression: the block path once forgot the per-source header cost,
    # silently overdrawing the budget by ~12 tokens per partial candidate
    filler = "INCOIS filler " + ("mid " * 300)            # one big block → never whole
    results = [_res(i, filler) for i in range(6)]
    budget = 600
    alloc = allocate_evidence(results, "INCOIS", budget)
    assert alloc.used_tokens <= alloc.budget_tokens
    # recompute independently: every admission must pay header + evidence
    recomputed = 0
    for a in alloc.admissions:
        header = f"[Source 1] (ID: {a.result.doc_id})\n\nQUESTION: {a.question_text}\nANSWER: "
        recomputed += estimate_tokens(header) + estimate_tokens(a.evidence_text)
    assert alloc.used_tokens == recomputed


def test_prose_starting_with_section_is_not_a_heading():
    # regression: "Section findings indicate…." is prose — must remain an
    # allocatable block, not be dropped as a trailing heading
    answer = "INCOIS report.\n" + "\n".join(
        f"Section finding {j} mentions INCOIS." for j in range(10)
    )
    blocks = segment_blocks(answer)
    assert len(blocks) == 11                                  # 1 + 10 paragraphs
    alloc = allocate_evidence([_res(0, answer)], "INCOIS", 200)
    adm = alloc.admissions[0]
    assert "Section finding" in adm.evidence_text             # evidence survives


# ── 6. clean_parliament_text heading tweak (declared change) ──────────────

def test_clean_keeps_structural_headings_strips_subject_titles():
    text = "RAINFALL FORECAST\nANNEXURE-II\nActual content follows.\nSECTION 4\nMore content."
    cleaned = clean_parliament_text(text)
    assert "RAINFALL FORECAST" not in cleaned              # subject boilerplate out
    assert "ANNEXURE-II" in cleaned                        # structural heading kept
    assert "SECTION 4" in cleaned
    assert "Actual content follows." in cleaned


# ── 7. neighbor-chunk pull-in (Deep) ──────────────────────────────────────

class _Chunk:
    def __init__(self, text):
        self.chunk_text = text


def test_neighbor_pull_in_bonds_heading_only():
    r = _res(0, "Matched chunk body about INCOIS.")
    r.metadata = {"chunk_ids": ["parent_L3"]}
    mapping = {"parent_L2": _Chunk("ANNEXURE-III: Station inventory")}
    n = enrich_deep_neighbors([r], mapping)
    assert n == 1
    assert r.answer.startswith("ANNEXURE-III: Station inventory\n")
    # idempotent — no double prefix
    assert enrich_deep_neighbors([r], mapping) == 0


def test_deep_sibling_window_pulls_body_siblings_and_skips_bad_ids():
    # SANCTIONED CHANGE (bridge task, scope item 7): the old heading-only
    # pull-in could never rescue torn table rows (they live in body blobs).
    # The Deep sibling window now pulls ±1 siblings regardless of shape,
    # bounded by DEEP_SIBLING_MAX. Bad ids are still rejected.
    r = _res(0, "Matched chunk body.")
    r.metadata = {"chunk_ids": ["parent_L2"]}
    blob = "Then the ministry elaborated. " * 40            # body blob sibling
    mapping = {"parent_L1": _Chunk(blob)}
    assert enrich_deep_neighbors([r], mapping) == 1
    assert r.answer.startswith(blob.rstrip())              # window strips edges
    r2 = _res(1, "body")
    r2.metadata = {"chunk_ids": ["weird-id"]}
    assert enrich_deep_neighbors([r2], mapping) == 0


# ── 8. generator plan path: shared assembly, parity, 413 heal ─────────────

class _StubLLM:
    def __init__(self, results_text="ok", raise_413_once=False):
        self.provider, self.model, self.think = "ollama", "stub", False
        self.results_text = results_text
        self.raise_413_once = raise_413_once
        self.calls: list[str] = []

    def _maybe_raise(self):
        if self.raise_413_once:
            self.raise_413_once = False
            req = httpx.Request("POST", "http://stub")
            resp = httpx.Response(413, request=req)
            raise httpx.HTTPStatusError("413", request=req, response=resp)

    def generate(self, prompt, system=None, **k):
        self.calls.append(prompt)
        self._maybe_raise()
        return LLMResponse(
            text=self.results_text, model="stub", prompt_tokens=1,
            completion_tokens=1, total_tokens=2, latency_ms=1.0,
        )

    def generate_stream(self, prompt, system=None, **k):
        self.calls.append(prompt)
        self._maybe_raise()
        yield {"type": "tokens", "text": self.results_text}
        yield {"type": "done"}


def _budgeted_generator(llm, mode="deep", ctx=32768, count=8):
    gen = AnswerGenerator(llm_client=llm)
    gen.plan = _plan(mode, ctx)
    results = [_res(i, f"Fact {i}: INCOIS runs project {i}.") for i in range(count)]
    return gen, results


def test_generate_and_stream_share_one_assembly():
    llm = _StubLLM()
    gen, results = _budgeted_generator(llm)
    nonstream = gen.generate("INCOIS projects?", results)
    assert nonstream.answer == "ok"
    events = list(gen.generate_stream("INCOIS projects?", results))
    meta = next(e for e in events if e.get("type") == "meta")
    # identical prompt text on both paths (one cached assembly)
    assert llm.calls[0] == llm.calls[1] == nonstream.prompt
    # same admitted sources on both paths
    assert meta["sources_used"] == nonstream.sources_used
    # admission followed the plan budget, not the legacy count cap
    assert nonstream.sources_used == [f"d{i}" for i in range(8)]  # > legacy 5


def test_plan_path_has_no_legacy_char_caps():
    big_answer = "INCOIS finding. " + ("detail " * 700)  # ~4.9k chars > legacy 3000
    llm = _StubLLM()
    gen = AnswerGenerator(llm_client=llm)
    gen.plan = _plan("deep", 32768)
    out = gen.generate("INCOIS?", [_res(0, big_answer)])
    # clean strips trailing whitespace per line — compare on the stripped form
    assert big_answer.strip() in out.prompt                # not cut to 3000 chars


def test_legacy_no_plan_path_is_unchanged():
    big_answer = "INCOIS finding. " + ("detail " * 700)
    llm = _StubLLM()
    gen = AnswerGenerator(llm_client=llm)                  # no plan
    gen.max_context_docs = 3
    gen.max_doc_chars = 1000
    results = [_res(i, big_answer) for i in range(5)]
    out = gen.generate("INCOIS?", results)
    assert out.sources_used == ["d0", "d1", "d2"]          # count slice intact
    assert "detail" not in out.prompt[out.prompt.index("ANSWER: ") + 8 + 1000:]


def test_413_triggers_budget_conform_heal_not_2x500():
    llm = _StubLLM(raise_413_once=True)
    gen = AnswerGenerator(llm_client=llm)
    gen.plan = _plan("deep", 32768)
    results = [_res(i, "INCOIS " + f"content {i}. " * 60) for i in range(6)]
    out = gen.generate("INCOIS?", results)
    assert out.answer == "ok"                              # healed, not graceful-fail
    assert len(llm.calls) == 2
    alloc = gen.last_allocation
    assert alloc is not None and alloc.used_tokens <= AGGRESSIVE_HEAL_TOKENS
    assert len(alloc.admissions) <= 2                      # aggressive pool
    assert estimate_tokens(llm.calls[1]) < estimate_tokens(llm.calls[0])


def test_stream_413_heal_and_think_empty_retry_coexist():
    llm = _StubLLM(raise_413_once=True)
    gen, results = _budgeted_generator(llm)
    events = list(gen.generate_stream("INCOIS projects?", results))
    kinds = [e.get("type") for e in events]
    assert "tokens" in kinds and "meta" in kinds
    meta = next(e for e in events if e.get("type") == "meta")
    assert len(meta["sources_used"]) <= 2                  # aggressive heal pool


# ── 9. RetrievedResult.score leak fix (pipeline grouping) ─────────────────

class _FakeEmbedder:
    model_name = "fake-test-embedder"
    embedding_dim = 8

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        v = rng.randn(8).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_batch(self, texts, batch_size=1, show_progress=False):
        return np.stack([self.embed(t) for t in texts])


class _PassthroughReranker:
    def rerank(self, query, candidates, k=5, doc_texts=None):
        return list(candidates)[:k]


def test_pipeline_scores_are_group_max_not_leaked_loop_var():
    from src.models.qa_record import QARecord, QARecordMetadata
    from src.retrieval.hybrid.pipeline import HybridRAGPipeline

    recs = [
        QARecord(
            question_id=f"q{i}", question_text=f"What is topic {i}?",
            answer_text=f"Answer explaining topic {i} in detail with facts.",
            metadata=QARecordMetadata(ministry="MOES", subject=f"topic {i}",
                                      document_type="parliamentary_qa"),
        )
        for i in range(4)
    ]
    pipe = HybridRAGPipeline(records=recs, embedder=_FakeEmbedder(),
                             reranker=_PassthroughReranker())
    hits, _ = pipe.retrieve("topic 1", top_k=4)
    assert len(hits) >= 2
    # every result's score must equal THE MAX fusion score of its own group —
    # the fix for the leaked-loop-variable bug (previously all equal / wrong).
    for h in hits:
        assert h.score == pytest.approx(h.rerank_score)
    assert len({h.score for h in hits}) > 1                # not a single leaked value


# ── 10. server helpers ─────────────────────────────────────────────────────

def test_server_pool_and_admitted_filters():
    from src.retrieval.frontend import server as srv

    deep = _plan("deep")
    fast = _plan("fast")
    assert srv._effective_top_k(5, deep) == 10             # profile pool floor
    assert srv._effective_top_k(3, deep) == 10
    assert srv._effective_top_k(20, deep) == 20            # explicit larger honored
    assert srv._effective_top_k(5, fast) == 5
    assert srv._effective_top_k(None, fast) == 5

    srcs = [{"doc_id": f"d{i}"} for i in range(6)]
    out = srv._filter_to_admitted(srcs, ["d1", "d3"], key=lambda s: s["doc_id"])
    assert [s["doc_id"] for s in out] == ["d1", "d3"]      # pool order kept


# ── 12. temporal awareness (R1: dated headers / R2: conflict note) ─────────

from src.generation.evidence import _source_header, _temporal_note, doc_signal_year
from src.generation.generator import build_user_prompt


def _dated(doc_id, year_stamp, body):
    return RetrievedResult(
        doc_id=doc_id, question=f"Question {doc_id}?", answer=body,
        score=0.9, retrieval_method="rrf_fusion",
        metadata={"ministry": "EARTH SCIENCES", "subject": "Weather Stations",
                  "date": year_stamp, "document_type": "parliamentary_qa"},
    )


def test_header_renders_date_when_present():
    items = [(_dated("d1", "2026-07-29", "There are 37 stations."), "Question d1?", "There are 37 stations.")]
    prompt = render_user_prompt("how many stations?", items)
    assert "Date: 2026-07-29" in prompt
    # it sits inside the [Source 1] header block, before QUESTION:
    src1 = prompt.split("[Source 1]")[1].split("QUESTION:")[0]
    assert "Date: 2026-07-29" in src1


def test_header_omits_date_line_when_absent_or_empty():
    for meta in ({}, {"date": None}, {"date": ""}, {"date": "  "}):
        items = [(_res(0, "body").__class__(
            doc_id="dX", question="Q?", answer="body", score=0.9,
            retrieval_method="rrf_fusion", metadata=meta), "Q?", "body")]
        prompt = render_user_prompt("q?", items)
        assert "\nDate:" not in prompt, f"phantom Date line for metadata={meta!r}"


def test_source_header_accounting_matches_renderer_exactly():
    # _source_header is what allocate_evidence charges against the budget;
    # it must BYTE-MATCH the renderer's header or the admission ledger lies.
    r = _dated("d1", "2026-07-29", "answer body")
    items = [(r, "Question d1?", "answer body")]
    prompt = render_user_prompt("q?", items)
    header = _source_header(r, 1, "Question d1?")
    assert header in prompt                       # exact block, incl. Date line
    # and it is the ONLY source header in this prompt
    assert prompt.count("[Source 1]") == 1


def test_temporal_note_fires_when_sources_span_years():
    results = [
        _dated("old", "2008-07-24", "There were 125 stations then."),
        _dated("new", "2011-08-10", "There were 550 stations."),
        _dated("newest", "2026-07-29", "There are 37 states covered."),
    ]
    items = [(r, "q", r.answer) for r in results]
    prompt = render_user_prompt("how many stations?", items)
    assert "NOTE:" in prompt
    note_line = next(l for l in prompt.splitlines() if l.startswith("NOTE:"))
    assert "2008" in note_line and "2026" in note_line
    # furniture position: banner first, note second, sources third
    assert prompt.index("RETRIEVED CONTEXT") < prompt.index("NOTE:") < prompt.index("[Source 1]") < prompt.index("[Source 2]")


def test_temporal_note_silent_for_same_year_consecutive_years_and_no_dates():
    same_year = [_dated("a", "2026-01-15", "x"), _dated("b", "2026-07-29", "y")]
    consecutive = [_dated("a", "2024-06-01", "x"), _dated("b", "2025-03-01", "y")]
    undated = [_res(0, "x"), _res(1, "y")]
    one_dated_only = [_dated("a", "2026-01-15", "x"), _res(1, "y")]
    for group in (same_year, consecutive, undated, one_dated_only):
        prompt = render_user_prompt("q?", [(r, "q", r.answer) for r in group])
        assert not any(l.startswith("NOTE:") for l in prompt.splitlines()), \
            f"spurious NOTE for group {[r.metadata.get('date') for r in group]}"


@pytest.mark.parametrize(("stamp", "year"), [
    ("2026-07-29", 2026),          # ISO
    ("2026", 2026),                # bare year
    ("2023-24", 2023),             # FY range stamp from annual-report ingest
    ("29.07.2026", 2026),          # dotted parliament-PDF stamp
    ("2023-2024", 2024),           # full range → coverage year (max)
    ("March 2024", 2024),          # month-year prose
    ("garbage-no-year", None),
    ("", None),
    (None, None),
])
def test_doc_signal_year_parses_common_stamps(stamp, year):
    assert doc_signal_year(stamp) == year


def test_temporal_note_function_returns_exact_advisory_wording():
    note = _temporal_note([_dated("a", "2008-01-01", "x"), _dated("b", "2026-06-01", "y")])
    assert note == (
        "NOTE: These sources are dated 2008 to 2026. Where figures conflict "
        "across them, prefer the newest dated source(s) for current values; "
        "treat older sources as historical."
    )


def test_legacy_builder_also_carries_dates_and_note():
    results = [_dated("old", "2011-08-10", "old count"), _dated("new", "2026-07-29", "new count")]
    prompt = build_user_prompt("stations?", results)
    assert "Date: 2011-08-10" in prompt and "Date: 2026-07-29" in prompt
    assert any(l.startswith("NOTE:") for l in prompt.splitlines())


def test_budgeted_and_legacy_identical_prompt_with_dated_sources():
    results = [_dated(f"d{i}", f"{2008 + i}-01-01", f"short fact {i}") for i in range(4)]
    alloc = allocate_evidence(results, "stations?", budget_tokens=10000)
    budgeted = assemble_budgeted_prompt("stations?", alloc.admissions)
    legacy = build_user_prompt("stations?", results)
    assert budgeted == legacy                        # renderer parity incl. R1/R2


def test_note_and_date_overhead_is_only_r1r2_furniture():
    # Same metadata in both groups EXCEPT dates, so the delta measures exactly
    # the R1/R2 furniture: one NOTE line + 3 short "Date:" header lines.
    def _group(use_dates):
        out = []
        for i in range(3):
            r = RetrievedResult(
                doc_id=f"d{i}", question="q", answer="body text", score=0.9,
                retrieval_method="rrf_fusion",
                metadata={"ministry": "EARTH SCIENCES", "subject": "Weather Stations",
                          "date": f"{2010 + i}-01-01" if use_dates else None,
                          "document_type": "parliamentary_qa"},
            )
            out.append((r, "q", "body text"))
        return out

    delta = estimate_tokens(render_user_prompt("q?", _group(True))) - estimate_tokens(
        render_user_prompt("q?", _group(False)))
    assert 0 < delta <= 70, f"unexpected R1/R2 furniture overhead: {delta} tok"
    # broken down: the note alone is the dominant piece (~40 tok by //4)
    note = _temporal_note([g[0] for g in _group(True)])
    assert estimate_tokens(note) <= 45


def test_generate_and_stream_parity_with_dated_sources():
    llm = _StubLLM()
    gen = AnswerGenerator(llm_client=llm)
    gen.plan = _plan("deep", 32768)
    results = [_dated(f"d{i}", f"{2008 + i * 4}-07-01", f"count is {100 + i}.") for i in range(4)]
    nonstream = gen.generate("stations?", results)
    events = list(gen.generate_stream("stations?", results))
    meta = next(e for e in events if e.get("type") == "meta")
    assert llm.calls[0] == llm.calls[1] == nonstream.prompt
    assert meta["sources_used"] == nonstream.sources_used
    assert "NOTE:" in nonstream.prompt               # deep admit-all spans 2008→2020


def test_server_source_payloads_carry_date():
    from src.retrieval.frontend import server as srv
    r = _dated("d1", "2026-07-29", "body")
    dicts = srv._to_sources([r])
    assert dicts[0]["date"] == "2026-07-29"
    item = srv.SourceItem(
        doc_id=r.doc_id, ministry=r.metadata["ministry"], subject=r.metadata["subject"],
        date=r.metadata["date"], score=0.9, question=r.question, answer=r.answer,
    )
    assert item.date == "2026-07-29"
    undated = srv.SourceItem(doc_id="d2", ministry="-", subject="-", score=0.1, question="q", answer="a")
    assert undated.date is None


# ── 11. no model-name conditionals in the new machinery ────────────────────

def test_no_model_name_conditionals_in_budgeting_layer():
    src = Path(__file__).resolve().parents[1] / "src"
    offending = re.compile(r"qwen|llama|gemma|mistral", re.IGNORECASE)
    hits = []
    for path in [src / "generation" / "evidence.py"]:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if offending.search(line):
                hits.append(f"{path.name}:{lineno}: {line.strip()}")
    assert hits == [], "model-name logic leaked into budgeting:\n" + "\n".join(hits)
