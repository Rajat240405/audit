"""Matched-chunk transparency and retrieval score attribution.

Covers the requirements in the matched-chunk task:
  * a child-only match keeps the parent id AND the matched chunk id/text
  * matched chunk text and assembled parent evidence are distinct fields
  * each matched unit keeps its OWN dense/BM25/RRF scores (no parent copy)
  * a summary score is labelled "parent" or "best_chunk", never presented as
    a parent score
  * a genuinely missing score stays None
  * multiple matched chunks are all reported (display cap, real total kept)
  * the serializer used by streaming and non-streaming responses is one path
  * the generator-facing evidence (``answer``) is unchanged by this layer

Pipeline tests reuse the 17-7-2936 fixture from ``test_evidence_bridge``.
They do not assume faiss is absent.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

import tests.conftest  # noqa: F401  (stubs torch/sentence-transformers)

from src.models.qa_record import QARecord, QARecordMetadata
from src.retrieval import attribution as attr
from src.retrieval.attribution import (
    MAX_MATCHED_CHUNKS,
    SCORE_FROM_BEST_CHUNK,
    SCORE_FROM_PARENT,
    UnitInfo,
    build_matched_chunks,
    first_scores,
    summarize_channel,
)
from src.retrieval.evidence_merge import merge_evidence
from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.retrieval.result import MatchedChunk, RetrievedResult
from tests.test_evidence_bridge import (
    QUERY,
    _KeywordFitReranker,
    _LexicalEmbedder,
    _aws_record,
)

PID = "17-7-2936"


class _StableEmbedder(_LexicalEmbedder):
    """Same lexical embedding, but with a stable token hash.

    The shared fixture uses Python's ``hash()``, which changes with
    PYTHONHASHSEED. That changes top-k between processes, so these tests use
    a stable hash to stay deterministic.
    """

    def _v(self, text):
        v = np.zeros(self.embedding_dim, dtype=np.float32)
        for w in text.lower().split():
            v[zlib.crc32(w.encode("utf-8")) % self.embedding_dim] += 1.0
        return v / (np.linalg.norm(v) + 1e-9)


# ── pure attribution rules (no pipeline) ─────────────────────────────────────

def test_first_scores_keeps_first_occurrence_in_rank_order():
    assert first_scores([("a", 3.0), ("b", 2.0), ("a", 9.0)]) == {"a": 3.0, "b": 2.0}


def test_summary_uses_parent_row_score_when_parent_is_in_channel():
    score, basis = summarize_channel(PID, [f"{PID}_L1", PID], {PID: 0.7, f"{PID}_L1": 0.9})
    assert (score, basis) == (0.7, SCORE_FROM_PARENT)


def test_summary_uses_best_chunk_when_parent_absent_and_labels_it():
    channel = {f"{PID}_L1": 0.4, f"{PID}_L2": 0.8, "other": 0.99}
    score, basis = summarize_channel(PID, [f"{PID}_L1", f"{PID}_L2"], channel)
    assert (score, basis) == (0.8, SCORE_FROM_BEST_CHUNK)


def test_summary_is_none_when_channel_returned_no_matched_unit():
    assert summarize_channel(PID, [f"{PID}_L1"], {"other": 1.0}) == (None, None)


def _lookup_from(texts: dict[str, str]):
    def lookup(unit_id):
        if unit_id not in texts:
            return None
        return UnitInfo(text=texts[unit_id], unit_type="annexure", page=None)
    return lookup


def test_matched_chunks_reranked_first_parent_row_last_and_per_unit_scores():
    pool = [(f"{PID}_L3", 0.03), (PID, 0.02), (f"{PID}_L1", 0.01)]
    dense = {f"{PID}_L3": 0.5, PID: 0.6, f"{PID}_L1": 0.2}
    bm25 = {f"{PID}_L3": 4.0, f"{PID}_L1": 1.0}
    rrf = {f"{PID}_L3": 0.03, PID: 0.02, f"{PID}_L1": 0.01}
    matched, total = build_matched_chunks(
        PID, pool, dense=dense, bm25=bm25, rrf=rrf,
        representative=f"{PID}_L3", rerank_score=8.0,
        lookup=_lookup_from({f"{PID}_L3": "chunk three", f"{PID}_L1": "chunk one"}),
    )
    assert total == 3
    assert [m.chunk_id for m in matched] == [f"{PID}_L3", f"{PID}_L1", PID]
    first, second, parent_row = matched
    assert first.reranked and first.rerank_score == 8.0 and first.text == "chunk three"
    assert (first.dense_score, first.bm25_score, first.rrf_score) == (0.5, 4.0, 0.03)
    assert second.reranked is False and second.rerank_score is None
    assert (second.dense_score, second.bm25_score) == (0.2, 1.0)
    assert parent_row.text is None and parent_row.unit_type == "parent"
    assert parent_row.dense_score == 0.6  # the parent row's own score, not a chunk's


def test_unknown_unit_is_skipped_never_invented():
    matched, total = build_matched_chunks(
        PID, [(f"{PID}_L9", 0.1), (f"{PID}_L1", 0.2)],
        dense={}, bm25={}, rrf={}, representative=None, rerank_score=None,
        lookup=_lookup_from({f"{PID}_L1": "ok"}),
    )
    assert [m.chunk_id for m in matched] == [f"{PID}_L1"]
    assert total == 1


def test_display_cap_keeps_true_total():
    units = [(f"{PID}_L{i}", 1.0 / (i + 1)) for i in range(MAX_MATCHED_CHUNKS + 5)]
    texts = {u: f"text {u}" for u, _ in units}
    matched, total = build_matched_chunks(
        PID, units, dense={}, bm25={}, rrf={}, representative=None, rerank_score=None,
        lookup=_lookup_from(texts),
    )
    assert len(matched) == MAX_MATCHED_CHUNKS
    assert total == MAX_MATCHED_CHUNKS + 5


def test_rerank_score_only_on_representative():
    matched, _ = build_matched_chunks(
        PID, [(f"{PID}_L1", 0.2), (f"{PID}_L2", 0.1)],
        dense={}, bm25={}, rrf={}, representative=f"{PID}_L2", rerank_score=3.0,
        lookup=_lookup_from({f"{PID}_L1": "a", f"{PID}_L2": "b"}),
    )
    by_id = {m.chunk_id: m for m in matched}
    assert by_id[f"{PID}_L2"].rerank_score == 3.0
    assert by_id[f"{PID}_L1"].rerank_score is None


# ── real pipeline (17-7-2936 fixture) ─────────────────────────────────────────

def _build(use_reranker: bool = True) -> HybridRAGPipeline:
    fillers = [
        QARecord(question_id=f"f{i}", question_text=f"Filler question {i} about ocean observation?",
                 answer_text=f"Filler answer {i} about ocean buoys and tsunami warning.",
                 metadata=QARecordMetadata(ministry="EARTH SCIENCES", subject=f"f{i}",
                                           document_type="parliamentary_qa"))
        for i in range(4)
    ]
    return HybridRAGPipeline(
        records=[_aws_record()] + fillers,
        embedder=_StableEmbedder(),
        reranker=_KeywordFitReranker() if use_reranker else None,
        use_reranker=use_reranker,
    )


def _find(results, doc_id):
    return next(r for r in results if r.doc_id == doc_id)


def _expected_dense(p) -> dict[str, float]:
    """Unfiltered dense scores by unit id (what the unit itself scored)."""
    return first_scores(p.vector_store.search(p.embedder.embed(QUERY), 1000))


def _drop_parent_row(search):
    return lambda q, k: [(d, s) for d, s in search(q, k) if d != PID]


@pytest.fixture
def child_only():
    """Parent row absent from dense and BM25; only its _L chunks can match."""
    p = _build()
    dense_search, bm25_search = p.vector_store.search, p.bm25_index.search
    expected = _expected_dense(p)
    p.vector_store.search = _drop_parent_row(dense_search)
    p.bm25_index.search = _drop_parent_row(bm25_search)
    results, _ = p.retrieve(QUERY, top_k=5)
    return p, _find(results, PID), expected


def test_child_match_keeps_parent_and_chunk_identity_and_text(child_only):
    p, r, _ = child_only
    assert r.doc_id == PID
    chunk_ids = [m.chunk_id for m in r.matched_chunks if m.chunk_id != PID]
    assert chunk_ids, "child-only match must report the matched chunk(s)"
    assert all(cid.startswith(f"{PID}_L") for cid in chunk_ids)
    for m in r.matched_chunks:
        if m.chunk_id == PID:
            continue
        stored = p._long_chunk_map[m.chunk_id].chunk_text
        assert m.text == stored, "matched text must be the stored chunk text"
        assert m.parent_doc_id == PID
        assert m.unit_type == "annexure"


def test_matched_text_and_parent_evidence_are_distinct(child_only):
    _, r, _ = child_only
    top = next(m for m in r.matched_chunks if m.text)
    assert top.text != r.answer
    assert len(r.answer) > len(top.text), "parent evidence stays the whole assembled parent"
    # the large parent keeps its full Annexure-II table (not reduced to the chunk)
    assert "DISTRICT-ARG-35" in r.answer
    assert "WEST MEDINIPUR" in r.answer


def test_child_only_summary_scores_are_not_none_and_labelled(child_only):
    _, r, _ = child_only
    assert r.dense_score is not None
    assert r.bm25_score is not None
    assert r.rrf_score is not None
    assert r.score_sources == {"dense": "best_chunk", "bm25": "best_chunk", "rrf": "best_chunk"}


def test_each_matched_unit_keeps_its_own_dense_score(child_only):
    _, r, expected = child_only
    scored = [m for m in r.matched_chunks if m.chunk_id != PID and m.dense_score is not None]
    assert len(scored) >= 2, "the fixture must yield several matched chunks"
    for m in scored:
        assert m.dense_score == pytest.approx(expected[m.chunk_id]), m.chunk_id
    # several distinct values: nothing was copied from one unit onto the others
    assert len({round(m.dense_score, 9) for m in scored}) > 1


def test_summary_score_is_the_best_matched_chunks_score(child_only):
    _, r, expected = child_only
    best = max(expected[m.chunk_id] for m in r.matched_chunks
               if m.chunk_id != PID and m.dense_score is not None)
    assert r.dense_score == pytest.approx(best)


def test_reranked_unit_carries_the_parent_rerank_score(child_only):
    _, r, _ = child_only
    reranked = [m for m in r.matched_chunks if m.reranked]
    assert len(reranked) == 1
    assert reranked[0].rerank_score == r.rerank_score
    assert all(m.rerank_score is None for m in r.matched_chunks if not m.reranked)


def test_parent_row_scores_come_from_parent_when_parent_is_in_pool():
    p = _build()
    expected = _expected_dense(p)
    assert PID in expected, "fixture: the parent row must be in the dense pool"
    results, _ = p.retrieve(QUERY, top_k=5)
    r = _find(results, PID)
    assert r.dense_score == pytest.approx(expected[PID])
    assert r.score_sources["dense"] == SCORE_FROM_PARENT
    # The parent row is in the dense pool but outside the 50-unit RRF pool in
    # this fixture, so it is not a matched unit. Its own score is still the
    # summary, labelled "parent". Any parent-row entry carries no text.
    for m in r.matched_chunks:
        if m.chunk_id == PID:
            assert m.text is None
        else:
            assert m.text is not None


def test_genuinely_missing_channel_stays_none():
    p = _build()
    bm25_search = p.bm25_index.search
    p.bm25_index.search = lambda q, k: [
        (d, s) for d, s in bm25_search(q, k)
        if d != PID and not d.startswith(PID + "_")
    ]
    results, _ = p.retrieve(QUERY, top_k=5)
    r = _find(results, PID)
    assert r.bm25_score is None
    assert r.score_sources["bm25"] is None
    assert all(m.bm25_score is None for m in r.matched_chunks)
    assert r.dense_score is not None  # the other system still reports its score


def test_without_reranker_no_rerank_score_is_invented():
    p = _build(use_reranker=False)
    p.vector_store.search = _drop_parent_row(p.vector_store.search)
    results, _ = p.retrieve(QUERY, top_k=5)
    r = _find(results, PID)
    assert r.rerank_score is None
    assert all(m.rerank_score is None and not m.reranked for m in r.matched_chunks)
    assert any(m.text for m in r.matched_chunks)


# ── result object defaults and fusion ───────────────────────────────────────

def test_result_defaults_keep_existing_callers_working():
    r = RetrievedResult(doc_id="d", question="q", answer="a", score=1.0,
                        retrieval_method="hybrid", metadata={})
    assert r.matched_chunks == [] and r.matched_chunk_total == 0 and r.score_sources == {}


def test_fuse_carries_matched_units_and_score_sources():
    m = MatchedChunk(chunk_id=f"{PID}_L1", parent_doc_id=PID, unit_type="annexure", text="t")
    hybrid = RetrievedResult(doc_id=PID, question="q", answer="a", score=1.0,
                             retrieval_method="hybrid", metadata={}, dense_score=0.5,
                             matched_chunks=[m], matched_chunk_total=3,
                             score_sources={"dense": "best_chunk", "bm25": None, "rrf": None})
    graph = RetrievedResult(doc_id=PID, question="q", answer="g", score=0.0,
                            retrieval_method="graph", metadata={}, bm25_score=2.0,
                            score_sources={"bm25": "parent"})
    fused = merge_evidence([hybrid], [graph])[0]
    assert fused.matched_chunks == [m]
    assert fused.matched_chunk_total == 3
    assert fused.score_sources["dense"] == "best_chunk"
    assert fused.score_sources["bm25"] == "parent"  # bm25 came from the graph side
    assert "rrf" not in fused.score_sources          # no rrf on either side


# ── server serialization (single path for streaming and non-streaming) ──────

def _server():
    from src.retrieval.frontend import server as srv
    return srv


def test_to_sources_emits_matched_fields_and_score_basis(child_only):
    srv = _server()
    _, r, _ = child_only
    d = srv._to_sources([r])[0]
    assert d["matched_chunk_total"] == r.matched_chunk_total
    assert d["score_sources"] == r.score_sources
    assert d["matched_chunks"][0]["text"] == r.matched_chunks[0].text
    assert d["matched_chunks"][0]["reranked"] is r.matched_chunks[0].reranked


def test_fallback_source_without_matched_metadata_still_validates():
    srv = _server()
    r = RetrievedResult(doc_id="d1", question="q", answer="a", score=0.5,
                        retrieval_method="graph", metadata={"ministry": "M"})
    d = srv._to_sources([r])[0]
    item = srv.SourceItem(**d)
    assert item.matched_chunks == [] and item.matched_chunk_total == 0
    assert item.score_sources == {}


def test_streaming_and_nonstreaming_source_payloads_match(child_only):
    """Both chat routes build SourceItem from ``_to_sources``. Check the
    model round-trip keeps every value the streaming event carries."""
    srv = _server()
    _, r, _ = child_only
    stream_item = srv._to_sources([r])[0]
    nonstream_item = srv.SourceItem(**srv._to_sources([r])[0]).model_dump()
    for key, value in stream_item.items():
        if key == "graph":
            continue
        assert nonstream_item[key] == value, key


def test_matched_chunk_item_model_accepts_dict_from_serializer(child_only):
    srv = _server()
    _, r, _ = child_only
    for raw in srv._to_sources([r])[0]["matched_chunks"]:
        item = srv.MatchedChunkItem(**raw)
        assert item.chunk_id == raw["chunk_id"]


# ── chunk-level path (use_chunking=True): Q/A chunk units, no parent row ─────

def _build_chunked() -> HybridRAGPipeline:
    fillers = [
        QARecord(question_id=f"f{i}", question_text=f"Filler question {i} about ocean observation?",
                 answer_text=f"Filler answer {i} about ocean buoys and tsunami warning.",
                 metadata=QARecordMetadata(ministry="EARTH SCIENCES", subject=f"f{i}",
                                           document_type="parliamentary_qa"))
        for i in range(4)
    ]
    return HybridRAGPipeline(records=[_aws_record()] + fillers, embedder=_StableEmbedder(),
                             reranker=_KeywordFitReranker(), use_chunking=True)


def test_chunk_path_attributes_each_unit_and_labels_summary_best_chunk():
    p = _build_chunked()
    expected = _expected_dense(p)
    results, _ = p.retrieve(QUERY, top_k=5)
    assert len({r.doc_id for r in results}) == len(results), "one result per parent"
    r = _find(results, PID)
    assert r.matched_chunks, "chunk path must report its matched units"
    assert {m.unit_type for m in r.matched_chunks} <= {"question", "answer"}
    for m in r.matched_chunks:
        assert m.text and m.text.strip(), "stored Q/A chunk text is reported"
        assert m.chunk_id.startswith(PID + "_")
        assert m.dense_score == pytest.approx(expected[m.chunk_id]), m.chunk_id
    assert sum(1 for m in r.matched_chunks if m.reranked) <= 1
    assert r.score_sources == {"dense": "best_chunk", "bm25": "best_chunk", "rrf": "best_chunk"}
    best = max(expected[m.chunk_id] for m in r.matched_chunks)
    assert r.dense_score == pytest.approx(best)
    # Legacy chunk assembly: `answer` is the matched ANSWER chunk's own text, so no
    # larger parent text exists on this path. The LLM context is unchanged.
    answer_units = [m for m in r.matched_chunks if m.unit_type == "answer"]
    assert answer_units and r.answer == answer_units[0].text


def test_chunk_path_summary_is_never_a_copy_of_one_unit_when_units_differ():
    p = _build_chunked()
    results, _ = p.retrieve(QUERY, top_k=5)
    r = _find(results, PID)
    values = {round(m.dense_score, 9) for m in r.matched_chunks if m.dense_score is not None}
    assert len(values) >= 2, "fixture must give the matched units distinct dense scores"
