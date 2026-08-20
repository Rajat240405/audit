"""Retrieval → evidence bridge (17-7-2936 regression suite).

Scope-protected assertions:
  A. The parent document's Annexure-I table (33 rows) reaches evidence
     assembly as bounded, caption-anchored row groups — never torn mid-row.
  B. ONE parent-evidence mechanism (assemble_parent_evidence) replaces the
     matched-window join and the keyword-paragraph fallback.
  C. Budget behavior: clean row-group admission/omission, no mid-row cut,
     explicit omission marker when prose promises dropped content.
  D. Deep sibling-window rescue (bounded, idempotent; skipped when the
     bridge already supplied the full parent).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

import tests.conftest  # noqa: F401  (stubs torch/sentence-transformers)

from src.generation import evidence as ev
from src.generation.evidence import (
    DEEP_SIBLING_MAX,
    PARENT_EVIDENCE_CAP_TOKENS,
    _OMISSION_TEMPLATE,
    allocate_evidence,
    assemble_parent_evidence,
    estimate_tokens,
    segment_blocks,
)
from src.models.qa_record import QARecord, QARecordMetadata
from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.retrieval.result import RetrievedResult


# ── 17-7-2936 fixture: real corpus-lineage spans (investigation ground truth) ─
INTRO = """(a) Yes Sir. On the whole, there are 33 Automatic Weather Stations and 35 Automatic Rain
Gauges installed in West Bengal.
(b) Details of automatic weather stations and automatic raingauges stations of West Bengal are
given inAnnexure I and Annexure II, respectively.
(c) Ministry of Earth Sciences does not have any plans to link theweather stations with micro
insurance programmes for farmers.
(d) No new projects are being sanctioned to Bishnupur Constituency. However, an automatic
weather station has already been established in Bankura district in which Bishnupur is
located.
(e) - (t) IMD has carried out an analysis of observed monsoon rainfall variability and changes of29
States & Union Territory at State and District levels based on the IMD's observational data
of recent 30 years (1989- 2018) during the Southwest monsoon season from June-July
August-September (JJAS). Five states viz., Uttar Pradesh, Bihar, West Bengal, Meghalaya
and Nagaland have shown significant decreasing trends in southwest monsoon rainfall during
the recent 30 years period (1989-2018)."""

AWS_ROWS = [
    "1 WEST BENGAL BANKURA BANKURA",
    "2 WEST BENGAL BARDHAMAN BURDWAN",
    "3 WEST BENGAL BHIRBHUM SURI",
    "SANTINIKETAN-",                       # wrap fragment of row 3 (real artifact)
    "4 WEST BENGAL BIRBHUM BOLPUR KVK",
] + [f"{i} WEST BENGAL DISTRICT-{i:02d} STATION-{i:02d}" for i in range(5, 27)] + [
    "27 WEST BENGAL SOUTH TWENTY FOUR PARGANAS BARUIPUR",
    "28 WEST BENGAL SOUTH TWENTY FOUR PARGANAS CANNIiNG",
    "29 WEST BENGAL SOUTH TWENTY FOUR PARGANAS KAKDWIP",
    "30 WEST BENGAL SOUTH TWENTY FOUR PARGANAS NIMPITH",
    "31 WEST BENGAL SOUTH TWENTY FOUR PARGANAS RAIDIGHI",
    "32 WEST BENGAL SOUTH TWENTY FOUR PARGANAS SAGAR ISLAND",
    "KHARAGPUR( lIT",                      # forward fragment of row 32 (real artifact)
    "33 WEST BENGAL WEST MEDINIPUR CAMPUS)",
]

ANNEXURE_I = (
    "Annexure-I\nDetails of Automatic Weather Stations of West Bengal\n"
    "SNO. STATE DISTRICT STATION\n" + "\n".join(AWS_ROWS)
)
ARG_ROWS = [f"{i} WEST BENGAL DISTRICT-ARG-{i:02d} STATION-ARG-{i:02d}" for i in range(1, 36)]
ANNEXURE_II = (
    "Annexure-II\nDetails of Automatic Raingauge Stations of West Bengal (TOTAL: 35)\n"
    "SNO. STATE DISTRICT STATION\n" + "\n".join(ARG_ROWS)
)
AWS_ANSWER = INTRO + "\n" + ANNEXURE_I + "\n" + ANNEXURE_II
AWS_QUESTION = ("Will the Minister of Earth Sciences be pleased to state: (a) whether any modern "
                "automatic weather stations have been set up in West Bengal; (b) if so, the "
                "details thereof, location-wise?")
QUERY = "Which automatic weather stations are installed in West Bengal? Give location-wise details."


def _aws_record():
    return QARecord(
        question_id="17-7-2936", question_text=AWS_QUESTION, answer_text=AWS_ANSWER,
        metadata=QARecordMetadata(ministry="EARTH SCIENCES", subject="Modern Automatic Weather Stations",
                                  date="2021-12-15", document_type="parliamentary_qa"),
    )


class _LexicalEmbedder:
    model_name = "lex-hash"
    embedding_dim = 512

    def _v(self, text):
        v = np.zeros(self.embedding_dim, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w) % self.embedding_dim] += 1.0
        return v / (np.linalg.norm(v) + 1e-9)

    def embed(self, text):
        return self._v(text)

    def embed_batch(self, texts, batch_size=1, show_progress=False):
        return np.stack([self._v(t) for t in texts])


class _KeywordFitReranker:
    """Ranks intro/caption chunks above row chunks — the live failure mode."""

    def rerank(self, query, candidates, k=5, doc_texts=None):
        kws = ev.query_keywords(query)
        scored = []
        for doc_id, _rrf in candidates:
            text = (doc_texts or {}).get(doc_id, "").lower()
            scored.append((doc_id, float(sum(1 for kw in kws if kw in text))))
        scored.sort(key=lambda s: -s[1])
        return scored[:k]


def _pipeline(**kw):
    fillers = [
        QARecord(question_id=f"f{i}", question_text=f"Filler question {i} about ocean observation services?",
                 answer_text=f"Filler answer {i} about ocean buoys and tsunami warning systems.",
                 metadata=QARecordMetadata(ministry="EARTH SCIENCES", subject=f"f{i}",
                                           document_type="parliamentary_qa"))
        for i in range(4)
    ]
    return HybridRAGPipeline(records=[_aws_record()] + fillers,
                             embedder=_LexicalEmbedder(),
                             reranker=kw.get("reranker") or _KeywordFitReranker())


# ── A. structural row-group units ────────────────────────────────────────────

def test_row_runs_become_bounded_groups_with_replicated_context():
    blocks = segment_blocks(ev.clean_parliament_text(AWS_ANSWER))
    tables = [b for b in blocks if b.kind == "table"]
    assert tables, "annexure row runs must become table blocks"
    aws_groups = [b for b in tables if "Automatic Weather Stations" in b.text]
    assert len(aws_groups) >= 2                     # bounded row groups
    for g in aws_groups:
        # caption + annexure identity + header ride with EVERY group
        assert "Annexure-I" in g.text
        assert "Details of Automatic Weather Stations of West Bengal" in g.text
        assert "SNO. STATE DISTRICT STATION" in g.text
        assert estimate_tokens(g.text) <= 300       # bounded (+context slack)


def test_all_33_rows_present_once_and_in_order():
    blocks = segment_blocks(AWS_ANSWER)
    emitted = "\n".join(b.text for b in blocks)
    rows = [l for l in emitted.splitlines()
            if re.match(r"^\d+ WEST BENGAL", l) and "DISTRICT-ARG" not in l]
    nums = [int(l.split()[0]) for l in rows]
    assert nums == list(range(1, 34))               # 33 complete rows, doc order
    # each row appears exactly once (context replication must not duplicate rows)
    assert len(rows) == 33


def test_wrap_fragments_glue_to_their_own_row():
    blocks = segment_blocks(AWS_ANSWER)
    for b in blocks:
        lines = b.text.splitlines()
        for idx, l in enumerate(lines):
            if l.strip() == "SANTINIKETAN-":
                # fragment must stand INSIDE its row's block, never be the
                # block-final orphan that 500-char chunking produced before
                assert any("3 WEST BENGAL BHIRBHUM SURI" in x for x in lines[: idx + 1])
                assert idx + 1 < len(lines), "fragment orphaned at block end"
                assert lines[idx + 1].startswith("4 WEST BENGAL BIRBHUM"), "torn row pair"
            if l.strip() == "KHARAGPUR( lIT":
                assert any("32 WEST BENGAL" in x for x in lines[: idx + 1])
                assert idx + 1 < len(lines) and lines[idx + 1].startswith("33 WEST BENGAL")


def test_blocks_stay_in_document_order():
    blocks = segment_blocks(AWS_ANSWER)
    text = "\n".join(b.text for b in blocks)
    intro_pos = text.index("On the whole, there are 33 Automatic Weather Stations")
    ann1 = text.index("1 WEST BENGAL BANKURA BANKURA")
    ann1_end = text.index("33 WEST BENGAL WEST MEDINIPUR CAMPUS)")
    ann2 = text.index("Annexure-II")
    assert intro_pos < ann1 < ann1_end < ann2


# ── B. assemble_parent_evidence — the ONE mechanism ───────────────────────────

def test_modest_parent_passes_through_whole():
    out = assemble_parent_evidence(AWS_ANSWER, QUERY)
    assert out == AWS_ANSWER                        # zero loss for corpus-scale docs


def test_giant_parent_is_structurally_windowed_not_keyword_filtered():
    body = "\n".join(
        ["Intro paragraph mentioning Automatic Weather Stations."]
        + [f"Row {i} station TOKEN_{i} " + "x" * 60 for i in range(1, 1200)]
    )
    assert estimate_tokens(body) > PARENT_EVIDENCE_CAP_TOKENS
    out = assemble_parent_evidence(body, "TOKEN_42 stations")
    assert estimate_tokens(out) <= PARENT_EVIDENCE_CAP_TOKENS
    assert len(out) < 20_000                        # retrieval-size invariant kept
    assert "TOKEN_42" in out                        # anchored by the relevant section
    assert "Row 1199" not in out                    # bounded — not a head dump


# ── C. pipeline integration — the money tests ────────────────────────────────

def test_retrieval_of_parent_recovers_full_annexure_table():
    pipe = _pipeline()
    results, _ = pipe.retrieve(QUERY, top_k=5)
    aws = next(r for r in results if r.doc_id == "17-7-2936")
    assert aws.metadata.get("evidence_source") == "parent_full"
    assert aws.metadata.get("chunk_ids")            # provenance kept
    rows = [l for l in aws.answer.splitlines() if re.match(r"^\d+ WEST BENGAL", l) and "DISTRICT-ARG" not in l]
    assert len(rows) == 33, f"rows lost: got {len(rows)}/33\n{aws.answer[:800]}"
    # document order inside evidence
    assert aws.answer.index("On the whole") < aws.answer.index("Annexure-I") < aws.answer.index("Annexure-II")


def test_rows_not_lost_merely_because_their_chunk_scored_low():
    # reranker pushes intro/caption chunks to the top (the live failure);
    # the bridge must recover row content from the identified parent anyway.
    pipe = _pipeline(reranker=_KeywordFitReranker())
    results, _ = pipe.retrieve(QUERY, top_k=5)
    aws = next(r for r in results if r.doc_id == "17-7-2936")
    mid_rows = [f"{i} WEST BENGAL DISTRICT-{i:02d}" for i in (7, 15, 24)]
    assert all(any(l.startswith(m) for l in aws.answer.splitlines()) for m in mid_rows)
    assert "32 WEST BENGAL SOUTH TWENTY FOUR PARGANAS SAGAR ISLAND" in aws.answer


def test_parent_only_fallback_no_longer_drops_every_row():
    class _ParentOnlyReranker:
        def rerank(self, query, candidates, k=5, doc_texts=None):
            return [(d, s) for d, s in candidates if d == "17-7-2936"]

    pipe = _pipeline(reranker=_ParentOnlyReranker())
    results, _ = pipe.retrieve(QUERY, top_k=5)
    aws = next(r for r in results if r.doc_id == "17-7-2936")
    assert aws.metadata.get("chunk_ids", []) == []    # the old P1 shape: no chunks
    # OLD path produced intro + captions + ZERO rows beyond the 2,000-char cut.
    rows = [l for l in aws.answer.splitlines() if re.match(r"^\d+ WEST BENGAL", l) and "DISTRICT-ARG" not in l]
    assert len(rows) == 33
    assert "30 WEST BENGAL SOUTH TWENTY FOUR PARGANAS NIMPITH" in aws.answer
    # and it is NOT the keyword-paragraph filter's disjointed line salad:
    assert "SNO. STATE DISTRICT STATION" in aws.answer


# ── D. budget behavior ───────────────────────────────────────────────────────

def _aws_result():
    return RetrievedResult(
        doc_id="17-7-2936", question=AWS_QUESTION, answer=AWS_ANSWER, score=0.99,
        retrieval_method="rrf_fusion",
        metadata={"ministry": "EARTH SCIENCES", "subject": "Modern Automatic Weather Stations",
                  "date": "2021-12-15", "document_type": "parliamentary_qa",
                  "evidence_source": "parent_full"},
    )


def test_tight_budget_admits_complete_row_groups_with_marker():
    # Room for header + intro + ~one table group, not all groups.
    alloc = allocate_evidence([_aws_result()], QUERY, budget_tokens=450)
    assert alloc.admissions, "top candidate must still yield evidence"
    adm = alloc.admissions[0]
    emitted = adm.evidence_text
    rows = [l for l in emitted.splitlines() if re.match(r"^\d+ WEST BENGAL", l)]
    fixture_rows = set("\n".join(b.text for b in segment_blocks(AWS_ANSWER)).splitlines())
    for l in rows:
        assert l in fixture_rows, f"mangled row line emitted: {l!r}"
    if adm.omitted_units:
        assert any(_OMISSION_TEMPLATE.split("{n}")[0] in l or "omitted" in l
                   for l in emitted.splitlines()), "dropped groups must be marked"


def test_caption_promising_dropped_content_gets_explicit_marker():
    # Budget admits the intro prose ("Details ... given inAnnexure I ...")
    # but NONE of the table groups → the promise must carry an omission note.
    alloc = allocate_evidence([_aws_result()], QUERY, budget_tokens=120)
    assert alloc.admissions
    emitted = alloc.admissions[0].evidence_text
    rows = [l for l in emitted.splitlines() if re.match(r"^\d+ WEST BENGAL", l)]
    if not rows:                                    # whole table omitted
        assert "[… " in emitted and "omitted" in emitted, \
            f"silent content-loss shape:\n{emitted}"


def test_no_row_is_cut_in_the_middle():
    alloc = allocate_evidence([_aws_result()], QUERY, budget_tokens=600)
    for adm in alloc.admissions:
        lines = adm.evidence_text.splitlines()
        for k, l in enumerate(lines):
            if re.match(r"^\d+ ", l):
                assert len(l) >= 15, f"half-row emitted: {l!r}"
            if l.strip().endswith("-"):
                # a torn fragment is only legal as an interior line glued to
                # its row — the numbered continuation must follow immediately
                assert k + 1 < len(lines), f"fragment orphaned at evidence end: {l!r}"
                assert re.match(r"^\d+ ", lines[k + 1]), f"fragment not glued: {l!r}"


# ── E. Deep sibling-window rescue ────────────────────────────────────────────

class _ChunkMap(dict):
    def __init__(self, d):
        super().__init__(d)
        for k, v in d.items():
            assert hasattr(v, "chunk_text")


class _C:
    def __init__(self, text):
        self.chunk_text = text


def test_deep_sibling_window_rejoins_torn_row_bounded_idempotent():
    r = RetrievedResult(doc_id="p", question="q", answer="4 WEST BENGAL BIRBHUM BOLPUR KVK",
                        score=1.0, retrieval_method="rrf_fusion",
                        metadata={"chunk_ids": ["p_L3"]})          # legacy partial (no flag)
    mapping = _ChunkMap({
        "p_L2": _C("3 WEST BENGAL BHIRBHUM SURI\nSANTINIKETAN-"),
        "p_L4": _C("5 WEST BENGAL COOCH BEHAR PUNDIBARI"),
    })
    n = ev.enrich_deep_neighbors([r], mapping)
    assert n == DEEP_SIBLING_MAX                                  # bounded at 2
    assert r.answer.index("SANTINIKETAN-") < r.answer.index("4 WEST BENGAL BIRBHUM")
    assert r.answer.index("4 WEST BENGAL BIRBHUM") < r.answer.index("5 WEST BENGAL COOCH")
    assert ev.enrich_deep_neighbors([r], mapping) == 0            # idempotent


def test_deep_window_skips_parent_full_and_bad_ids():
    flagged = RetrievedResult(doc_id="p", question="q", answer="full parent text here",
                              score=1.0, retrieval_method="rrf_fusion",
                              metadata={"chunk_ids": ["p_L3"], "evidence_source": "parent_full"})
    legacy_bad = RetrievedResult(doc_id="q", question="q", answer="body", score=0.9,
                                 retrieval_method="rrf_fusion", metadata={"chunk_ids": ["weird"]})
    mapping = _ChunkMap({"p_L2": _C("sibling text")})
    assert ev.enrich_deep_neighbors([flagged], mapping) == 0       # bridge subsumes
    assert ev.enrich_deep_neighbors([legacy_bad], mapping) == 0    # unparseable ids


def test_deep_window_capped_across_multiple_anchors():
    r = RetrievedResult(doc_id="p", question="q", answer="mid body", score=1.0,
                        retrieval_method="rrf_fusion", metadata={"chunk_ids": ["p_L2", "p_L4"]})
    mapping = _ChunkMap({f"p_L{i}": _C(f"body {i}") for i in range(0, 7)})
    n = ev.enrich_deep_neighbors([r], mapping)
    assert n <= DEEP_SIBLING_MAX
