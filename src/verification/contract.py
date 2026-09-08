"""Normalized verification contract (Phase 3 / WS3).

The ONE representation the Verification Authority consumes, regardless of
which retrieval mode produced the evidence (Hybrid RAG or GraphRAG).
Retrieval-specific code produces these; only the authority interprets them.

Verdict semantics (strict):
  verified     — genuine textual support in a cited source (normalized +
                 alias-aware, number+word swap aware) AND the citation is
                 traceable (hybrid: doc_id among the sources; graph: the
                 Document→SUPPORTS→Fact chain resolves).
  rejected     — stronger negative: the LLM judge said unsupported, or a
                 graph fact is presented as grounded without supports.
  insufficient — no support found, no judge upgrade (default for ungrounded
                 claims; a weak/partial overlap is NEVER sufficient).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "Claim",
    "EvidenceItem",
    "EvidenceSet",
    "GraphProvenance",
    "ClaimVerdict",
    "VerificationReport",
    "STATUS_VERIFIED",
    "STATUS_REJECTED",
    "STATUS_INSUFFICIENT",
    "MODE_HYBRID",
    "MODE_GRAPH",
]

STATUS_VERIFIED = "verified"
STATUS_REJECTED = "rejected"
STATUS_INSUFFICIENT = "insufficient"

MODE_HYBRID = "hybrid"
MODE_GRAPH = "graph"


@dataclass(frozen=True)
class Claim:
    """One checkable claim extracted from an answer."""

    text: str
    kind: str = "generic"     # figure | num_word | quote | named_abbr | acronym | generic
    context: str = ""         # sentence the claim came from (optional)

    def to_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind, "context": self.context}


@dataclass
class GraphProvenance:
    """Graph-mode provenance attached to an evidence item (WS2 §12):
    the fact(s) that linked the document to the query anchor, and the
    full SUPPORTS chains for each fact."""

    fact_keys: list[str] = field(default_factory=list)
    anchors: list[dict] = field(default_factory=list)
    # per-fact provenance as resolved from the store:
    # [{fact_key, rel, origin, doc_count,
    #   supports: [{fact_key, evidence, origin, doc_key, extracted_at}]}]
    facts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"fact_keys": self.fact_keys, "anchors": self.anchors,
                "facts": self.facts}


@dataclass
class EvidenceItem:
    """One source of evidence, normalized across retrieval modes."""

    doc_id: str
    question: str = ""
    answer: str = ""
    mode: str = MODE_HYBRID            # "hybrid" | "graph"
    scores: Optional[dict] = None      # hybrid sub-scores (optional, passthrough)
    graph: Optional[GraphProvenance] = None  # graph mode only
    # graph: the document node's stored text may differ from the corpus row
    # when the corpus changed after the graph was built — kept when supplied.
    stored_date: Optional[str] = None

    def text(self) -> str:
        return f"{self.question} {self.answer}".strip()

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "scores": self.scores,
            "graph": self.graph.to_dict() if self.graph else None,
        }


@dataclass
class EvidenceSet:
    """All evidence for one verification run (one mode, never mixed)."""

    items: list[EvidenceItem] = field(default_factory=list)
    mode: str = MODE_HYBRID

    def as_sources(self) -> list[dict]:
        """The legacy {doc_id, question, answer} dict shape (what the
        /api/verify engine and the text_support core consume)."""
        return [
            {"doc_id": i.doc_id, "question": i.question, "answer": i.answer}
            for i in self.items
        ]

    def doc_ids(self) -> set[str]:
        return {i.doc_id for i in self.items}


@dataclass
class ClaimVerdict:
    """The authority's verdict for one claim."""

    claim: str
    status: str                     # verified | rejected | insufficient
    source_doc_id: Optional[str] = None
    method: str = "text"            # text | llm_judge | graph_provenance
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.claim,
            "found": self.status == STATUS_VERIFIED,
            "status": self.status,
            "source": self.source_doc_id,
            "method": self.method,
            "note": self.note,
        }


@dataclass
class VerificationReport:
    """Full report for one verification run."""

    claims: list[ClaimVerdict] = field(default_factory=list)
    method: str = "light"                     # light | full
    provenance_ok: bool = True                # every verified claim traceable
    graph_checks: Optional[dict] = None       # graph integrity check results
    final_text: Optional[str] = None          # rewritten answer (full mode)
    judge_rewritten: bool = False
    judge_removed: list[str] = field(default_factory=list)
    citation_dropped: list[str] = field(default_factory=list)

    @property
    def sufficiency(self) -> dict:
        n = {STATUS_VERIFIED: 0, STATUS_REJECTED: 0, STATUS_INSUFFICIENT: 0}
        for c in self.claims:
            n[c.status] = n.get(c.status, 0) + 1
        total = max(1, len(self.claims))
        n["verified_ratio"] = round(n[STATUS_VERIFIED] / total, 4)
        n["total"] = len(self.claims)
        return n

    def to_dict(self) -> dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "sufficiency": self.sufficiency,
            "method": self.method,
            "provenance_ok": self.provenance_ok,
            "graph_checks": self.graph_checks,
            "final_text": self.final_text,
            "judge_rewritten": self.judge_rewritten,
            "judge_removed": self.judge_removed,
            "citation_dropped": self.citation_dropped,
        }
