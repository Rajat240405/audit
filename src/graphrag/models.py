"""Canonical graph data models (Phase 2 / WS2).

OPEN-WORLD identity model (spec §4): the Phase 1 vocabulary is NOT a closed
list of everything that can exist. Every entity reference carries:

  * ``key``        — the stable identity used for MERGE (deterministic)
  * ``name``       — display name (a property, NEVER the identity)
  * ``raw``        — the original raw surface (provenance; always preserved)
  * ``resolution`` — "canonical" (known identity) or "unresolved" (unknown)

Unknown entities get a stable ``u:<sha1-16 of folded raw>`` key: exactly
equal folded surfaces merge (that is a safe identity rule), but nothing is
ever fuzzy-merged because names merely look similar. Promotion of an
unresolved key to a canonical one is a later, explicit operation — this layer
never invents it.

Key namespaces (documented in docs/phase2/DESIGN.md):
  * Document   : question_id
  * Ministry   : vocab canonical slug | u:<hash>
  * Organization: vocab org slug | c:<concept canonical> | u:<hash>
  * House      : vocab canonical
  * LokSabhaTerm: ls-term-<n>
  * Session    : <house>:<n>
  * Member     : m:<phase1 canonical key> | u:<hash>
  * Programme  : c:<concept canonical> | u:<hash>
  * Facility   : u:<hash>          (no controlled vocabulary in corpus yet)
  * Place      : u:<hash>          (no controlled vocabulary in corpus yet)
  * Year       : year-<n>
  * Fact       : "<REL>:<src_key>-><dst_key>"
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "fold_surface",
    "unknown_key",
    "EntityRef",
    "GraphFact",
    "Support",
    "GraphContribution",
    "FactView",
    "NodeView",
    "DocumentRef",
]

_WS_RE = re.compile(r"\s+")


def fold_surface(text: str) -> str:
    """Comparison/identity folding: casefold + whitespace collapse."""
    return _WS_RE.sub(" ", str(text).casefold().strip())


def unknown_key(raw: str) -> str:
    """Stable key for an UNRESOLVED entity: hash of the folded surface.

    Deterministic and collision-safe for this corpus size (sha1-16). Two
    different raw strings that fold to the same surface share a key (exact
    identity only — never similarity)."""
    h = hashlib.sha1(fold_surface(raw).encode("utf-8")).hexdigest()[:16]
    return f"u:{h}"


@dataclass(frozen=True)
class EntityRef:
    """One entity node reference (open-world)."""

    label: str
    key: str
    name: str
    resolution: str = "canonical"          # "canonical" | "unresolved"
    raw: Optional[str] = None              # original raw surface (provenance)
    props: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError(f"EntityRef({self.label}) needs a stable key")
        if self.resolution not in ("canonical", "unresolved"):
            raise ValueError(f"bad resolution: {self.resolution!r}")
        if self.raw is None:
            object.__setattr__(self, "raw", self.name)

    def to_node(self) -> dict:
        """Property bag for the node (key + name + resolution + raw + props)."""
        out = {"key": self.key, "name": self.name,
               "resolution": self.resolution, "raw": self.raw or self.name}
        out.update(self.props)
        return out


@dataclass(frozen=True)
class GraphFact:
    """One factual relationship between two entity nodes.

    ``origin`` = "deterministic" (derived from structured metadata — no LLM)
    or "llm" (semantic extraction, grounded + validated). ``source_field``
    names the metadata field for deterministic facts (e.g.
    "metadata.ministry") — the field IS the source; LLM facts instead carry
    per-document evidence snippets on their SUPPORTS edges."""

    rel: str
    src: EntityRef
    dst: EntityRef
    origin: str = "deterministic"
    source_field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.origin not in ("deterministic", "llm"):
            raise ValueError(f"bad origin: {self.origin!r}")

    @property
    def fact_key(self) -> str:
        return f"{self.rel}:{self.src.key}->{self.dst.key}"

    def rel_properties(self, *, sample_evidence: Optional[str] = None) -> dict:
        return {
            "fact_key": self.fact_key,
            "origin": self.origin,
            "source_field": self.source_field,
            "sample_evidence": sample_evidence,
        }


@dataclass(frozen=True)
class Support:
    """One (document, fact) provenance edge.

    A fact may be supported by MANY documents; each document contributes one
    Support with its own evidence snippet (verbatim text for LLM facts) and
    timestamp. This is what makes multi-source relationships possible and
    what lets a changed document's retraction leave other documents' facts
    intact."""

    doc_key: str
    fact_key: str
    evidence: Optional[str] = None
    origin: str = "deterministic"
    extracted_at: str = ""


@dataclass
class GraphContribution:
    """Everything ONE document contributes to the graph.

    The pipeline applies the whole contribution atomically (one transaction)
    and can retract it the same way (document-scoped reconciliation)."""

    doc_key: str
    nodes: list[EntityRef] = field(default_factory=list)
    facts: list[GraphFact] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)

    def fact_keys(self) -> set[str]:
        return {f.fact_key for f in self.facts}


# ── read-side views (store → application) ───────────────────────────────────

@dataclass(frozen=True)
class NodeView:
    key: str
    label: str
    name: str
    resolution: str
    props: dict
    # The surface form the SOURCE actually used for this entity. Identity is
    # the key; ``raw`` is provenance (what a document called it) and must stay
    # readable — especially for unresolved "u:" entities, where it is the only
    # human-legible record of the open-world value. Exposed BOTH as an
    # attribute and inside ``props`` so either access style works.
    raw: Optional[str] = None

    def __post_init__(self) -> None:
        if self.raw is None:
            object.__setattr__(self, "raw", self.props.get("raw"))
        elif "raw" not in self.props:
            # keep props authoritative-in-sync without mutating the caller's dict
            object.__setattr__(self, "props", {**self.props, "raw": self.raw})

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "name": self.name,
                "resolution": self.resolution, **self.props}


@dataclass(frozen=True)
class FactView:
    fact_key: str
    rel: str
    src_key: str
    dst_key: str
    origin: str
    doc_count: int
    sample_evidence: Optional[str]
    first_seen_at: str = ""
    updated_at: str = ""
    source_field: Optional[str] = None  # deterministic facts name their field

    def to_dict(self) -> dict:
        return {
            "fact_key": self.fact_key, "rel": self.rel, "src_key": self.src_key,
            "dst_key": self.dst_key, "origin": self.origin,
            "doc_count": self.doc_count, "sample_evidence": self.sample_evidence,
            "first_seen_at": self.first_seen_at, "updated_at": self.updated_at,
            "source_field": self.source_field,
        }


@dataclass
class DocumentRef:
    """Document node properties (identity = question_id)."""

    doc_key: str
    question_text: str = ""
    answer_text: str = ""
    content_hash: str = ""
    source_url: Optional[str] = None
    date: Optional[str] = None            # canonical ISO (or raw if unparseable)
    date_year: Optional[int] = None       # temporal facet (int, for indexing)
    ls_term: Optional[int] = None         # Lok Sabha term (LS docs only)
    document_type: Optional[str] = None   # canonical document type
    question_type: Optional[str] = None   # canonical (or raw) question type
    subject_raw: Optional[str] = None
    ministry_raw: Optional[str] = None

    def to_node(self) -> dict:
        return {
            "key": self.doc_key,
            "question_text": self.question_text,
            "answer_text": self.answer_text,
            "content_hash": self.content_hash,
            "source_url": self.source_url,
            "date": self.date,
            "date_year": self.date_year,
            "ls_term": self.ls_term,
            "document_type": self.document_type,
            "question_type": self.question_type,
            "subject_raw": self.subject_raw,
            "ministry_raw": self.ministry_raw,
        }
