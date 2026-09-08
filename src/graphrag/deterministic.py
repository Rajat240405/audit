"""Deterministic graph population (Phase 2 / WS2-B).

Turns ONE canonical corpus row (QARecord) into a document node + a
document-scoped GraphContribution using ONLY structured metadata + the Phase 1
vocabulary (src/vocabulary). NO LLM involved — everything here is
deterministic and auditable (every fact names its source_field).

Open-world (spec §4): metadata values the vocabulary does not know are
preserved as unresolved nodes (stable "u:" keys, raw kept) — never rejected,
never merged.

Relationships produced (all origin="deterministic"):
  Document -[:ADDRESSED_TO]        -> Ministry      (metadata.ministry)
  Document -[:PUBLISHED_BY]        -> Organization  (metadata.org)
  Document -[:ASKED_IN]            -> House         (metadata.house / id prefix)
  Document -[:IN_SESSION]          -> Session       (metadata.session + house)
  Document -[:IN_LOK_SABHA_TERM]   -> LokSabhaTerm  (ls-<term>-… doc id)
  Document -[:ASKED_BY]            -> Member        (metadata.member, one per
                                                     clubbed-list token)
  Document -[:DATED]               -> Year          (metadata.date, year facet)

Deliberately NOT produced here:
  * Subject entities — deferred (WS1 decision: free text, no controlled map)
  * semantic relationships — extract.py (LLM, grounded)
"""
from __future__ import annotations

import re
from typing import Optional

from src.models.qa_record import QARecord
from src.scripts.ingest_folder import qa_content_hash
from src.vocabulary import Vocabulary
from src.vocabulary.normalize import (
    ls_term_from_doc_id,
    normalize_question_type,
    normalize_date,
    normalize_document_type,
    normalize_house,
    normalize_house_from_doc_id,
    normalize_member,
    normalize_ministry,
    normalize_org,
)

from src.graphrag.models import (
    DocumentRef,
    EntityRef,
    GraphContribution,
    GraphFact,
    Support,
    unknown_key,
)
from src.graphrag.schema import (
    key_ls_term,
    key_member,
    key_session,
    key_year,
)

__all__ = ["build_contribution", "document_ref"]


def _concept_surface_index(voc: Vocabulary) -> dict:
    """folded surface form → concept canonical (built once per call; the
    concept list is tiny). Shared with the slug-unification rule below."""
    idx: dict[str, str] = {}
    for c in voc.concepts:
        forms = [c.get("canonical")] + list(c.get("surfaces") or [])
        for s in forms:
            if s:
                idx[_fold(s)] = c["canonical"]
    return idx


def _concept_canonical_for(voc: Vocabulary, raw: Optional[str]) -> Optional[str]:
    """Exact folded-surface match of a controlled org slug against concept
    surfaces (e.g. slug "incois" → concept "INCOIS"). Returns None when the
    slug matches no concept (e.g. "sansad", "moes_hq")."""
    if not raw:
        return None
    return _concept_surface_index(voc).get(_fold(raw))


def _concept_label(voc: Vocabulary, canonical: str) -> str:
    for c in voc.concepts:
        if c["canonical"] == canonical:
            return c.get("label") or canonical
    return canonical


def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").casefold().strip())


def _entry_label(voc: Vocabulary, match, raw: Optional[str]) -> str:
    """Display name for a controlled value (vocab label; raw when unknown)."""
    if match is not None:
        return match.label
    return (raw or "").strip()


def document_ref(rec: QARecord, voc: Vocabulary) -> DocumentRef:
    """The Document node for a corpus row (identity = question_id)."""
    meta = rec.metadata
    date_n = normalize_date(voc, meta.date)
    date_value: Optional[str] = date_n.iso or (
        date_n.canonical if date_n.canonical else meta.date)
    date_year: Optional[int] = None
    if date_n.precision == "day" and date_n.iso:
        date_year = int(date_n.iso[:4])
    elif date_n.precision == "year" and date_n.canonical:
        date_year = int(date_n.canonical)

    dt = normalize_document_type(voc, meta.document_type)
    # canonical when controlled; otherwise the raw value is preserved as a
    # plain property (open world — document_type is data, not an entity)
    doc_type = dt.canonical if dt.status == "canonical" else meta.document_type
    qt = normalize_question_type(voc, meta.question_type)
    question_type = qt.canonical if qt.status == "canonical" else str(meta.question_type)
    return DocumentRef(
        doc_key=rec.question_id,
        question_text=rec.question_text or "",
        answer_text=rec.answer_text or "",
        content_hash=qa_content_hash(rec),
        source_url=meta.source_url,
        date=date_value,
        date_year=date_year,
        ls_term=ls_term_from_doc_id(rec.question_id),
        document_type=doc_type,
        question_type=question_type,
        subject_raw=meta.subject,
        ministry_raw=meta.ministry,
    )


def build_contribution(
    rec: QARecord, voc: Vocabulary, *, now: str
) -> tuple[DocumentRef, GraphContribution]:
    """Deterministic contribution of one corpus row to the graph.

    Returns (DocumentRef, GraphContribution). The contribution's nodes list
    holds the entity nodes (Document itself is applied via upsert_document);
    every fact carries origin="deterministic" + source_field, and every fact
    has exactly one support (this document)."""
    meta = rec.metadata
    doc_key = rec.question_id
    nodes: list[EntityRef] = []
    facts: list[GraphFact] = []
    supports: list[Support] = []

    doc_ref = EntityRef("Document", doc_key, doc_key)  # endpoint stub
    doc = document_ref(rec, voc)

    def add(fact: GraphFact, source_field: str) -> None:
        fact = GraphFact(rel=fact.rel, src=fact.src, dst=fact.dst,
                         origin="deterministic", source_field=source_field)
        facts.append(fact)
        supports.append(Support(doc_key=doc_key, fact_key=fact.fact_key,
                                evidence=None, origin="deterministic",
                                extracted_at=now))

    # ── Ministry (ADDRESSED_TO) ─────────────────────────────────────────
    m_n = normalize_ministry(voc, meta.ministry)
    if m_n.status == "canonical":
        entry = voc.match_ministry(meta.ministry)
        ministry = EntityRef(
            label="Ministry", key=m_n.canonical,
            name=_entry_label(voc, entry, meta.ministry),
            resolution="canonical", raw=m_n.raw,
            props={"label": entry.label if entry else None,
                   "sansad_code_ls": (entry.extra.get("sansad_codes") or {}).get("lok-sabha") if entry else None,
                   "sansad_code_rs": (entry.extra.get("sansad_codes") or {}).get("rajya-sabha") if entry else None},
        )
    elif m_n.status == "unknown":
        ministry = EntityRef(
            label="Ministry", key=unknown_key(m_n.raw), name=(m_n.raw or "").strip(),
            resolution="unresolved", raw=m_n.raw)
    else:
        ministry = None
    if ministry is not None:
        nodes.append(ministry)
        add(GraphFact("ADDRESSED_TO", doc_ref, ministry), "metadata.ministry")

    # ── Organization (PUBLISHED_BY) ─────────────────────────────────────
    # Identity rule: a controlled org slug that is ALSO a controlled concept
    # (e.g. org slug "incois" = concept INCOIS) shares the concept key
    # "c:<canonical>" — one node, one identity, so deterministic PUBLISHED_BY
    # edges and LLM MENTIONS edges meet at the same node. Slugs that match
    # no concept keep their vocabulary slug (e.g. "sansad", "moes_hq").
    o_n = normalize_org(voc, meta.org)
    org = None
    concept_canon = _concept_canonical_for(voc, meta.org)
    if o_n.status == "canonical":
        if concept_canon is not None:
            org = EntityRef(
                label="Organization", key=f"c:{concept_canon}",
                name=_concept_label(voc, concept_canon),
                resolution="canonical", raw=o_n.raw,
                props={"matched_concept": concept_canon})
        else:
            entry = voc.match_org(meta.org)
            org = EntityRef(
                label="Organization", key=o_n.canonical,
                name=_entry_label(voc, entry, meta.org),
                resolution="canonical", raw=o_n.raw)
    elif o_n.status == "unknown":
        org = EntityRef(
            label="Organization", key=unknown_key(o_n.raw),
            name=(o_n.raw or "").strip(), resolution="unresolved", raw=o_n.raw)
    if org is not None:
        nodes.append(org)
        add(GraphFact("PUBLISHED_BY", doc_ref, org), "metadata.org")

    # ── House (ASKED_IN) ────────────────────────────────────────────────
    # metadata.house wins; the doc-id prefix is a fallback ONLY for ids that
    # actually carry a known ls-/rs- prefix (incdoc-… rows have no house).
    if meta.house:
        h_n = normalize_house(voc, meta.house)
    else:
        h_n = normalize_house_from_doc_id(voc, doc_key)
        if h_n.status == "unknown":  # no recognized prefix (e.g. incdoc-…)
            h_n = None
    house = None
    house_canonical: Optional[str] = None
    if h_n is not None and h_n.status == "canonical":
        entry = voc.match_house(meta.house or "")
        house = EntityRef(
            label="House", key=h_n.canonical,
            name=_entry_label(voc, entry, meta.house or doc_key),
            resolution="canonical", raw=h_n.raw or doc_key)
        house_canonical = h_n.canonical
    elif h_n is not None and h_n.status == "unknown":
        raw_house = meta.house or doc_key
        house = EntityRef(
            label="House", key=unknown_key(raw_house), name=raw_house,
            resolution="unresolved", raw=raw_house)
    if house is not None:
        nodes.append(house)
        add(GraphFact("ASKED_IN", doc_ref, house),
            "metadata.house" if meta.house else "doc-id prefix")

    # ── Session (IN_SESSION) — needs a controlled house for the key ─────
    if meta.session is not None and house_canonical:
        sess = EntityRef(
            label="Session",
            key=key_session(house_canonical, meta.session),
            name=f"session {meta.session}",
            resolution="canonical",
            raw=f"session {meta.session}",
            props={"house": house_canonical, "number": meta.session},
        )
        nodes.append(sess)
        add(GraphFact("IN_SESSION", doc_ref, sess), "metadata.session")

    # ── Lok Sabha term (IN_LOK_SABHA_TERM) — from ls-<term>-… doc id ────
    term = ls_term_from_doc_id(doc_key)
    if term is not None:
        t_entry = voc.ls_term(term)
        if t_entry is not None:
            term_node = EntityRef(
                label="LokSabhaTerm", key=key_ls_term(term),
                name=str(t_entry.get("label") or f"{term}th Lok Sabha"),
                resolution="canonical", raw=f"{term}th Lok Sabha",
                props={"number": term, "house": "lok-sabha",
                       "year_span": t_entry.get("years")},
            )
        else:
            # open world: a term the vocabulary does not cover yet
            term_node = EntityRef(
                label="LokSabhaTerm", key=key_ls_term(term),
                name=f"{term}th Lok Sabha", resolution="unresolved",
                raw=f"{term}th Lok Sabha",
                props={"number": term, "house": "lok-sabha"},
            )
        nodes.append(term_node)
        add(GraphFact("IN_LOK_SABHA_TERM", doc_ref, term_node), "doc-id term")

    # ── Members (ASKED_BY) — one fact per clubbed-list token ────────────
    mem = normalize_member(voc, meta.member)
    for token in mem.tokens:
        member = EntityRef(
            label="Member",
            key=key_member(token.canonical_key, token.raw),
            name=token.raw,           # display the raw form (provenance)
            resolution="canonical" if mem.status == "canonical" else "unresolved",
            raw=token.raw,
            props={"canonical_key": token.canonical_key,
                   "honorific": token.honorific},
        )
        nodes.append(member)
        add(GraphFact("ASKED_BY", doc_ref, member), "metadata.member")

    # ── Year (DATED) — temporal facet ───────────────────────────────────
    if doc.date_year is not None:
        year_node = EntityRef(
            label="Year", key=key_year(doc.date_year),
            name=str(doc.date_year), resolution="canonical",
            raw=str(doc.date_year), props={"number": doc.date_year},
        )
        nodes.append(year_node)
        add(GraphFact("DATED", doc_ref, year_node), "metadata.date")

    return doc, GraphContribution(
        doc_key=doc_key, nodes=nodes, facts=facts, supports=supports)
