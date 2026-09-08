"""Canonical graph schema (Phase 2 / WS2).

Node identities, relationship catalog, and the Neo4j DDL (constraints +
indexes) as data. This module is backend-agnostic: the InMemory store uses
the catalogs for validation; Neo4jGraphStore compiles the DDL.

IDENTITY STRATEGY (spec §11) — display names are NEVER the identity:

  Document      question_id (corpus stable id; content-derived for incdoc)
  Ministry      vocabulary canonical slug (e.g. "earth-sciences");
                unknowns: "u:" + sha1(folded raw)[:16]
  Organization  vocabulary org slug (sansad/moes_hq/incois); known concepts:
                "c:" + concept canonical; unknowns: "u:" hash
  House         vocabulary canonical (lok-sabha / rajya-sabha)
  LokSabhaTerm  the term number itself ("ls-term-18")
  Session       "<house>:<n>" (LS and RS session numbers share a label but
                are disambiguated by house in the key)
  Member        "m:" + Phase-1 canonical key (casefold + honorific strip);
                unknowns: "u:" hash
  Programme     "c:" + concept canonical (e.g. "c:Deep Ocean Mission");
                unknowns: "u:" hash
  Facility      "u:" hash (no controlled facility vocabulary in the corpus
                yet — exact-surface identity only, safe by construction)
  Place         "u:" hash (same rationale as Facility)
  Year          "year-<n>"
  Fact          fact_key = "<REL>:<src_key>-><dst_key>" (the provenance hub)

OPEN-WORLD (spec §4): every unknown raw value is preserved under its stable
"u:" key with resolution="unresolved" — never rejected, never fuzzy-merged.
"""
from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "NODE_LABELS",
    "DETERMINISTIC_RELS",
    "SEMANTIC_RELS",
    "ALL_RELS",
    "SEMANTIC_ENDPOINTS",
    "DocumentNode",
    "schema_ddl",
]

# Node labels present in the production schema (Fact = provenance hub).
NODE_LABELS: tuple[str, ...] = (
    "Document",
    "Ministry",
    "Organization",
    "House",
    "LokSabhaTerm",
    "Session",
    "Member",
    "Programme",
    "Facility",
    "Place",
    "Year",
    "Fact",
)

# Deterministic relationships — derived from structured metadata, no LLM.
# (rel, src_label, dst_label)
DETERMINISTIC_RELS: tuple[tuple[str, str, str], ...] = (
    ("ADDRESSED_TO", "Document", "Ministry"),
    ("PUBLISHED_BY", "Document", "Organization"),
    ("ASKED_IN", "Document", "House"),
    ("IN_SESSION", "Document", "Session"),
    ("IN_LOK_SABHA_TERM", "Document", "LokSabhaTerm"),
    ("ASKED_BY", "Document", "Member"),
    ("DATED", "Document", "Year"),
    ("OF_HOUSE", "LokSabhaTerm", "House"),
    ("OF_HOUSE", "Session", "House"),
)

# Semantic relationships — LLM-extracted, grounded, validated.
SEMANTIC_RELS: tuple[tuple[str, str, str], ...] = (
    ("MENTIONS", "Document", "Programme"),
    ("MENTIONS", "Document", "Organization"),
    ("MENTIONS", "Document", "Facility"),
    ("MENTIONS", "Document", "Place"),
    ("OPERATED_BY", "Programme", "Organization"),
    ("OPERATED_BY", "Facility", "Organization"),
    ("LOCATED_IN", "Facility", "Place"),
    ("LOCATED_IN", "Organization", "Place"),
    ("PART_OF", "Programme", "Programme"),
    ("PART_OF", "Programme", "Organization"),
    ("FUNDED_BY", "Programme", "Organization"),
    ("FUNDED_BY", "Facility", "Organization"),
    ("COLLABORATES_WITH", "Organization", "Organization"),
)

ALL_RELS: frozenset[str] = frozenset({r for r, _, _ in DETERMINISTIC_RELS + SEMANTIC_RELS})

# Labels an LLM-extracted entity may use (persons are deterministic from
# metadata — LLM person extraction is deliberately out of scope).
SEMANTIC_ENDPOINTS: frozenset[str] = frozenset(
    {"Programme", "Organization", "Facility", "Place"}
)

# Labels with a `name` property that gets a non-unique lookup index.
_NAME_INDEXED_LABELS: tuple[str, ...] = (
    "Ministry", "Organization", "House", "LokSabhaTerm", "Session",
    "Member", "Programme", "Facility", "Place", "Year",
)


def _safe_ident(name: str) -> str:
    """Sanitize a label into a safe Cypher identifier (validation only —
    labels come from this module's constants, never from data)."""
    return re.sub(r"[^A-Za-z0-9_]", "", name)


def schema_ddl() -> list[str]:
    """Idempotent Cypher DDL: UNIQUE constraints on every node key + Fact key,
    plus lookup indexes. Order is stable (deterministic output)."""
    stmts: list[str] = []
    # 1) uniqueness constraints — one per label (key is the identity)
    for label in NODE_LABELS:
        ident = _safe_ident(label).lower()
        stmts.append(
            f"CREATE CONSTRAINT {ident}_key_unique IF NOT EXISTS "
            f"FOR (n:`{label}`) REQUIRE n.key IS UNIQUE"
        )
    # 2) lookup indexes
    stmts.append(
        "CREATE INDEX document_year IF NOT EXISTS FOR (d:Document) ON (d.date_year)"
    )
    stmts.append(
        "CREATE INDEX document_ls_term IF NOT EXISTS FOR (d:Document) ON (d.ls_term)"
    )
    stmts.append(
        "CREATE INDEX document_source_url IF NOT EXISTS FOR (d:Document) ON (d.source_url)"
    )
    for label in _NAME_INDEXED_LABELS:
        ident = _safe_ident(label).lower()
        stmts.append(
            f"CREATE INDEX {ident}_name IF NOT EXISTS "
            f"FOR (n:`{label}`) ON (n.name)"
        )
    # 3) relationship-property index (B3) — PERFORMANCE, not correctness.
    # _apply_contribution_tx probes existing supports with
    #   MATCH (d:Document {key:$doc})-[s:SUPPORTS]->(f:Fact) WHERE f.key IN $fks
    # and SUPPORTS carries fact_key. Without this index the planner has no
    # relationship-property lookup and falls back to expanding/scanning
    # SUPPORTS edges, which grow to ~documents x facts-per-document at the full
    # 2,648-document build. Verified valid 5.26.4 DDL; IF NOT EXISTS keeps the
    # whole schema_ddl() idempotent.
    stmts.append(
        "CREATE INDEX supports_fact_key IF NOT EXISTS "
        "FOR ()-[s:SUPPORTS]-() ON (s.fact_key)"
    )
    return stmts


class DocumentNode:
    """Tiny namespace for Document identity helpers."""

    @staticmethod
    def key(doc_id: str) -> str:
        return str(doc_id)


# ── key constructors (single source for every key shape in the codebase) ──

from src.graphrag.models import unknown_key  # noqa: E402  (after use in docs)


def key_ministry(canonical: Optional[str], raw: Optional[str]) -> str:
    return canonical or (unknown_key(raw) if raw else "")


def key_org(slug: Optional[str], raw: Optional[str]) -> str:
    return slug or (unknown_key(raw) if raw else "")


def key_concept(canonical: str) -> str:
    return f"c:{canonical}"


def key_house(canonical: str) -> str:
    return canonical


def key_ls_term(term: int) -> str:
    return f"ls-term-{int(term)}"


def key_session(house: str, n: int) -> str:
    return f"{house}:{int(n)}"


def key_member(canonical_key: Optional[str], raw: Optional[str]) -> str:
    return (f"m:{canonical_key}" if canonical_key
            else (unknown_key(raw) if raw else ""))


def key_year(n: int) -> str:
    return f"year-{int(n)}"


def fact_key(rel: str, src_key: str, dst_key: str) -> str:
    return f"{rel}:{src_key}->{dst_key}"
