"""Graph query layer (Phase 2 / WS2-F).

A thin, backend-agnostic API over the GraphStore ABC: entity lookup,
one-hop / multi-hop traversal, temporal/facet filtering, provenance
retrieval — plus DETERMINISTIC anchor resolution from free-text queries
(the WS5 graph-mode entry point). No LLM at query time (spec §8: the LLM is
for build-time extraction only), no automatic routing (spec §19).

Anchor resolution rules (all deterministic, corpus-justified):
  1. Phase-1 controlled concepts (surface forms, word-boundary, casefold)
     → nodes keyed "c:<canonical>" (if present in the graph)
  2. controlled org / ministry / house canonical names (exact folded
     match on word boundaries) → their node keys
  3. entity-name matching: the graph's own entity names (substring,
     casefold) — lets LLM-built entities (facilities, places) be found by
     the names the corpus uses
Results are ranked: controlled anchors first, then name matches, by
specificity (longer match = better) then name order (deterministic).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from src.vocabulary import Vocabulary
from src.vocabulary.normalize import concept_mentions

from src.graphrag.models import FactView, NodeView, fold_surface
from src.graphrag.store import GraphStore

__all__ = ["GraphQuery", "Anchor"]


@dataclass(frozen=True)
class Anchor:
    """A resolved query anchor: a graph node the query points at."""

    key: str
    label: str
    name: str
    via: str          # "concept" | "ministry" | "org" | "house" | "member" | "name-match"
    match_length: int

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "name": self.name,
                "via": self.via}


_WORD = r"[a-z0-9][a-z0-9'\-\.]*"


def _boundary_pattern(surface: str) -> re.Pattern:
    """Word-boundary casefold pattern for a controlled surface form."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(fold_surface(surface))}(?![a-z0-9])")


class GraphQuery:
    """Query API over any GraphStore implementation."""

    def __init__(self, store: GraphStore, voc: Vocabulary) -> None:
        self.store = store
        self.voc = voc

    # ── entity lookup ─────────────────────────────────────────────────────

    def lookup_entity(self, name: str, *, label: Optional[str] = None,
                      limit: int = 20) -> list[NodeView]:
        """Exact (folded) then substring entity search."""
        needle = fold_surface(name)
        exact = [n for n in self.store.find_nodes(label=label, limit=1000)
                 if fold_surface(n.name) == needle]
        if exact:
            return exact[:limit]
        return self.store.find_nodes(label=label, name_contains=name, limit=limit)

    def get_node(self, key: str) -> Optional[NodeView]:
        return self.store.get_node(key)

    # ── traversal ─────────────────────────────────────────────────────────

    def one_hop(self, key: str, *, rel: Optional[str] = None,
                labels: Optional[Iterable[str]] = None) -> list[NodeView]:
        return self.store.neighbors(key, depth=1, rel=rel, labels=labels)

    def multi_hop(self, key: str, depth: int, *, rel: Optional[str] = None,
                  labels: Optional[Iterable[str]] = None) -> list[NodeView]:
        return self.store.neighbors(key, depth=depth, rel=rel, labels=labels)

    def facts(self, key: str, *, rel: Optional[str] = None,
              direction: str = "out") -> list[FactView]:
        if direction == "in":
            return self.store.facts_in(key, rel=rel)
        return self.store.facts_out(key, rel=rel)

    # ── documents (temporal / facet filters) ──────────────────────────────

    def documents(self, entity_key: str, *, rel: Optional[str] = None,
                  year: Optional[int] = None, ls_term: Optional[int] = None,
                  limit: int = 100) -> list[dict]:
        return self.store.documents_for_entity(
            entity_key, rel=rel, year=year, ls_term=ls_term, limit=limit)

    # ── provenance ────────────────────────────────────────────────────────

    def provenance(self, fact_key: str) -> Optional[dict]:
        return self.store.provenance(fact_key)

    def stats(self) -> dict:
        return self.store.stats()

    # ── anchor resolution from free text ──────────────────────────────────

    def resolve_anchors(self, text: str, *, limit: int = 8) -> list[Anchor]:
        """Deterministic query → graph-node anchors (see module docstring)."""
        anchors: list[Anchor] = []
        seen: set[str] = set()
        folded = fold_surface(text)

        def add(key: str, label: str, name: str, via: str, length: int) -> None:
            if key in seen:
                return
            node = self.store.get_node(key)
            if node is None:
                return  # not in the graph — never invent anchors
            seen.add(key)
            anchors.append(Anchor(key=key, label=label, name=node.name,
                                  via=via, match_length=length))

        # 1) controlled concepts (longest surface wins span claims)
        for canon, _count in concept_mentions(self.voc, text).items():
            node = self.store.get_node(f"c:{canon}")
            if node is not None:
                add(f"c:{canon}", node.label, canon, "concept", len(canon))

        # 2) controlled org / ministry / house canonical surfaces
        pairs = ([ (e, "org") for e in self.voc.org ]
                 + [(e, "ministry") for e in self.voc.ministry]
                 + [(e, "house") for e in self.voc.house])
        for entry, via in pairs:
            for surface in entry.surface_forms():
                if _boundary_pattern(surface).search(folded):
                    add(entry.canonical, _label_for(via), entry.label, via,
                        len(surface))
                    break

        # 3) graph entity names (LLM-built facilities/places/orgs/people)
        for word in re.findall(_WORD, folded):
            if len(word) < 4:
                continue
            for node in self.store.find_nodes(name_contains=word, limit=50):
                if node.label in ("Document", "Year", "Fact"):
                    continue
                add(node.key, node.label, node.name, "name-match", len(word))

        # controlled first, then by specificity, then by name (deterministic)
        rank = {"concept": 0, "org": 0, "ministry": 0, "house": 0, "member": 0,
                "name-match": 1}
        anchors.sort(key=lambda a: (rank.get(a.via, 2), -a.match_length, a.name))
        return anchors[:limit]


def _label_for(via: str) -> str:
    return {"org": "Organization", "ministry": "Ministry", "house": "House",
            "member": "Member"}.get(via, "Organization")
