"""GraphStore abstraction (Phase 2 / WS2).

Application code talks to this ABC — never to raw Cypher. Two implementations:

  * ``InMemoryGraphStore`` — pure Python, no dependencies. Serves as the unit
    test double AND the local-validation backend (``GRAPHRAG_BACKEND=inmemory``).
    It implements the SAME semantics as the Neo4j store, which is what makes
    the test suite Neo4j-free (spec §24).
  * ``Neo4jGraphStore`` (neo4j_store.py) — production backend, same ABC.

Semantics both backends must honor:
  * upserts are idempotent (MERGE on node key / fact key / (fact, doc))
  * a document's contribution is document-SCOPED: ``withdraw_document``
    removes exactly that document's supports, decrements fact doc_counts, and
    deletes relationships/facts that lose their last supporting document —
    facts still supported by other documents survive
  * ``apply_contribution`` is one atomic unit (single transaction on Neo4j)
  * timestamps are injected by the caller (``now``) — deterministic tests
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from src.graphrag.models import (
    DocumentRef,
    EntityRef,
    FactView,
    GraphContribution,
    NodeView,
    Support,
)

__all__ = ["GraphStore", "InMemoryGraphStore", "NEIGHBORS_LIMIT"]

# Default cap on neighbour expansion. Hub entities can otherwise return
# thousands of nodes, and callers issue further per-neighbour queries. Sized
# in line with the existing find_nodes(limit=200) contract; callers that need
# fewer pass an explicit limit.
NEIGHBORS_LIMIT = 200


class GraphStore(ABC):
    """Backend-agnostic graph store (see module docstring for semantics)."""

    # ── lifecycle ─────────────────────────────────────────────────────────
    @abstractmethod
    def init_schema(self) -> None:
        """Create constraints/indexes (idempotent)."""

    @abstractmethod
    def close(self) -> None:
        """Release driver/session resources (no-op where inapplicable)."""

    # ── writes (idempotent) ───────────────────────────────────────────────
    @abstractmethod
    def upsert_document(self, doc: DocumentRef) -> None:
        """MERGE the Document node (identity = doc.doc_key = question_id).

        The dataclass field is ``doc_key``; ``DocumentRef.to_node()`` maps it to
        the Neo4j node property ``key``. Backends must read ``doc.doc_key`` and
        write it as the ``key`` property — do not expect a ``.key`` attribute.
        """

    @abstractmethod
    def apply_contribution(self, contrib: GraphContribution, *, now: str,
                           doc: Optional[DocumentRef] = None) -> dict:
        """Apply one document's nodes+facts+supports atomically.

        ``doc`` (optional) upserts the Document node in the SAME transaction
        (one atomic unit per document). Returns counters {"nodes": n,
        "facts": n, "supports": n, "new_supports": n} (new_supports = facts
        whose doc_count grew)."""

    @abstractmethod
    def withdraw_document(self, doc_key: str, *, now: str) -> dict:
        """Retract ONE document's contribution (document-scoped).

        Removes the document's SUPPORTS edges, decrements each affected fact's
        doc_count, deletes relationships + Fact nodes that reach 0 (their last
        source is gone), and removes the Document node. Facts supported by
        other documents are untouched. Returns counters."""

    # ── reads ─────────────────────────────────────────────────────────────
    @abstractmethod
    def get_document(self, doc_key: str) -> Optional[dict]:
        """Document node properties (or None)."""

    @abstractmethod
    def list_document_keys(self) -> set[str]:
        """All Document keys present in the store."""

    @abstractmethod
    def get_node(self, key: str) -> Optional[NodeView]:
        """Node by identity key (any label) or None."""

    @abstractmethod
    def find_nodes(
        self,
        *,
        label: Optional[str] = None,
        name_contains: Optional[str] = None,
        resolution: Optional[str] = None,
        limit: int = 200,
    ) -> list[NodeView]:
        """Entity search by label / name substring / resolution (deterministic
        ordering: by name)."""

    @abstractmethod
    def facts_out(self, key: str, rel: Optional[str] = None) -> list[FactView]:
        """Relationships FROM the node (optionally filtered by type)."""

    @abstractmethod
    def facts_in(self, key: str, rel: Optional[str] = None) -> list[FactView]:
        """Relationships TO the node (optionally filtered by type)."""

    @abstractmethod
    def neighbors(self, key: str, *, depth: int = 1, rel: Optional[str] = None,
                  labels: Optional[Iterable[str]] = None,
                  limit: int = NEIGHBORS_LIMIT) -> list[NodeView]:
        """BFS expansion over relationship endpoints (undirected by default;
        ``depth>=1``; deduplicated; deterministic order by (depth, key)).

        At most ``limit`` nodes are returned, truncating the deterministic
        (depth, key) ordering — nearest neighbours are kept first."""

    @abstractmethod
    def documents_for_entity(
        self,
        entity_key: str,
        *,
        rel: Optional[str] = None,
        year: Optional[int] = None,
        ls_term: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Documents linked to the entity (direct deterministic or MENTIONS
        edge), optionally filtered by temporal facets. Returns document
        property dicts (deterministic order: by doc key)."""

    @abstractmethod
    def provenance(self, fact_key: str) -> Optional[dict]:
        """Fact + all its supports: {fact: FactView, supports: [{doc_key,
        evidence, origin, extracted_at, document: {…props}}]} or None."""

    @abstractmethod
    def stats(self) -> dict:
        """{labels: {label: count}, relationships: {rel: count},
        facts: n, supports: n, documents: n}."""


# ─────────────────────────────────────────────────────────────────────────────
# In-memory reference implementation (test double + local validation)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryGraphStore(GraphStore):
    """Deterministic pure-Python store implementing the full ABC.

    Data layout:
      _nodes:     key -> {"label", "name", "resolution", "raw", **props}
      _docs:      doc_key -> Document properties
      _facts:     fact_key -> {rel, src_key, dst_key, origin, source_field,
                               sample_evidence, first_seen_at, updated_at,
                               doc_count}
      _supports:  fact_key -> {doc_key -> Support}
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._docs: dict[str, dict] = {}
        self._facts: dict[str, dict] = {}
        self._supports: dict[str, dict[str, Support]] = {}
        self._schema_ready = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        self._schema_ready = True  # identity enforced in code (dict keys)

    def close(self) -> None:
        pass

    # ── writes ────────────────────────────────────────────────────────────

    def upsert_document(self, doc: DocumentRef) -> None:
        props = doc.to_node()
        self._docs[doc.doc_key] = props
        # Documents are nodes for traversal purposes too:
        node = self._nodes.setdefault(doc.doc_key, {
            "label": "Document", "name": doc.doc_key,
            "resolution": "canonical", "raw": doc.doc_key,
        })
        node["label"] = "Document"

    def _sync_doc_node(self, doc_key: str) -> None:
        """Keep the traversal-node view of a document in step with _docs."""
        if doc_key in self._nodes:
            self._nodes[doc_key]["label"] = "Document"

    def _upsert_node(self, ref: EntityRef) -> bool:
        """MERGE semantics: create if absent; if present, promotion-safe
        updates (canonical resolution wins over unresolved; name/raw kept
        from first writer unless absent). Returns True if created."""
        existing = self._nodes.get(ref.key)
        if existing is None:
            node = ref.to_node()
            node["label"] = ref.label
            self._nodes[ref.key] = node
            return True
        if existing["resolution"] == "unresolved" and ref.resolution == "canonical":
            existing["resolution"] = "canonical"
            existing["name"] = ref.name
            existing["raw"] = ref.raw or ref.name
        return False

    def apply_contribution(self, contrib: GraphContribution, *, now: str,
                           doc: Optional[DocumentRef] = None) -> dict:
        if doc is not None:
            self.upsert_document(doc)
        created = 0
        for ref in contrib.nodes:
            if self._upsert_node(ref):
                created += 1
        new_supports = 0
        for fact in contrib.facts:
            fk = fact.fact_key
            entry = self._facts.get(fk)
            if entry is None:
                self._facts[fk] = {
                    "fact_key": fk, "rel": fact.rel,
                    "src_key": fact.src.key, "dst_key": fact.dst.key,
                    "origin": fact.origin, "source_field": fact.source_field,
                    "sample_evidence": None, "first_seen_at": now,
                    "updated_at": now, "doc_count": 0,
                }
                entry = self._facts[fk]
            entry["updated_at"] = now
        for s in contrib.supports:
            entry = self._facts.get(s.fact_key)
            if entry is None:
                raise ValueError(
                    f"support for unknown fact {s.fact_key!r} in contribution "
                    f"{contrib.doc_key!r} — the facts list must cover all supports"
                )
            per_doc = self._supports.setdefault(s.fact_key, {})
            if s.doc_key not in per_doc:
                entry["doc_count"] += 1
                new_supports += 1
            per_doc[s.doc_key] = s  # (doc, fact) merge: the doc's own slot
            if s.evidence:
                entry["sample_evidence"] = s.evidence
        return {
            "nodes": created,
            "facts": len(contrib.facts),
            "supports": len(contrib.supports),
            "new_supports": new_supports,
        }

    def withdraw_document(self, doc_key: str, *, now: str) -> dict:
        removed_supports = 0
        deleted_facts = 0
        for fk, per_doc in list(self._supports.items()):
            if doc_key in per_doc:
                del per_doc[doc_key]
                removed_supports += 1
                entry = self._facts[fk]
                entry["doc_count"] = max(0, entry["doc_count"] - 1)
                entry["updated_at"] = now
                if entry["doc_count"] == 0:
                    del self._facts[fk]
                    deleted_facts += 1
        # drop empty per-doc maps
        self._supports = {fk: m for fk, m in self._supports.items() if m}
        doc_existed = doc_key in self._docs
        self._docs.pop(doc_key, None)
        self._nodes.pop(doc_key, None)  # traversal node view too
        return {
            "supports_removed": removed_supports,
            "facts_deleted": deleted_facts,
            "document_removed": doc_existed,
        }

    # ── reads ─────────────────────────────────────────────────────────────

    def get_document(self, doc_key: str) -> Optional[dict]:
        d = self._docs.get(doc_key)
        return dict(d) if d else None

    def list_document_keys(self) -> set[str]:
        return set(self._docs)

    def get_node(self, key: str) -> Optional[NodeView]:
        n = self._nodes.get(key)
        if n is None:
            return None
        if n["label"] == "Document" and key in self._docs:
            # surface full document properties in the node view
            doc = dict(self._docs[key])
            doc.pop("key", None)
            props = {k: v for k, v in doc.items()
                     if k not in ("name", "resolution", "raw")}
            return NodeView(key=key, label="Document",
                            name=doc.get("key", key), resolution="canonical",
                            props=props)
        return self._node_view(key, n)

    def _node_view(self, key: str, n: dict) -> NodeView:
        props = {k: v for k, v in n.items() if k not in ("label", "key", "name", "resolution")}
        return NodeView(key=key, label=n["label"], name=n["name"],
                        resolution=n["resolution"], props=props,
                        raw=n.get("raw"))

    def find_nodes(self, *, label=None, name_contains=None,
                   resolution=None, limit=200) -> list[NodeView]:
        out = []
        needle = name_contains.casefold() if name_contains else None
        for key in sorted(self._nodes, key=lambda k: (
                self._nodes[k]["name"].casefold(), k)):
            n = self._nodes[key]
            if label and n["label"] != label:
                continue
            if resolution and n["resolution"] != resolution:
                continue
            if needle and needle not in n["name"].casefold():
                continue
            out.append(self._node_view(key, n))
            if len(out) >= limit:
                break
        return out

    def _facts_where(self, key: str, rel: Optional[str], direction: str) -> list[FactView]:
        out = []
        for fk, f in self._facts.items():
            if rel and f["rel"] != rel:
                continue
            if direction == "out" and f["src_key"] == key:
                out.append(self._fact_view(f))
            elif direction == "in" and f["dst_key"] == key:
                out.append(self._fact_view(f))
        return sorted(out, key=lambda v: v.fact_key)

    def _fact_view(self, f: dict) -> FactView:
        return FactView(
            fact_key=f["fact_key"], rel=f["rel"], src_key=f["src_key"],
            dst_key=f["dst_key"], origin=f["origin"], doc_count=f["doc_count"],
            sample_evidence=f["sample_evidence"], first_seen_at=f["first_seen_at"],
            updated_at=f["updated_at"], source_field=f.get("source_field"),
        )

    def facts_out(self, key: str, rel: Optional[str] = None) -> list[FactView]:
        return self._facts_where(key, rel, "out")

    def facts_in(self, key: str, rel: Optional[str] = None) -> list[FactView]:
        return self._facts_where(key, rel, "in")

    def neighbors(self, key: str, *, depth=1, rel=None, labels=None,
                  limit=NEIGHBORS_LIMIT) -> list[NodeView]:
        """BFS over relationship endpoints. ``labels`` filters the RESULT
        (traversal passes through all node types); ``rel`` filters edges.

        Bounded by ``limit`` for backend parity with Neo4j (same truncation of
        the same deterministic ordering)."""
        limit = max(1, int(limit))
        label_set = set(labels) if labels else None
        seen: set[str] = {key}
        depth_of: dict[str, int] = {}
        frontier = {key}
        for d in range(1, max(1, int(depth)) + 1):
            nxt: set[str] = set()
            for k in frontier:
                for f in (self._facts_where(k, rel, "out")
                          + self._facts_where(k, rel, "in")):
                    for endpoint in (f.src_key, f.dst_key):
                        if endpoint in seen or endpoint not in self._nodes:
                            continue
                        seen.add(endpoint)
                        depth_of[endpoint] = d
                        nxt.add(endpoint)
            frontier = nxt
        out = []
        for k, d in depth_of.items():
            n = self._nodes[k]
            if label_set and n["label"] not in label_set:
                continue
            out.append(self._node_view(k, n))
        return sorted(out, key=lambda v: (depth_of[v.key], v.key))[:limit]

    def documents_for_entity(self, entity_key, *, rel=None, year=None,
                             ls_term=None, limit=100) -> list[dict]:
        # direct edges: Document-[:rel]->Entity  or  Entity-[:rel]->Document
        doc_keys: set[str] = set()
        for f in self._facts.values():
            if rel and f["rel"] != rel:
                continue
            if f["dst_key"] == entity_key and self._is_doc(f["src_key"]):
                doc_keys.add(f["src_key"])
            elif f["src_key"] == entity_key and self._is_doc(f["dst_key"]):
                doc_keys.add(f["dst_key"])
        out = []
        for dk in sorted(doc_keys):
            d = self._docs.get(dk)
            if d is None:
                continue
            if year is not None and d.get("date_year") != year:
                continue
            if ls_term is not None and d.get("ls_term") != ls_term:
                continue
            out.append(dict(d))
            if len(out) >= limit:
                break
        return out

    def _is_doc(self, key: str) -> bool:
        return key in self._docs

    def provenance(self, fact_key: str) -> Optional[dict]:
        f = self._facts.get(fact_key)
        if f is None:
            return None
        supports = []
        for dk in sorted(self._supports.get(fact_key, {})):
            s = self._supports[fact_key][dk]
            supports.append({
                "doc_key": dk,
                "evidence": s.evidence,
                "origin": s.origin,
                "extracted_at": s.extracted_at,
                "document": self.get_document(dk),
            })
        return {"fact": self._fact_view(f), "supports": supports}

    def stats(self) -> dict:
        labels: dict[str, int] = {}
        for n in self._nodes.values():
            labels[n["label"]] = labels.get(n["label"], 0) + 1
        rels: dict[str, int] = {}
        for f in self._facts.values():
            rels[f["rel"]] = rels.get(f["rel"], 0) + 1
        n_supports = sum(len(m) for m in self._supports.values())
        return {
            "labels": dict(sorted(labels.items())),
            "relationships": dict(sorted(rels.items())),
            "facts": len(self._facts),
            "supports": n_supports,
            "documents": len(self._docs),
        }

    # ── test helpers ──────────────────────────────────────────────────────

    def reset(self) -> None:
        self._nodes.clear()
        self._docs.clear()
        self._facts.clear()
        self._supports.clear()


def _doc_from_contribution(contrib: GraphContribution) -> DocumentRef:  # pragma: no cover
    raise NotImplementedError("document upsert is explicit (upsert_document)")
