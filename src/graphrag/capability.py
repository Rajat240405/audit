"""GraphRAG capability — controlled graph access (Phase 2 / WS5).

Explicit graph mode: the user's query is resolved to deterministic anchors
(no LLM at query time, no automatic routing — spec §18/§19), documents are
pulled by direct and one expansion-hop graph edges, ranked, and returned as
the SHARED ``RetrievedResult`` contract (src/retrieval/result.py) so the
existing serving paths and the downstream answer/verification pipeline can
consume them unchanged (spec §17).

Graph evidence (spec §12/§17): every retrieved document carries the facts
that produced it (fact keys, relationship types, origins) and
``GraphCapability.explain`` returns full ``GraphEvidence`` records
(fact + supporting documents + evidence snippets + origin + timestamps) in a
shape Phase 3's centralized verification can consume.

Graceful degradation (WS0 freeze preserved): when the graph backend is not
configured or unreachable, ``build_graph_capability`` returns a disabled
capability that logs loudly and returns ``[]`` — the server's honest-empty
graph-mode behavior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.retrieval.result import RetrievedResult
from src.vocabulary import Vocabulary, load_vocabulary

from src.graphrag.config import GraphConfig, GraphConfigError, load_graph_config
from src.graphrag.models import FactView, NodeView
from src.graphrag.query import Anchor, GraphQuery
from src.graphrag.store import GraphStore, InMemoryGraphStore

__all__ = [
    "GraphCapability",
    "GraphEvidence",
    "DisabledGraphCapability",
    "build_graph_capability",
]

logger = logging.getLogger(__name__)

# Relationship directness → base score (deterministic metadata edges are the
# strongest signal; MENTIONS is LLM-derived; expanded hops decay)
_DET_REL_SCORE = 1.0
_MENTION_SCORE = 0.8
_EXPAND_SCORE = 0.5


@dataclass
class GraphEvidence:
    """One graph fact with full provenance (Phase-3 verification input)."""

    fact_key: str
    rel: str
    src: dict
    dst: dict
    origin: str
    doc_count: int
    supports: list[dict] = field(default_factory=list)
    first_seen_at: str = ""
    updated_at: str = ""
    source_field: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "fact_key": self.fact_key, "rel": self.rel, "src": self.src,
            "dst": self.dst, "origin": self.origin, "doc_count": self.doc_count,
            "supports": self.supports,
            "first_seen_at": self.first_seen_at, "updated_at": self.updated_at,
            "source_field": self.source_field,
        }


class DisabledGraphCapability:
    """No usable graph backend — retrieval returns [] (honest empty)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedResult]:
        return []

    def explain(self, results: list[RetrievedResult]) -> list[GraphEvidence]:
        return []

    def verify_answer(self, answer: str, results: list[RetrievedResult],
                      depth: str = "light", llm_client=None):
        """Honest-empty verification: no graph, so no graph evidence and no
        graph integrity claim — an EMPTY report, never a passing one."""
        from src.verification.contract import VerificationReport

        return VerificationReport(method=depth)

    def stats(self) -> dict:
        return {"disabled": True, "reason": self.reason}

    def close(self) -> None:
        pass


class GraphCapability:
    """Explicit graph-mode retrieval over any GraphStore backend."""

    def __init__(self, store: GraphStore, voc: Optional[Vocabulary] = None) -> None:
        self.store = store
        self.voc = voc or load_vocabulary()
        self._query = GraphQuery(store, self.voc)

    # ── retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedResult]:
        anchors = self._query.resolve_anchors(query, limit=8)
        if not anchors:
            return []
        candidates: dict[str, dict] = {}  # doc_key -> {score, via, anchors, facts}
        for anchor in anchors:
            self._collect(anchor, 1.0, candidates, depth=1)
        ordered = sorted(candidates.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
        out: list[RetrievedResult] = []
        for doc_key, info in ordered[: max(1, int(top_k))]:
            d = self.store.get_document(doc_key)
            if d is None:
                continue
            out.append(RetrievedResult(
                doc_id=doc_key,
                question=d.get("question_text") or "",
                answer=d.get("answer_text") or "",
                score=round(info["score"], 4),
                retrieval_method="graph_traversal",
                metadata={
                    "ministry": d.get("ministry_raw"),
                    "subject": d.get("subject_raw"),
                    "date": d.get("date"),
                    "question_type": d.get("question_type"),
                    "document_type": d.get("document_type"),
                    "source_url": d.get("source_url"),
                    "graph": {
                        "anchors": [a.to_dict() for a in info["anchors"]],
                        "via": info["via"],
                        "fact_keys": sorted(info["facts"]),
                    },
                },
            ))
        return out

    def _collect(self, anchor: Anchor, base: float,
                 out: dict[str, dict], *, depth: int,
                 hop_facts: frozenset | None = None) -> None:
        docs = self.store.documents_for_entity(anchor.key)
        for d in docs:
            doc_key = d["key"]
            score = base
            via = f"{anchor.key} (depth {depth})"
            # the document's own facts toward this anchor (both directions)
            doc_facts = {f.fact_key for f in self.store.facts_out(doc_key)
                         if f.dst_key == anchor.key} | \
                        {f.fact_key for f in self.store.facts_in(doc_key)
                         if f.src_key == anchor.key}
            if depth == 1 and doc_facts:
                # refine by edge type (deterministic > MENTIONS-only)
                rels = {f.rel for f in self.store.facts_out(doc_key)
                        if f.dst_key == anchor.key}
                if "MENTIONS" in rels and not (rels - {"MENTIONS"}):
                    score = _MENTION_SCORE
            self._merge(out, doc_key, score, anchor, via,
                        doc_facts | (hop_facts or frozenset()))
        # one expansion hop (depth-2) — e.g. Org→(Facility)→MENTIONS→Doc
        if depth == 1:
            for nbr in self.store.neighbors(anchor.key, depth=1):
                if nbr.label in ("Document", "Year", "Fact"):
                    continue
                hop = frozenset(
                    f.fact_key for f in self.store.facts_out(anchor.key)
                    if f.dst_key == nbr.key) | frozenset(
                    f.fact_key for f in self.store.facts_in(anchor.key)
                    if f.src_key == nbr.key)
                self._collect(Anchor(key=nbr.key, label=nbr.label, name=nbr.name,
                                     via="expand", match_length=0),
                              _EXPAND_SCORE, out, depth=2, hop_facts=hop)

    @staticmethod
    def _merge(out: dict[str, dict], doc_key: str, score: float,
               anchor: Anchor, via: str, fact_keys: frozenset | set) -> None:
        cur = out.get(doc_key)
        if cur is None:
            out[doc_key] = {"score": score, "via": via, "anchors": [anchor],
                            "facts": set(fact_keys)}
            return
        if score > cur["score"]:
            cur["score"] = score
            cur["via"] = via
        if all(a.key != anchor.key for a in cur["anchors"]):
            cur["anchors"].append(anchor)
        cur["facts"] |= set(fact_keys)

    # ── evidence ──────────────────────────────────────────────────────────

    def explain(self, results: list[RetrievedResult]) -> list[GraphEvidence]:
        """Full provenance for every fact cited by the given results."""
        seen: set[str] = set()
        out: list[GraphEvidence] = []
        for r in results:
            for fk in r.metadata.get("graph", {}).get("fact_keys", []):
                if fk in seen:
                    continue
                seen.add(fk)
                ev = self.evidence_for_fact(fk)
                if ev is not None:
                    out.append(ev)
        return out

    def evidence_for_fact(self, fact_key: str) -> Optional[GraphEvidence]:
        prov = self.store.provenance(fact_key)
        if prov is None:
            return None
        f: FactView = prov["fact"]
        return GraphEvidence(
            fact_key=f.fact_key, rel=f.rel,
            src=self._node_dict(f.src_key), dst=self._node_dict(f.dst_key),
            origin=f.origin, doc_count=f.doc_count,
            supports=prov["supports"],
            first_seen_at=f.first_seen_at, updated_at=f.updated_at,
            source_field=f.source_field,
        )

    # ── verification (WS3: delegates to the ONE authority) ────────────────

    def verify_answer(self, answer: str, results: list[RetrievedResult],
                      depth: str = "light", llm_client=None):
        """Verify an answer against THIS capability's graph evidence.

        Thin entry point only: the graph results are normalized into the
        shared verification contract (``graph_results`` resolves each cited
        fact's Document→SUPPORTS→Fact provenance through the store) and handed
        to the central ``VerificationAuthority``. GraphRAG contains NO
        verification logic of its own.

        ``llm_client`` is injected (the caller passes the client built by the
        existing generation architecture); ``depth="light"`` makes zero LLM
        calls. No provider or model is referenced here.
        """
        from src.verification.authority import VerificationAuthority
        from src.verification.sources import graph_results

        evidence = graph_results(results, capability=self)
        return VerificationAuthority(llm_client=llm_client).verify(
            evidence, answer, depth=depth)

    def _node_dict(self, key: str) -> dict:
        n: Optional[NodeView] = self.store.get_node(key)
        if n is None:
            d = self.store.get_document(key)
            return {"key": key, "label": "Document",
                    "name": key, "resolution": "canonical",
                    **{k: v for k, v in (d or {}).items() if k != "key"}}
        return n.to_dict()

    # ── passthroughs ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        return self.store.stats()

    def close(self) -> None:
        self.store.close()


# ── factory (server + CLI + tests) ─────────────────────────────────────────


def build_graph_capability(
    config: Optional[GraphConfig] = None,
    log: logging.Logger = logger,
) -> GraphCapability | DisabledGraphCapability:
    """Build the graph capability from configuration.

    Never raises for deployment-readiness reasons: missing config or an
    unreachable Neo4j yields a DisabledGraphCapability (loud log line,
    honest-empty retrieval) so the serving path degrades gracefully exactly
    as the WS0-frozen graph mode does."""
    try:
        cfg = config or load_graph_config()
    except GraphConfigError as e:
        log.warning("[graph] capability disabled: %s", e)
        return DisabledGraphCapability(str(e))

    if cfg.backend == "inmemory":
        log.warning("[graph] inmemory backend — local validation only, "
                    "no persistence")
        return GraphCapability(InMemoryGraphStore())

    try:
        from src.graphrag.neo4j_store import Neo4jGraphStore  # lazy: driver

        store = Neo4jGraphStore(cfg)
        if not store.ping():
            store.close()
            reason = f"Neo4j unreachable at {cfg.neo4j_uri}"
            log.warning("[graph] capability disabled: %s", reason)
            return DisabledGraphCapability(reason)
        return GraphCapability(store)
    except Exception as e:  # noqa: BLE001 — degraded mode is the contract
        log.warning("[graph] capability disabled: %s: %s",
                    type(e).__name__, e)
        return DisabledGraphCapability(f"{type(e).__name__}: {e}")

