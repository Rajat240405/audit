"""
GraphRAG query path.

Combines:
1. Entity expansion — extract entities from the user query (via the same
   grounded extractor) and pull documents connected to those entities.
2. Vector search — embed the query with bge-m3 and use the Neo4j vector index.

Results are de-duplicated and ranked. This is a standalone GraphRAG retrieval
path (separate from Hybrid RAG); the two will be fused later.
"""

from __future__ import annotations

from typing import Optional

from src.graphrag.config import GraphRAGConfig
from src.graphrag.embeddings import GraphEmbedder
from src.graphrag.extractor import EntityRelationshipExtractor
from src.graphrag.neo4j_client import Neo4jGraphStore


class GraphRAGQueryResult:
    """One retrieved document with its graph-derived score."""

    def __init__(
        self,
        doc_id: str,
        subject: Optional[str],
        ministry: Optional[str],
        date: Optional[str],
        score: float,
        matched_entities: list[str],
        via: str,
    ) -> None:
        self.doc_id = doc_id
        self.subject = subject
        self.ministry = ministry
        self.date = date
        self.score = score
        self.matched_entities = matched_entities
        self.via = via

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "subject": self.subject,
            "ministry": self.ministry,
            "date": self.date,
            "score": round(self.score, 4),
            "matched_entities": self.matched_entities,
            "via": self.via,
        }


class GraphRAGQuerier:
    """Graph-aware retrieval over the Neo4j GraphRAG store."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.store = Neo4jGraphStore(config)
        self.embedder = GraphEmbedder(config)
        self.extractor = EntityRelationshipExtractor(config)

    def query(self, query_text: str, top_k: int = 10) -> list[GraphRAGQueryResult]:
        if not self.store.ping():
            raise RuntimeError("Neo4j is not reachable — cannot query the graph.")

        scored: dict[str, GraphRAGQueryResult] = {}

        # ── 1. Entity expansion ─────────────────────────────────────────
        try:
            entities, _ = self.extractor.extract(
                self._query_as_document(query_text)
            )
            for ent in entities:
                docs = self.store.documents_by_entity(ent.label, ent.name, limit=top_k)
                for d in docs:
                    res = scored.get(d["id"])
                    if res is None:
                        res = GraphRAGQueryResult(
                            doc_id=d["id"],
                            subject=d.get("subject"),
                            ministry=d.get("ministry"),
                            date=d.get("date"),
                            score=0.0,
                            matched_entities=[ent.name],
                            via="entity",
                        )
                        scored[d["id"]] = res
                    else:
                        if ent.name not in res.matched_entities:
                            res.matched_entities.append(ent.name)
                        res.score += 1.0
        except Exception:  # noqa: BLE001 - entity extraction must not break querying
            pass

        # ── 2. Vector search ────────────────────────────────────────────
        try:
            qv = self.embedder.embed(query_text)
            hits = self.store.vector_search(qv, k=top_k)
            for i, h in enumerate(hits):
                doc_id = h["id"]
                res = scored.get(doc_id)
                if res is None:
                    res = GraphRAGQueryResult(
                        doc_id=doc_id,
                        subject=h.get("subject"),
                        ministry=h.get("ministry"),
                        date=h.get("date"),
                        score=float(h.get("score", 0.0)),
                        matched_entities=[],
                        via="vector",
                    )
                    scored[doc_id] = res
                else:
                    res.score = max(res.score, float(h.get("score", 0.0)))
                    res.via = "entity+vector"
        except Exception:  # noqa: BLE001 - vector search must not break querying
            pass

        results = sorted(scored.values(), key=lambda r: r.score, reverse=True)[:top_k]
        return results

    def _query_as_document(self, text: str):
        """Wrap the raw query so the extractor can process it."""
        from src.graphrag.models import DocumentRecord

        return DocumentRecord(
            question_id="__query__",
            question_text=text,
            answer_text="",
        )

    def close(self) -> None:
        self.store.close()
