"""
Neo4j client for the GraphRAG pipeline.

Responsibilities:
- Connect to Neo4j (bolt) with the official ``neo4j`` driver.
- Apply the schema: uniqueness constraints, node-key indexes, and the vector
  index on ``Document.embedding``.
- Insert documents / entities / relationships in batched transactions.
- Expose stats (counts by label, relationship counts, index info).
- Expose vector search + entity-expansion queries for ``graphrag query``.

Neo4j Community Edition is the production source of truth; this module is the
only place that talks to it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from src.graphrag.config import GraphRAGConfig
from src.graphrag.models import (
    ENTITY_LABELS,
    Entity,
    Relationship,
)

logger = logging.getLogger(__name__)

# Labels that get a unique-name constraint + index.
ENTITY_LABELS_LIST = sorted(ENTITY_LABELS.values())
# Relationship types we manage.
RELATION_TYPES = [
    "MENTIONS",
    "LOCATED_IN",
    "IMPLEMENTED_BY",
    "OPERATED_BY",
    "FUNDED_BY",
    "RELATED_TO",
    "PART_OF",
    "MONITORS",
    "FORECASTS",
    "USES",
    "COLLABORATES_WITH",
    "REPORTS_TO",
]

_SAFE_LABEL = __import__("re").compile(r"[^A-Za-z0-9_]")


def _label(name: str) -> str:
    """Sanitize a label/type string into a safe Neo4j identifier."""
    return _SAFE_LABEL.sub("", name)


class Neo4jGraphStore:
    """Production Neo4j store for GraphRAG."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        auth = None
        if config.neo4j_user:
            auth = (config.neo4j_user, config.neo4j_password or "")
        self._driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=auth,
            connection_timeout=15,
            # Silence informational query notifications (e.g. "relationship type
            # does not exist" while counting types that are simply absent).
            notifications_min_severity="OFF",
        )
        self._database = config.neo4j_database

    def close(self) -> None:
        self._driver.close()

    # ── connection / schema ─────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:  # noqa: BLE001
            return False

    def apply_schema(self, embedding_dim: int) -> None:
        """Create constraints, indexes and the vector index (idempotent)."""
        queries = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT document_question_id IF NOT EXISTS FOR (d:Document) REQUIRE d.question_id IS UNIQUE",
        ]
        for label in ENTITY_LABELS_LIST:
            queries.append(
                f"CREATE CONSTRAINT {_label(label).lower()}_name IF NOT EXISTS "
                f"FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
            )
        # Vector index for Document embeddings (bge-m3 dim).
        queries.append(
            "CREATE VECTOR INDEX document_embeddings IF NOT EXISTS "
            "FOR (d:Document) ON (d.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: %d, "
            "`vector.similarity_function`: 'cosine'}}" % int(embedding_dim)
        )
        for q in queries:
            self._run_write(q)

    def reset_graph(self) -> None:
        """Drop all GraphRAG nodes/relationships (used by ``graphrag rebuild``)."""
        for label in ["Document"] + ENTITY_LABELS_LIST:
            self._run_write(f"MATCH (n:`{label}`) DETACH DELETE n")
        self._drop_vector_index()

    def _drop_vector_index(self) -> None:
        try:
            self._run_write("DROP INDEX document_embeddings IF EXISTS")
        except Neo4jError:
            pass

    # ── inserts ─────────────────────────────────────────────────────────

    def upsert_document(
        self,
        doc_id: str,
        question_id: str,
        question_text: str,
        answer_text: str,
        embedding: list[float],
        *,
        ministry: Optional[str] = None,
        subject: Optional[str] = None,
        session: Optional[int] = None,
        question_number: Optional[int] = None,
        parliament_number: Optional[int] = None,
        date: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> dict[str, int]:
        q = """
        MERGE (d:Document {id: $id})
        SET d.question_id = $question_id,
            d.question_text = $question_text,
            d.answer_text = $answer_text,
            d.embedding = $embedding,
            d.ministry = $ministry,
            d.subject = $subject,
            d.session = $session,
            d.question_number = $question_number,
            d.parliament_number = $parliament_number,
            d.date = $date,
            d.source_url = $source_url
        RETURN count(d) AS n
        """
        result = self._run_write(
            q,
            id=doc_id,
            question_id=question_id,
            question_text=question_text,
            answer_text=answer_text,
            embedding=embedding,
            ministry=ministry,
            subject=subject,
            session=session,
            question_number=question_number,
            parliament_number=parliament_number,
            date=date,
            source_url=source_url,
        )
        return {"nodes_created": 1}

    def upsert_entities(self, entities: list[Entity]) -> dict[str, int]:
        """Batch MERGE entity nodes grouped by label."""
        if not entities:
            return {"nodes_created": 0}
        by_label: dict[str, list[dict]] = {}
        for e in entities:
            by_label.setdefault(e.label, []).append({"name": e.name})
        total = 0
        for label, rows in by_label.items():
            result = self._run_write(
                f"""
                UNWIND $rows AS row
                MERGE (n:`{label}` {{name: row.name}})
                RETURN count(n) AS n
                """,
                rows=rows,
            )
            total += 1
        return {"nodes_created": total}

    def upsert_relationships(self, rels: list[Relationship]) -> dict[str, int]:
        if not rels:
            return {"relationships_created": 0}
        created = 0
        for r in rels:
            created += self._merge_relationship(
                r.source_type.value, r.source_name,
                r.relation.value,
                r.target_type.value, r.target_name,
                r.evidence,
            )
        return {"relationships_created": created}

    def _merge_relationship(
        self,
        source_label: str,
        source_name: str,
        rel: str,
        target_label: str,
        target_name: str,
        evidence: Optional[str],
    ) -> int:
        q = f"""
        MATCH (a:`{source_label}` {{name: $s}})
        MATCH (b:`{target_label}` {{name: $t}})
        MERGE (a)-[r:`{rel}`]->(b)
        SET r.evidence = coalesce(r.evidence, $evidence)
        RETURN count(r) AS n
        """
        result = self._run_write(q, s=source_name, t=target_name, evidence=evidence)
        return 1

    def link_document_entities(
        self, doc_id: str, entities: list[Entity]
    ) -> dict[str, int]:
        """Create (:Document)-[:MENTIONS]->(:Entity) edges for a document."""
        if not entities:
            return {"relationships_created": 0}
        rows = [{"label": e.label, "name": e.name} for e in entities]
        created = 0
        for r in rows:
            q = f"""
            MATCH (d:Document {{id: $id}})
            MATCH (e:`{r['label']}` {{name: $name}})
            MERGE (d)-[:MENTIONS]->(e)
            """
            self._run_write(q, id=doc_id, name=r["name"])
            created += 1
        return {"relationships_created": created}

    # ── queries ─────────────────────────────────────────────────────────

    def vector_search(self, query_vector: list[float], k: int = 10) -> list[dict]:
        result = self._run_read(
            """
            CALL db.index.vector.queryNodes('document_embeddings', $k, $vec)
            YIELD node AS d, score
            RETURN d.id AS id, d.question_id AS question_id,
                   d.subject AS subject, d.ministry AS ministry,
                   d.date AS date, score
            ORDER BY score DESC
            """,
            k=k,
            vec=query_vector,
        )
        return [dict(r) for r in result]

    def find_entity(self, label: str, name: str) -> Optional[dict]:
        result = self._run_read(
            f"MATCH (n:`{label}` {{name: $name}}) RETURN n.name AS name LIMIT 1",
            name=name,
        )
        return dict(result[0]) if result else None

    def documents_by_entity(self, label: str, name: str, limit: int = 10) -> list[dict]:
        result = self._run_read(
            f"""
            MATCH (e:`{label}` {{name: $name}})<-[:MENTIONS]-(d:Document)
            RETURN d.id AS id, d.subject AS subject, d.ministry AS ministry,
                   d.date AS date, d.source_url AS source_url
            LIMIT {int(limit)}
            """,
            name=name,
        )
        return [dict(r) for r in result]

    def stats(self) -> dict[str, Any]:
        labels = {}
        for label in ["Document"] + ENTITY_LABELS_LIST:
            result = self._run_read(f"MATCH (n:`{label}`) RETURN count(n) AS c")
            labels[label] = result[0]["c"] if result else 0
        rels: dict[str, int] = {}
        for rt in RELATION_TYPES:
            result = self._run_read(
                f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c"
            )
            rels[rt] = result[0]["c"] if result else 0
        indexes = self._run_read(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
            "RETURN name, type, labelsOrTypes, properties"
        )
        index_list = [dict(r) for r in indexes]
        return {
            "labels": labels,
            "relationships": rels,
            "indexes": index_list,
            "total_nodes": sum(labels.values()),
            "total_relationships": sum(rels.values()),
        }

    # ── internals ───────────────────────────────────────────────────────

    def _run_write(self, cypher: str, **params) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    def _run_read(self, cypher: str, **params) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]
