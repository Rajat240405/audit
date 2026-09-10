"""Neo4jGraphStore — production backend (Phase 2 / WS2).

Implements the exact ``GraphStore`` ABC over Cypher. Same semantics as the
in-memory store (idempotent upserts, document-scoped retraction, multi-source
facts via SUPPORTS edges). The driver is imported lazily so that merely
importing this module never requires the ``neo4j`` package to be importable
at module-load time in minimal environments.

Identity: every MERGE is on the node ``key`` property (UNIQUE-constrained) —
never on display names. Relationship merges are keyed by ``fact_key``.
SUPPORTS edges are keyed by (document, fact_key).

Label/relationship identifiers interpolated into Cypher come ONLY from the
validated catalog in ``src.graphrag.schema`` (never from data).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from src.graphrag.config import GraphConfig
from src.graphrag.models import (
    DocumentRef,
    FactView,
    GraphContribution,
    NodeView,
)
from src.graphrag.schema import (
    ALL_RELS,
    NODE_LABELS,
    schema_ddl,
)
from src.graphrag.store import GraphStore, NEIGHBORS_LIMIT

__all__ = ["Neo4jGraphStore"]

_ALLOWED_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    """Validate a catalog identifier before interpolation."""
    if not _ALLOWED_IDENT.match(name):
        raise ValueError(f"unsafe graph identifier: {name!r}")
    return name


def _props(node) -> dict:
    """Node -> plain property dict (driver metadata keys stripped)."""
    return {k: v for k, v in dict(node).items()
            if k not in ("element_id", "identity")}


class Neo4jGraphStore(GraphStore):
    """Production Neo4j backend for the canonical graph."""

    def __init__(self, config: GraphConfig) -> None:
        if config.backend != "neo4j":
            raise ValueError("Neo4jGraphStore requires backend='neo4j'")
        from neo4j import GraphDatabase  # lazy: driver is a production dep

        self._config = config
        self._driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=config.auth,
            connection_timeout=15,
            notifications_min_severity="OFF",
        )
        self._db = config.neo4j_database

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()

    def ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:  # noqa: BLE001
            return False

    def init_schema(self) -> None:
        for stmt in schema_ddl():  # DDL: idempotent, one statement per run
            self._run_write(stmt)

    # ── writes ────────────────────────────────────────────────────────────

    def upsert_document(self, doc: DocumentRef) -> None:
        self._run_write(
            """
            MERGE (d:Document {key: $key})
            SET d += $props
            """,
            key=doc.doc_key, props=doc.to_node(),
        )

    def apply_contribution(self, contrib: GraphContribution, *, now: str,
                           doc: Optional[DocumentRef] = None) -> dict:
        with self._driver.session(database=self._db) as session:
            return session.execute_write(_apply_contribution_tx, contrib, now, doc)

    def withdraw_document(self, doc_key: str, *, now: str) -> dict:
        with self._driver.session(database=self._db) as session:
            return session.execute_write(_withdraw_document_tx, doc_key, now)

    # ── reads ─────────────────────────────────────────────────────────────

    def get_document(self, doc_key: str) -> Optional[dict]:
        rows = self._run_read(
            "MATCH (d:Document {key: $k}) RETURN d", k=doc_key)
        if not rows:
            return None
        return _props(rows[0]["d"])

    def list_document_keys(self) -> set[str]:
        rows = self._run_read(
            "MATCH (d:Document) RETURN d.key AS k")
        return {r["k"] for r in rows}

    def get_node(self, key: str) -> Optional[NodeView]:
        rows = self._run_read(
            "MATCH (n {key: $k}) RETURN n", k=key)
        if not rows:
            return None
        return self._node_view(rows[0]["n"])

    @staticmethod
    def _node_view(n) -> NodeView:
        labels = list(n.labels)
        d = dict(n)
        props = {k: v for k, v in d.items()
                 if k not in ("key", "name", "resolution")}
        return NodeView(key=d["key"], label=labels[0] if labels else "?",
                        name=d.get("name", ""),
                        resolution=d.get("resolution", "unresolved"),
                        props=props, raw=d.get("raw"))

    def find_nodes(self, *, label=None, name_contains=None,
                   resolution=None, limit=200) -> list[NodeView]:
        where = []
        params: dict = {"limit": int(limit)}
        if label:
            where.append(f"n:`{_ident(label)}`")
        if resolution:
            where.append("n.resolution = $resolution")
            params["resolution"] = resolution
        if name_contains:
            where.append("toLower(n.name) CONTAINS toLower($needle)")
            params["needle"] = name_contains
        cy = ("MATCH (n) "
              + ("WHERE " + " AND ".join(where) if where else "")
              + " RETURN n ORDER BY toLower(n.name), n.key LIMIT $limit")
        return [self._node_view(r["n"]) for r in self._run_read(cy, **params)]

    #: Label disjunction that lets the planner use the per-label
    #: ``<label>_key_unique`` constraints. An unlabelled ``(a {key: $k})``
    #: pattern cannot use ANY of them, which is why PROFILE showed
    #: ``AllNodesScan`` over 69,878 nodes for facts_out/facts_in/neighbors.
    #: Built from NODE_LABELS so new labels are covered automatically; no new
    #: (redundant) global index is introduced.
    _KEYED_LABELS = " OR ".join(f"{{v}}:`{_ident(l)}`" for l in NODE_LABELS)

    @classmethod
    def _keyed(cls, var: str) -> str:
        """`WHERE` fragment pinning ``var`` to an indexed label + key lookup."""
        return "(" + cls._KEYED_LABELS.format(v=var) + ")"

    def _facts_batch(
        self, keys: Sequence[str], rel: Optional[str], direction: str
    ) -> dict[str, list[FactView]]:
        """Fetch facts for MANY keys in ONE round trip.

        Replaces the N+1 pattern where GraphCapability issued a separate
        facts_out/facts_in query per document (profiled: 1,761 calls / 65.4 s).
        ``UNWIND`` keeps it a single query while the label disjunction keeps
        each lookup an index seek.

        Returns ``{queried_key: [FactView, ...]}`` — every requested key is
        present (empty list when it has no matching edges), and each list keeps
        the same ``fact_key`` ordering as the single-key path.
        """
        unique = sorted({k for k in keys if k})
        if not unique:
            return {}
        if direction == "in":
            pat = "(a)-[r]->(b)"
            keyed, other = "b", "a"
        else:
            pat = "(a)-[r]->(b)"
            keyed, other = "a", "b"
        relwhere = " AND type(r) = $rel" if rel else ""
        rows = self._run_read(
            "UNWIND $keys AS k "
            f"MATCH {pat} "
            f"WHERE {self._keyed(keyed)} AND {keyed}.key = k{relwhere} "
            "RETURN k AS qk, r AS r, a.key AS src, b.key AS dst",
            keys=unique, rel=rel,
        )
        grouped: dict[str, list[FactView]] = {k: [] for k in unique}
        for row in rows:
            grouped.setdefault(row["qk"], []).append(self._fact_view_row(row))
        for k in grouped:
            grouped[k] = sorted(grouped[k], key=lambda v: v.fact_key)
        # `other` is unused beyond documenting the pattern's orientation.
        del other
        return grouped

    @staticmethod
    def _fact_view_row(row) -> FactView:
        """Map one driver row to a FactView (shared by single + batch paths)."""
        r = row["r"]
        props = dict(r)
        return FactView(
            fact_key=props["fact_key"], rel=r.type,
            src_key=row["src"], dst_key=row["dst"],
            origin=props.get("origin", "deterministic"),
            doc_count=int(props.get("doc_count", 0)),
            sample_evidence=props.get("sample_evidence"),
            first_seen_at=str(props.get("first_seen_at", "")),
            updated_at=str(props.get("updated_at", "")),
            source_field=props.get("source_field"),
        )

    def facts_for_keys(
        self, keys: Sequence[str], *, direction: str = "out",
        rel: Optional[str] = None,
    ) -> dict[str, list[FactView]]:
        """Batched facts lookup — see :meth:`_facts_batch`."""
        return self._facts_batch(keys, rel, direction)

    def _facts_where(self, key: str, rel: Optional[str], direction: str) -> list[FactView]:
        keyed = "b" if direction == "in" else "a"
        relwhere = " AND type(r) = $rel" if rel else ""
        rows = self._run_read(
            "MATCH (a)-[r]->(b) "
            f"WHERE {self._keyed(keyed)} AND {keyed}.key = $k{relwhere} "
            "RETURN r AS r, a.key AS src, b.key AS dst",
            k=key, rel=rel,
        )
        out = []
        for row in rows:
            r = row["r"]
            props = dict(r)
            out.append(FactView(
                # B1: `type(r)` called the Python builtin (shadowing Cypher's
                # type()), yielding the driver's Relationship CLASS instead of
                # the relationship type string. `r.type` is the driver's
                # accessor for the actual type name.
                fact_key=props["fact_key"], rel=r.type,
                # FactView.src_key/dst_key describe the TRUE edge direction,
                # not the direction the caller queried from. The Cypher pattern
                # already binds `a` as the edge source and `b` as its target in
                # BOTH branches ((a)-[r]->(b {key:$k}) for "in"), so the row is
                # correctly oriented as-is. Swapping it for "in" reversed the
                # endpoints, contradicting the fact_key stored on the same
                # relationship ("REL:src->dst") and diverging from the
                # InMemory backend. Callers such as GraphCapability filter
                # facts_in(x) on `f.src_key == other`, so the reversal made
                # those provenance fact_keys silently unmatchable.
                src_key=row["src"],
                dst_key=row["dst"],
                origin=props.get("origin", "deterministic"),
                doc_count=int(props.get("doc_count", 0)),
                sample_evidence=props.get("sample_evidence"),
                first_seen_at=str(props.get("first_seen_at", "")),
                updated_at=str(props.get("updated_at", "")),
                source_field=props.get("source_field"),
            ))
        return sorted(out, key=lambda v: v.fact_key)

    def facts_out(self, key: str, rel: Optional[str] = None) -> list[FactView]:
        return self._facts_where(key, rel, "out")

    def facts_in(self, key: str, rel: Optional[str] = None) -> list[FactView]:
        return self._facts_where(key, rel, "in")

    def neighbors(self, key: str, *, depth=1, rel=None, labels=None,
                  limit=NEIGHBORS_LIMIT) -> list[NodeView]:
        depth = max(1, int(depth))
        # Bounded expansion. Hub entities (a ministry MENTIONS-linked from most
        # of the corpus) otherwise return every neighbour, and callers such as
        # GraphCapability issue two further queries PER neighbour — unbounded
        # work and memory driven by graph shape. Follows the same convention as
        # find_nodes(limit=200) / documents_for_entity(limit=100): a keyword
        # arg materialised as a bound $limit parameter, applied AFTER the
        # deterministic ORDER BY so the truncation is stable (nearest first,
        # then key) rather than arbitrary.
        limit = max(1, int(limit))
        label_filter = ""
        params: dict = {"k": key, "depth": depth, "limit": limit}
        if rel:
            # `r` is already the relationship LIST of the variable-length
            # match; `relationships(path)` needed the dropped `path` binding.
            label_filter += " AND ALL(x IN r WHERE type(x) = $rel)"
            params["rel"] = rel
        if labels:
            allowed = [l for l in labels if l in NODE_LABELS]
            label_filter += (
                " AND any(l IN labels(m) WHERE l IN $labels)")
            params["labels"] = allowed
        # A1: Cypher forbids parameters in variable-length bounds — they are
        # compiled into the query plan, not bound at runtime ("Parameter maps
        # cannot be used in MATCH patterns"). `depth` is already coerced by
        # max(1, int(depth)) above, so it is an int literal and NEVER user text.
        # Every data value (key, rel, labels) remains a bound parameter.
        params.pop("depth", None)
        rows = self._run_read(
            f"MATCH (n)-[r *1..{depth}]-(m) "
            f"WHERE {self._keyed('n')} AND n.key = $k "
            "AND m.key <> $k" + label_filter + " "
            # `size(r)` is the hop count of the variable-length match — the
            # same value the removed `length(path)` produced. The `path =`
            # binding was dropped so the keyed endpoint can be an index seek.
            "WITH m, min(size(r)) AS d "
            "RETURN m, d ORDER BY d, m.key LIMIT $limit",
            **params,
        )
        return [self._node_view(r["m"]) for r in rows]

    def documents_for_entity(self, entity_key, *, rel=None, year=None,
                             ls_term=None, limit=100) -> list[dict]:
        where = ["($rel IS NULL OR type(r) = $rel)"]
        params: dict = {"k": entity_key, "limit": int(limit)}
        if rel:
            params["rel"] = rel
        else:
            params["rel"] = None
        if year is not None:
            where.append("d.date_year = $year")
            params["year"] = year
        if ls_term is not None:
            where.append("d.ls_term = $ls_term")
            params["ls_term"] = ls_term
        # Seek the KEYED endpoint first (indexed), then expand to Documents.
        # Previously `(n {key: $k})` was unlabelled, so the planner fell back to
        # NodeByLabelScan over every :Document (PROFILE: 2,649 scanned).
        rows = self._run_read(
            "MATCH (d:Document)-[r]->(n) "
            f"WHERE {self._keyed('n')} AND n.key = $k AND "
            + " AND ".join(where) + " "
            "RETURN d ORDER BY d.key LIMIT $limit",
            **params,
        )
        return [_props(r["d"]) for r in rows]

    def provenance(self, fact_key: str) -> Optional[dict]:
        rows = self._run_read(
            """
            MATCH (f:Fact {key: $fk})
            // B2: the two OPTIONAL MATCHes were joined into a cartesian
            // product — N supports x M fact relationships — so every support
            // appeared M times in `sup`. Aggregate the supports to a single
            // row FIRST, then look up source_field separately.
            OPTIONAL MATCH (d:Document)-[s:SUPPORTS]->(f)
            WITH f,
                 collect(DISTINCT {doc_key: d.key,
                                   evidence: s.evidence,
                                   origin: s.origin,
                                   extracted_at: s.extracted_at,
                                   document: d}) AS sup
            OPTIONAL MATCH (a)-[r {fact_key: $fk}]->(b)
            WITH f, sup, max(r.source_field) AS source_field
            RETURN f AS f, sup AS sup, source_field AS source_field
            """,
            fk=fact_key,
        )
        if not rows:
            return None
        r = rows[0]
        f = dict(r["f"])
        fact_view = FactView(
            fact_key=f["key"], rel=f.get("rel_type", ""),
            src_key=f.get("src_key", ""), dst_key=f.get("dst_key", ""),
            origin=f.get("origin", "deterministic"),
            doc_count=int(f.get("doc_count", 0)),
            sample_evidence=None,
            first_seen_at=str(f.get("first_seen_at", "")),
            updated_at=str(f.get("updated_at", "")),
            source_field=r.get("source_field"),
        )
        supports = []
        for s in r["sup"] or []:
            if s.get("doc_key") is None:
                continue
            supports.append({
                "doc_key": s["doc_key"],
                "evidence": s.get("evidence"),
                "origin": s.get("origin"),
                "extracted_at": s.get("extracted_at"),
                "document": _props(s["document"]) if s.get("document") is not None else None,
            })
        return {"fact": fact_view, "supports": supports}

    def stats(self) -> dict:
        labels = {}
        for label in NODE_LABELS:
            rows = self._run_read(f"MATCH (n:`{_ident(label)}`) RETURN count(n) AS c")
            labels[label] = int(rows[0]["c"]) if rows else 0
        rels = {}
        for rt in sorted(ALL_RELS):
            rows = self._run_read(
                f"MATCH ()-[r:`{_ident(rt)}`]->() RETURN count(r) AS c")
            rels[rt] = int(rows[0]["c"]) if rows else 0
        # SUPPORTS is a RELATIONSHIP (Document)-[:SUPPORTS]->(Fact), not a node
        # label — node syntax matches nothing and silently reports 0.
        rows = self._run_read(
            "MATCH (:Document)-[s:SUPPORTS]->(:Fact) RETURN count(s) AS c")
        n_supports = int(rows[0]["c"]) if rows else 0
        return {
            "labels": labels,
            "relationships": rels,
            "facts": labels.get("Fact", 0),
            "supports": n_supports,
            "documents": labels.get("Document", 0),
        }

    # ── internals ─────────────────────────────────────────────────────────

    def _run_write(self, cypher: str, **params) -> list[dict]:
        # A2: execute_write forwards extra args to the CALLABLE, not to tx.run().
        # Passing **params here made the lambda raise TypeError (and the values
        # never reached Cypher). Params travel as ONE positional dict instead.
        with self._driver.session(database=self._db) as session:
            result = session.execute_write(
                lambda tx, c, p: [dict(r) for r in tx.run(c, **p)],
                cypher, params)
        return result

    def _run_read(self, cypher: str, **params) -> list[dict]:
        with self._driver.session(database=self._db) as session:
            return [dict(r) for r in session.run(cypher, **params)]

# ── transaction functions (module-level: required for driver pickling) ─────

def _apply_contribution_tx(tx, contrib: GraphContribution, now: str,
                           doc: Optional[DocumentRef] = None) -> dict:
    doc_key = contrib.doc_key

    # 0) the Document node itself — same transaction (atomic unit)
    if doc is not None:
        tx.run(
            "MERGE (d:Document {key: $key}) SET d += $props",
            key=doc.doc_key, props=doc.to_node(),
        )

    # 1) nodes — grouped by label (MERGE needs the label in the pattern)
    by_label: dict[str, list[dict]] = {}
    for ref in contrib.nodes:
        by_label.setdefault(ref.label, []).append(ref.to_node())
    for label, rows in by_label.items():
        tx.run(
            f"UNWIND $rows AS row MERGE (n:`{_ident(label)}` {{key: row.key}}) "
            f"SET n += row, "
            "n.resolution = CASE WHEN n.resolution = 'unresolved' "
            "AND row.resolution = 'canonical' THEN 'canonical' ELSE n.resolution END",
            rows=rows,
        )

    # 2) facts + relationships — grouped by (src_label, rel, dst_label)
    groups: dict[tuple[str, str, str], list] = {}
    for fact in contrib.facts:
        groups.setdefault((fact.src.label, fact.rel, fact.dst.label), []).append(fact)
    for (src_label, rel, dst_label), facts in groups.items():
        rows = [
            {
                "fk": f.fact_key, "a": f.src.key, "b": f.dst.key,
                "origin": f.origin, "source_field": f.source_field,
                "now": now,
            }
            for f in facts
        ]
        tx.run(
            f"""
            UNWIND $rows AS row
            MERGE (f:Fact {{key: row.fk}})
            SET f.rel_type = $rel, f.src_key = row.a, f.dst_key = row.b,
                f.origin = row.origin, f.updated_at = row.now
            WITH f, row
            MERGE (a:`{_ident(src_label)}` {{key: row.a}})
            MERGE (b:`{_ident(dst_label)}` {{key: row.b}})
            MERGE (a)-[r:`{_ident(rel)}`]->(b)
            SET r.fact_key = row.fk, r.origin = row.origin,
                r.source_field = row.source_field, r.updated_at = row.now
            """,
            rows=rows, rel=rel,
        )

    # 3) supports — MERGE-keyed on (doc, fact); doc_count grows only for new
    fks = sorted({s.fact_key for s in contrib.supports})
    existing: set[str] = set()
    if fks:
        res = tx.run(
            "MATCH (d:Document {key: $doc})-[s:SUPPORTS]->(f:Fact) "
            "WHERE f.key IN $fks RETURN DISTINCT f.key AS fk",
            doc=doc_key, fks=fks,
        )
        existing = {r["fk"] for r in res}
    rows = [
        {
            "fk": s.fact_key, "doc": s.doc_key, "evidence": s.evidence,
            "origin": s.origin, "ts": s.extracted_at or now,
            "is_new": s.fact_key not in existing,
        }
        for s in contrib.supports
    ]
    if rows:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (d:Document {key: row.doc})
            MERGE (f:Fact {key: row.fk})
            MERGE (d)-[s:SUPPORTS {fact_key: row.fk}]->(f)
            SET s.evidence = row.evidence, s.origin = row.origin,
                s.extracted_at = row.ts
            WITH row, f
            WHERE row.is_new
            SET f.doc_count = coalesce(f.doc_count, 0) + 1
            """,
            rows=rows,
        )
        # first_seen_at + sample_evidence maintenance for touched facts
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (f:Fact {key: row.fk})
            SET f.first_seen_at = coalesce(f.first_seen_at, row.ts),
                f.updated_at = row.ts,
                f.sample_evidence = coalesce(row.evidence, f.sample_evidence)
            """,
            rows=rows,
        )

    counters = {
        "nodes": len(contrib.nodes),
        "facts": len(contrib.facts),
        "supports": len(contrib.supports),
        "new_supports": sum(1 for s in contrib.supports if s.fact_key not in existing),
    }
    Neo4jGraphStore._last_apply_counters = counters
    return counters


def _withdraw_document_tx(tx, doc_key: str, now: str) -> dict:
    # 1) decrement doc_count on the facts this document supports
    tx.run(
        """
        MATCH (d:Document {key: $doc})-[s:SUPPORTS]->(f:Fact)
        SET f.doc_count = coalesce(f.doc_count, 1) - 1, f.updated_at = $now
        """,
        doc=doc_key, now=now,
    )
    # 2) remove the document's SUPPORTS edges
    # A3: the old form aggregated `count(s)` in a WITH that dropped `s` from
    # scope, so `DETACH DELETE s` referenced an undefined variable (hard
    # SyntaxError on 5.26.4 — nothing was ever deleted). Collect the edges
    # first, keep them in scope, then delete each one. DETACH is wrong for a
    # relationship anyway; plain DELETE removes exactly these SUPPORTS edges
    # and leaves both endpoint nodes intact (document-scoped semantics).
    res = tx.run(
        """
        MATCH (d:Document {key: $doc})-[s:SUPPORTS]->()
        WITH collect(s) AS ss
        WITH ss, size(ss) AS removed
        FOREACH (x IN ss | DELETE x)
        RETURN removed
        """,
        doc=doc_key,
    )
    removed = int(res.single()["removed"])
    # 3) facts that lost their last supporting document go away — WITH their
    #    relationships. Facts still supported by other documents survive.
    res = tx.run(
        """
        MATCH (f:Fact) WHERE coalesce(f.doc_count, 0) <= 0
        // A4: the old inner MATCH was NON-optional, so any orphan Fact whose
        // projected relationship was already gone matched nothing and the
        // whole row was dropped — the Fact leaked and `deleted` under-counted.
        // OPTIONAL MATCH keeps the row; the projected (a)-[r]->(b) edge is NOT
        // attached to f, so it must be deleted explicitly, while DETACH DELETE
        // f clears f's own edges (e.g. surviving SUPPORTS from other docs
        // cannot exist here, since doc_count <= 0).
        OPTIONAL MATCH (a)-[r {fact_key: f.key}]->(b)
        WITH collect(DISTINCT f) AS fs, collect(DISTINCT r) AS rs
        WITH fs, rs, size(fs) AS deleted
        FOREACH (x IN rs | DELETE x)
        FOREACH (x IN fs | DETACH DELETE x)
        RETURN deleted
        """,
    )
    deleted = int(res.single()["deleted"])
    # 4) remove the Document node itself
    res = tx.run(
        """
        MATCH (d:Document {key: $doc})
        DETACH DELETE d
        RETURN count(d) AS n
        """,
        doc=doc_key,
    )
    doc_removed = int(res.single()["n"]) > 0
    return {"supports_removed": removed, "facts_deleted": deleted,
            "document_removed": doc_removed}
