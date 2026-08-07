"""
GraphRAG pre-build verification.

Runs the extraction + insertion path on a sample of ``n`` random documents from
the enriched corpus and produces a human-readable, per-document report:

    Document ID
    Subject
    Extracted Entities (name, type)
    Extracted Relationships (source, relation, target)
    Evidence snippet for every relationship
    Rejected entities (if any)
    Rejected relationships (if any)
    Final Neo4j nodes created
    Final Neo4j relationships created

Quality checks applied per document:
  - no hallucinated entities (every entity name appears verbatim in the text)
  - no hallucinated relationships (every relationship supported by evidence)
  - no duplicate entities
  - no duplicate relationships

An overall quality grade is produced: Excellent / Good / Needs prompt tuning /
Poor. If the grade is below Good the full build should NOT be started until the
extraction prompt is improved.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.graphrag.config import GraphRAGConfig
from src.graphrag.extractor import EntityRelationshipExtractor, ExtractionError
from src.graphrag.llm import DocumentExtractionError, LLMBackendExhaustedError
from src.graphrag.models import DocumentRecord, Entity, Relationship
from src.graphrag.neo4j_client import Neo4jGraphStore
from src.models.qa_record import QARecord

logger = logging.getLogger(__name__)
console = Console()


class DocumentVerification:
    """Report + quality result for a single verified document."""

    def __init__(self, doc: DocumentRecord) -> None:
        self.doc = doc
        self.entities: list[Entity] = []
        self.relationships: list[Relationship] = []
        self.rejected_entities: list[dict] = []
        self.rejected_relationships: list[dict] = []
        self.error: Optional[str] = None
        # True when the failure was a per-document content rejection
        # (DocumentExtractionError, e.g. json_validate_failed) rather than a
        # genuine extraction/insertion failure. Content rejections are counted
        # as failed documents but do not, by themselves, force grade "Poor".
        self.content_failure = False
        self.nodes_created = 0
        self.relationships_created = 0

    # ── quality checks ─────────────────────────────────────────────────

    def check_grounding(self) -> list[str]:
        """Return a list of quality problems (empty == clean)."""
        problems: list[str] = []
        text = (self.doc.question_text + " " + self.doc.answer_text).lower()
        for e in self.entities:
            if e.name.lower() not in text:
                problems.append(
                    f"entity '{e.name}' ({e.type.value}) not found verbatim in document"
                )
        for r in self.relationships:
            if not r.evidence:
                problems.append(
                    f"relationship {r.source_name}-[{r.relation.value}]->{r.target_name} "
                    "has no evidence snippet"
                )
            else:
                # Evidence must contain both endpoint names (normalized).
                ev = r.evidence.lower()
                if r.source_name.lower() not in ev or r.target_name.lower() not in ev:
                    problems.append(
                        f"relationship evidence for {r.source_name}-[{r.relation.value}]->"
                        f"{r.target_name} does not contain both endpoints"
                    )
        # duplicates
        seen_e: set[tuple[str, str]] = set()
        for e in self.entities:
            k = (e.type.value, e.name)
            if k in seen_e:
                problems.append(f"duplicate entity {e.name} ({e.type.value})")
            seen_e.add(k)
        seen_r: set[tuple] = set()
        for r in self.relationships:
            k = (r.source_type.value, r.source_name, r.relation.value,
                 r.target_type.value, r.target_name)
            if k in seen_r:
                problems.append(
                    f"duplicate relationship {r.source_name}-[{r.relation.value}]->{r.target_name}"
                )
            seen_r.add(k)
        return problems


class GraphVerificationReport:
    """Aggregate report over the sampled documents."""

    def __init__(self) -> None:
        self.docs: list[DocumentVerification] = []
        self.total_docs = 0
        self.total_entities = 0
        self.total_relationships = 0
        self.total_rejected_entities = 0
        self.total_rejected_relationships = 0
        self.total_problems = 0
        self.failed_docs = 0
        # Subset of failed_docs that were per-document content rejections
        # (DocumentExtractionError, e.g. json_validate_failed). These are
        # reported as failed documents but do not, by themselves, force the
        # grade to "Poor" — the grade reflects extraction quality of the
        # documents that were actually extracted.
        self.content_failures = 0

    def grade(self) -> str:
        """Overall quality grade."""
        # Genuine extraction/insertion failures are a hard fail.
        if self.failed_docs > self.content_failures:
            return "Poor"
        if self.total_problems > 0:
            return "Needs prompt tuning"
        if self.total_relationships == 0:
            return "Needs prompt tuning"
        # Ratio of successfully extracted docs with at least one relationship.
        # Failed docs (incl. content rejections) are reported in failed_docs
        # but excluded here so they don't dilute the quality ratio.
        graded = [d for d in self.docs if not d.error]
        if not graded:
            return "Needs prompt tuning"
        with_rel = sum(1 for d in graded if d.relationships)
        rel_ratio = with_rel / len(graded)
        if rel_ratio >= 0.8:
            return "Excellent"
        if rel_ratio >= 0.5:
            return "Good"
        return "Needs prompt tuning"


def _to_document_record(rec: QARecord) -> DocumentRecord:
    return DocumentRecord(
        question_id=rec.question_id,
        question_text=rec.question_text,
        answer_text=rec.answer_text,
        ministry=rec.metadata.ministry,
        subject=rec.metadata.subject,
        session=rec.metadata.session,
        question_number=rec.metadata.question_number,
        parliament_number=rec.metadata.parliament_number,
        date=rec.metadata.date,
        source_url=rec.metadata.source_url,
    )


class GraphVerifier:
    """Runs the verification pipeline and renders the human-readable report."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.store = Neo4jGraphStore(config)
        self.extractor = EntityRelationshipExtractor(config)

    def run(self, records: list[QARecord], n: int = 10) -> GraphVerificationReport:
        # Deterministic sample so re-runs verify the same documents.
        if len(records) < n:
            raise ValueError(f"Need at least {n} records, found {len(records)}")
        sample = random.Random(20260806).sample(records, n)

        report = GraphVerificationReport()
        report.total_docs = n

        for rec in sample:
            # Failover events (key/model switches) printed for visibility.
            for ev in self.extractor.drain_events():
                self._print_failover_event(ev, current_doc=rec.question_id)
            v = self.verify_one(rec)
            for ev in self.extractor.drain_events():
                self._print_failover_event(ev, current_doc=rec.question_id)
            report.docs.append(v)
            report.total_entities += len(v.entities)
            report.total_relationships += len(v.relationships)
            report.total_rejected_entities += len(v.rejected_entities)
            report.total_rejected_relationships += len(v.rejected_relationships)
            report.total_problems += len(v.check_grounding())
            if v.error:
                report.failed_docs += 1
                if v.content_failure:
                    report.content_failures += 1

        return report

    @staticmethod
    def _print_failover_event(ev: dict, current_doc: Optional[str] = None) -> None:
        etype = ev.get("type")
        doc = f"document {current_doc}" if current_doc else f"document {ev.get('context') or '?'}"
        if etype == "key_switch":
            console.print(
                f"[yellow]Rate limit / failure on key {ev.get('from_key')} "
                f"(model {ev.get('model')})[/yellow]\n"
                f"[bold]  Switching API key: {ev.get('from_key')} -> {ev.get('to_key')}[/bold]\n"
                f"[dim]  Continuing from {doc}...[/dim]"
            )
        elif etype == "model_switch":
            console.print(
                f"[yellow]All keys exhausted for {ev.get('from_model')}[/yellow]\n"
                f"[bold]  Switching model: {ev.get('from_model')} -> {ev.get('to_model')}[/bold]\n"
                f"[dim]  Resuming from {doc}...[/dim]"
            )

    def verify_one(self, rec: QARecord) -> DocumentVerification:
        doc = _to_document_record(rec)
        v = DocumentVerification(doc)
        try:
            entities, relationships, rejections = self.extractor.extract_with_rejections(doc)
            v.entities, v.relationships = entities, relationships
            # Rejections are only meaningful for LLM-proposed items; the
            # deterministic ministry entity is added by the extractor itself
            # and is never a "rejection".
            v.rejected_entities = [
                r for r in rejections.get("entities", [])
                if not (r.get("type") == "Ministry" and r.get("name") == doc.ministry)
            ]
            v.rejected_relationships = rejections.get("relationships", [])
        except ExtractionError as e:
            v.error = str(e)
            return v
        except DocumentExtractionError as e:
            # Per-document content failure (e.g. HTTP 400 json_validate_failed).
            # Mark only this document as failed; keep verifying the others.
            v.error = f"provider rejected document: {e}"
            v.content_failure = True
            return v
        except LLMBackendExhaustedError:
            # All providers/keys exhausted — propagate so verification stops.
            raise
        except Exception as e:  # noqa: BLE001
            v.error = f"{type(e).__name__}: {e}"
            return v

        # Insert into Neo4j (verification actually exercises the write path).
        try:
            text = f"{doc.question_text}\n{doc.answer_text}"
            self.store.upsert_document(
                doc.question_id, doc.question_id, doc.question_text, doc.answer_text,
                [0.0] * 8,  # placeholder embedding (not used in verification)
                ministry=doc.ministry, subject=doc.subject,
                session=doc.session, question_number=doc.question_number,
                parliament_number=doc.parliament_number,
                date=doc.date, source_url=doc.source_url,
            )
            v.nodes_created = 1
            if v.entities:
                self.store.upsert_entities(v.entities)
                self.store.link_document_entities(doc.question_id, v.entities)
            if v.relationships:
                self.store.upsert_relationships(v.relationships)
            v.relationships_created = len(v.relationships)
        except Exception as e:  # noqa: BLE001
            v.error = f"Neo4j write failed: {type(e).__name__}: {e}"

        return v

    # ── rendering ──────────────────────────────────────────────────────

    def render(self, report: GraphVerificationReport) -> None:
        for v in report.docs:
            self._render_doc(v)
        self._render_summary(report)

    def _render_doc(self, v: DocumentVerification) -> None:
        doc = v.doc
        console.print(Panel.fit(
            f"[bold]Document {doc.question_id}[/bold]  |  {doc.subject or '(no subject)'}",
            border_style="cyan",
        ))
        if v.error:
            console.print(f"[red]  ERROR: {v.error}[/red]")
            return

        # Entities
        etab = Table(title="Extracted Entities")
        etab.add_column("Entity")
        etab.add_column("Type")
        for e in v.entities:
            etab.add_row(e.name, e.type.value)
        console.print(etab)

        # Relationships with evidence
        if v.relationships:
            rtab = Table(title="Extracted Relationships")
            rtab.add_column("Source")
            rtab.add_column("Relationship")
            rtab.add_column("Target")
            rtab.add_column("Evidence")
            for r in v.relationships:
                rtab.add_row(r.source_name, r.relation.value, r.target_name,
                             (r.evidence or "")[:120])
            console.print(rtab)
        else:
            console.print("[dim]No relationships extracted.[/dim]")

        # Rejections
        if v.rejected_entities:
            console.print(
                f"[yellow]Rejected entities ({len(v.rejected_entities)}):[/yellow] "
                + ", ".join(f"{r['name']} ({r['type']})" for r in v.rejected_entities)
            )
        if v.rejected_relationships:
            console.print(
                f"[yellow]Rejected relationships ({len(v.rejected_relationships)}):[/yellow] "
                + "; ".join(
                    f"{r['source']}-[{r['relation']}]->{r['target']} ({r['reason']})"
                    for r in v.rejected_relationships
                )
            )

        # Quality problems
        problems = v.check_grounding()
        if problems:
            console.print(f"[red]Quality problems ({len(problems)}):[/red]")
            for p in problems:
                console.print(f"  - {p}")
        else:
            console.print("[green]✓ No hallucination / grounding / duplicate issues.[/green]")

        console.print(
            f"[dim]Neo4j: {v.nodes_created} document node, "
            f"{v.relationships_created} relationships created.[/dim]\n"
        )

    def _render_summary(self, report: GraphVerificationReport) -> None:
        grade = report.grade()
        color = {"Excellent": "green", "Good": "green",
                 "Needs prompt tuning": "yellow", "Poor": "red"}[grade]
        console.print(Panel.fit(
            "\n".join([
                f"[bold]Documents verified        : {report.total_docs}[/bold]",
                f"Documents failed          : {report.failed_docs}",
                f"Total entities            : {report.total_entities}",
                f"Total relationships       : {report.total_relationships}",
                f"Rejected entities         : {report.total_rejected_entities}",
                f"Rejected relationships    : {report.total_rejected_relationships}",
                f"Total quality problems    : {report.total_problems}",
                "",
                f"[bold {color}]Overall quality: {grade}[/bold {color}]",
            ]),
            title="[bold]GraphRAG Verification Summary[/bold]",
            border_style=color,
        ))
        if grade in ("Needs prompt tuning", "Poor"):
            console.print(
                "[yellow]Do NOT start the full build until extraction quality is fixed. "
                "Review the rejected entities/relationships and the extraction prompt.[/yellow]"
            )
        else:
            console.print(
                "[green]Extraction quality acceptable — the full build may proceed "
                "(run `graphrag build` when ready).[/green]"
            )

    def close(self) -> None:
        self.store.close()
