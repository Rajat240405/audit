"""
GraphRAG build pipeline.

Pipeline (per the production spec):

    Load JSONL → Extract entities → Extract relationships → Insert into Neo4j
    → Generate embeddings → Create vector index → Verify graph → Statistics

Checkpointing is mandatory and automatic: every successfully processed
document is checkpointed immediately, so an interrupted ``graphrag build``
resumes exactly where it stopped.

Progress is reported per document with elapsed / ETA / nodes created /
relationships created / failures / retries.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live

from src.data.loader import DataLoader
from src.graphrag.checkpoint import GraphCheckpoint
from src.graphrag.config import GraphRAGConfig
from src.graphrag.display import BuildStatusPanel
from src.graphrag.embeddings import GraphEmbedder
from src.graphrag.extractor import EntityRelationshipExtractor, ExtractionError
from src.graphrag.llm import DocumentExtractionError, LLMBackendExhaustedError
from src.graphrag.models import DocumentRecord, Entity, Relationship
from src.graphrag.neo4j_client import Neo4jGraphStore
from src.models.qa_record import QARecord

logger = logging.getLogger(__name__)
console = Console()


class GraphBuildResult:
    """Summary of a graph build / rebuild run."""

    def __init__(self) -> None:
        self.documents_processed = 0
        self.nodes_created = 0
        self.relationships_created = 0
        self.failures = 0
        self.retries = 0
        self.duration_seconds = 0.0
        self.checkpoint_counts: dict = {}
        self.embedding_count = 0
        self.skipped_from_checkpoint = 0
        self.failed_docs: list[str] = []
        self.stopped_reason: Optional[str] = None   # set when the build stops early
        self.last_completed_doc: Optional[str] = None
        # Observability / failover accounting
        self.llm_requests = 0
        self.provider_model_usage: dict = {}
        self.key_usage: dict = {}
        self.provider_switches = 0
        self.model_switches = 0

    def to_dict(self) -> dict:
        return {
            "documents_processed": self.documents_processed,
            "nodes_created": self.nodes_created,
            "relationships_created": self.relationships_created,
            "failures": self.failures,
            "retries": self.retries,
            "duration_seconds": round(self.duration_seconds, 2),
            "embedding_count": self.embedding_count,
            "skipped_from_checkpoint": self.skipped_from_checkpoint,
            "failed_docs": self.failed_docs,
            "stopped_reason": self.stopped_reason,
            "last_completed_doc": self.last_completed_doc,
            "checkpoint": self.checkpoint_counts,
            "llm_requests": self.llm_requests,
            "provider_model_usage": self.provider_model_usage,
            "key_usage": self.key_usage,
            "provider_switches": self.provider_switches,
            "model_switches": self.model_switches,
        }


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


class GraphRAGPipeline:
    """Orchestrates the GraphRAG build (extract → insert → embed → verify)."""

    def __init__(self, config: GraphRAGConfig) -> None:
        self.config = config
        self.store = Neo4jGraphStore(config)
        self.embedder = GraphEmbedder(config)
        self.extractor = EntityRelationshipExtractor(config)
        self.checkpoint = GraphCheckpoint(
            config.checkpoint_file, retry_failed=config.retry_failed
        )

    # ── input loading ──────────────────────────────────────────────────

    def load_enriched(self) -> list[QARecord]:
        import glob

        files = sorted(glob.glob(self.config.enriched_glob))
        if not files:
            raise FileNotFoundError(
                f"No enriched JSONL found matching {self.config.enriched_glob!r}. "
                "Run `ingest` first or pass --enriched."
            )
        records: list[QARecord] = []
        for fp in files:
            records.extend(DataLoader.load_jsonl(fp))
        return records

    # ── verification (10 random documents) ──────────────────────────────

    def verify_sample(self, records: list[QARecord], n: int = 10) -> GraphBuildResult:
        """
        Verify the full path on ``n`` random documents BEFORE the full build.

        Exercises: load → extract → insert → embed → vector index → query.

        Failure policy (consistent with the production build):
        - ``DocumentExtractionError`` (e.g. HTTP 400 json_validate_failed) marks
          ONLY that sampled document as failed and verification CONTINUES with
          the remaining documents.
        - Genuine infrastructure failures (Neo4j unreachable, provider
          exhaustion, unexpected exceptions) still abort immediately.
        - After the sample, the overall verification grade is computed from the
          completed sample; the build is blocked ONLY when the grade falls
          below ``verify_min_grade`` (default Good).
        """
        if len(records) < n:
            raise ValueError(f"Need at least {n} records to verify, found {len(records)}")
        sample = random.Random(20260806).sample(records, n)
        console.print(f"[cyan]Verifying on {n} random documents...[/cyan]")

        # Genuine infrastructure failures abort immediately (preserved).
        if not self.store.ping():
            raise RuntimeError("Neo4j is not reachable — cannot verify.")

        # Ensure schema exists before writing anything.
        self.store.apply_schema(self.embedder.embedding_dim)

        result = GraphBuildResult()
        # Track per-doc quality for the grade (same model as `graphrag verify`).
        from src.graphrag.verify import GraphVerificationReport, DocumentVerification

        report = GraphVerificationReport()
        report.total_docs = len(sample)

        for rec in sample:
            try:
                ok = self._process_one(rec, result, verify_mode=True)
                if ok == "content":
                    # Per-document content failure (e.g. json_validate_failed):
                    # mark ONLY this doc failed, report it, and CONTINUE. It IS
                    # a failed sampled document (counted in report.failed_docs
                    # so the final summary matches the console output), but it
                    # is NOT a genuine extraction-quality failure — the grade
                    # reflects how well the successfully-extracted docs were
                    # extracted, and a content rejection alone must not block
                    # the build.
                    result.failures += 1
                    result.failed_docs.append(rec.question_id)
                    dv = DocumentVerification(_to_document_record(rec))
                    dv.error = "provider rejected document (json_validate/content)"
                    dv.content_failure = True
                    report.docs.append(dv)
                    report.failed_docs += 1
                    report.content_failures += 1
                    console.print(
                        f"[yellow]  verification doc {rec.question_id} failed "
                        f"(json_validate/content)[/yellow]"
                    )
                    continue
                if ok is not None:
                    # A non-content failure (extraction / insert) — count it as
                    # a failed sampled document; keep verifying the rest. It
                    # lowers the grade (failed docs -> Poor) as it should.
                    result.failures += 1
                    result.failed_docs.append(rec.question_id)
                    dv = DocumentVerification(_to_document_record(rec))
                    dv.error = f"document failed during verification ({ok})"
                    report.docs.append(dv)
                    report.failed_docs += 1
                    continue
                # Success: record extraction quality for the grade.
                dv = DocumentVerification(_to_document_record(rec))
                try:
                    ents, rels, _ = self.extractor.extract_with_rejections(dv.doc)
                    dv.entities, dv.relationships = ents, rels
                except Exception:  # noqa: BLE001 - quality re-derivation must not fail the gate
                    pass
                report.docs.append(dv)
                report.total_entities += len(dv.entities)
                report.total_relationships += len(dv.relationships)
                report.total_problems += len(dv.check_grounding())
            except LLMBackendExhaustedError:
                # Genuine infrastructure failure — abort immediately (preserved).
                raise
            except Exception as e:  # noqa: BLE001 - unexpected infrastructure failure
                raise RuntimeError(
                    f"Verification FAILED on document {rec.question_id}: "
                    f"{type(e).__name__}: {e}. This is an infrastructure/backend "
                    "failure — fix it before starting the full build."
                ) from e

        # Vector search sanity check on the verified documents.
        if result.embedding_count:
            qv = self.embedder.embed("cyclone warning system")
            hits = self.store.vector_search(qv, k=3)
            console.print(f"[green]Vector index verified: {len(hits)} hits returned.[/green]")

        # Overall grade from the completed sample; block only if below threshold.
        grade = report.grade()
        min_grade = getattr(self.config, "verify_min_grade", "Good")
        console.print(
            f"[cyan]Verification grade: {grade} (minimum required: {min_grade})[/cyan]"
        )
        if grade in ("Needs prompt tuning", "Poor"):
            raise RuntimeError(
                f"Verification grade '{grade}' is below the required "
                f"'{min_grade}'. Fix extraction quality before starting the "
                f"full build. ({report.failed_docs} of {len(sample)} sample "
                f"documents failed; {report.total_problems} quality problems.)"
            )
        return result

    # ── full build ─────────────────────────────────────────────────────

    def build(
        self,
        records: list[QARecord],
        *,
        verify_first: bool = True,
        n_verify: int = 10,
    ) -> GraphBuildResult:
        started = time.monotonic()
        result = GraphBuildResult()

        if verify_first:
            self.verify_sample(records, n=n_verify)

        if not self.store.ping():
            raise RuntimeError("Neo4j is not reachable — cannot build.")
        self.store.apply_schema(self.embedder.embedding_dim)

        # Filter to records that need processing (checkpoint resume).
        todo = []
        for rec in records:
            if self.config.resume and self.checkpoint.is_done(rec.question_id):
                result.skipped_from_checkpoint += 1
                continue
            if self.config.resume and not self.checkpoint.should_retry(rec.question_id):
                result.skipped_from_checkpoint += 1
                continue
            todo.append(rec)
        if self.config.limit is not None:
            todo = todo[: self.config.limit]

        total = len(todo)
        console.print(
            f"[cyan]Graph build: {total} documents to process "
            f"({result.skipped_from_checkpoint} skipped via checkpoint)[/cyan]"
        )
        if total == 0:
            console.print("[green]Nothing to do — all documents already processed.[/green]")
            self._finalize(result, started)
            return result

        status = BuildStatusPanel(total=total, started=started)
        with Live(status.render(), console=console, refresh_per_second=10) as live:
            for i, rec in enumerate(todo, start=1):
                # Failover events from the previous document (if any).
                for ev in self.extractor.drain_events():
                    self._print_failover_event(ev)
                status.update(current_id=rec.question_id)
                live.update(status.render())
                try:
                    ok = self._process_one(rec, result)
                except LLMBackendExhaustedError as e:
                    # All providers/models/keys exhausted → stop cleanly;
                    # checkpoint was already saved for the current doc.
                    for ev in self.extractor.drain_events():
                        self._print_failover_event(ev)
                    result.stopped_reason = str(e)
                    console.print()
                    console.print(
                        "[bold red]LLM backends exhausted — stopping build cleanly.[/bold red]"
                    )
                    console.print(f"[red]{e}[/red]")
                    console.print(
                        f"[yellow]Last completed document: "
                        f"{result.last_completed_doc or '(none)'}[/yellow]"
                    )
                    console.print(
                        "[yellow]Checkpoint saved. Re-run `graphrag build` to resume "
                        "from where it stopped.[/yellow]"
                    )
                    break
                # Drain any failover events raised during this document.
                for ev in self.extractor.drain_events():
                    self._print_failover_event(ev, current_doc=rec.question_id)

                status.update(
                    completed=i,
                    current_id=rec.question_id,
                    provider=self.extractor.stats.get("provider"),
                    model=self.extractor.stats.get("model"),
                    key=self.extractor.stats.get("key"),
                    nodes=result.nodes_created,
                    rels=result.relationships_created,
                    failures=result.failures,
                    retries=result.retries,
                )
                live.update(status.render())

                if ok is None:
                    # Success (None == no failure reason).
                    result.last_completed_doc = rec.question_id
                else:
                    # Any failure reason (content/extraction/insert) counts as a
                    # failed document; the build continues (resume retries it).
                    result.failures += 1
                    result.failed_docs.append(rec.question_id)
                    if result.failures > self.config.max_failures:
                        console.print(
                            f"[red]Aborting build: failures exceeded {self.config.max_failures}.[/red]"
                        )
                        break

        self._finalize(result, started)
        return result

    def _print_failover_event(self, ev: dict, current_doc: Optional[str] = None) -> None:
        """Print a human-readable failover (key/model switch) message."""
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
        elif etype == "schema_downgrade":
            console.print(
                f"[yellow]{ev.get('model')} rejected the configured JSON format "
                f"({ev.get('from_level')})[/yellow]\n"
                f"[bold]  Downgrading to {ev.get('to_level')} "
                f"(same model, same key)[/bold]\n"
                f"[dim]  Continuing from {doc}...[/dim]"
            )

    def _finalize(self, result: GraphBuildResult, started: float) -> None:
        result.duration_seconds = time.monotonic() - started
        result.checkpoint_counts = self.checkpoint.counts()
        stats = self.store.stats()
        result.nodes_created = stats["total_nodes"]
        result.relationships_created = stats["total_relationships"]
        result.embedding_count = self.checkpoint.counts()["done"]

        # LLM observability / failover accounting from the extractor.
        result.llm_requests = self.extractor.stats.get("calls", 0)
        result.provider_model_usage = self.extractor.usage_summary()
        # Flatten key usage: provider -> model -> masked_key -> count
        result.key_usage = self.extractor.usage_summary()
        switches = self.extractor.switch_counts()
        result.provider_switches = switches.get("key_switches", 0)
        result.model_switches = switches.get("model_switches", 0)
        result.schema_downgrades = switches.get("schema_downgrades", 0)
        # Retries reported by the extractor (already added to result.retries in
        # _process_one, but ensure it is consistent).
        result.retries = max(result.retries, self.extractor.stats.get("retries", 0))

    # ── per-document processing ─────────────────────────────────────────

    def _process_one(
        self, rec: QARecord, result: GraphBuildResult, verify_mode: bool = False
    ) -> Optional[str]:
        """Process one document.

        Returns ``None`` on success, or a failure-reason string:
        - ``"content"``    : per-document content failure (DocumentExtractionError,
                             e.g. HTTP 400 json_validate_failed)
        - ``"extraction"`` : generic extraction failure (ExtractionError)
        - ``"insert"``     : embedding / Neo4j insertion failure

        Truthiness is preserved for existing call sites (``if ok:`` works the
        same as before); the reason string lets the verification gate
        distinguish per-document content failures from other failures.
        """
        doc = _to_document_record(rec)
        try:
            # 1. Extract entities + relationships (grounded).
            entities, relationships = self.extractor.extract(doc)
            # NOTE: extractor retries are cumulative across all documents;
            # they are read once in _finalize, not accumulated per document.

            # 2. Embed the document.
            embedding = self.embedder.embed(doc.question_text + "\n" + doc.answer_text)

            # 3. Insert into Neo4j (single transaction per document is safe
            #    and keeps checkpoints accurate).
            self.store.upsert_document(
                doc.question_id,
                doc.question_id,
                doc.question_text,
                doc.answer_text,
                embedding,
                ministry=doc.ministry,
                subject=doc.subject,
                session=doc.session,
                question_number=doc.question_number,
                parliament_number=doc.parliament_number,
                date=doc.date,
                source_url=doc.source_url,
            )
            if entities:
                self.store.upsert_entities(entities)
            if relationships:
                self.store.upsert_relationships(relationships)
            if entities:
                self.store.link_document_entities(doc.question_id, entities)

            result.documents_processed += 1
            result.embedding_count += 1
            if not verify_mode:
                self.checkpoint.mark_done(doc.question_id)
            return None
        except LLMBackendExhaustedError as e:
            # All providers/keys exhausted: record this doc as failed so the
            # next run retries it, then let the build loop stop cleanly.
            if not verify_mode:
                self.checkpoint.mark_failed(doc.question_id, str(e))
            raise
        except DocumentExtractionError as e:
            # Per-document content failure (e.g. HTTP 400 json_validate_failed).
            # Mark ONLY this document as failed, save the checkpoint, and let
            # the build loop continue with the remaining documents — this is
            # NOT backend exhaustion and must not stop the build.
            if not verify_mode:
                self.checkpoint.mark_failed(doc.question_id, str(e))
            logger.warning(
                "Document %s skipped (extraction rejected by provider): %s",
                doc.question_id, e,
            )
            return "content"
        except ExtractionError as e:
            if not verify_mode:
                self.checkpoint.mark_failed(doc.question_id, str(e))
            logger.warning("Extraction failed for %s: %s", doc.question_id, e)
            return "extraction"
        except Exception as e:  # noqa: BLE001 - insertion/embedding failures
            if not verify_mode:
                self.checkpoint.mark_failed(doc.question_id, str(e))
            logger.exception("Document %s failed: %s", doc.question_id, e)
            return "insert"

    # ── cleanup ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self.store.close()
