"""Incremental graph ingestion pipeline (Phase 2 / WS2-E).

Canonical corpus (``data/corpus_reports.jsonl``) → GraphStore, driven by the
Phase-1 canonical content hash (``qa_content_hash`` — same identity the corpus
ingestion uses):

    new document        → extract + apply contribution          (ADD)
    unchanged document  → skip (checkpoint hash matches)        (SKIP)
    changed document    → withdraw old contribution → re-extract → apply
                                                        (RECONCILE)
    removed document    → withdraw contribution + Document node (WITHDRAW)

Properties (spec §5):
  * idempotent  — a second run over unchanged documents changes nothing
    (skip at checkpoint level; upserts MERGE even when forced)
  * restartable — per-document atomic checkpoint; crash at N resumes at N
  * checkpointable — storage/graphrag/checkpoint.json (crash-safe writes)
  * deterministic where possible — the deterministic pass needs no LLM;
    semantic extraction runs through the existing generation architecture
    and only when a client is provided (``deterministic_only`` / no llm)

Every document is ONE atomic unit: upsert Document + entity nodes + facts +
SUPPORTS provenance in a single transaction.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.models.qa_record import QARecord
from src.scripts.ingest_folder import qa_content_hash
from src.vocabulary import Vocabulary, load_vocabulary

from src.graphrag.checkpoint import (
    EXTRACTION_DETERMINISTIC,
    EXTRACTION_SEMANTIC,
    GraphCheckpoint,
    normalize_extraction,
)
from src.graphrag.config import GraphConfig
from src.graphrag.deterministic import build_contribution
from src.graphrag.extract import ExtractionError, SemanticExtractor
from src.graphrag.identity import stable_content_hash
from src.graphrag.models import GraphContribution
from src.graphrag.store import GraphStore

__all__ = ["GraphBuilder", "BuildReport", "load_corpus"]

logger = logging.getLogger(__name__)


def load_corpus(path: str | Path) -> list[QARecord]:
    """Read the canonical corpus JSONL (one QARecord per line)."""
    out: list[QARecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(QARecord.model_validate(json.loads(line)))
    return out


@dataclass
class BuildReport:
    total: int = 0
    added: int = 0
    skipped_unchanged: int = 0
    reconciled: int = 0
    withdrawn: int = 0
    rekeyed: int = 0    # same content moved to a new question_id (no LLM)
    ambiguous: int = 0  # several ids claim one content -> sent to the LLM
    failed: int = 0
    extraction_errors: int = 0
    documents: int = 0  # Document nodes applied
    facts_applied: int = 0
    supports_applied: int = 0
    failures: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total, "added": self.added,
            "skipped_unchanged": self.skipped_unchanged,
            "reconciled": self.reconciled, "withdrawn": self.withdrawn,
            "rekeyed": self.rekeyed, "ambiguous": self.ambiguous,
            "failed": self.failed, "extraction_errors": self.extraction_errors,
            "documents": self.documents, "facts_applied": self.facts_applied,
            "supports_applied": self.supports_applied,
            "failures": self.failures[:20], "seconds": round(self.seconds, 2),
        }


def graph_confirms(ev: dict | None, content_hash: str, mode: str) -> bool:
    """True iff the GRAPH itself shows this exact content is incorporated.

    The graph is the authority, not the checkpoint. A checkpoint entry can
    outlive the contribution it describes — a crash between
    ``withdraw_document()`` and ``apply_contribution()`` leaves the old proof in
    the file with nothing behind it in Neo4j, and skipping on that proof would
    silently lose the document.

    Conversely the graph can hold a contribution the checkpoint never recorded —
    a crash between ``apply_contribution()`` committing and ``mark_done()`` —
    and that document must NOT be re-extracted.

    Module-level so the read-only audit tool predicts the build by calling this
    very function rather than a copy that can drift.
    """
    if not ev or ev.get("content_hash") != content_hash:
        return False
    if normalize_extraction(mode) != EXTRACTION_SEMANTIC:
        return True
    return bool(ev.get("has_llm_facts"))


def churn_source(checkpoint, stable: str, doc_key: str, mode: str,
                 corpus_keys: set, graph_ev: dict) -> tuple:
    """(source key to re-key from, number of live claimants).

    A candidate only counts when its contribution is still IN the graph —
    re-keying from a key whose contribution was withdrawn would silently drop
    the LLM facts.
    """
    cands = checkpoint.churn_candidates(stable, doc_key, mode, corpus_keys)
    live = [k for k in cands
            if graph_ev.get(k) is not None
            and (normalize_extraction(mode) != EXTRACTION_SEMANTIC
                 or graph_ev[k].get("has_llm_facts"))]
    # None, never "": an empty string is falsy but IS NOT None, and the caller
    # tests `is not None` before re-keying
    return (live[0] if len(live) == 1 else None), len(live)


class _ExtractionPrefetcher:
    """Runs semantic extraction for upcoming documents concurrently.

    WHY only this step: extraction is the sole slow operation (~63 s/doc on a
    27B model) and the sole *pure* one — ``SemanticExtractor.extract`` reads
    the record plus read-only vocabulary maps and returns a fresh
    ``ExtractionResult``; it mutates no shared state. Everything that DOES
    mutate state (withdraw, apply_contribution, checkpoint) stays on the main
    thread, in corpus order.

    Semantics deliberately preserved:
      * results are consumed strictly in corpus order via ``take(doc_key)``,
        so the graph is written in the same sequence as a serial run;
      * a failure is captured and re-raised on ``take()``, i.e. at exactly the
        point the serial code would have raised — so one document failing
        never affects another's checkpoint entry;
      * at most ``concurrency`` extractions are ever in flight, matching the
        vLLM server's ``--max-num-seqs``;
      * ``concurrency <= 1`` disables threading entirely (synchronous path).
    """

    def __init__(self, extractor: SemanticExtractor, records: list,
                 concurrency: int, *, now_fn, log) -> None:
        self._extractor = extractor
        self._records = records
        self._now = now_fn
        self._log = log
        self._concurrency = max(1, int(concurrency))
        self._pool = None
        self._futures: "dict[str, object]" = {}
        self._next = 0
        if self._concurrency > 1 and records:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(
                max_workers=self._concurrency,
                thread_name_prefix="graphrag-extract")
            self._fill()

    # ── internals ─────────────────────────────────────────────────────────

    def _submit(self, rec) -> None:
        now = self._now()
        self._futures[rec.question_id] = (
            self._pool.submit(self._extractor.extract, rec, now=now), now)

    def _fill(self) -> None:
        """Keep the in-flight window full (bounded by concurrency)."""
        while self._pool is not None and len(self._futures) < self._concurrency \
                and self._next < len(self._records):
            self._submit(self._records[self._next])
            self._next += 1

    # ── main-thread API ───────────────────────────────────────────────────

    def take(self, doc_key: str):
        """Return ``(ExtractionResult, now)`` for ``doc_key``.

        Blocks until that document's extraction finishes. Re-raises its
        exception unchanged, so ExtractionError still classifies as an
        extraction error upstream.
        """
        if self._pool is None:
            # synchronous fallback (concurrency == 1): identical behaviour to
            # the original serial code path
            rec = next(r for r in self._records if r.question_id == doc_key)
            now = self._now()
            return self._extractor.extract(rec, now=now), now

        entry = self._futures.pop(doc_key, None)
        if entry is None:
            # Not prefetched (e.g. skipped ahead) — do it inline.
            rec = next(r for r in self._records if r.question_id == doc_key)
            now = self._now()
            return self._extractor.extract(rec, now=now), now
        future, now = entry
        try:
            return future.result(), now
        finally:
            # refill only after a slot frees, keeping the window bounded
            self._fill()

    def close(self) -> None:
        if self._pool is not None:
            for future, _now in self._futures.values():
                future.cancel()
            self._futures.clear()
            self._pool.shutdown(wait=True)
            self._pool = None


class GraphBuilder:
    """Incremental corpus → graph builder (store-agnostic)."""

    def __init__(
        self,
        store: GraphStore,
        config: GraphConfig,
        *,
        voc: Optional[Vocabulary] = None,
        llm_client=None,
        now_fn: Optional[Callable[[], str]] = None,
        checkpoint: Optional[GraphCheckpoint] = None,
        log=logger,
    ) -> None:
        self.store = store
        self.config = config
        self.voc = voc or load_vocabulary()
        self._llm = llm_client
        self._now = now_fn or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._log = log
        self.last_counters: dict = {"facts": 0, "supports": 0}
        # per-document LLM usage samples (instrumentation)
        self.usage_samples: list[dict] = []
        self.checkpoint = checkpoint or GraphCheckpoint(
            config.checkpoint_path,
            retry_failed=True,
            max_attempts=max(1, config.extract_attempts),
        )

    # ── public ────────────────────────────────────────────────────────────

    def run(
        self,
        corpus_path: str | Path,
        *,
        limit: Optional[int] = None,
        deterministic_only: bool = False,
        prune: bool = False,
        resume: bool = True,
        semantic_backfill: bool = False,
        migrate: bool = True,
    ) -> BuildReport:
        started = time.monotonic()
        report = BuildReport()
        all_records = load_corpus(corpus_path)
        # corpus membership for prune is the FULL corpus, never the slice
        corpus_keys = {r.question_id for r in all_records}

        # ── checkpoint migration (idempotent, additive) ────────────────────
        # Checkpoints written before the incremental fix carry no proof of
        # incorporation and no stable identity. Without them nothing is
        # provable and the whole corpus would go to the LLM — the full rebuild
        # this step exists to prevent. It never changes status, never rewrites
        # a hash, never deletes an entry, and re-running it is a no-op.
        if resume and migrate:
            from src.graphrag.migrate import migrate_checkpoint
            stats = migrate_checkpoint(self.checkpoint, all_records,
                                       store=self.store)
            self._log.info("[graph] checkpoint migration: %s", stats)

        if semantic_backfill:
            if deterministic_only:
                raise ValueError(
                    "--semantic-backfill requires the LLM: it cannot be "
                    "combined with --deterministic-only")
            if self._llm is None:
                raise ValueError(
                    "--semantic-backfill requires an LLM client (none was "
                    "configured)")
            # Select FIRST, slice SECOND: --limit N means "N documents that
            # still need semantic extraction", not "the first N corpus rows,
            # then filtered" (which would re-visit the same rows every run).
            records = [
                r for r in all_records
                if self.checkpoint.needs_extraction(
                    r.question_id, qa_content_hash(r), EXTRACTION_SEMANTIC,
                    stable_content_hash(r), corpus_keys)
            ]
            self._log.info(
                "[graph] semantic backfill: %d of %d document(s) still need "
                "semantic extraction", len(records), len(all_records))
        else:
            records = all_records

        if limit is not None:
            records = records[: max(0, int(limit))]
        report.total = len(records)

        # Publish run state for /api/graph/build-status. The build usually runs
        # detached, so the checkpoint file is the only channel back to the API.
        self.checkpoint.begin_run(
            total=report.total, now=self._now(), pid=os.getpid(),
            mode="prune" if (prune and self._llm is None) else
                 ("semantic-backfill" if semantic_backfill else
                  ("deterministic" if deterministic_only else "build")),
        )

        extractor = (
            SemanticExtractor(self._llm, self.voc,
                              max_chars=self.config.extract_max_chars)
            if (self._llm is not None and not deterministic_only)
            else None
        )
        # What this run is capable of producing — persisted per document only
        # after its contribution is actually applied.
        run_extraction = (EXTRACTION_SEMANTIC if extractor is not None
                          else EXTRACTION_DETERMINISTIC)

        # Pre-extraction pipelining. The LLM call is the only slow, purely
        # functional step (~63 s/doc vs ~1 ms for everything else), so it is
        # the only thing parallelized. Results are consumed IN CORPUS ORDER by
        # the single main thread below, which keeps merging, Neo4j writes,
        # provenance and checkpoint updates serialized and deterministic —
        # no concurrent writers, no second job system.
        #
        # Only documents that will ACTUALLY be processed are prefetched:
        # eagerly extracting a document the loop is about to skip would burn a
        # real LLM call (and GPU time) for nothing. This mirrors the loop's own
        # skip rule; anything mis-predicted still falls back to an inline call
        # inside take(), so the two can never disagree on correctness.
        # One batched round-trip for what the graph actually holds right now.
        # This is the evidence the skip decision rests on — see
        # _graph_confirms for why the checkpoint alone is not trusted.
        # The key set must include checkpoint keys that are NOT in the corpus:
        # a re-key source is by definition absent from the corpus, and deciding
        # whether its contribution is still present is exactly what tells a
        # rename apart from a document whose contribution was withdrawn.
        evidence_keys = set(corpus_keys)
        if resume:
            evidence_keys |= self.checkpoint.keys()
        graph_ev = (self.store.incorporated_evidence(evidence_keys)
                    if resume else {})

        candidates = (
            [r for r in records
             if resume is False
             or self._needs_llm(r, graph_ev, corpus_keys, run_extraction)]
            if extractor is not None else []
        )
        prefetch = _ExtractionPrefetcher(
            extractor, candidates, self.config.llm_concurrency,
            now_fn=self._now, log=self._log,
        ) if extractor is not None else None

        try:
            for rec in records:
                doc_key = rec.question_id
                h = qa_content_hash(rec)
                stable = stable_content_hash(rec)
                entry = self.checkpoint.get(doc_key) if resume else None
                ev = graph_ev.get(doc_key)
                in_graph = ev is not None

                # ── SKIP: the GRAPH shows this exact content is already
                # incorporated at the strength this run requires.
                #
                # The graph is the authority and the checkpoint is only a
                # cache, which is what makes both crash windows safe:
                #   * crash after apply_contribution committed but before
                #     mark_done -> no checkpoint entry, but the graph has it,
                #     so 0 LLM calls instead of paying twice for one document
                #   * crash after withdraw_document but before apply -> the
                #     stale applied_* proof is still in the checkpoint, but the
                #     graph no longer has it, so the document is rebuilt
                #
                # A deterministic-only pass does NOT satisfy a semantic run
                # (no llm-origin support), so backfill remains possible.
                if resume and self._graph_confirms(ev, h, run_extraction):
                    report.skipped_unchanged += 1
                    if not self.checkpoint.incorporation_for(
                            doc_key, h, stable, run_extraction):
                        # heal a checkpoint that lost track of a contribution
                        # the graph demonstrably holds (the crash window above)
                        self.checkpoint.mark_done(
                            doc_key, h, now=self._now(),
                            facts=entry.facts if entry else 0,
                            supports=entry.supports if entry else 0,
                            extraction=run_extraction, stable_hash=stable)
                    # In-memory only: skips do not write the checkpoint (they change
                    # nothing), so flush periodically to keep the UI moving during
                    # long unchanged stretches on a re-run.
                    self.checkpoint.update_run(
                        now=self._now(), last_doc=doc_key,
                        skipped_unchanged=report.skipped_unchanged,
                        processed=report.skipped_unchanged + report.documents + report.failed,
                    )
                    if report.skipped_unchanged % self.config.write_batch_size == 0:
                        self.checkpoint.flush_run()
                    continue

                # ── RE-KEY: same content, regenerated question_id ──────────
                # Exactly one other checkpoint key already holds this content,
                # so the document is unchanged and needs NO LLM call. But the
                # graph is keyed by question_id, so the contribution has to
                # move to the new key — otherwise the prune pass below sees
                # the old key missing from the corpus and withdraws a document
                # that is still very much present.
                #
                # Several claimants are AMBIGUOUS: they fall through to the
                # LLM rather than being silently merged into one document.
                churn_from = None
                if resume:
                    churn_from, n_live = self._churn_source(
                        stable, doc_key, run_extraction, corpus_keys, graph_ev)
                    if not churn_from and n_live > 1:
                        report.ambiguous += 1
                        self._log.warning(
                            "[graph] %s: %d checkpoint keys claim this exact "
                            "content — not assuming unchanged, sending to the "
                            "LLM", doc_key, n_live)
                if churn_from is not None:
                    self._rekey_document(churn_from, rec, h, stable,
                                         run_extraction)
                    report.rekeyed += 1
                    self.checkpoint.update_run(
                        now=self._now(), current_doc=None, last_doc=doc_key,
                        processed=report.skipped_unchanged + report.documents
                                  + report.failed + report.rekeyed,
                    )
                    continue

                # ── action classification ───────────────────────────────────
                # A semantic upgrade of an already-deterministic document lands
                # here as "reconcile": the document IS in the graph, so its
                # previous (deterministic-only) contribution is withdrawn
                # document-scoped before the combined deterministic+semantic
                # contribution is applied. That is what prevents stale
                # deterministic facts coexisting with the new semantic ones,
                # and it reuses the existing reconcile path unchanged.
                if entry is None or entry.status != "done":
                    action = "add" if (entry is None and not in_graph) else \
                        ("reconcile" if in_graph else "retry")
                else:  # done, but hash differs OR a weaker extraction pass
                    action = "reconcile" if in_graph else "retry"

                # in-memory; persisted by the mark_done/mark_failed save below
                self.checkpoint.update_run(now=self._now(), current_doc=doc_key)
                try:
                    self._process_one(rec, h, action, extractor,
                                      prefetch=prefetch)
                    if action in ("add", "retry"):
                        report.added += 1
                    elif action == "reconcile":
                        report.reconciled += 1
                    counters = self.last_counters
                    self.checkpoint.update_run(
                        now=self._now(), current_doc=None, last_doc=doc_key,
                        added=report.added, reconciled=report.reconciled,
                        failed=report.failed,
                        processed=report.skipped_unchanged + report.documents + 1
                                  + report.failed,
                    )
                    # Recorded ONLY here — after _process_one() applied the
                    # contribution to the store without raising. A document is
                    # never marked "semantic" on the strength of an attempt.
                    self.checkpoint.mark_done(
                        doc_key, h, now=self._now(),
                        facts=counters["facts"], supports=counters["supports"],
                        extraction=run_extraction, stable_hash=stable)
                    report.documents += 1
                    report.facts_applied += counters["facts"]
                    report.supports_applied += counters["supports"]
                except ExtractionError as e:
                    report.failed += 1
                    report.extraction_errors += 1
                    report.failures.append({"doc": doc_key, "error": str(e)[:300]})
                    self._run_progress(report, doc_key)
                    self.checkpoint.mark_failed(doc_key, h, str(e),
                                                now=self._now(),
                                                stable_hash=stable)
                    if report.failed >= self.config.max_failures:
                        self._log.warning("aborting: failures exceed max_failures=%d",
                                          self.config.max_failures)
                        break
                except Exception as e:  # noqa: BLE001
                    report.failed += 1
                    report.failures.append({"doc": doc_key, "error": str(e)[:300]})
                    self._run_progress(report, doc_key)
                    self.checkpoint.mark_failed(doc_key, h, str(e),
                                                now=self._now(),
                                                stable_hash=stable)
                    self._log.exception("graph ingest failed for %s", doc_key)

            # ── WITHDRAW: checkpointed docs missing from the corpus ─────────
            if prune:
                for doc_key in sorted(self.checkpoint.keys() - corpus_keys):
                    self.store.withdraw_document(doc_key, now=self._now())
                    self.checkpoint.remove(doc_key)
                    report.withdrawn += 1

            report.seconds = time.monotonic() - started
            self.checkpoint.update_run(
                now=self._now(), withdrawn=report.withdrawn,
                seconds=round(report.seconds, 2),
                processed=report.skipped_unchanged + report.documents + report.failed,
            )
        except BaseException as e:  # noqa: BLE001
            # SIGINT/SIGTERM or an unexpected fault: record a terminal state so
            # the UI shows "failed" instead of a permanently "running" build.
            self.checkpoint.end_run(now=self._now(), state="failed", error=repr(e))
            raise
        finally:
            if prefetch is not None:
                prefetch.close()
        self.checkpoint.end_run(now=self._now(), state="completed")
        summary = self.usage_summary()
        if summary:
            self._log.info("[graph] LLM usage: %s", summary)
        self._log.info("[graph] build done: %s", report.to_dict())
        return report

    def _run_progress(self, report: "BuildReport", doc_key: str) -> None:
        """Mirror report counters into the checkpoint run block (in-memory;
        the following mark_done/mark_failed performs the atomic save)."""
        self.checkpoint.update_run(
            now=self._now(), current_doc=None, last_doc=doc_key,
            added=report.added, reconciled=report.reconciled,
            failed=report.failed,
            processed=report.skipped_unchanged + report.documents + report.failed,
        )

    # ── per-document unit ─────────────────────────────────────────────────

    @staticmethod
    def _graph_confirms(ev: dict | None, content_hash: str,
                        mode: str) -> bool:
        """Delegates to the module-level rule the audit tool also calls."""
        return graph_confirms(ev, content_hash, mode)

    def _churn_source(self, stable: str, doc_key: str, mode: str,
                      corpus_keys: set, graph_ev: dict) -> tuple:
        """Delegates to the module-level rule the audit tool also calls."""
        return churn_source(self.checkpoint, stable, doc_key, mode,
                            corpus_keys, graph_ev)

    def _needs_llm(self, rec: QARecord, graph_ev: dict, corpus_keys: set,
                   mode: str) -> bool:
        """Single source of truth for "will this document reach the LLM?".

        The prefetcher and the main loop both call this, so the prefetcher can
        never burn a real GPU call on a document the loop is about to skip.
        """
        h = qa_content_hash(rec)
        if self._graph_confirms(graph_ev.get(rec.question_id), h, mode):
            return False
        src, n = self._churn_source(stable_content_hash(rec), rec.question_id,
                                    mode, corpus_keys, graph_ev)
        return not (src and n == 1)

    def _rekey_document(self, old_key: str, rec: QARecord, h: str,
                        stable: str, extraction: str) -> None:
        """Move an existing contribution to a new question_id — no LLM call.

        The content is provably unchanged (same stable hash) and exactly one
        other checkpoint key holds it, so re-extracting would burn GPU time to
        regenerate the same facts. But the graph is keyed by question_id, so
        the contribution has to MOVE: the old key is withdrawn and the new one
        applied. Withdraw-then-apply keeps a fact shared with other documents
        from ever being observed at doc_count 0 by another reader.

        The deterministic half is recomputed from the current record — cheaper
        than reading it back and it keeps ``source_field`` provenance exact.
        Only the LLM half is read back from the graph. If there is none (a
        deterministic-only document), the deterministic half alone is applied.
        """
        now = self._now()
        llm = self.store.read_llm_contribution(
            old_key, rename_to=rec.question_id)
        self.store.withdraw_document(old_key, now=now)
        doc, det = build_contribution(rec, self.voc, now=now)
        contrib = (self._merge_contributions(det, llm)
                   if llm is not None else det)
        counters = self.store.apply_contribution(contrib, now=now, doc=doc)
        # The old key is absent from the corpus, so its checkpoint entry must
        # go too — leaving it behind would make the next prune pass withdraw
        # the very document we just re-keyed.
        self.checkpoint.remove(old_key)
        self.checkpoint.mark_done(
            rec.question_id, h, now=now, facts=counters["facts"],
            supports=counters["supports"], extraction=extraction,
            stable_hash=stable)
        self.last_counters = {"facts": counters["facts"],
                              "supports": counters["supports"]}
        self._log.info("[graph] re-keyed %s -> %s (content unchanged, "
                       "no LLM call, %d facts)", old_key, rec.question_id,
                       counters["facts"])

    def _process_one(self, rec: QARecord, h: str, action: str,
                     extractor: Optional[SemanticExtractor],
                     prefetch: "Optional[_ExtractionPrefetcher]" = None) -> None:
        doc_key = rec.question_id

        # Take the concurrently-computed extraction BEFORE any store mutation:
        # if it failed, the exception propagates here and the document is left
        # entirely untouched (no half-written state), exactly as in the serial
        # path. Falls back to a synchronous call when there is no prefetcher.
        sem_res = None
        if extractor is not None:
            if prefetch is not None:
                sem_res, now = prefetch.take(doc_key)
            else:
                now = self._now()
                sem_res = extractor.extract(rec, now=now)
        else:
            now = self._now()

        # reconcile: retract this document's PREVIOUS contribution first
        # (document-scoped — facts supported by other documents survive)
        if action == "reconcile":
            self.store.withdraw_document(doc_key, now=now)

        # deterministic pass (metadata only — always)
        doc, det = build_contribution(rec, self.voc, now=now)

        # semantic pass (LLM via the existing generation architecture)
        contrib = det
        if extractor is not None:
            sem = GraphContribution(
                doc_key=doc_key,
                nodes=list(sem_res.entities),
                facts=list(sem_res.facts),
                supports=list(sem_res.supports),
            )
            contrib = self._merge_contributions(det, sem)
            self._record_usage(doc_key, sem_res)
            if sem_res.rejected:
                self._log.debug(
                    "[graph] %s: %d LLM item(s) rejected: %s",
                    doc_key, len(sem_res.rejected),
                    sorted({r["reason"] for r in sem_res.rejected}))

        counters = self.store.apply_contribution(contrib, now=now, doc=doc)
        self.last_counters = {
            "facts": counters["facts"], "supports": counters["supports"]}

    def _record_usage(self, doc_key: str, sem_res) -> None:
        """Log + accumulate per-document LLM usage (successful extractions).

        Exists to answer one open question with data instead of estimates: how
        many OUTPUT tokens does extraction actually need? ``completion_tokens``
        is that number per document, and ``finish_reason == "length"`` marks a
        response still being capped. The run summary reports the distribution
        so the 18000 budget can be right-sized from evidence.
        """
        usage = dict(getattr(sem_res, "llm_usage", None) or {})
        if not usage:
            return
        completion = usage.get("completion_tokens")
        latency_ms = usage.get("latency_ms")
        finish = usage.get("finish_reason")
        self._log.info(
            "[graph] %s: completion_tokens=%s latency_ms=%s finish_reason=%s "
            "facts=%d",
            doc_key, completion,
            None if latency_ms is None else round(float(latency_ms)),
            finish, len(getattr(sem_res, "facts", []) or []))
        if isinstance(completion, int):
            self.usage_samples.append({
                "doc": doc_key,
                "completion_tokens": completion,
                "latency_ms": (None if latency_ms is None
                               else round(float(latency_ms))),
                "finish_reason": finish,
            })
        if finish == "length":
            self._log.warning(
                "[graph] %s: response hit the output cap "
                "(finish_reason=length, completion_tokens=%s) — extraction "
                "may be truncated; consider raising "
                "GRAPHRAG_EXTRACT_MAX_TOKENS", doc_key, completion)

    def usage_summary(self) -> dict:
        """Aggregate of the per-document samples (empty when none collected)."""
        toks = sorted(s["completion_tokens"] for s in self.usage_samples)
        if not toks:
            return {}
        lats = sorted(s["latency_ms"] for s in self.usage_samples
                      if s["latency_ms"] is not None)

        def pct(vals, q):
            return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else None

        return {
            "samples": len(toks),
            "completion_tokens": {
                "min": toks[0], "median": pct(toks, 0.5),
                "p90": pct(toks, 0.9), "p99": pct(toks, 0.99), "max": toks[-1],
            },
            "latency_ms": ({"min": lats[0], "median": pct(lats, 0.5),
                            "p90": pct(lats, 0.9), "max": lats[-1]}
                           if lats else {}),
            "hit_output_cap": sum(1 for s in self.usage_samples
                                  if s["finish_reason"] == "length"),
        }

    @staticmethod
    def _merge_contributions(a: GraphContribution, b: GraphContribution) -> GraphContribution:
        nodes = list(a.nodes) + [n for n in b.nodes
                                 if n.key not in {x.key for x in a.nodes}]
        facts = list(a.facts) + [f for f in b.facts
                                 if f.fact_key not in {x.fact_key for x in a.facts}]
        supports = list(a.supports) + [s for s in b.supports
                                       if (s.fact_key, s.doc_key) not in
                                       {(x.fact_key, x.doc_key) for x in a.supports}]
        return GraphContribution(doc_key=a.doc_key, nodes=nodes, facts=facts,
                                 supports=supports)
