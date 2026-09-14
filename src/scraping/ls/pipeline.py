"""Per-(loksabha, session) orchestration for the Lok Sabha crawler.

LS sessions reset per Lok Sabha (unlike RS's global session numbers), so the
staging unit is the composite (loksabha × session)::

    data/parliamentary-qa/lok-sabha/
      lok-<N>/session-<M>/
        qa.jsonl         # QARecord-shaped rows merged by id (byte-stable)
        documents/       # content-keyed official files (one fetch, one file,
                         # N record references — grouped-answer annexes)
        manifest.json    # sha-256 state for incremental byte-stable re-runs
      last_run.json      # run summary at the house root

Flow per loksabha:
  1. inventory   one era-routed discovery walk (api_ls listing OR DSpace
                 discover search) per ministry — cached for the whole run; a
                 failed inventory marks THAT loksabha's sessions failed and
                 the run continues with the next loksabha
  2. sessions    target sessions = calendar ∪ inventory rows ∩ config/CLI
                 window; sessions without rows are honestly "empty"
  3. normalize   RawLsQuestion → QARecord-shaped dicts (ids ls-<lok>-<ses>-*
  4. documents   eng/hin slots: blank-link parking (document-not-published),
                 annex suffix retry, DSpace resolution, magic sniff, stage
  5. answers     per-field selection: question → inline primary,
                 answer → document primary, deterministic arbitration
                 (src/scraping/ls/text_selection.py)
  6. emit        qa.jsonl merged by id + manifest.json — written ONLY when
                 content changed (byte-stable no-change re-runs)

CLI: ``python -m src.scraping.ls.pipeline [--loksabha 18,16-17] [--sessions …]
[--ministry slug] [--dry-run] [--no-documents] [--no-retry-failed]
[--max-records N] [--coverage-ref workbook.xlsx] [--config …] [--root …]``

Exit codes: 0 = success, 2 = usage/config error, 3 = partial run (some
loksabha inventories/sessions failed — details in last_run.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from src.scraping import records as rec_utils
from src.scraping.formats import DocFacts
from src.scraping.http import CrawlHttpClient, HttpApiError, HttpTransportError
from src.scraping.ls import discovery
from src.scraping.ls.client import LsClient
from src.scraping.ls.config import (
    ERA_API_LS,
    LsConfigError,
    api_endpoint,
    elib_endpoint,
    era_for,
    http_headers,
    http_kwargs,
    load_config,
    output_root,
    resolve_ministries,
    window,
)
from src.scraping.ls.discovery import Inventory, dedupe_rows
from src.scraping.ls.documents import UrlCache, plan_slots, process_slot
from src.scraping.ls.extract import extract_qa
from src.scraping.ls.normalize import build_record, sort_key, utcnow_iso
from src.scraping.ls.text_selection import (
    in_scope_text,
    is_richer,
    measure,
    select_answer,
    select_question,
    split_was_boundary_based,
    structural_signals,
)
from src.scraping.manifest import load_manifest, manifests_equal, write_manifest
from src.utils.atomic_io import write_bytes_atomic

CRAWLER_VERSION = "ls-1.0"
QA_JSONL = "qa.jsonl"

#: Retain an already-staged document answer when a fresh extraction of the same
#: document comes back structurally poorer (lost annexure rows / sub-parts).
#: Turn off to always trust the newest extraction.
GUARD_EXTRACTION_REGRESSIONS = True

#: manifest entries carried from the documents stage
_DOC_EXTRA_KEYS = ("suffix_retried", "retry_url", "dspace_resolution", "resolved_url")

API_SOURCE_LABEL = "sansad.in/api_ls/qetFilteredQuestionsAns"
DSPACE_SOURCE_LABEL = "elibrary.sansad.in/discover-search"


# ── run wiring ───────────────────────────────────────────────────────────────

@dataclass
class RunOptions:
    loksabhas: list[int] | None = None        # None => calendar ∩ config window
    sessions: list[int] | None = None         # None => calendar ∪ inventory ∩ config window
    ministry_slugs: list[str] | None = None   # None => all configured
    fetch_documents: bool = True
    dry_run: bool = False
    retry_failed: bool = True
    max_records: int | None = None            # per-session cap (live-smoke aid)
    coverage_ref: str | None = None           # xlsx for the coverage audit


@dataclass
class CrawlContext:
    cfg: dict[str, Any]
    opts: RunOptions
    http: CrawlHttpClient
    root: Path
    ministries: list[dict[str, Any]]
    ls: LsClient = field(init=False)
    _calendar: dict[int, list[int]] | None = None
    _calendar_error: str | None = None
    _inventory: dict[int, Inventory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        api_base, api_size, _referer = api_endpoint(self.cfg)
        elib_base, elib_size = elib_endpoint(self.cfg)
        self.ls = LsClient(
            self.http,
            api_base_url=api_base,
            api_page_size=api_size,
            elib_base_url=elib_base,
            elib_page_size=elib_size,
        )

    @property
    def policy(self) -> dict[str, Any]:
        return self.cfg.get("policy") or {}


@dataclass
class SessionReport:
    loksabha: int
    session: int | None
    status: str = "ok"      # ok | empty | unchanged | updated | dry-run | failed
    ministries: dict[str, int] = field(default_factory=dict)
    zero_ministries: list[str] = field(default_factory=list)
    records: int = 0
    capped_from: int = 0                     # >0 when --max-records truncated
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    docs: dict[str, int] = field(
        default_factory=lambda: {"good": 0, "partial": 0, "broken": 0, "missing": 0}
    )
    doc_files_written: int = 0
    attention: list[dict[str, str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loksabha": self.loksabha, "session": self.session,
            "status": self.status,
            "ministries": self.ministries,
            "zero_ministries": self.zero_ministries,
            "records": self.records, "capped_from": self.capped_from,
            "added": self.added, "changed": self.changed,
            "unchanged": self.unchanged,
            "docs": self.docs, "doc_files_written": self.doc_files_written,
            "attention": self.attention, "failures": self.failures,
        }


# ── calendar / inventory caches ──────────────────────────────────────────────

def calendar(ctx: CrawlContext) -> dict[int, list[int]]:
    """Official Lok Sabha → sessions calendar (fetched once per run)."""
    if ctx._calendar is None and ctx._calendar_error is None:
        try:
            ctx._calendar = ctx.ls.loksabha_calendar()
        except (HttpApiError, HttpTransportError) as exc:
            ctx._calendar = {}
            ctx._calendar_error = str(exc)
    return ctx._calendar or {}


def inventory(ctx: CrawlContext, lok: int) -> Inventory:
    """Era-routed per-loksabha inventory (cached per run; raises on failure)."""
    if lok not in ctx._inventory:
        era = era_for(ctx.cfg, lok)
        ctx._inventory[lok] = discovery.build_inventory(ctx.ls, lok, ctx.ministries, era)
    return ctx._inventory[lok]


def _target_sessions(ctx: CrawlContext, lok: int, inv: Inventory | None) -> list[int] | None:
    """Session universe for one loksabha. None when NEITHER the calendar nor
    an inventory could determine it (fail-closed, reported)."""
    if ctx.opts.sessions is not None:
        return sorted(set(ctx.opts.sessions))
    lo, hi, exclude = window(ctx.cfg, "sessions")
    universe = set(calendar(ctx).get(lok, []))
    if inv is not None:
        universe |= set(inv.sessions)
    if not universe:
        return None
    return sorted(
        s for s in universe
        if (lo is None or s >= lo) and (hi is None or s <= hi) and s not in exclude
    )


# ── per-field text selection (inline question / document answer, arbitrated) ─

def select_record_text(
    rec: dict[str, Any],
    inline_q: str,
    inline_a: str,
    eng_facts: DocFacts | None,
    eng_body: bytes | None,
    fallback_wanted: bool,
    *,
    documents_enabled: bool = True,
) -> str | None:
    """Fill question/answer by PER-FIELD arbitration. Returns the failure
    cause for the attention log, or None.

    Supersedes the old inline-first ladder (``apply_extraction``). The old
    ladder only parsed the official English PDF when ``answer_text`` was
    empty, so a non-empty inline ``answerText`` silently discarded the richer
    PDF answer — including its annexure tables. Selection now always parses a
    usable document and then decides per field (see ``text_selection``):

      * ``question_text`` → inline primary, document fallback
      * ``answer_text``   → document primary, inline fallback

    The cause vocabulary is unchanged (``extract-disabled`` /
    ``legacy-format-not-extracted`` / ``extract-failed`` /
    ``english-document-unavailable``, plus ``documents-disabled``), so the
    manifest and downstream consumers keep working.
    """
    meta = rec["metadata"]

    doc_q: str | None = None
    doc_a: str | None = None
    boundary_split = True          # only meaningful once extraction succeeds
    meta_cause: str | None = None  # stored in answer_unavailable_cause
    log_cause: str | None = None   # richer form for the attention log

    if not documents_enabled:
        meta_cause = log_cause = "documents-disabled"
    elif not fallback_wanted:
        meta_cause = log_cause = "extract-disabled"
    elif eng_facts is None or eng_facts.doc_class not in ("good", "partial"):
        meta_cause = log_cause = "english-document-unavailable"
    elif eng_facts.format not in ("pdf", "docx"):
        meta_cause = log_cause = "legacy-format-not-extracted"
    else:
        qa, reason = extract_qa(eng_body or b"", eng_facts.format)
        if not qa:
            meta_cause = "extract-failed"
            log_cause = f"extract-failed: {reason}"
        else:
            doc_q, doc_a = qa
            boundary_split = split_was_boundary_based(doc_a)

    q_sel = select_question(inline_q, doc_q, boundary_split=boundary_split)
    a_sel = select_answer(inline_a, doc_a, boundary_split=boundary_split,
                          unavailable_cause=log_cause)

    rec["question_text"] = q_sel.text
    rec["answer_text"] = a_sel.text

    # answer_source keeps its existing vocabulary (incl. 'unavailable')
    if a_sel.text:
        meta["answer_source"] = a_sel.source
        meta.pop("answer_unavailable_cause", None)
    else:
        meta["answer_source"] = "unavailable"
        meta["answer_unavailable_cause"] = meta_cause or "no-usable-answer"

    # Per-field provenance. Written ONLY when a real choice was made between
    # two non-empty candidates, so single-source records stay byte-stable and
    # do not churn qa_content_hash. See text_selection.ALWAYS_RECORD_PROVENANCE.
    if q_sel.provenance:
        meta["question_text_source"] = q_sel.source
    if a_sel.provenance:
        meta["answer_text_source"] = a_sel.source
    if q_sel.had_choice or a_sel.had_choice:
        meta["text_selection_reason"] = f"q={q_sel.reason}; a={a_sel.reason}"

    return None if a_sel.text else (log_cause or "no-usable-answer")


def guard_staged_answer(new_row: dict[str, Any],
                        old_row: dict[str, Any] | None) -> dict[str, Any]:
    """Keep an already-staged richer document answer over a poorer inline one.

    Covers the re-crawl-with-documents-disabled (or extract-failed) case: the
    fresh row would otherwise carry only inline text and ``merge_by_id`` would
    overwrite the richer staged answer on byte difference.

    Fires only when the OLD staged row's answer is document-extracted and
    materially richer than the new one. A fresh document-extracted answer
    always wins over an old one (the normal update path). Only ``answer_text``
    is guarded — ``question_text`` deliberately prefers the cleaner inline form.

    ``merge_by_id`` is shared across houses and is intentionally left
    untouched; this reconciliation is LS-local.
    """
    if not old_row:
        return new_row
    old_meta = old_row.get("metadata") or {}
    if old_meta.get("answer_source") != "document-extract":
        return new_row
    new_meta = new_row.setdefault("metadata", {})

    if new_meta.get("answer_source") == "document-extract":
        # Both document-derived. A fresh extraction normally wins — but it must
        # not be allowed to silently drop an annexure, which is what a partial
        # parse or a re-paginated PDF looks like. Retain the staged copy ONLY
        # on a structural regression (numeric / annexure / sub-part loss);
        # never on length or spacing, which shift with re-pagination and with
        # PDF text-extraction whitespace artifacts.
        if not GUARD_EXTRACTION_REGRESSIONS:
            return new_row
        # Compare only the in-scope prefix: _split_question_answer sweeps the
        # NEXT question of a multi-question PDF into this record's answer, and
        # that out-of-scope tail must not count as information the fresh
        # extraction lost (ls-16-14-4262 carries question 4247's whole Q&A).
        lost = structural_signals(measure(in_scope_text(old_row.get("answer_text"))),
                                  measure(in_scope_text(new_row.get("answer_text"))))
        if not lost:
            return new_row
        reason = "staged-document-retained:extraction-regression:" + ",".join(lost)
    elif is_richer(old_row.get("answer_text"), new_row.get("answer_text")):
        reason = "staged-document-retained"
    else:
        return new_row

    new_row["answer_text"] = old_row["answer_text"]
    new_meta["answer_source"] = "document-extract"
    new_meta["answer_text_source"] = "document-extract"
    new_meta["text_selection_reason"] = reason
    new_meta.pop("answer_unavailable_cause", None)
    return new_row


# ── failed-slot bookkeeping (mirror of the RS shape) ─────────────────────────

def _prior_failed_entry(old_manifest: dict[str, Any] | None, rid: str, lang: str) -> dict | None:
    if not old_manifest:
        return None
    for entry in old_manifest.get("failed_slots") or []:
        if entry.get("id") == rid and entry.get("lang") == lang:
            return entry
    return None


def _failed_slot_entry(rid: str, url: str | None, facts: DocFacts, lang: str) -> dict[str, Any]:
    if facts.cause == "document-not-published":
        # blank upstream link — parked permanently per operator directive
        # (the 69 OCEAN DEVELOPMENT rows are unrecoverable BY DESIGN); the
        # sparse Hindi links share the same honest marker.
        alternate = {"source": None, "url": None, "status": "unrecoverable"}
    else:
        # transient/upstream failures are re-attempted on the next crawl run
        alternate = {"source": "next-crawl-retry", "url": None, "status": "pending"}
    return {
        "id": rid, "lang": lang,
        "class": facts.doc_class,             # broken | missing
        "http": facts.http_status,
        "cause": facts.cause,
        "url": url,
        "alternate": alternate,
    }


# ── session orchestration ────────────────────────────────────────────────────

def session_dir_of(root: Path, lok: int, ses: int) -> Path:
    return root / f"lok-{lok}" / f"session-{ses}"


def crawl_session(
    ctx: CrawlContext, inv: Inventory, ses: int, rows: list
) -> SessionReport:
    lok = inv.loksabha
    report = SessionReport(loksabha=lok, session=ses)

    mcfg_by_slug = {m["slug"]: m for m in ctx.ministries}
    rows = [q for q in dedupe_rows(rows) if q.ministry_slug in mcfg_by_slug]
    for m in ctx.ministries:
        n = sum(1 for q in rows if q.ministry_slug == m["slug"])
        report.ministries[m["slug"]] = n
        if n == 0:
            report.zero_ministries.append(m["slug"])

    total_rows = len(rows)
    if ctx.opts.max_records is not None and total_rows > ctx.opts.max_records:
        rows = rows[: ctx.opts.max_records]     # deterministic: rows pre-sorted
        report.capped_from = total_rows

    pairs = [(build_record(q, mcfg_by_slug[q.ministry_slug]), q) for q in rows]
    report.records = len(pairs)

    if ctx.opts.dry_run:
        report.status = "dry-run"
        return report
    if not pairs:
        report.status = "empty"
        return report

    session_dir = session_dir_of(ctx.root, lok, ses)
    old_manifest = load_manifest(session_dir)
    cache = UrlCache()
    doc_manifest: list[dict[str, Any]] = []
    failed_slots: list[dict[str, Any]] = []
    fallback_wanted = ctx.policy.get("extract_fallback", True)
    suffix_retry = ctx.policy.get("annex_suffix_retry", True)

    for rec, q in pairs:
        meta = rec["metadata"]
        rid = rec["question_id"]
        # Capture the inline API candidates BEFORE any document extraction:
        # selection is per field, so both candidates must stay visible. The
        # old ladder stamped answer_source="inline" here and then only parsed
        # the PDF when answer_text was empty, which is what let a shorter
        # inline answer silently win.
        inline_q, inline_a = rec["question_text"], rec["answer_text"]

        eng_facts: DocFacts | None = None
        eng_body: bytes | None = None

        if ctx.opts.fetch_documents:
            for slot in plan_slots(q):
                carried = _prior_failed_entry(old_manifest, rid, slot.lang)
                if carried and not ctx.opts.retry_failed:
                    failed_slots.append(carried)
                    report.docs[carried["class"]] += 1
                    meta["documents"][slot.lang] = {
                        "class": carried["class"], "cause": carried["cause"],
                    }
                    continue

                outcome = process_slot(
                    ctx.http, q, slot, session_dir, rid,
                    suffix_retry=suffix_retry, cache=cache,
                )
                facts = outcome.result.facts
                report.docs[facts.doc_class] += 1
                if outcome.written:
                    report.doc_files_written += 1
                doc_summary: dict[str, Any] = {"class": facts.doc_class}
                if outcome.result.path:
                    doc_summary["path"] = outcome.result.path
                    doc_summary["format"] = facts.format
                if facts.cause:
                    doc_summary["cause"] = facts.cause
                meta["documents"][slot.lang] = doc_summary

                if facts.doc_class in ("broken", "missing"):
                    failed_slots.append(
                        _failed_slot_entry(rid, slot.url, facts, slot.lang)
                    )
                else:
                    entry = {
                        "key": outcome.result.key, "id": rid,
                        "path": outcome.result.path,
                        **{k: v for k, v in outcome.extra.items()
                           if k in _DOC_EXTRA_KEYS},
                        **facts.as_manifest(),
                    }
                    doc_manifest.append(entry)

                if slot.lang == "eng":
                    eng_facts, eng_body = facts, outcome.body

        # Per-field selection. Always runs; `documents_enabled` gates whether a
        # document candidate exists at all. Returns the attention-log cause.
        cause = select_record_text(
            rec, inline_q, inline_a, eng_facts, eng_body, fallback_wanted,
            documents_enabled=bool(ctx.opts.fetch_documents),
        )
        if cause is not None:
            report.attention.append(
                {"id": rid, "reason": f"no usable answer ({cause})"}
            )

    # emit qa.jsonl (merge by id; write only on byte difference)
    existing_rows = rec_utils.load_jsonl(session_dir / QA_JSONL)
    staged_by_id = {r.get("question_id"): r for r in existing_rows}
    # LS-local guard: a poorer re-crawl (documents disabled / extract failed)
    # must not overwrite a richer already-staged document answer. merge_by_id
    # is shared across houses and is deliberately left untouched.
    new_rows = [guard_staged_answer(rec, staged_by_id.get(rec["question_id"]))
                for rec, _ in pairs]
    merged, stats = rec_utils.merge_by_id(
        existing_rows,
        new_rows,
        key=lambda r: r["question_id"],
        sort_key=sort_key,
    )
    report.added, report.changed, report.unchanged = (
        stats["added"], stats["changed"], stats["unchanged"])

    now = utcnow_iso()
    for row in merged:
        if "scraped_at" not in row:
            row["scraped_at"] = now
    payload = rec_utils.serialize_jsonl(merged)
    qa_path = session_dir / QA_JSONL
    qa_written = not qa_path.exists() or qa_path.read_bytes() != payload
    if qa_written:
        session_dir.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(qa_path, payload)

    # manifest (write only on semantic change)
    manifest = {
        "source": API_SOURCE_LABEL if inv.era == ERA_API_LS else DSPACE_SOURCE_LABEL,
        "house": "lok-sabha",
        "loksabha": lok,
        "session": ses,
        "crawler_version": CRAWLER_VERSION,
        "generated_at": now,
        "ministries": sorted({q.ministry_slug for _, q in pairs}),
        "policy": {
            "extract_fallback": bool(fallback_wanted),
            "annex_suffix_retry": bool(suffix_retry),
        },
        "records": [
            {
                "id": row["question_id"],
                "row_sha256": rec_utils.row_sha256(row),
                "qtype": (row["metadata"] or {}).get("question_type"),
                "ministry": (row["metadata"] or {}).get("ministry"),
                "answer_source": (row["metadata"] or {}).get("answer_source"),
            }
            for row in merged
        ],
        "documents": sorted(doc_manifest, key=lambda d: d["key"]),
        "failed_slots": sorted(failed_slots, key=lambda d: (d["id"], d["lang"])),
        "attention": sorted(report.attention, key=lambda d: d["id"]),
    }
    manifest_written = not manifests_equal(old_manifest, manifest)
    if manifest_written:
        session_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(session_dir, manifest)

    report.status = (
        "updated" if (qa_written or manifest_written or report.doc_files_written)
        else "unchanged"
    )
    return report


def crawl_loksabha(ctx: CrawlContext, lok: int) -> list[SessionReport]:
    try:
        inv = inventory(ctx, lok)
    except (HttpApiError, HttpTransportError) as exc:
        era = era_for(ctx.cfg, lok)
        sessions = _target_sessions(ctx, lok, None)
        if sessions is None:
            return [SessionReport(
                loksabha=lok, session=None, status="failed",
                failures=[f"inventory failed for loksabha {lok} ({era}): {exc} "
                          f"(calendar sessions were also undeterminable)"],
            )]
        return [SessionReport(
            loksabha=lok, session=ses, status="failed",
            failures=[f"inventory failed for loksabha {lok} ({era}): {exc}"],
        ) for ses in sessions]

    sessions = _target_sessions(ctx, lok, inv)
    if sessions is None:
        return [SessionReport(
            loksabha=lok, session=None, status="empty",
            failures=["no sessions determinable: empty inventory and no calendar"],
        )]

    n_ses = len(sessions)
    total_rows = sum(len(v) for v in inv.sessions.values())
    print(f"[lok-{lok}] {total_rows} rows across {n_ses} sessions", flush=True)
    out: list[SessionReport] = []
    for j, ses in enumerate(sessions, 1):
        rows = inv.sorted_rows(ses)
        if not rows:
            rep = SessionReport(loksabha=lok, session=ses, status="empty")
            for m in ctx.ministries:
                rep.ministries[m["slug"]] = 0
            rep.zero_ministries = [m["slug"] for m in ctx.ministries]
            out.append(rep)
            print(f"  [lok-{lok}] {j}/{n_ses} session-{ses}: empty", flush=True)
            continue
        print(f"  [lok-{lok}] {j}/{n_ses} session-{ses}: {len(rows)} rows ...", flush=True)
        rep = crawl_session(ctx, inv, ses, rows)
        out.append(rep)
        print(f"  [lok-{lok}] {j}/{n_ses} session-{ses}: {rep.status}"
              f" records={rep.records} +{rep.added} ~{rep.changed}", flush=True)
    return out


# ── coverage audit (Excel as REFERENCE — never the source of truth) ──────────

def coverage_audit(reports: list[SessionReport], xlsx_path: str) -> dict[str, Any]:
    """Compare per-(loksabha, session) staged counts against the frozen
    workbook. Read-only reference check: divergences are REPORTED, never
    reconciled from the Excel."""
    import pandas as pd  # lazy: only needed for the optional audit

    df = pd.read_excel(xlsx_path)
    reference: dict[tuple[int, int], int] = {}
    for (lok, ses), n in df.groupby(["lokNo", "sessionNo"]).size().items():
        reference[(int(lok), int(ses))] = int(n)
    staged = {
        (r.loksabha, r.session): r
        for r in reports if r.session is not None and r.status not in ("failed",)
    }
    entries = []
    for key in sorted(set(reference) | set(staged)):
        rep = staged.get(key)
        found = rep.records if rep else 0
        ref = reference.get(key, 0)
        entries.append({
            "loksabha": key[0], "session": key[1],
            "discovered": found, "reference": ref, "delta": found - ref,
            "crawl_status": rep.status if rep else "not-crawled",
            "capped_from": rep.capped_from if rep else 0,
        })
    return {
        "reference": str(xlsx_path),
        "note": "frozen workbook used as coverage REFERENCE only — it is never "
                "the scraper's source of truth",
        "sessions": entries,
        "totals": {
            "discovered": sum(e["discovered"] for e in entries),
            "reference": sum(e["reference"] for e in entries),
            "delta": sum(e["discovered"] - e["reference"] for e in entries),
        },
    }


# ── run ──────────────────────────────────────────────────────────────────────

def run(ctx: CrawlContext) -> dict[str, Any]:
    """Run all target loksabhas × sessions; write <root>/last_run.json."""
    started = utcnow_iso()
    run_failures: list[str] = []
    if ctx.opts.loksabhas is not None:
        loks = sorted(set(ctx.opts.loksabhas))
    else:
        cal = calendar(ctx)
        lo, hi, exclude = window(ctx.cfg, "loksabhas")
        loks = sorted(
            n for n in cal
            if (lo is None or n >= lo) and (hi is None or n <= hi) and n not in exclude
        )
        if not cal and ctx._calendar_error:
            run_failures.append(
                f"loksabha calendar unavailable (fail-closed: pass --loksabha "
                f"explicitly to run without it): {ctx._calendar_error}"
            )

    reports: list[SessionReport] = []
    for lok in loks:
        reports.extend(crawl_loksabha(ctx, lok))

    totals: dict[str, int] = {}
    for rep in reports:
        for k, v in rep.docs.items():
            totals[k] = totals.get(k, 0) + v
    out_of_scope = [
        entry for lok in loks for entry in
        (ctx._inventory.get(lok).out_of_scope if lok in ctx._inventory else [])
    ]
    failures_n = sum(1 for r in reports if r.status == "failed") + len(run_failures)
    run_report = {
        "crawler": f"crawl_lok_sabha_qa/{CRAWLER_VERSION}",
        "house": ctx.cfg.get("house"),
        "started_at": started,
        "finished_at": utcnow_iso(),
        "options": {
            "loksabhas": ctx.opts.loksabhas,
            "sessions": ctx.opts.sessions,
            "ministry_slugs": ctx.opts.ministry_slugs,
            "fetch_documents": ctx.opts.fetch_documents,
            "dry_run": ctx.opts.dry_run,
            "retry_failed": ctx.opts.retry_failed,
            "max_records": ctx.opts.max_records,
            "coverage_ref": ctx.opts.coverage_ref,
        },
        "config": str(ctx.cfg.get("_path", "?")),
        "root": str(ctx.root),
        "http_requests": ctx.http.request_count,
        "eras": {"api_ls_min_loksabha": (ctx.cfg.get("eras") or {})
                 .get("api_ls_min_loksabha", 16)},
        "ministries": [
            {"slug": m["slug"], "label": m["label"],
             "api_ministry_code": int(m["api_ministry_code"])}
            for m in ctx.ministries
        ],
        "sessions": [r.as_dict() for r in reports],
        "out_of_scope_rows": out_of_scope,
        "doc_totals": totals,
        "failures": failures_n,
        "run_failures": run_failures,
    }
    if ctx._calendar_error:
        # the run may still have succeeded from inventory-derived sessions —
        # the degraded calendar is surfaced, never silently swallowed
        run_report["calendar_error"] = ctx._calendar_error
    if ctx.opts.coverage_ref:
        try:
            run_report["coverage"] = coverage_audit(reports, ctx.opts.coverage_ref)
        except Exception as exc:  # noqa: BLE001 — audit failure must not fail the crawl
            run_report["coverage"] = {
                "reference": str(ctx.opts.coverage_ref), "error": str(exc),
            }
    if not ctx.opts.dry_run:
        ctx.root.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(
            ctx.root / "last_run.json",
            (json.dumps(run_report, indent=1, ensure_ascii=True) + "\n").encode("utf-8"),
        )
    return run_report


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_http(cfg: dict[str, Any], *, transport=None, sleeper=None) -> CrawlHttpClient:
    """CrawlHttpClient carrying the browser-identity headers the live
    contracts demand (api_ls: Referer + browser UA — 403 without)."""
    kw = http_kwargs(cfg)
    inner = httpx.Client(
        timeout=kw["timeout"],
        follow_redirects=True,
        headers=http_headers(cfg),
        transport=transport,
    )
    return CrawlHttpClient(
        client=inner,
        delay=kw["delay"],
        retries=kw["retries"],
        backoff=kw["backoff"],
        sleeper=sleeper,  # type: ignore[arg-type]
    )


def _parse_int_spec(spec: str | None) -> list[int] | None:
    """'18,16-17' → sorted unique ints; None/'all' → None."""
    if not spec or spec.strip().lower() == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"bad range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crawl_lok_sabha_qa",
        description="Lok Sabha Q&A crawler (staging only — no indexing/embedding).",
    )
    p.add_argument("--config", type=Path, default=None,
                   help="crawler config YAML (default: config/crawlers/lok_sabha_qa.yaml)")
    p.add_argument("--root", type=Path, default=None,
                   help="output root override (default: <data>/parliamentary-qa/lok-sabha)")
    p.add_argument("--loksabha", default=None,
                   help="loksabha filter, e.g. '18,16-17' or 'all' (default: calendar ∩ config)")
    p.add_argument("--sessions", default=None,
                   help="session filter within each loksabha, e.g. '8,2-4' or 'all'")
    p.add_argument("--ministry", default=None,
                   help="ministry slug filter, comma-separated (default: all configured)")
    p.add_argument("--dry-run", action="store_true",
                   help="inventory only: no writes, no document downloads")
    p.add_argument("--no-documents", dest="documents", action="store_false",
                   help="records only: skip document downloads/extraction")
    p.add_argument("--no-retry-failed", dest="retry_failed", action="store_false",
                   help="keep previously failed document slots as-is (no re-attempt)")
    p.add_argument("--max-records", type=int, default=None,
                   help="cap records processed per session (smoke-test aid)")
    p.add_argument("--coverage-ref", default=None,
                   help="frozen workbook xlsx for the read-only coverage audit "
                        "(reference only — never the source of truth)")
    return p


def main(argv: list[str] | None = None, *, transport=None, sleeper=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        ministries = resolve_ministries(
            cfg, [s.strip() for s in args.ministry.split(",")] if args.ministry else None)
        loksabhas = _parse_int_spec(args.loksabha)
        sessions = _parse_int_spec(args.sessions)
    except (LsConfigError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    root = output_root(cfg, args.root)
    opts = RunOptions(
        loksabhas=loksabhas,
        sessions=sessions,
        ministry_slugs=[m["slug"] for m in ministries],
        fetch_documents=args.documents,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        max_records=args.max_records,
        coverage_ref=args.coverage_ref,
    )
    http = build_http(cfg, transport=transport, sleeper=sleeper)
    ctx = CrawlContext(cfg=cfg, opts=opts, http=http, root=root, ministries=ministries)

    lok_hint = f"loksabhas={loksabhas}" if loksabhas else "loksabhas=calendar∩config"
    sess_hint = f"sessions={sessions}" if sessions else "sessions=calendar∪inventory∩config"
    min_hint = f"ministries={[m['slug'] for m in ministries]}"
    print(f"[start] lok-sabha root={root} {lok_hint} {sess_hint} {min_hint}",
          flush=True)
    if opts.dry_run:
        print(f"[dry-run] root={root} (nothing will be written)", flush=True)
    try:
        report = run(ctx)
    finally:
        http.close()

    n_sessions = len(report["sessions"])
    for i, s in enumerate(report["sessions"], 1):
        label = f"lok-{s['loksabha']}/session-{s['session']}"
        line = (f"  [{i}/{n_sessions}] {label}: {s['status']:9s} records={s['records']} "
                f"+{s['added']} ~{s['changed']} "
                f"docs(g/p/b/m)={s['docs']['good']}/{s['docs']['partial']}/"
                f"{s['docs']['broken']}/{s['docs']['missing']}")
        if s["capped_from"]:
            line += f" (capped from {s['capped_from']})"
        if s["zero_ministries"] and s["status"] not in ("empty",):
            line += f" zero-ministries={','.join(s['zero_ministries'])}"
        if s["failures"]:
            line += f" FAILURES={s['failures']}"
        print(line)
    print(json.dumps({
        "sessions": len(report["sessions"]),
        "out_of_scope_rows": len(report["out_of_scope_rows"]),
        "doc_totals": report["doc_totals"],
        "http_requests": report["http_requests"],
        "failures": report["failures"],
    }, indent=1))
    for failure in report["run_failures"]:
        print(f"  [run-failure] {failure}")
    if not opts.dry_run:
        print(f"[done] root={root} last_run.json updated")
    return 3 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
