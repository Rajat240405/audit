"""Per-session orchestration for the Rajya Sabha crawler (design §5, §8).

Flow per (session × configured ministries) — mirroring exactly the workflow
that produced the validated dataset:

1. inventory      Search_Questions per ministry (empty list = checked-empty,
                  never a failure)
2. normalize      rows → QARecord-shaped dicts (metadata incl. ministry slug)
3. documents      eng/hin slots: download, magic-sniff, classify, stage
                  byte-preserved, then merge manifest-side
4. answers        inline-first policy (REC-P1): ans_text if present, else
                  table-aware text extraction from the English PDF
5. emit           qa.jsonl merged by id + manifest.json — both written ONLY
                  when content changed (byte-stable no-change re-runs)

Failure ladder: transport errors are retried inside the HTTP client; a
ministry inventory call that still fails aborts THAT SESSION (recorded, run
continues); document-slot failures are recorded as failed_slots with an
eParlib back-fill stub and never abort anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scraping import records as rec_utils
from src.scraping.manifest import load_manifest, manifests_equal, write_manifest
from src.scraping.rs.client import RsClient
from src.scraping.rs.documents import plan_slots, process_slot
from src.scraping.rs.normalize import build_record, sort_key, utcnow_iso
from src.scraping.http import CrawlHttpClient, HttpApiError, HttpTransportError
from src.utils.atomic_io import write_bytes_atomic

CRAWLER_VERSION = "rs-1.0"
QA_JSONL = "qa.jsonl"
EPARLIB_BASE = "https://eparlib.sansad.in/"


@dataclass
class RunOptions:
    sessions: list[int] | None = None        # None => server list intersected with config range
    ministry_slugs: list[str] | None = None  # None => all configured
    fetch_documents: bool = True
    dry_run: bool = False
    retry_failed: bool = True


@dataclass
class CrawlContext:
    cfg: dict[str, Any]
    opts: RunOptions
    http: CrawlHttpClient
    root: Path
    ministries: list[dict[str, Any]]
    rs: RsClient = field(init=False)

    def __post_init__(self) -> None:
        self.rs = RsClient(self.http, (self.cfg.get("rsdoc_base_url") or "https://rsdoc.nic.in"))

    @property
    def policy(self) -> dict[str, Any]:
        return self.cfg.get("policy") or {}


@dataclass
class SessionReport:
    session: int
    status: str = "ok"      # ok | empty | unchanged | updated | dry-run | failed
    ministries: dict[str, int] = field(default_factory=dict)
    zero_ministries: list[str] = field(default_factory=list)
    records: int = 0
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    docs: dict[str, int] = field(default_factory=lambda: {"good": 0, "partial": 0, "broken": 0, "missing": 0})
    doc_files_written: int = 0
    attention: list[dict[str, str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session": self.session, "status": self.status,
            "ministries": self.ministries,
            "zero_ministries": self.zero_ministries,
            "records": self.records, "added": self.added,
            "changed": self.changed, "unchanged": self.unchanged,
            "docs": self.docs, "doc_files_written": self.doc_files_written,
            "attention": self.attention, "failures": self.failures,
        }


def _extract_answer_fallback(pdf_body: bytes) -> str | None:
    """Table-aware whole-document text; None when the PDF is a scan (no text)."""
    try:
        from src.data.pdf_table_extract import extract_pdf_text
    except ImportError:
        return None
    try:
        text = extract_pdf_text(pdf_body)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001 — unopenable/corrupt: treat as unavailable
        return None
    return (text or "").strip() or None


def _prior_failed_entry(old_manifest: dict[str, Any] | None, rid: str, lang: str) -> dict | None:
    if not old_manifest:
        return None
    for entry in old_manifest.get("failed_slots") or []:
        if entry.get("id") == rid and entry.get("lang") == lang:
            return entry
    return None


def _failed_slot_entry(rid: str, slot_result, url: str | None) -> dict[str, Any]:
    facts = slot_result.facts
    return {
        "id": rid,
        "lang": slot_result.lang,
        "class": facts.doc_class,           # broken | missing
        "http": facts.http_status,
        "cause": facts.cause,
        "url": url,
        "alternate": {"source": EPARLIB_BASE, "url": None, "status": "pending"},
    }


def crawl_session(ctx: CrawlContext, ses: int) -> SessionReport:
    report = SessionReport(session=ses)

    # 1. inventory (per ministry; a failed inventory aborts this session only)
    fetched: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for mcfg in ctx.ministries:
        try:
            rows = ctx.rs.search(ses_no=ses, min_code=mcfg["code"])
        except (HttpApiError, HttpTransportError) as exc:
            report.status = "failed"
            report.failures.append(
                f"inventory failed for ministry {mcfg['slug']} (code {mcfg['code']}): {exc}"
            )
            return report
        report.ministries[mcfg["slug"]] = len(rows)
        if rows:
            fetched.append((mcfg, rows))
        else:
            report.zero_ministries.append(mcfg["slug"])

    if not fetched:
        report.status = "empty"
        return report

    # 2. normalize (kept with their raw rows for document planning)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for mcfg, rows in fetched:
        for raw in rows:
            pairs.append((build_record(raw, mcfg), raw))
    report.records = len(pairs)

    if ctx.opts.dry_run:
        report.status = "dry-run"
        return report

    session_dir = ctx.root / f"session-{ses}"
    old_manifest = load_manifest(session_dir)

    # 3. documents + 4. answers
    doc_manifest: list[dict[str, Any]] = []
    failed_slots: list[dict[str, Any]] = []
    fallback_wanted = ctx.policy.get("extract_fallback", True)

    for rec, raw in pairs:
        meta = rec["metadata"]
        rid = rec["question_id"]
        qslno = meta["qslno"]

        if rec["answer_text"]:
            meta["answer_source"] = "inline"

        if ctx.opts.fetch_documents:
            for slot in plan_slots(raw):
                carried = _prior_failed_entry(old_manifest, rid, slot.lang)
                if carried and not ctx.opts.retry_failed:
                    failed_slots.append(carried)
                    report.docs[carried["class"]] += 1
                    meta["documents"][slot.lang] = {"class": carried["class"],
                                                    "cause": carried["cause"]}
                    continue

                result, body, written = process_slot(ctx.http, slot, session_dir, qslno)
                facts = result.facts
                report.docs[facts.doc_class] += 1
                if written:
                    report.doc_files_written += 1
                doc_summary: dict[str, Any] = {"class": facts.doc_class}
                if result.path:
                    doc_summary["path"] = result.path
                    doc_summary["format"] = facts.format
                if facts.cause:
                    doc_summary["cause"] = facts.cause
                meta["documents"][slot.lang] = doc_summary

                if facts.doc_class in ("broken", "missing"):
                    failed_slots.append(_failed_slot_entry(rid, result, slot.url))
                else:
                    doc_manifest.append({"key": result.key, "id": rid, "path": result.path,
                                         **facts.as_manifest()})

                # inline-first → PDF-extract fallback (REC-P1)
                if (
                    not rec["answer_text"]
                    and slot.lang == "eng"
                    and facts.doc_class in ("good", "partial")
                ):
                    if not fallback_wanted:
                        meta["answer_source"] = "unavailable"
                        meta["answer_unavailable_cause"] = "extract-disabled"
                    elif facts.format != "pdf":
                        meta["answer_source"] = "unavailable"
                        meta["answer_unavailable_cause"] = "legacy-format-not-extracted"
                    else:
                        text = _extract_answer_fallback(body) if body else None
                        if text:
                            rec["answer_text"] = text
                            meta["answer_source"] = "document-extract"
                        else:
                            meta["answer_source"] = "unavailable"
                            meta["answer_unavailable_cause"] = "extract-failed"
        elif not rec["answer_text"]:
            meta["answer_source"] = "unavailable"
            meta["answer_unavailable_cause"] = "documents-disabled"

        if not rec["answer_text"] and meta["answer_source"] is None:
            meta["answer_source"] = "unavailable"
            meta["answer_unavailable_cause"] = "english-document-unavailable"
        if not rec["answer_text"]:
            report.attention.append(
                {"id": rid, "reason": f"no usable answer ({meta['answer_unavailable_cause']})"}
            )

    # 5a. emit qa.jsonl (merge by id; write only on byte difference)
    existing_rows = rec_utils.load_jsonl(session_dir / QA_JSONL)
    merged, stats = rec_utils.merge_by_id(
        existing_rows,
        [rec for rec, _ in pairs],
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

    # 5b. manifest (write only on semantic change)
    manifest = {
        "source": "rsdoc.nic.in/Search_Questions",
        "house": "rajya-sabha",
        "session": ses,
        "crawler_version": CRAWLER_VERSION,
        "generated_at": now,
        "ministries": sorted({m["slug"] for m, _ in fetched}),
        "policy": {
            "extract_fallback": bool(fallback_wanted),
            "append_doc_annexures": bool(ctx.policy.get("append_doc_annexures", False)),
            "hindi_legacy_text": bool(ctx.policy.get("hindi_legacy_text", False)),
            "extract_legacy_text": bool(ctx.policy.get("extract_legacy_text", False)),
        },
        "records": [
            {
                "id": row["question_id"],
                "qslno": (row["metadata"] or {}).get("qslno"),
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
        "updated" if (qa_written or manifest_written or report.doc_files_written) else "unchanged"
    )
    return report


def _config_session_range(cfg: dict[str, Any]) -> tuple[int | None, int | None, list[int]]:
    ses = cfg.get("sessions") or {}
    return ses.get("min"), ses.get("max"), list(ses.get("exclude") or [])


def run(ctx: CrawlContext) -> dict[str, Any]:
    """Run all target sessions; write <root>/last_run.json. Returns the run report."""
    started = utcnow_iso()
    if ctx.opts.sessions is not None:
        sessions = sorted(set(ctx.opts.sessions))
    else:
        lo, hi, exclude = _config_session_range(ctx.cfg)
        sessions = [s for s in ctx.rs.sessions()
                    if (lo is None or s >= lo) and (hi is None or s <= hi) and s not in exclude]

    reports: list[SessionReport] = []
    for ses in sessions:
        reports.append(crawl_session(ctx, ses))

    totals: dict[str, int] = {}
    for rep in reports:
        for k, v in rep.docs.items():
            totals[k] = totals.get(k, 0) + v
    run_report = {
        "crawler": f"crawl_parliamentary_qa/{CRAWLER_VERSION}",
        "house": ctx.cfg.get("house"),
        "started_at": started,
        "finished_at": utcnow_iso(),
        "options": {
            "sessions": ctx.opts.sessions,
            "ministry_slugs": ctx.opts.ministry_slugs,
            "fetch_documents": ctx.opts.fetch_documents,
            "dry_run": ctx.opts.dry_run,
            "retry_failed": ctx.opts.retry_failed,
        },
        "config": str(ctx.cfg.get("_path", "?")),
        "root": str(ctx.root),
        "http_requests": ctx.http.request_count,
        "ministries": [{"code": m["code"], "slug": m["slug"], "label": m["label"]}
                       for m in ctx.ministries],
        "other_sources": [
            {"kind": "short-notice-questions",
             "detail": "sansad api_rs SNQ corpus: 0 records for configured ministries "
                       "(verified 2026-08-23) — not crawled"}
        ],
        "sessions": [r.as_dict() for r in reports],
        "doc_totals": totals,
        "failures": sum(1 for r in reports if r.status == "failed"),
    }
    if not ctx.opts.dry_run:
        ctx.root.mkdir(parents=True, exist_ok=True)
        import json as _json

        write_bytes_atomic(ctx.root / "last_run.json",
                           (_json.dumps(run_report, indent=1, ensure_ascii=True) + "\n")
                           .encode("utf-8"))
    return run_report


def pending_backfill(root: Path) -> list[dict[str, Any]]:
    """All failed slots awaiting eParlib back-fill across session manifests.

    Read-only scan; the back-fill itself is the operator-network hook
    (design §9) and deliberately not part of the initial crawl.
    """
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for session_dir in sorted(root.glob("session-*")):
        manifest = load_manifest(session_dir)
        if not manifest:
            continue
        for entry in manifest.get("failed_slots") or []:
            alt = entry.get("alternate") or {}
            if alt.get("status") == "pending":
                out.append({"session": manifest.get("session"), **entry})
    return out
