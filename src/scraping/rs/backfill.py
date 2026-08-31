"""eParlib back-fill for failed official document slots (design §9, audit §7).

Why this exists: the RS staging crawl records every broken/missing official
document slot in the session ``manifest.json`` ``failed_slots[]`` with an
``alternate`` stub (``eparlib, url: None, status: pending``). eParlib
(eparlib.sansad.in, a DSpace-7 instance) mirrors the SAME official answer
documents; the rsdoc inventory row for each question carries DSpace handle
URLs in ``eng_file_dsp`` / ``hindi_file_dsp``. This module executes the
recovery:

    pending failed slot -> qslno -> re-fetch upstream row (rsdoc permalink,
    source of truth) -> DSpace handle -> DSpace-7 REST resolution
    (pid/find -> item -> bundles -> bitstreams, English pdf>docx>hindi
    preference; HTML citation_pdf_url fallback) -> download CONTENT bytes ->
    magic-sniff/classify (formats.classify_document) -> stage byte-preserved
    as documents/<qslno>-<lang>.<ext> (same naming as the crawl) -> update
    session manifest + qa.jsonl IN PLACE (slot leaves failed_slots, a
    documents[] entry with a ``backfill`` provenance marker appears, an
    empty inline answer is repaired from the recovered PDF under the same
    REC-P1 policy as the crawl via pipeline.apply_answer_fallback, the
    record's attention entry is resolved when the answer became usable).

Guarantees:
  * IDEMPOTENT — re-running with recovered bytes already on disk writes
    nothing (byte-stable, like the crawl itself).
  * NO FAKES — only bytes actually downloaded & classified are staged; a
    slot that cannot be fetched stays in failed_slots with a fresh,
    honest failure note; the summary reports recovered vs pending exactly.
  * DSpace resolution is fully generéric by base URL (config
    ``backfill.eparlib_base_url``) — no hardcoded handles; the HTTP layer is
    the shared CrawlHttpClient (transport injectable for offline tests).
  * RE-CRAWL SAFE — the crawl carries recovered documents forward when the
    official URL re-fails (see pipeline._prior_recovered_document): a later
    normal crawl never demotes a backfilled slot back into failed_slots.

The bitstream preference walk mirrors the Lok Sabha DSpace resolver
(RealArchiveScraper) via a direct import of its pure static helper — one
implementation, shared across eras.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.data.scraper import RealArchiveScraper  # shared DSpace bitstream pick
from src.scraping import records as rec_utils
from src.scraping.http import CrawlHttpClient, HttpApiError, HttpTransportError
from src.scraping.manifest import load_manifest, write_manifest
from src.scraping.rs.client import RsClient
from src.scraping.rs.documents import Slot, process_slot
from src.scraping.rs.normalize import sort_key, utcnow_iso
from src.scraping.rs.pipeline import apply_answer_fallback
from src.utils.atomic_io import write_bytes_atomic

UUID_RE = (r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_DSP_KEY = {"eng": "eng_file_dsp", "hin": "hindi_file_dsp"}


class BackfillResolutionError(RuntimeError):
    """A DSpace handle could not be resolved to a document content URL."""


# ─────────────────────────────────────────────────────────────────────────────
# DSpace-7 resolution (generic by base URL — no hardcoded handles)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dspace_document_url(handle_url: str, base_url: str,
                                http: CrawlHttpClient) -> dict[str, str]:
    """Resolve a DSpace handle page to the direct content URL of its document.

    Returns {"handle": ..., "content_url": ...}. Raises
    BackfillResolutionError with a machine- and human-readable reason.

    Strategy (same algorithm as RealArchiveScraper._resolve_dspace_handle,
    re-expressed over the shared CrawlHttpClient transport/retry seam so it
    runs in offline tests too):
      1. REST API: pid/find -> item uuid -> bundles -> bitstreams; pick the
         primary document bitstream (English pdf>docx, then Hindi) and return
         its /content URL.
      2. Fallback: scrape the handle page HTML for a citation_pdf_url meta
         bitstream uuid, else a bitstreams/<uuid>/download link.
    """
    base = base_url.rstrip("/")
    # 1) REST API path
    try:
        resp = http.get(f"{base}/server/api/pid/find?id={handle_url}")
        if resp.status == 200:
            item_uuid = (json.loads(resp.body.decode("utf-8", "replace")) or {}).get("uuid")
            if item_uuid:
                rb = http.get(f"{base}/server/api/core/items/{item_uuid}/bundles")
                if rb.status == 200:
                    bundles = (json.loads(rb.body.decode("utf-8", "replace"))
                               .get("_embedded", {}) or {}).get("bundles", [])
                    for bundle in bundles:
                        buuid = bundle.get("uuid")
                        if not buuid:
                            continue
                        rbs = http.get(
                            f"{base}/server/api/core/bundles/{buuid}/bitstreams")
                        if rbs.status != 200:
                            continue
                        bitstreams = (json.loads(rbs.body.decode("utf-8", "replace"))
                                      .get("_embedded", {}) or {}).get("bitstreams", [])
                        uuid = RealArchiveScraper._pick_document_bitstream(bitstreams)  # noqa: SLF001
                        if uuid:
                            return {"handle": handle_url,
                                    "content_url": f"{base}/server/api/core/bitstreams/"
                                                   f"{uuid}/content"}
    except HttpTransportError as exc:
        raise BackfillResolutionError(f"REST resolution transport failure: {exc}") from exc
    except (ValueError, AttributeError, TypeError) as exc:
        raise BackfillResolutionError(f"REST resolution parse failure: {exc}") from exc

    # 2) HTML page fallback
    try:
        page = http.get(handle_url)
    except HttpTransportError as exc:
        raise BackfillResolutionError(f"handle page transport failure: {exc}") from exc
    if page.status == 200:
        text = page.body.decode("utf-8", errors="replace")
        m = re.search(r'citation_pdf_url"\s+content="([^"]+)"', text, re.I)
        if m:
            mu = re.search(UUID_RE, m.group(1))
            if mu:
                return {"handle": handle_url,
                        "content_url": f"{base}/server/api/core/bitstreams/"
                                       f"{mu.group(0)}/content"}
        m2 = re.search(rf"bitstreams/({UUID_RE})/download", text)
        if m2:
            return {"handle": handle_url,
                    "content_url": f"{base}/server/api/core/bitstreams/"
                                   f"{m2.group(1)}/content"}
    elif page.status != 200:
        raise BackfillResolutionError(f"handle page HTTP {page.status}")
    raise BackfillResolutionError("no document bitstream found via REST or HTML")


# ─────────────────────────────────────────────────────────────────────────────
# Planning (local state only)
# ─────────────────────────────────────────────────────────────────────────────

def _record_qslno(manifest: dict[str, Any], rid: str) -> int | None:
    for entry in manifest.get("records") or []:
        if entry.get("id") == rid and entry.get("qslno") is not None:
            try:
                return int(entry["qslno"])
            except (TypeError, ValueError):
                return None
    return None


def _manifest_document(manifest: dict[str, Any], key: str) -> dict[str, Any] | None:
    for entry in manifest.get("documents") or []:
        if entry.get("key") == key:
            return entry
    return None


def plan_backfill(cfg: dict[str, Any], root: Path,
                  sessions: list[int] | None) -> list[dict[str, Any]]:
    """Work items for every pending failed slot (local state only).

    Each item carries everything the executor needs BEFORE any network:
    session dir, manifest, record id, slot lang, the failed entry itself and
    the best-known qslno (failed-entry key first, manifest records record
    second — qa.jsonl is consulted later by the executor).
    """
    items: list[dict[str, Any]] = []
    if not root.exists():
        return items
    for session_dir in sorted(root.glob("session-*")):
        manifest = load_manifest(session_dir)
        if not manifest:
            continue
        ses = manifest.get("session")
        if sessions is not None and ses not in sessions:
            continue
        for entry in manifest.get("failed_slots") or []:
            if (entry.get("alternate") or {}).get("status") != "pending":
                continue
            rid = entry.get("id")
            items.append({
                "session": ses,
                "session_dir": session_dir,
                "id": rid,
                "lang": entry.get("lang"),
                "qslno": entry.get("qslno") if entry.get("qslno") is not None
                         else _record_qslno(manifest, rid),
                "failed_entry": entry,
            })
    return items


def _fetch_upstream_row(rs: RsClient, qslno: int) -> dict[str, Any] | None:
    """Re-fetch the rsdoc row for one qslno (upstream = source of truth for
    DSpace handles; the staged qa.jsonl intentionally drops raw dsp fields)."""
    rows = rs.http.get_json(RsClient.record_permalink(qslno))
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def _qa_row_qslno(session_dir: Path, rid: str) -> int | None:
    for row in rec_utils.load_jsonl(session_dir / "qa.jsonl"):
        if row.get("question_id") == rid:
            qslno = ((row.get("metadata") or {}).get("qslno"))
            try:
                return int(qslno) if qslno is not None else None
            except (TypeError, ValueError):
                return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill(*, http: CrawlHttpClient, cfg: dict[str, Any], root: Path,
                 sessions: list[int] | None = None, dry_run: bool = False,
                 now: str | None = None) -> dict[str, Any]:
    """Execute (or, with dry_run, plan) the eParlib back-fill over a staging root.

    Returns the summary report dict (also written to <root>/last_backfill.json
    when not a dry run). ``now`` is injectable for deterministic tests.
    """
    now = now or utcnow_iso()
    bcfg = cfg.get("backfill") or {}
    eparlib_base = str(bcfg.get("eparlib_base_url") or "https://eparlib.sansad.in/")
    rsdoc_base = str(cfg.get("rsdoc_base_url") or "https://rsdoc.nic.in")
    fallback_wanted = bool((cfg.get("policy") or {}).get("extract_fallback", True))
    rs = RsClient(http, rsdoc_base)

    items = plan_backfill(cfg, root, sessions)
    outcomes: list[dict[str, Any]] = []
    # per-session mutation state: session_dir -> {"manifest", "rows_by_id", "touched"}
    state: dict[Path, dict[str, Any]] = {}

    def session_state(session_dir: Path) -> dict[str, Any]:
        st = state.get(session_dir)
        if st is None:
            st = {
                "manifest": load_manifest(session_dir),
                "rows_by_id": {r.get("question_id"): r
                               for r in rec_utils.load_jsonl(session_dir / "qa.jsonl")},
                "touched": False,
            }
            state[session_dir] = st
        return st

    for item in items:
        session_dir: Path = item["session_dir"]
        rid, lang = item["id"], item["lang"]
        key_qslno = item["qslno"]
        if key_qslno is None:
            key_qslno = _qa_row_qslno(session_dir, rid)
        outcome: dict[str, Any] = {
            "session": item["session"], "id": rid, "qslno": key_qslno,
            "lang": lang, "status": "pending", "detail": "",
        }

        st = session_state(session_dir)
        row = st["rows_by_id"].get(rid)
        if row is None:
            outcome.update(status="error",
                           detail="qa.jsonl row missing (corrupt staging) — "
                                  "left pending instead of faking a record")
            outcomes.append(outcome)
            continue
        if key_qslno is None:
            outcome.update(status="unfetchable",
                           detail="no qslno in failed slot, manifest records or "
                                  "qa row — cannot address the rsdoc inventory")
            outcomes.append(outcome)
            continue

        # 1) upstream row (source of truth for handles)
        try:
            upstream = _fetch_upstream_row(rs, int(key_qslno))
        except (HttpApiError, HttpTransportError) as exc:
            outcome.update(status="unfetchable", detail=f"inventory fetch failed: {exc}")
            outcomes.append(outcome)
            continue
        if not upstream:
            outcome.update(status="unfetchable",
                           detail=f"rsdoc returned no row for qslno={key_qslno}")
            outcomes.append(outcome)
            continue

        handle = (upstream.get(_DSP_KEY.get(lang, "")) or "").strip() or None
        outcome["handle"] = handle
        if handle is None:
            outcome.update(status="unresolvable",
                           detail="upstream row carries no DSpace handle for "
                                  f"this slot ({_DSP_KEY.get(lang)} is empty)")
            outcomes.append(outcome)
            continue

        # 2) DSpace resolution
        try:
            resolved = resolve_dspace_document_url(handle, eparlib_base, http)
        except BackfillResolutionError as exc:
            outcome.update(status="unresolvable", detail=str(exc))
            outcomes.append(outcome)
            continue
        outcome["content_url"] = resolved["content_url"]

        if dry_run:
            outcome.update(status="planned",
                           detail="dry-run: would download, stage and update "
                                  "manifest in place")
            outcomes.append(outcome)
            continue

        # 3) download + stage (same Slot/process_slot machinery as the crawl:
        #    magic-sniffed classification, byte-preserved promotion, idempotent)
        slot = Slot(lang, filename=f"{key_qslno}-{lang}.via-eparlib", url=resolved["content_url"])
        try:
            result, body, written = process_slot(http, slot, session_dir, int(key_qslno))
        except HttpTransportError as exc:
            outcome.update(status="error", detail=f"download transport failure: {exc}")
            outcomes.append(outcome)
            continue
        facts = result.facts
        outcome["class"] = facts.doc_class
        if facts.doc_class not in ("good", "partial"):
            outcome.update(status="error",
                           detail=f"downloaded payload classified '{facts.doc_class}' "
                                  f"({facts.cause}) — nothing staged; slot stays pending")
            outcomes.append(outcome)
            continue

        sha = facts.sha256
        entry = {
            "key": result.key,
            "id": rid,
            "path": result.path,
            **facts.as_manifest(),
            "backfill": {"source": "eparlib", "handle": handle,
                         "url": resolved["content_url"], "recovered_at": now},
        }

        manifest = st["manifest"]
        doc_entry = _manifest_document(manifest, result.key)
        if (doc_entry and doc_entry.get("sha256") == sha
                and (doc_entry.get("backfill") or {}).get("source") == "eparlib"):
            outcome["status"] = "already"
        else:
            outcome["status"] = "recovered"
        if written:
            outcome["detail"] = f"staged {result.path} ({facts.bytes} bytes)"
        else:
            outcome["detail"] = outcome["detail"] or "bytes already on disk (sha match)"

        # 4) manifest in-place update
        st["touched"] = True
        manifest["documents"] = sorted(
            [d for d in (manifest.get("documents") or []) if d.get("key") != result.key]
            + [entry],
            key=lambda d: d["key"])
        manifest["failed_slots"] = [
            f for f in (manifest.get("failed_slots") or [])
            if not (f.get("id") == rid and f.get("lang") == lang)
        ]

        # 5) qa.jsonl row in-place update — stamp exactly what a re-crawl with
        #    carry-forward would produce for this slot (byte-stable by design)
        meta = row.get("metadata") or {}
        row["metadata"] = meta
        meta.setdefault("documents", {})[lang] = {
            "class": facts.doc_class, "path": result.path, "format": facts.format,
        }

        attention = list(manifest.get("attention") or [])
        answer_usable = bool((row.get("answer_text") or "").strip())
        if lang == "eng" and not answer_usable:
            apply_answer_fallback(meta, row, facts, body, fallback_wanted)
            answer_usable = bool((row.get("answer_text") or "").strip())
            if answer_usable:
                outcome["note"] = "inline answer repaired from recovered PDF"
            else:
                # recovered file did not yield an answer (e.g. scanned PDF):
                # restamp attention with the NEW cause, exactly as the crawl
                # would after carrying this slot forward
                cause = meta.get("answer_unavailable_cause") or "extract-failed"
                attention = [a for a in attention if a.get("id") != rid]
                attention.append({"id": rid, "reason": f"no usable answer ({cause})"})
        remaining_failed = [f for f in manifest["failed_slots"] if f.get("id") == rid]
        if answer_usable and not remaining_failed:
            attention = [a for a in attention if a.get("id") != rid]
        manifest["attention"] = sorted(attention, key=lambda a: a["id"])
        outcomes.append(outcome)

    # ── write phase (per touched session; atomic; skipped entirely on dry-run)
    per_session: list[dict[str, Any]] = []
    if not dry_run:
        for session_dir, st in sorted(state.items(), key=lambda kv: str(kv[0])):
            if not st["touched"]:
                continue
            manifest = st["manifest"]
            merged_rows = sorted(st["rows_by_id"].values(), key=sort_key)
            payload = rec_utils.serialize_jsonl(merged_rows)
            qa_path = session_dir / "qa.jsonl"
            if not qa_path.exists() or qa_path.read_bytes() != payload:
                write_bytes_atomic(qa_path, payload)
            manifest["generated_at"] = now
            write_manifest(session_dir, manifest)
    totals_session: dict[Any, dict[str, Any]] = {}
    for o in outcomes:
        agg = totals_session.setdefault(o["session"], {"session": o["session"],
                                                       "recovered": 0, "already": 0,
                                                       "planned": 0, "pending": 0})
        agg[o["status"] if o["status"] in ("recovered", "already", "planned")
            else "pending"] += 1
    per_session = [totals_session[k] for k in sorted(totals_session)]
    recovered_total = sum(1 for o in outcomes if o["status"] == "recovered")
    already_total = sum(1 for o in outcomes if o["status"] == "already")
    planned_total = sum(1 for o in outcomes if o["status"] == "planned")
    pending_after = sum(1 for o in outcomes
                        if o["status"] in ("unfetchable", "unresolvable", "error", "pending"))

    report = {
        "tool": "rs-eparlib-backfill/1.0",
        "generated_at": now,
        "root": str(root),
        "dry_run": dry_run,
        "sessions": per_session,
        "recovered": recovered_total,
        "already": already_total,
        "planned": planned_total,
        "pending_after": pending_after,
        "http_requests": http.request_count,
        "outcomes": outcomes,
    }
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(root / "last_backfill.json",
                           (json.dumps(report, indent=1, ensure_ascii=True) + "\n")
                           .encode("utf-8"))
    return report


# Re-export for the CLI/report (avoids a private-import at the call site).
__all__ = [
    "BackfillResolutionError",
    "plan_backfill",
    "resolve_dspace_document_url",
    "run_backfill",
]
