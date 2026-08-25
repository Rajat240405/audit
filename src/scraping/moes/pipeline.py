"""MoES website crawl pipeline: per-category run, manifests, idempotency.

Semantics frozen from the approved boundary review (adapted RS design):

- byte-stable no-change runs: record.json / manifest.json / attachment-map.json are
  written ONLY on semantic change; document bytes are never rewritten when
  hashes match (duplicate-from-RS staging/promote idiom, see documents.py).
- replace-per-id on change; NO deletions ever. A post disappearing from its
  listing is carried forward as a tombstone (bytes kept); 3 consecutive
  absent runs escalate to ``attention`` for human review.
- deterministic everywhere: sorted emission, listing-order processing
  (server ordering post_date DESC is deterministic), wall-clock kept out of
  hashed content; failure/attention entries carry no timestamps (retry
  evidence lives in the run-artifact ``last_run.json``).
- failure grammar: taxonomy failure aborts the RUN (scope unprovable);
  listing failure aborts that CATEGORY (others continue, exit 3), as does a
  taxonomy failure (run-fatal — scope unprovable); a broken file slot or an
  unresolvable attachment id fails only the SLOT (per-id isolation; exit 3
  via failures) and failed resolutions are retried on the next run.
  a broken file slot fails the SLOT (run continues, exit 3 via failures).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.scraping import records as recs
from src.scraping.http import HttpApiError, HttpTransportError
from src.scraping.manifest import load_manifest, manifests_equal, write_manifest
from src.scraping.moes import normalize
from src.scraping.moes.config import (
    download_languages,
    resolve_categories,
    resolve_report_families,
)
from src.scraping.moes.documents import process_record_documents, resolve_attachment
from src.utils.atomic_io import write_bytes_atomic

CRAWLER_VERSION = "moes-1.0"
RECORD_NAME = "record.json"
ATTACHMENT_MAP_NAME = "attachment-map.json"
LAST_RUN_NAME = "last_run.json"
TOMBSTONE_ATTENTION_AFTER = 3


@dataclass
class RunOptions:
    categories: list[str] | None = None        # None = all configured
    report_families: list[str] | None = None   # None = all configured families
    fetch_documents: bool = True
    dry_run: bool = False
    resolve_attachments: bool = False          # dry-run: also probe attachment
                                               # resolution (post-page/post?id=)


@dataclass
class CrawlContext:
    cfg: dict[str, Any]
    opts: RunOptions
    http: Any                                   # CrawlHttpClient
    api: Any                                    # MoesApi
    root: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_json_bytes(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode()


class AttachmentMap:
    """attachment-id → resolved pdf-links cache (``attachment-map.json`` run
    artefact). Only POSITIVE resolutions are cached (ids are immutable
    upstream); failures stay in ``errors`` for this run and are retried next
    run (self-healing, boundary review §13)."""

    def __init__(self, root: Path) -> None:
        self.path = root / ATTACHMENT_MAP_NAME
        self.links: dict[int, dict[str, Any]] = {}
        self.errors: dict[int, dict[str, str]] = {}      # this run only
        self.resolved_this_run = 0
        self._loaded_semantic: dict[str, Any] | None = None
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for k, v in (raw.get("attachments") or {}).items():
                    self.links[int(k)] = v
            except Exception:  # noqa: BLE001 — corrupt cache rebuilt from scratch
                self.links = {}

    def semantic(self) -> dict[str, Any]:
        return {"attachments": {str(k): self.links[k] for k in sorted(self.links)}}

    def view(self) -> dict[int, dict[str, Any]]:
        """links + this-run negative markers (process_record_documents input)."""
        merged: dict[int, dict[str, Any]] = dict(self.links)
        for aid, err in self.errors.items():
            merged.setdefault(aid, err)
        return merged

    def ensure(self, api, needed: set[int]) -> None:
        """Resolve each missing id via post-page/post?id= (+ revision chains).

        Per-id failure ISOLATION (a resolver outage on one id can never
        mislabel another): resolve errors (``attachment-resolve-failed``) and
        no-pdf-object results (``attachment-missing-upstream``) become that
        id's slot-level cause. Deterministic: ids ascending, capped chains.
        """
        for aid in sorted(set(needed) - set(self.links)):
            self.errors.pop(aid, None)
            self.resolved_this_run += 1
            try:
                links, reason = resolve_attachment(api, aid)
            except (HttpApiError, HttpTransportError) as exc:
                self.errors[aid] = {"error": "attachment-resolve-failed",
                                    "note": f"attachment resolver failed: {exc}"}
                continue
            if links:
                self.links[aid] = links
            else:
                self.errors[aid] = {
                    "error": "attachment-missing-upstream",
                    "note": "resolved post carries no inline pdf object "
                            f"(revision chain walked): {reason}"}

    def save_if_changed(self) -> bool:
        payload = {"crawler_version": CRAWLER_VERSION, "generated_at": _now(),
                   **self.semantic()}
        if self._loaded_semantic is not None and self._loaded_semantic == self.semantic():
            return False
        if self.path.exists() and self._loaded_semantic is None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                prior = {"attachments": raw.get("attachments") or {}}
                if prior == self.semantic():
                    return False
            except Exception:  # noqa: BLE001
                pass
        text = json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=True) + "\n"
        write_bytes_atomic(self.path, text.encode())
        return True

    def mark_loaded(self) -> None:
        self._loaded_semantic = self.semantic()


# ─────────────────────────────────────────────────────────────────────────────
# category run
# ─────────────────────────────────────────────────────────────────────────────

def _verify_prior_entries(record_dir: Path, entries: list[dict[str, Any]]) -> bool:
    """True iff every good/partial prior entry's bytes are intact on disk."""
    for e in entries:
        if e.get("class") in ("good", "partial"):
            f = record_dir / str(e.get("path") or "")
            if not e.get("path") or not f.exists():
                return False
            if hashlib.sha256(f.read_bytes()).hexdigest() != e.get("sha256"):
                return False
    return True


def category_run(
    ctx: CrawlContext,
    category: str,
    *,
    terms: list[dict[str, Any]],
    fam_terms: dict[str, list[int]] | None,
    active_families: list[str] | None,
    attach: AttachmentMap,
    run_date: str,
    scraped_at: str,
    central_attachment_resolver=None,
) -> dict[str, Any]:
    cfg_families = ((ctx.cfg.get("categories") or {}).get(category) or {}).get("families")
    # the central-documents category may exist WITHOUT any taxonomy node
    # upstream (live-verified 2026-08-25: the tree carries no central term),
    # so only reports-style categories may treat a missing root as run-fatal
    if category == normalize.CENTRAL_DOCUMENTS_PARENT_SLUG:
        parent_id = normalize.top_term_id(terms, category)
    else:
        parent_id = normalize.parent_term_id(terms, category)
    cat_dir = ctx.root / category
    prior = load_manifest(cat_dir)
    prior_records = {r["id"]: r for r in (prior or {}).get("records") or []}
    prior_docs_by_record: dict[str, list[dict[str, Any]]] = {}
    for e in (prior or {}).get("documents") or []:
        prior_docs_by_record.setdefault(e.get("record_id"), []).append(e)
    prior_skipped_by_record: dict[str, list[dict[str, Any]]] = {}
    for s in (prior or {}).get("skipped_external") or []:
        prior_skipped_by_record.setdefault(s.get("record_id"), []).append(s)
    prior_attention_by_record: dict[str, list[dict[str, Any]]] = {}
    for a in (prior or {}).get("attention") or []:
        if a.get("type") == "empty-file-row":  # row-level; tombstone notes re-derived
            prior_attention_by_record.setdefault(a.get("record_id"), []).append(a)

    summary: dict[str, Any] = {
        "category": category, "status": "ok", "discovered": 0, "in_scope": 0,
        "scoped_out": [], "added": 0, "changed": 0, "unchanged": 0,
        "docs": {"good": 0, "partial": 0, "broken": 0},
        "failed": 0, "skipped_external": 0, "attention": 0, "tombstoned": 0,
        "pq_titled": 0, "failures": [], "bytes_changed": False,
    }

    # ── listing ───────────────────────────────────────────────────────────────
    try:
        posts, lst = ctx.api.listing_posts(category)
    except (HttpApiError, HttpTransportError) as exc:
        summary["status"] = "aborted"
        summary["failures"].append(f"listing failed: {exc}")
        return summary
    summary["listing"] = lst
    summary["discovered"] = len(posts)

    # ── normalize + scope partition ───────────────────────────────────────────
    central_scope_cfg: dict[str, Any] = (
        (ctx.cfg.get("central_documents") or {}).get("scopes") or {}
    )
    central_scope_families = (
        normalize.resolve_central_family_terms(terms, central_scope_cfg)
        if category == normalize.CENTRAL_DOCUMENTS_PARENT_SLUG else None
    )
    in_scope: list[dict[str, Any]] = []
    scoped_out: list[dict[str, Any]] = []
    for post in posts:
        children = normalize.child_terms_of(post, parent_id)
        child_ids = [int(t.get("term_id") or 0) for t in children]
        family = None
        if central_scope_families is not None:
            # central-documents: ONLY the approved scopes pass — first by
            # taxonomy, else by attachment-content evidence (pure; the
            # resolver callable supplies the REST result when needed).
            family = normalize.post_central_family(
                post, central_scope_families, central_scope_cfg)
            if family is None and central_attachment_resolver is not None:
                family = normalize.post_central_family(
                    post, central_scope_families, central_scope_cfg,
                    resolved_attachment=central_attachment_resolver(post))
            if family is None:
                scoped_out.append({
                    "id": f"moes-web-{int(post['ID'])}",
                    "slug": post.get("post_name"),
                    "children": [t.get("slug") for t in children],
                    "reason": "excluded-central-documents-family",
                })
                continue
            rec = normalize.normalize_central_record(
                post, category_slug=category, family=family,
                listing_url=f"{ctx.api._ep['listing']}?document_category={category}",
                scraped_at=scraped_at, scope_cfg=central_scope_cfg)
        else:
            if cfg_families:
                family = normalize.post_family(child_ids, fam_terms or {})
                reason = None
                if family is None:
                    reason = "excluded-reports-family"
                elif active_families is not None and family not in active_families:
                    reason = "family-not-requested"
                if reason:
                    scoped_out.append({
                        "id": f"moes-web-{int(post['ID'])}",
                        "slug": post.get("post_name"),
                        "children": [t.get("slug") for t in children],
                        "reason": reason,
                    })
                    continue
            rec = normalize.normalize_post(
                post, category=category, family=family, child_terms=children,
                listing_url=f"{ctx.api._ep['listing']}?document_category={category}",
                scraped_at=scraped_at,
            )
        in_scope.append(rec)

    summary["in_scope"] = len(in_scope)
    summary["scoped_out"] = sorted(scoped_out, key=lambda s: str(s["id"]))
    summary["pq_titled"] = sum(1 for r in in_scope if r["is_parliament_question"])
    if cfg_families or central_scope_families is not None:
        fam_stats: dict[str, dict[str, int]] = {}
        for r in in_scope:
            fs = fam_stats.setdefault(r["family"], {"posts": 0, "file_rows": 0})
            fs["posts"] += 1
            fs["file_rows"] += len(r["files"])
        summary["families"] = fam_stats

    planned_downloads = sum(1 for r in in_scope for f in r["files"]
                            if f.get("attachment_id") is not None)
    summary["file_rows"] = {
        "attachment": planned_downloads,
        "external": sum(1 for r in in_scope for f in r["files"]
                        if f.get("attachment_id") is None and f.get("external_url")),
        "empty": sum(1 for r in in_scope for f in r["files"]
                     if f.get("attachment_id") is None and not f.get("external_url")),
    }

    # ── dry-run stops here (no writes; optional read-only resolve probe) ─────
    if ctx.opts.dry_run:
        summary["status"] = "dry-run"
        if ctx.opts.resolve_attachments and planned_downloads:
            needed = {int(f["attachment_id"]) for r in in_scope for f in r["files"]
                      if f.get("attachment_id") is not None}
            attach.ensure(ctx.api, needed)          # per-id failure isolation
            resolved = len(needed & set(attach.links))
            summary["attachments"] = {
                "needed": len(needed), "resolved": resolved,
                "unresolved": sorted(a for a in needed if a not in attach.links),
            }
            if attach.errors:
                summary["failed"] = len(attach.errors)
        return summary

    # ── attachment resolution for downloads (per-id failure isolation) ───────
    if ctx.opts.fetch_documents and planned_downloads:
        needed_ids = {int(f["attachment_id"]) for r in in_scope for f in r["files"]
                      if f.get("attachment_id") is not None}
        attach.ensure(ctx.api, needed_ids)

    # ── per-record processing ────────────────────────────────────────────────
    sha_seen: dict[str, str] = {}
    for e in sorted((prior or {}).get("documents") or [], key=lambda x: str(x.get("key"))):
        if e.get("sha256") and e.get("sha256") not in sha_seen:
            sha_seen[str(e["sha256"])] = str(e["key"])

    new_records: list[dict[str, Any]] = []
    new_docs: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record in in_scope:
        seen_ids.add(record["id"])
        rel_dir = Path(*record["category_path"]) / record["slug"]
        record_dir = ctx.root / rel_dir
        prior_rec = prior_records.get(record["id"])

        # slug renamed upstream → move the directory (bytes preserved)
        if prior_rec and prior_rec.get("slug") != record["slug"]:
            old_dir = cat_dir.parent / Path(*prior_rec.get("category_path") or [category]) \
                / str(prior_rec["slug"])
            if old_dir.exists() and not record_dir.exists():
                record_dir.parent.mkdir(parents=True, exist_ok=True)
                old_dir.rename(record_dir)

        row_hash = recs.row_sha256(record)
        rec_entry = {
            "id": record["id"], "wp_id": record["wp_id"], "slug": record["slug"],
            "title": record["title"], "family": record["family"],
            "category_path": record["category_path"], "row_sha256": row_hash,
            "post_modified": record["post_modified"],
        }
        prior_entries = prior_docs_by_record.get(record["id"], [])
        unchanged = bool(prior_rec) and prior_rec.get("row_sha256") == row_hash

        rewrite_record = not unchanged or not (record_dir / RECORD_NAME).exists()
        reprocess = ctx.opts.fetch_documents and (
            not unchanged
            or any(e.get("class") == "broken" for e in prior_entries)   # retry failed
            or not _verify_prior_entries(record_dir, prior_entries)
        )

        if rewrite_record:
            record_dir.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic(record_dir / RECORD_NAME, _record_json_bytes(record))

        if reprocess:
            outcome = process_record_documents(
                ctx.api, record, record_dir, attach.view(), sha_seen,
                languages=download_languages(ctx.cfg))
            new_docs.extend(outcome.entries)
            failed.extend(outcome.failed)
            skipped.extend(outcome.skipped_external)
            attention.extend(outcome.attention)
            summary["bytes_changed"] = summary["bytes_changed"] or outcome.bytes_changed \
                or rewrite_record
        else:
            for e in prior_entries:
                new_docs.append(e)
                if e.get("sha256") and str(e["sha256"]) not in sha_seen:
                    sha_seen[str(e["sha256"])] = str(e["key"])
            failed.extend([e for e in prior_entries if e.get("class") == "broken"])
            skipped.extend(prior_skipped_by_record.get(record["id"], []))
            attention.extend(prior_attention_by_record.get(record["id"], []))
            summary["bytes_changed"] = summary["bytes_changed"] or rewrite_record

        if unchanged:
            summary["unchanged"] += 1
        elif prior_rec:
            summary["changed"] += 1
        else:
            summary["added"] += 1
        new_records.append(rec_entry)

    # ── tombstones ────────────────────────────────────────────────────────────
    for pid, prec in prior_records.items():
        if pid in seen_ids:
            continue
        n = int((prec.get("tombstone") or {}).get("consecutive_absent_runs", 0)) + 1
        tomb = dict(prec)
        tomb["tombstone"] = {"consecutive_absent_runs": n, "last_seen": run_date,
                             "reason": "absent-from-listing"}
        new_records.append(tomb)
        summary["tombstoned"] += 1
        # prior docs/skip/attention of tombstoned records are carried forward
        for e in prior_docs_by_record.get(pid, []):
            new_docs.append(e)
            if e.get("class") == "broken":
                failed.append(e)
        skipped.extend(prior_skipped_by_record.get(pid, []))
        attention.extend(prior_attention_by_record.get(pid, []))
        if n >= TOMBSTONE_ATTENTION_AFTER:
            attention.append({
                "type": "tombstoned-record-absent", "record_id": pid,
                "consecutive_absent_runs": n, "slug": prec.get("slug"),
                "note": "post absent from listing for 3+ consecutive runs — human review"
                        " required; bytes are kept, never auto-deleted",
            })

    # ── manifest ──────────────────────────────────────────────────────────────
    new_records.sort(key=lambda r: int(r["wp_id"]))
    new_docs.sort(key=lambda e: str(e["key"]))
    failed = sorted({e["key"]: e for e in failed}.values(), key=lambda e: str(e["key"]))
    failed_slots = [{
        "key": e["key"], "record_id": e["record_id"], "row": e.get("row"),
        "attachment_id": e.get("attachment_id"), "url": e.get("url"),
        "class": "broken", "cause": e.get("cause"), "note": e.get("note", ""),
    } for e in failed]
    skipped.sort(key=lambda s: (str(s["record_id"]), int(s["row"])))
    attention.sort(key=lambda a: (str(a.get("type")), str(a.get("record_id"))))

    summary["failed"] = len(failed_slots)
    summary["skipped_external"] = len(skipped)
    summary["attention"] = len(attention)
    for e in new_docs:
        cls = e.get("class")
        if cls in summary["docs"]:
            summary["docs"][cls] += 1

    manifest = {
        "source": "moes-website",
        "site": "www.moes.gov.in",
        "category": category,
        "crawler_version": CRAWLER_VERSION,
        "generated_at": scraped_at,                     # volatile (diff-excluded)
        "listing": {"base": ctx.api._ep["listing"], "page_size": ctx.api.listing_page_size,
                    **summary["listing"]},
        "families": ({f: fam_terms[f] for f in (active_families or fam_terms or {})}
                     if fam_terms else {}) if cfg_families else {},
        "scoped_out": summary["scoped_out"],
        "records": new_records,
        "documents": new_docs,
        "failed_slots": failed_slots,
        "skipped_external": skipped,
        "attention": attention,
    }
    cat_dir.mkdir(parents=True, exist_ok=True)
    if prior is None or not manifests_equal(prior, manifest):
        write_manifest(cat_dir, manifest)
        summary["bytes_changed"] = True
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# full run
# ─────────────────────────────────────────────────────────────────────────────

def run(ctx: CrawlContext) -> dict[str, Any]:
    started = _now()
    run_date = started[:10]
    scraped_at = started
    categories = resolve_categories(ctx.cfg, ctx.opts.categories)
    if "reports" in categories:
        # validates subset too (fail closed on unknown family names)
        active_families = resolve_report_families(ctx.cfg, ctx.opts.report_families)
    else:
        active_families = None

    report: dict[str, Any] = {
        "source": "moes-website", "crawler_version": CRAWLER_VERSION,
        "started_at": started, "dry_run": ctx.opts.dry_run,
        "root": str(ctx.root), "categories": [], "failures": False,
        "robots": None, "counts": None, "pq_stats": None,
        "overlap": "NOT COMPUTED — cross-source SHA-256 matching against "
                   "data/parliamentary-qa/ is a later, reviewed integration step "
                   "(boundary review §5D2/D4; titles-only stats in pq_stats).",
    }

    # compliance + inventory artefacts (both non-fatal on failure)
    try:
        status, body = ctx.api.robots()
        report["robots"] = {"status": status, "body_prefix": body}
    except (HttpApiError, HttpTransportError) as exc:
        report["robots"] = {"error": str(exc)}
    try:
        report["counts"] = ctx.api.counts()
    except (HttpApiError, HttpTransportError) as exc:
        report["counts"] = {"error": str(exc)}

    # taxonomy is run-fatal: without it the approved scope cannot be proven
    terms = ctx.api.taxonomy()
    fam_terms = None
    if "reports" in categories:
        fam_terms = normalize.resolve_family_terms(
            terms, (ctx.cfg["categories"]["reports"] or {}).get("families") or {})
    else:
        # still validate reports parent exists? No — reports not in this run's scope.
        pass

    attach = AttachmentMap(ctx.root)
    attach.mark_loaded()

    # central-documents scope resolution (fail-closed): taxonomy-driven family
    # terms + one bounded REST probe per scope-unknown post, whose result is
    # fed back into the pure scoper (attachment-content fallback for
    # uncategorized uploads; a resolver outage fails closed to scoped-out).
    # Memoized per run (listing order = deterministic); a plain dry-run gets
    # NO resolver at all — it stays listing+taxonomy read-only, and posts
    # that need REST evidence then fail closed to scoped-out (use
    # --resolve-attachments for a full probe).
    central_att_cache: dict[int, dict[str, Any] | None] = {}

    def _central_attachment(post: dict[str, Any]) -> dict[str, Any] | None:
        """attachment_post(aid) of the post's first ACF file row; None on any
        failure or when no id exists (deterministic — never fabricated)."""
        aid = normalize.first_file_attachment_id(post)
        if aid is None:
            return None
        if aid not in central_att_cache:
            try:
                central_att_cache[aid] = ctx.api.attachment_post(aid)
            except (HttpApiError, HttpTransportError):
                central_att_cache[aid] = None
        return central_att_cache[aid]

    central_resolver = (
        None if (ctx.opts.dry_run and not ctx.opts.resolve_attachments)
        else _central_attachment
    )

    pq_stats: dict[str, int] = {}
    for category in categories:
        summary = category_run(
            ctx, category, terms=terms, fam_terms=fam_terms,
            active_families=active_families if category == "reports" else None,
            attach=attach, run_date=run_date, scraped_at=scraped_at,
            central_attachment_resolver=central_resolver)
        report["categories"].append(summary)
        if summary.get("failures") or summary.get("failed"):
            report["failures"] = True
        if category == "press-release":
            pq_stats = {
                "discovered": summary["discovered"],
                "pq_titled": summary["pq_titled"],
                "pq_sha_exact_match_in_parliamentary_corpus": None,   # NOT COMPUTED
                "pq_not_present_in_parliamentary_corpus": None,       # NOT COMPUTED
                "uncomparable_missing_or_broken":
                    None if ctx.opts.dry_run else summary["docs"]["broken"],
            }

    if pq_stats:
        report["pq_stats"] = pq_stats

    if not ctx.opts.dry_run:
        attach.save_if_changed()
        finished = _now()
        last_run = {
            **{k: v for k, v in report.items() if k != "categories"},
            "categories": {c["category"]: {k: v for k, v in c.items()} for c in
                           report["categories"]},
            "finished_at": finished,
            "http_requests": ctx.http.request_count,
        }
        ctx.root.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(
            ctx.root / LAST_RUN_NAME,
            (json.dumps(last_run, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode())
    report["http_requests"] = ctx.http.request_count
    return report
