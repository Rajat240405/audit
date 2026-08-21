"""
Unified SOURCE SYNC for public reports — weekly auto, manual, or check-only.

Modes:
  --weekly  : full sync (scan -> download new -> OCR -> convert -> ingest ->
              rebuild index -> log). Intended for a cron job (HPC, Sat/Sun).
  --manual  : same as weekly, run on demand (don't wait for the schedule).
  --check   : ONLY scan public sources and report what's NEW (no download,
              no ingest). Lets you see "3 new documents arrived" anytime.

How it knows what's new: a manifest file (data/sync_manifest.json) stores
{url: sha256} for every PDF already downloaded. Anything on the source page
that's not in the manifest (or whose hash changed) is "new".

Corpus safety (Phase 3): the ingest step writes its conversion to a scratch
file and MERGES it into the canonical data/corpus_reports.jsonl (id-union:
sync re-conversions refresh their own records; records from every other
source — parliament pipeline, inbox uploads, hierarchical trees — are always
preserved). The sync can never shrink the canonical corpus; wholesale
replacement stays explicit (run ingest_all --out ... yourself).

Internal/private documents are NEVER auto-scraped — scientists put those in
data/inbox/ and run ingest_inbox.py (or the UI Ingest button).

Usage:
    # see what's new (no side effects)
    python -m src.scripts.sync_sources --check
    # full manual sync
    python -m src.scripts.sync_sources --manual
    # cron (Sat/Sun): full auto sync
    python -m src.scripts.sync_sources --weekly
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# R2: SINGLE source for INCOIS section discovery — import SECTIONS + the
# discover() helper from the crawler and wrap them here (was: a 3rd copy of
# the same page/pattern config -> drift whenever the site changed).
from src.scripts.crawl_incois_reports import SECTIONS, discover as _crawl_discover
from src.models.qa_record import QARecord
from src.utils.atomic_io import write_text_atomic


def discover_incois() -> dict[str, str]:
    """{url: label} for every INCOIS report section (thin wrapper)."""
    found: dict[str, str] = {}
    for sec in SECTIONS:
        try:
            for url in _crawl_discover(sec):
                found[url] = sec
        except Exception as e:  # noqa: BLE001
            log(f"  [warn] section {sec}: {e}")
    return found

MOES_MEDIA = "https://ccps.digifootprint.gov.in/wp-json/wp/v2/media"
MOES_POSTS = "https://ccps.digifootprint.gov.in/wp-json/wp/v2/posts"

MANIFEST = Path("data/sync_manifest.json")
LOG = Path("data/sync.log")
DOWNLOAD_DIR = Path("data/incois_reports")
MOES_DIR = Path("data/moes_reports")
OCR_DIR = Path("data/scanned_ocr")
CORPUS = Path("data/corpus_reports.jsonl")
INBOX = Path("data/inbox")


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_manifest(mf: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(mf, indent=1), encoding="utf-8")


def discover_moes() -> dict[str, str]:
    """{url: 'moes'} from CCPS media + posts APIs."""
    import re
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    found: dict[str, str] = {}
    with httpx.Client(timeout=20, verify=ctx) as c:
        for api in (MOES_MEDIA, MOES_POSTS):
            for page in range(1, 16):
                try:
                    r = c.get(f"{api}?per_page=100&page={page}")
                    if r.status_code != 200:
                        break
                    items = r.json()
                    if not items:
                        break
                    for it in items:
                        src = it.get("source_url") or ""
                        if src.lower().endswith(".pdf"):
                            found[src] = "moes"
                        content = (it.get("content") or {}).get("rendered", "")
                        for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\']', content, re.I):
                            found[m.group(1)] = "moes"
                except Exception:  # noqa: BLE001
                    break
    return found


def new_pdfs(manifest: dict) -> dict[str, str]:
    """URLs in sources that are not yet in the manifest."""
    urls = {}
    urls.update(discover_incois())
    urls.update(discover_moes())
    return {u: s for u, s in urls.items() if u not in manifest}


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(timeout=120, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            if len(r.content) < 10_000:
                return False
            dest.write_bytes(r.content)
        return True
    except Exception:  # noqa: BLE001
        return False


def merge_scratch_into_corpus(scratch: Path, corpus: Path) -> dict:
    """Safely merge a freshly-generated sync corpus into the CANONICAL corpus.

    Phase-3 ingestion-safety fix (audit F.1): previously the sync ingest step
    ran `ingest_all --out data/corpus_reports.jsonl`, which REWRITES the file
    from only the sync-managed folders — silently dropping every record the
    sync pipeline did not produce (parliament Q&A from data/processed, inbox
    uploads, hierarchical tree uploads). Append/merge mode must never shrink
    the canonical corpus.

    Semantics (id-union, order-stable, atomic):
      * every EXISTING canonical line survives — foreign records (any id the
        sync output doesn't contain) are preserved verbatim;
      * an id the sync pipeline re-generated takes the FRESH copy (converter
        improvements keep flowing through — the legacy rewrite behavior for
        sync-managed records, without the collateral damage);
      * new sync ids append at the end;
      * unparseable historical lines are preserved verbatim (never silently
        dropped);
      * invariant guard: every existing question_id must be present in the
        result — a violation raises instead of writing (replacement of the
        canonical corpus stays EXPLICIT: run ingest_all --out yourself).
    """
    stats = {"existing": 0, "updated": 0, "added": 0, "preserved_raw": 0,
             "total": 0, "skipped": False}
    if not scratch.exists():
        # ingest_all exits(1) without writing when its folders produce nothing
        log("  merge: no sync output produced — canonical corpus untouched")
        stats["skipped"] = True
        return stats

    final: list[str] = []
    pos_by_id: dict[str, int] = {}
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            stats["existing"] += 1
            try:
                qid = json.loads(line).get("question_id")
            except Exception:  # noqa: BLE001
                qid = None
            if qid and qid not in pos_by_id:
                pos_by_id[qid] = len(final)
            elif qid is None:
                stats["preserved_raw"] += 1
            final.append(line)

    for line in scratch.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = QARecord.model_validate_json(line)
            qid = rec.question_id
        except Exception:  # noqa: BLE001 — never trust scratch blindly
            log(f"  merge: skipped malformed sync line ({line[:60]}...)")
            continue
        if qid in pos_by_id:
            if final[pos_by_id[qid]] != line:
                stats["updated"] += 1
            final[pos_by_id[qid]] = line
        else:
            pos_by_id[qid] = len(final)
            final.append(line)
            stats["added"] += 1

    # non-shrink invariant (by unique id, not raw line count)
    merged_ids = {qid for qid in pos_by_id}
    existing_ids = set()
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            try:
                qid = json.loads(line.strip()).get("question_id")
            except Exception:  # noqa: BLE001
                continue
            if qid:
                existing_ids.add(qid)
    missing = existing_ids - merged_ids
    if missing:
        raise RuntimeError(
            f"Refusing to write canonical corpus: merge would DROP "
            f"{len(missing)} existing record(s) (e.g. {sorted(missing)[:3]}). "
            "Append mode never shrinks the corpus; replacement must be explicit."
        )

    corpus.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(corpus, "\n".join(final) + "\n")
    stats["total"] = len(final)
    return stats


def _server_running(port: int = 4000) -> bool:
    """Cheap liveness probe: is the FastAPI server up on :port?"""
    try:
        import socket

        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except Exception:  # noqa: BLE001
        return False


def run(*cmd: str) -> None:
    log("  running: " + " ".join(cmd))
    # UTF-8 child output — Windows cp1252 console crashes on rich's unicode
    # arrows (→) otherwise.
    import os as _os

    env = dict(_os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([sys.executable, *cmd], check=False,
                   env=env, encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--weekly", action="store_true", help="Full auto sync (cron)")
    group.add_argument("--manual", action="store_true", help="Full sync now")
    group.add_argument("--check", action="store_true", help="Only report what's new")
    ap.add_argument("--sections", default=None,
                    help="Restrict sync to comma list (annual,general,tech,research,moes)")
    args = ap.parse_args()

    log(f"=== sync_sources ({'weekly' if args.weekly else 'manual' if args.manual else 'check'}) ===")
    t0 = time.time()

    mf = load_manifest()
    new = new_pdfs(mf)
    log(f"Source scan complete. {len(new)} NEW document(s) found.")

    if args.check:
        for url, sec in sorted(new.items()):
            log(f"  NEW [{sec}] {url.split('/')[-1]}")
        if not new:
            log("  Nothing new — corpus is up to date.")
        return

    # ── Full sync: download new, ingest ──
    if not new:
        log("Nothing new to download — skipping ingest (index stays as is).")
        return

    downloaded = 0
    for url, sec in sorted(new.items()):
        fname = url.split("/")[-1].split("?")[0]
        fname = re.sub(r'[<>:"/\\|?*]', "_", fname)  # Windows-safe
        dest = (DOWNLOAD_DIR / sec if sec != "moes" else MOES_DIR) / fname
        if download(url, dest):
            mf[url] = sha256_file(dest)
            downloaded += 1
            log(f"  downloaded {fname} ({sec})")
        else:
            log(f"  FAILED {fname}")

    save_manifest(mf)
    log(f"Downloaded {downloaded} new file(s).")

    # OCR any scanned PDFs among the new annual reports
    run("-m", "src.scripts.ocr_pdfs", "--folder", str(DOWNLOAD_DIR / "AnnualReports"),
        "--output", str(OCR_DIR))

    # Ingest: convert the sync-managed folders, then MERGE into the canonical
    # corpus — never rewrite it (Phase-3 safety fix, audit F.1). ingest_all
    # writes a SCRATCH file; merge_scratch_into_corpus does an id-union that
    # preserves foreign records (parliament, inbox, tree uploads) while sync
    # re-conversions still refresh their own records. Explicit replacement of
    # the canonical corpus remains possible only by running ingest_all
    # --out data/corpus_reports.jsonl directly (operator intent).
    scratch = CORPUS.with_name(".sync_ingest_scratch.jsonl")
    run("-m", "src.scripts.ingest_all",
        "--annual", str(DOWNLOAD_DIR / "AnnualReports"),
        "--reports", str(DOWNLOAD_DIR / "Others"),
        "--reports", str(DOWNLOAD_DIR / "TechnicalReports"),
        "--reports", str(DOWNLOAD_DIR / "ResearchPublications"),
        "--documents", str(MOES_DIR / "knowledge"),
        "--scanned", str(OCR_DIR),
        "--parliament", "data/does_not_exist",
        "--out", str(scratch))
    stats = merge_scratch_into_corpus(scratch, CORPUS)
    if stats["skipped"]:
        log("Nothing merged — corpus unchanged.")
    else:
        log(f"Corpus merge: {stats['existing']} existing preserved "
            f"({stats['preserved_raw']} raw), {stats['updated']} refreshed, "
            f"{stats['added']} new -> {stats['total']} total")
    scratch.unlink(missing_ok=True)

    # Rebuild index (embeddings) so newly synced docs are queryable.
    # If the server is running on :4000 it holds the index in memory — rebuild
    # on disk then, and tell the user to swap via the UI Ingest button
    # (or restart). If the server is down, rebuild here.
    if _server_running():
        log("Server is running on :4000 — index NOT swapped live. "
            "Run 'retrieve build --data data/corpus_reports.jsonl --rebuild' with the "
            "server stopped, or use the UI Ingest button to swap live.")
    else:
        run("-m", "src.retrieval.cli", "build", "--data", str(CORPUS), "--rebuild")
        log("Index rebuilt (embeddings created) — new docs are queryable on next server start.")
    log(f"=== sync done in {time.time()-t0:.0f}s — {downloaded} new, {len(new)-downloaded} failed ===")


if __name__ == "__main__":
    main()
