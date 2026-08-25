"""
COMPAT SHIM — INCOIS annual reports download (pre-consolidation era CLI).

Classification: thin compatibility wrapper (workspace cleanup, audit §4).
The discovery patterns and the download idiom live in ONE place —
``src/scripts/crawl_incois_reports.py`` (SECTIONS/discover/download_pdf).
This shim keeps the historical command working with its historical
defaults and exactly two own behaviors:

  1. ``--lang Hindi`` — the Hindi annual-report page variant. This is the
     ONLY capability the canonical crawler does not cover on its main path;
     it is implemented here by passing ``lang`` into the shared
     ``discover("annual", lang=...)`` (no duplicated page/pattern config).
  2. Default output dir ``data/annual_reports/`` — the legacy flat folder,
     still a registered ingestion source (``config/sources.yaml`` →
     ``incois.folders``). Canonical English AR acquisition nowadays lands in
     ``data/incois_reports/AnnualReports/`` via::

         python -m src.scripts.crawl_incois_reports --sections annual

Identical files under both trees dedupe by content hash at ingest, so
running both is safe during migration.

Usage (run on any machine with internet, e.g. the local dev PC):
    python -m src.scripts.crawl_annual_reports \
        --out data/annual_reports \
        --lang English          # or Hindi

Idempotent: files already present (by name, >10KB) are skipped. Prints a
summary of what was downloaded / skipped / failed.
"""

from __future__ import annotations

import argparse
import re

import httpx

from src.scripts.crawl_incois_reports import discover, download_pdf


def fetch_pdf_links(lang: str = "English") -> list[tuple[str, str]]:
    """Return [(year_label, pdf_url)] for the annual-report page.

    Kept for backward compatibility with the pre-consolidation module API;
    discovery itself delegates to the canonical implementation.
    """
    out = []
    for url in discover("annual", lang=lang):
        m = re.search(r"AR_([\d-]+)_", url)
        out.append((m.group(1) if m else url.rsplit("/", 1)[-1], url))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/annual_reports", help="Download dir")
    ap.add_argument("--lang", default="English", choices=["English", "Hindi"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from pathlib import Path

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[shim] crawl_annual_reports delegates discovery/download to "
          "crawl_incois_reports (single implementation, audit §4).")
    print(f"Fetching {args.lang} annual report links ...")
    links = fetch_pdf_links(args.lang)
    if args.limit:
        links = links[: args.limit]
    print(f"Found {len(links)} reports.\n")

    downloaded, skipped, failed = 0, 0, 0
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for _year, url in links:
            fname = url.split("/")[-1].split("?")[0]
            fname = re.sub(r'[<>:\"/\\|?*]', "_", fname)  # Windows-safe
            if not fname.lower().endswith(".pdf"):
                fname += ".pdf"
            dest = out_dir / fname
            status, detail = download_pdf(client, url, dest)
            if status == "downloaded":
                print(f"  [ok]   {fname} ({dest.stat().st_size/1e6:.1f} MB)")
                downloaded += 1
            elif status == "skipped":
                print(f"  [skip] {fname} (already on disk)")
                skipped += 1
            else:
                print(f"  [fail] {fname} ({detail or 'tiny/empty response'})")
                failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
