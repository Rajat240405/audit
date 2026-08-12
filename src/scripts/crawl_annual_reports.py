"""
Crawl INCOIS public annual reports from the official site.

Source: https://incois.gov.in/site/annual_report.jsp
The AR_* PDFs are public (verified: same filenames as found on the
scientist's machine — these are the exact official files). We download them
to data/annual_reports/ so HPC (no internet) never needs to fetch anything.

Usage (run on any machine with internet, e.g. the local dev PC):
    python -m src.scripts.crawl_annual_reports \
        --out data/annual_reports \
        --lang English          # or Hindi

Idempotent: files already present (by name) are skipped. Print a summary of
what was downloaded / skipped / failed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx

PAGE_URL = "https://incois.gov.in/site/annual_report.jsp"
BASE = "https://incois.gov.in"


def fetch_pdf_links(lang: str = "English") -> list[tuple[str, str]]:
    """Return [(year_label, pdf_url)] parsed from the annual report page."""
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        r = client.get(f"{PAGE_URL}?lang={lang}")
        r.raise_for_status()
    html = r.text
    links = re.findall(
        r"href=\"(/documents/Reports/AnnualReports/[^\"]+\.pdf)\"",
        html,
    )
    # de-dup, keep order
    seen: set[str] = set()
    out = []
    for href in links:
        url = BASE + href if href.startswith("/") else href
        if url in seen:
            continue
        seen.add(url)
        year = re.search(r"AR_([\d-]+)_", href)
        year = year.group(1) if year else Path(href).stem
        out.append((year, url))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/annual_reports", help="Download dir")
    ap.add_argument("--lang", default="English", choices=["English", "Hindi"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.lang} annual report links from {PAGE_URL} ...")
    links = fetch_pdf_links(args.lang)
    if args.limit:
        links = links[: args.limit]
    print(f"Found {len(links)} reports.\n")

    downloaded, skipped, failed = 0, 0, 0
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for year, url in links:
            fname = url.split("/")[-1]
            dest = out_dir / fname
            if dest.exists() and dest.stat().st_size > 10_000:
                print(f"  [skip] {fname} (already on disk)")
                skipped += 1
                continue
            try:
                r = client.get(url)
                r.raise_for_status()
                if len(r.content) < 10_000:
                    print(f"  [fail] {fname} (tiny/empty response)")
                    failed += 1
                    continue
                dest.write_bytes(r.content)
                print(f"  [ok]   {fname} ({len(r.content)/1e6:.1f} MB)")
                downloaded += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {fname}: {e}")
                failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
