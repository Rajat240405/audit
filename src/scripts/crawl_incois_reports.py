"""
Crawl the high-value public INCOIS report sections from incois.gov.in.

Sections (each maps to a page that lists PDFs):
  annual   /site/annual_report.jsp     -> documents/Reports/AnnualReports/AR_*
  general  /site/general_reports.jsp   -> documents/Reports/Others/Report_*
  research /site/research_pub.jsp      -> documents/Reports/ResearchPublications/RP_*
  tech     /site/technical_report.jsp  -> documents/Reports/TechnicalReports/TR_*

(RTI disclosures and News PDFs are intentionally excluded — procedural/low
value for the audit corpus.)

This module is the SINGLE implementation of INCOIS section discovery +
the download idiom (workspace cleanup, audit §4): sync_sources wraps
SECTIONS/discover for its scan phase, and crawl_annual_reports is a thin
compat shim over discover("annual", lang=...) + download_pdf. Do not grow
a fourth copy of the page/pattern config anywhere.

The files are PUBLIC (verified: the scientist's Report_* files are exactly
the ones listed on general_reports.jsp, and 98% of annual report pages are
text-extractable — no OCR). We download them on the local dev PC so HPC
(no internet) never needs to fetch anything.

Usage (run on any machine with internet):
    python -m src.scripts.crawl_incois_reports --sections annual general tech
    # or all high-value sections:
    python -m src.scripts.crawl_incois_reports --sections all
    # options: --out <dir>, --limit <N>, --extract (write text files too)

Idempotent: files already on disk (by name) are skipped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import httpx

BASE = "https://incois.gov.in"

SECTIONS: dict[str, dict] = {
    "annual": {
        "page": "/site/annual_report.jsp",
        "pat": r"/documents/Reports/AnnualReports/[^\"']+\.pdf",
        "label": "AnnualReports",
    },
    "general": {
        "page": "/site/general_reports.jsp",
        "pat": r"/documents/Reports/Others/[^\"']+\.pdf",
        "label": "Others",
    },
    "research": {
        "page": "/site/research_pub.jsp",
        "pat": r"/documents/Reports/ResearchPublications/[^\"']+\.pdf",
        "label": "ResearchPublications",
    },
    "tech": {
        "page": "/site/technical_report.jsp",
        "pat": r"/documents/Reports/TechnicalReports/[^\"']+\.pdf",
        "label": "TechnicalReports",
    },
}


def discover(section: str, lang: str | None = None) -> list[str]:
    """Return absolute PDF urls listed on the section's page.

    ``lang`` (additive; default None = legacy English behavior) fetches the
    site's language variant via ``?lang=<Lang>`` — the annual-report page
    swaps its PDF list under ?lang=Hindi. Only a query-string append;
    section patterns/labels are untouched.
    """
    cfg = SECTIONS[section]
    page = cfg["page"] + (f"?lang={lang}" if lang else "")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        r = client.get(BASE + page)
        r.raise_for_status()
    links = sorted(set(re.findall(cfg["pat"], r.text)))
    return [u if u.startswith("http") else BASE + u for u in links]


def download_pdf(client: httpx.Client, url: str, dest: Path) -> tuple[str, str]:
    """One download attempt — THE shared idiom (workspace cleanup, audit §4).

    Returns (status, detail) with status in {"downloaded", "skipped",
    "failed"}: >10KB existing file -> skipped; <10KB response or any
    request error -> failed (detail carries the reason for logging).
    """
    if dest.exists() and dest.stat().st_size > 10_000:
        return "skipped", ""
    try:
        r = client.get(url)
        r.raise_for_status()
        if len(r.content) < 10_000:
            return "failed", "tiny"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return "downloaded", ""
    except Exception as e:  # noqa: BLE001
        return "failed", f"{type(e).__name__}: {str(e)[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="all",
                    help="comma list: annual,general,research,tech | all")
    ap.add_argument("--out", default="data/incois_reports", help="Download dir")
    ap.add_argument("--limit", type=int, default=0, help="0 = all (per section)")
    ap.add_argument("--extract", action="store_true",
                    help="Also write extracted .txt per PDF (via pypdf)")
    args = ap.parse_args()

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    sections = list(SECTIONS) if args.sections.lower() == "all" else [
        s.strip().lower() for s in args.sections.split(",") if s.strip()
    ]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[start] incois-reports sections={sections} out={out_root.resolve()}")
    total_down = total_skip = total_fail = 0
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for sec in sections:
            if sec not in SECTIONS:
                print(f"[skip] unknown section '{sec}'")
                continue
            out_dir = out_root / SECTIONS[sec]["label"]
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                urls = discover(sec)
            except Exception as e:  # noqa: BLE001
                print(f"[{sec}] discover failed: {e}")
                continue
            if args.limit:
                urls = urls[: args.limit]
            total_urls = len(urls)
            print(f"[{sec}] {total_urls} PDFs -> {out_dir}")

            d = s = f = 0
            for i, url in enumerate(urls, 1):
                fname = url.split("/")[-1].split("?")[0]
                fname = re.sub(r'[<>:"/\\|?*]', "_", fname)  # Windows-safe
                if not fname.lower().endswith(".pdf"):
                    fname += ".pdf"
                dest = out_dir / fname
                status, detail = download_pdf(client, url, dest)
                if status == "downloaded":
                    d += 1
                    if args.extract:
                        txt = extract_pdf_text(dest)
                        if txt.strip():
                            (dest.with_suffix(".txt")).write_text(txt, encoding="utf-8")
                elif status == "skipped":
                    s += 1
                else:
                    print(f"   [fail] {fname} ({detail})")
                    f += 1
                if i % 10 == 0 or i == total_urls:
                    print(f"   [{sec}] {i}/{total_urls} processed"
                          f" (downloaded={d} skipped={s} failed={f})")
            print(f"   -> {d} downloaded, {s} skipped, {f} failed")
            total_down += d
            total_skip += s
            total_fail += f

    print(f"\nTOTAL: {total_down} downloaded, {total_skip} skipped, {total_fail} failed")
    print(f"[done] saved under: {out_root.resolve()}")


def extract_pdf_text(path: Path) -> str:
    """Best-effort text extraction (pypdf); returns '' if it fails."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "").strip() for p in reader.pages if p.extract_text())
    except Exception:  # noqa: BLE001
        return ""


if __name__ == "__main__":
    main()
