"""
Crawl the high-value public INCOIS report sections from incois.gov.in.

Sections (each maps to a page that lists PDFs):
  annual   /site/annual_report.jsp     -> documents/Reports/AnnualReports/AR_*
  general  /site/general_reports.jsp   -> documents/Reports/Others/Report_*
  research /site/research_pub.jsp      -> documents/Reports/ResearchPublications/RP_*
  tech     /site/technical_report.jsp  -> documents/Reports/TechnicalReports/TR_*

(RTI disclosures and News PDFs are intentionally excluded — procedural/low
value for the audit corpus.)

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
import sys
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


def discover(section: str) -> list[str]:
    """Return absolute PDF urls listed on the section's page."""
    cfg = SECTIONS[section]
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        r = client.get(BASE + cfg["page"])
        r.raise_for_status()
    links = sorted(set(re.findall(cfg["pat"], r.text)))
    return [u if u.startswith("http") else BASE + u for u in links]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="all",
                    help="comma list: annual,general,research,tech | all")
    ap.add_argument("--out", default="data/incois_reports", help="Download dir")
    ap.add_argument("--limit", type=int, default=0, help="0 = all (per section)")
    ap.add_argument("--extract", action="store_true",
                    help="Also write extracted .txt per PDF (via pypdf)")
    args = ap.parse_args()

    sections = list(SECTIONS) if args.sections.lower() == "all" else [
        s.strip().lower() for s in args.sections.split(",") if s.strip()
    ]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

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
            print(f"[{sec}] {len(urls)} PDFs -> {out_dir}")

            d = s = f = 0
            for url in urls:
                fname = url.split("/")[-1].split("?")[0]
                fname = re.sub(r'[<>:"/\\|?*]', "_", fname)  # Windows-safe
                if not fname.lower().endswith(".pdf"):
                    fname += ".pdf"
                dest = out_dir / fname
                if dest.exists() and dest.stat().st_size > 10_000:
                    s += 1
                    continue
                try:
                    r = client.get(url)
                    r.raise_for_status()
                    if len(r.content) < 10_000:
                        print(f"   [fail] {fname} (tiny)")
                        f += 1
                        continue
                    dest.write_bytes(r.content)
                    d += 1
                    if args.extract:
                        txt = extract_pdf_text(dest)
                        if txt.strip():
                            (dest.with_suffix(".txt")).write_text(txt, encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    print(f"   [fail] {fname}: {type(e).__name__}: {str(e)[:80]}")
                    f += 1
            print(f"   -> {d} downloaded, {s} skipped, {f} failed")
            total_down += d
            total_skip += s
            total_fail += f

    print(f"\nTOTAL: {total_down} downloaded, {total_skip} skipped, {total_fail} failed")
    print(f"Saved under: {out_root.resolve()}")


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
