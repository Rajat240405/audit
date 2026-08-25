"""
Crawl MoES reports/documents from the public CCPS portal.

Classification: LEGACY-REQUIRED (workspace cleanup, audit §4). Do not
extend; do not delete — its staging layout is a registered ingestion source
(``config/sources.yaml`` → ``moes_reports`` → ``moes_reports/knowledge``)
and its knowledge JSONs may still be re-converted on the operator machine.

Ownership boundary (one responsibility per path):
  * moes.gov.in website content (annual/monthly reports, demands-for-grants,
    press releases) is owned by the DEDICATED website crawler
    (``src/scripts/crawl_moes_website.py`` → ``data/.moes-website/`` →
    ``python -m src.scripts.ingest moes_website``);
  * sync_sources.py no longer scans this portal either (its MoES leg is
    retired — it raced this crawler for the same documents);
  * THIS crawler remains the acquisition path only for the legacy CCPS
    mirror tree (data/moes_reports/), until an explicit operator migration
    supersedes it.

Source: https://ccps.digifootprint.gov.in (WordPress REST API — public).
This is a cleaned, structured port of the scientist's crawl_moes_reports.py:
it walks the media + posts endpoints, collects PDF URLs, downloads the PDFs
to data/moes_reports/, extracts text with PyMuPDF, and writes each one as a
document JSON in the format the knowledge converter consumes
({title, category, source_url, file_name, content}) so it can be merged into
the corpus with convert_sirs_knowledge.py --documents.

Usage (run on any machine with internet, e.g. the local dev PC):
    python -m src.scripts.crawl_moes_reports \
        --out data/moes_reports \
        --max-downloads 50
Idempotent: files already on disk are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

BASE = "https://ccps.digifootprint.gov.in"
MEDIA_API = f"{BASE}/wp-json/wp/v2/media"
POSTS_API = f"{BASE}/wp-json/wp/v2/posts"

# CCPS serves a self-signed cert sometimes; mirror sir's ssl.CERT_NONE behaviour
_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def _json(url: str, client: httpx.Client) -> list[dict] | None:
    try:
        r = client.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception as e:  # noqa: BLE001
        print(f"  [fetch] {url[:80]}... -> {type(e).__name__}")
    return None


def collect_pdf_urls(max_pages_media: int = 15, max_pages_posts: int = 10) -> dict[str, dict]:
    """Return {url: {title, date, slug}} for all PDFs found on the portal."""
    found: dict[str, dict] = {}
    with httpx.Client(timeout=20, verify=_ctx) as client:
        for page in range(1, max_pages_media + 1):
            data = _json(f"{MEDIA_API}?per_page=100&page={page}", client)
            if not data:
                break
            for item in data:
                src = item.get("source_url") or ""
                mime = item.get("mime_type") or ""
                if src.lower().endswith(".pdf") or "pdf" in mime.lower():
                    found[src] = {
                        "title": (item.get("title") or {}).get("rendered", "Untitled MoES Document"),
                        "date": item.get("date", ""),
                        "slug": item.get("slug", ""),
                    }
        for page in range(1, max_pages_posts + 1):
            data = _json(f"{POSTS_API}?per_page=100&page={page}", client)
            if not data:
                break
            for item in data:
                content = (item.get("content") or {}).get("rendered", "")
                title = (item.get("title") or {}).get("rendered", "Untitled MoES Document")
                for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\']', content, re.I):
                    url = m.group(1)
                    found[url] = {"title": title, "date": item.get("date", ""), "slug": item.get("slug", "")}
    return found


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz). Falls back to pypdf."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        parts = []
        for i in range(len(doc)):
            t = doc[i].get_text().strip()
            if t:
                parts.append(f"--- Page {i+1} ---\n{t}")
        doc.close()
        return "\n\n".join(parts)
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n\n".join((p.extract_text() or "").strip() for p in reader.pages if p.extract_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/moes_reports", help="PDF download dir")
    ap.add_argument("--json-out", default=None, help="Dir for extracted knowledge JSONs (default: <out>/knowledge)")
    ap.add_argument("--max-downloads", type=int, default=50)
    ap.add_argument("--max-pages-media", type=int, default=15)
    ap.add_argument("--max-pages-posts", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = Path(args.json_out or out_dir / "knowledge")
    json_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {BASE} for PDFs...")
    pdfs = collect_pdf_urls(args.max_pages_media, args.max_pages_posts)
    print(f"Found {len(pdfs)} unique PDF links.\n")

    done = skipped = failed = 0
    with httpx.Client(timeout=120, verify=_ctx) as client:
        for i, (url, info) in enumerate(pdfs.items(), start=1):
            if done >= args.max_downloads:
                print("  [stop] reached --max-downloads")
                break
            fname = Path(urlparse(url).path).name or f"doc_{i}.pdf"
            if not fname.lower().endswith(".pdf"):
                fname += ".pdf"
            dest = out_dir / fname
            if dest.exists() and dest.stat().st_size > 10_000:
                print(f"  [skip] {fname}")
                skipped += 1
                continue
            try:
                r = client.get(url)
                r.raise_for_status()
                if len(r.content) < 10_000:
                    print(f"  [fail] {fname} (tiny)")
                    failed += 1
                    continue
                dest.write_bytes(r.content)
                text = extract_text(dest)
                if len(text.strip()) > 50:
                    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (info["title"] or "").lower())[:60] or f"doc_{i}"
                    entry = {
                        "title": f"MoES Document: {info['title']}",
                        "category": "Ministry of Earth Sciences (MoES) Official Knowledge",
                        "source_url": url,
                        "file_name": fname,
                        "content": f"Title: {info['title']}\nDate: {info.get('date', '')}\nURL: {url}\n\nContent:\n{text[:50000]}",
                    }
                    (json_dir / f"moes_report_{slug}.json").write_text(
                        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                print(f"  [ok]   {fname} ({len(r.content)/1e6:.1f} MB, {len(text):,} chars)")
                done += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {fname}: {type(e).__name__}: {e}")
                failed += 1

    print(f"\nDone: {done} downloaded, {skipped} skipped, {failed} failed")
    print(f"PDFs in: {out_dir.resolve()} | knowledge JSONs in: {json_dir.resolve()}")


if __name__ == "__main__":
    main()
