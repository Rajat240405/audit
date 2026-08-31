"""
Extract TABLES + FIGURES/DIAGRAMS from PDFs into structured, reusable files.

This is the "handle tables & diagrams" mandate. For each PDF it produces:
  <out>/<pdf_stem>/tables.json    — every detected table (rows/cols + readable text)
  <out>/<pdf_stem>/figures.json   — every detected image (page, bbox, caption if found)
  <out>/<pdf_stem>/images/        — the actual image files (.png)

Tables are also flattened to readable text (the form that goes into the RAG
corpus), so the LLM can answer "what did the 2022 table say about buoys?".

Usage:
    python -m src.scripts.extract_structured_content \
        --folder data/incois_reports --out data/extracted
    # or single file:
    python -m src.scripts.extract_structured_content \
        --input file.pdf --out data/extracted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _looks_like_table(rows: list[list]) -> bool:
    """Heuristic: a real table has >1 row with structured columns, or a
    caption-like first cell. pymupdf's find_tables sometimes flags text
    layout; this filters obvious false positives."""
    if not rows or len(rows) < 2:
        return False
    # require at least 2 columns on the header row
    header = [str(c or "").strip() for c in rows[0]]
    non_empty = [h for h in header if h]
    if len(non_empty) < 2:
        return False
    return True


def extract_pdf(pdf_path: Path, min_fig_px: int = 20000) -> dict:
    """Extract tables + figures from one PDF. Returns a report dict.

    ``min_fig_px``: minimum image area (w*h) to count as a figure. Smaller
    images (logos, icons, buttons, header graphics) are ignored so the
    figure count reflects actual diagrams/charts, not decoration.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    tables: list[dict] = []
    figures: list[dict] = []
    seen_xrefs: set[int] = set()  # dedup same image reused across pages
    total_pages = len(doc)
    pages_with_text = 0
    total_chars = 0

    for i in range(total_pages):
        page = doc[i]
        text = page.get_text().strip()
        if len(text) > 50:
            pages_with_text += 1
        total_chars += len(text)

        # ── Tables ──
        try:
            found = page.find_tables()
            for t in found.tables:
                rows = t.extract()
                if _looks_like_table(rows):
                    # readable text form
                    lines = []
                    for row in rows:
                        cells = [str(c or "").strip() for c in row]
                        lines.append(" | ".join(cells))
                    tables.append({
                        "page": i + 1,
                        "bbox": list(t.bbox) if t.bbox else None,
                        "rows": len(rows),
                        "cols": len(rows[0]) if rows else 0,
                        "text": "\n".join(lines),
                    })
        except Exception:  # noqa: BLE001
            pass

        # ── Figures / diagrams (embedded raster images, real-size only) ──
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                w, h = img[2], img[3]
                if w * h < min_fig_px:
                    continue  # logo/icon/decoration — skip
                if xref in seen_xrefs:
                    continue  # same image on another page — skip dup
                # try to find a caption: text right below the image bbox
                rects = page.get_image_rects(xref)
                caption = None
                for r in rects[:1]:
                    # look ~40pt below the image for a "Figure N" line
                    region = fitz.Rect(r.x0, r.y1, r.x1, min(r.y1 + 40, page.rect.y1))
                    snippet = page.get_text(clip=region).strip()
                    if snippet and ("fig" in snippet.lower() or len(snippet) < 120):
                        caption = " ".join(snippet.split())[:200]
                seen_xrefs.add(xref)
                figures.append({
                    "page": i + 1,
                    "xref": xref,
                    "size": (w, h),
                    "caption": caption,
                    "rect": [list(r) for r in rects[:1]],
                })
        except Exception:  # noqa: BLE001
            pass

    doc.close()
    return {
        "file": pdf_path.name,
        "pages": total_pages,
        "pages_with_text": pages_with_text,
        "total_chars": total_chars,
        "text_coverage_pct": round(100.0 * pages_with_text / total_pages, 1) if total_pages else 0,
        "tables": tables,
        "figures": figures,
    }


def save_images(pdf_path: Path, report: dict, out_dir: Path) -> int:
    """Save detected figure images as PNG files. Returns count saved."""
    import fitz

    doc = fitz.open(str(pdf_path))
    saved = 0
    for idx, fig in enumerate(report["figures"], start=1):
        try:
            pix = fitz.Pixmap(doc, fig["xref"])
            if pix.n - pix.alpha > 3:  # CMYK → RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out = out_dir / f"fig_{fig['page']:03d}_{idx}.png"
            pix.save(str(out))
            fig["image_file"] = out.name
            saved += 1
        except Exception:  # noqa: BLE001
            continue
    doc.close()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None, help="Single PDF")
    ap.add_argument("--folder", default=None, help="Folder of PDFs (recursive)")
    ap.add_argument("--out", default="data/extracted", help="Output dir")
    ap.add_argument("--save-images", action="store_true", help="Also save image files")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.input:
        targets.append(Path(args.input))
    elif args.folder:
        targets = sorted(Path(args.folder).rglob("*.pdf"))
    else:
        print("Pass --input or --folder")
        sys.exit(1)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for pdf in targets:
        print(f"[extract] {pdf.name} ...", flush=True)
        try:
            report = extract_pdf(pdf)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            continue

        # write structured files
        stem_dir = out_root / pdf.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        tbl = report.pop("tables")
        figs = report.pop("figures")
        (stem_dir / "tables.json").write_text(
            json.dumps(tbl, ensure_ascii=False, indent=1), encoding="utf-8")
        (stem_dir / "figures.json").write_text(
            json.dumps(figs, ensure_ascii=False, indent=1), encoding="utf-8")

        # save images if asked
        img_saved = 0
        if args.save_images:
            img_dir = stem_dir / "images"
            img_dir.mkdir(exist_ok=True)
            img_saved = save_images(pdf, {"figures": figs}, img_dir)

        # summary line
        report["tables"] = len(tbl)
        report["figures"] = len(figs)
        report["images_saved"] = img_saved
        summary.append(report)
        print(f"  -> {len(tbl)} tables, {len(figs)} figures, "
              f"text {report['text_coverage_pct']}%")

    # write aggregate summary
    summary_path = out_root / "_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSummary: {len(summary)} PDFs processed -> {summary_path}")

    # aggregate numbers
    tot_tables = sum(s["tables"] for s in summary)
    tot_figs = sum(s["figures"] for s in summary)
    avg_text = sum(s["text_coverage_pct"] for s in summary) / len(summary) if summary else 0
    print(f"TOTAL: {tot_tables} tables, {tot_figs} figures, avg text coverage {avg_text:.0f}%")


if __name__ == "__main__":
    main()
