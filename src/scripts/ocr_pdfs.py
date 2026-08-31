"""
OCR scanned/image-only PDFs -> text files, ready for ingestion.

Windows-friendly: uses PyMuPDF (fitz) to render pages to images + pytesseract
to OCR — NO poppler needed (pdf2image is avoided). Works the same on Linux.

Usage:
    # OCR a single PDF:
    python -m src.scripts.ocr_pdfs --input path/to/scanned.pdf --output dir

    # OCR every scanned PDF in a folder (skips PDFs that already have text):
    python -m src.scripts.ocr_pdfs --folder data/scanned --output data/scanned_ocr

    # Higher quality for dense small text:
    python -m src.scripts.ocr_pdfs --input x.pdf --output dir --dpi 300

Output: one .txt per PDF (same name, .txt extension) with "--- Page N ---"
separators — exactly the format convert_sirs_knowledge.py --scanned consumes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _tools_ready() -> bool:
    if shutil.which("tesseract") is None:
        print("[!] tesseract not found on PATH. Install it first:")
        print("    Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("    Linux:   sudo apt-get install -y tesseract-ocr")
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        print("[!] pip install pytesseract  (and pymupdf)")
        return False
    return True


def has_extractable_text(pdf_path: Path) -> bool:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        chars = sum(len(p.get_text().strip()) for p in doc)
        doc.close()
        return chars > 50
    except Exception:  # noqa: BLE001
        return False


def ocr_pdf(pdf_path: Path, dpi: int = 200) -> str:
    """OCR a PDF page-by-page using PyMuPDF rendering + tesseract.

    Page-by-page keeps memory flat (a 78-page scanned report won't OOM).
    Returns '' on failure — never raises.
    """
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        pages_text: list[str] = []
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            try:
                txt = pytesseract.image_to_string(img)
                if txt.strip():
                    pages_text.append(f"--- Page {i+1} (OCR) ---\n{txt.strip()}")
            except Exception:  # noqa: BLE001
                continue
            del img, pix  # free memory
        doc.close()
        return "\n\n".join(pages_text)
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None, help="Single scanned PDF to OCR")
    ap.add_argument("--folder", default=None, help="Folder of PDFs to OCR (skips text PDFs)")
    ap.add_argument("--output", default="data/scanned_ocr", help="Where to write .txt files")
    ap.add_argument("--dpi", type=int, default=200, help="Render resolution (200-300)")
    args = ap.parse_args()

    if not _tools_ready():
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = []
    if args.input:
        targets.append(Path(args.input))
    elif args.folder:
        targets = sorted(Path(args.folder).rglob("*.pdf"))
    else:
        print("Pass --input <file.pdf> or --folder <dir>")
        sys.exit(1)

    ok = skipped = failed = 0
    for pdf in targets:
        if not pdf.exists():
            print(f"[missing] {pdf}")
            failed += 1
            continue
        if has_extractable_text(pdf):
            print(f"[skip] {pdf.name} (already has text — no OCR needed)")
            skipped += 1
            continue
        print(f"[ocr] {pdf.name} ...", flush=True)
        text = ocr_pdf(pdf, dpi=args.dpi)
        if not text.strip():
            print(f"  -> FAILED (no text extracted)")
            failed += 1
            continue
        out = out_dir / (pdf.stem + ".txt")
        out.write_text(text, encoding="utf-8")
        print(f"  -> {len(text):,} chars -> {out.name}")
        ok += 1

    print(f"\nDone: {ok} OCR'd, {skipped} skipped (had text), {failed} failed")
    print(f"Text files in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
