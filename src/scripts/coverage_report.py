"""
Coverage / comparison report — "how much of the source data did we
extract + ingest?"

Compares THREE things per PDF and overall:
  A. Source: total pages, pages with extractable text, tables, figures
     (from extract_structured_content's _summary.json, or scans fresh)
  B. Extracted: text coverage %, tables extracted, figures found
  C. Ingested: did a corresponding record make it into the corpus JSONL?
     (matched by filename stem in the record's source_url/subject)

Outputs:
  - console summary
  - data/coverage_report.csv  (per-file row)
  - prints the overall % coverage — the number sir asked for

Usage:
    python -m src.scripts.coverage_report \
        --pdfs data/incois_reports \
        --corpus data/corpus_reports.jsonl \
        --extracted data/extracted \
        --out data/coverage_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def scan_pdfs(folder: Path, ocr_dir: Path | None = None) -> dict[str, dict]:
    """Quick scan: pages, text pages, chars, tables, figures per PDF.

    If ``ocr_dir`` is given, PDFs with no extractable text are cross-checked
    against OCR'd ``.txt`` files (same stem) so a scanned-but-OCR'd report
    counts as extracted, not 0%."""
    import fitz

    ocr_stems = {p.stem: p for p in ocr_dir.glob("*.txt")} if ocr_dir and ocr_dir.exists() else {}

    out = {}
    for pdf in sorted(folder.rglob("*.pdf")):
        try:
            doc = fitz.open(str(pdf))
            pages = len(doc)
            text_pages = 0
            chars = 0
            tables = 0
            for i in range(pages):
                t = doc[i].get_text().strip()
                if len(t) > 50:
                    text_pages += 1
                chars += len(t)
                try:
                    tables += len(doc[i].find_tables().tables)
                except Exception:  # noqa: BLE001
                    pass
            doc.close()

            # OCR fallback: scanned PDF whose text lives in an OCR .txt
            ocr_txt = ocr_stems.get(pdf.stem)
            ocr_chars = 0
            if ocr_txt:
                try:
                    ocr_chars = len(ocr_txt.read_text(encoding="utf-8", errors="ignore"))
                except Exception:  # noqa: BLE001
                    ocr_chars = 0
            if chars < 50 and ocr_chars > 50:
                chars = ocr_chars
                text_pages = pages  # fully OCR'd -> all pages have text

            out[pdf.stem] = {
                "file": pdf.name,
                "pages": pages,
                "pages_with_text": text_pages,
                "total_chars": chars,
                "text_coverage_pct": round(100.0 * text_pages / pages, 1) if pages else 0,
                "tables_detected": tables,
                "ocr_applied": 1 if ocr_chars > 50 else 0,
            }
        except Exception as e:  # noqa: BLE001
            out[pdf.stem] = {"file": pdf.name, "error": str(e)}
    return out


def load_extracted_summary(path: Path) -> dict[str, dict]:
    """If extract_structured_content ran, load its _summary.json."""
    if not path.exists():
        return {}
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        return {Path(i["file"]).stem: i for i in items}
    except Exception:  # noqa: BLE001
        return {}


def load_corpus_stems(corpus_path: Path) -> set[str]:
    """Stems of files that made it into the corpus.

    REAL matching, not a guess: for every corpus record we collect
    (a) the stem of its metadata.source_url, (b) the stem of its
    metadata.subject, and (c) its question_id (incdoc-* for converted docs).
    A source PDF is "ingested" iff any of these matches its filename stem
    (or vice versa). This replaced the old always-yes/"maybe" stub.
    """
    stems: set[str] = set()
    if not corpus_path.exists():
        return stems
    for line in open(corpus_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        for needle in ("source_url", "subject"):
            v = meta.get(needle, "")
            if v:
                stem = Path(str(v)).stem.lower()
                if stem:
                    stems.add(stem)
        qid = rec.get("question_id", "")
        if qid:
            stems.add(qid.lower())
            # also the incdoc stem itself
            if qid.startswith("incdoc-"):
                stems.add(qid[len("incdoc-"):].lower())
    return stems


def is_ingested(pdf_stem: str, pdf_file: str, corpus_stems: set[str]) -> str:
    """True match between a source PDF and the corpus stems."""
    if not corpus_stems:
        return "no"  # no corpus -> definitely not ingested
    stem = pdf_stem.lower()
    fname = pdf_file.lower()
    for s in corpus_stems:
        if not s:
            continue
        if s == stem or s in fname or stem in s:
            return "yes"
    return "no"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdfs", required=True, help="Folder of source PDFs")
    ap.add_argument("--corpus", default="data/corpus_reports.jsonl", help="Merged corpus JSONL")
    ap.add_argument("--extracted", default="data/extracted", help="extract_structured_content output dir")
    ap.add_argument("--ocr", default="data/scanned_ocr", help="Folder of OCR'd .txt files (for scanned PDFs)")
    ap.add_argument("--out", default="data/coverage_report.csv")
    args = ap.parse_args()

    pdf_folder = Path(args.pdfs)
    if not pdf_folder.exists():
        print(f"PDFs folder not found: {pdf_folder}")
        sys.exit(1)

    print("Scanning source PDFs...")
    sources = scan_pdfs(pdf_folder, ocr_dir=Path(args.ocr))
    extracted = load_extracted_summary(Path(args.extracted) / "_summary.json")
    corpus_stems = load_corpus_stems(Path(args.corpus))

    rows = []
    for stem, src in sources.items():
        if "error" in src:
            rows.append({**src, "tables_extracted": "", "figures_found": "",
                         "ingested": "error", "extract_pct": "", "overall_pct": ""})
            continue
        ext = extracted.get(stem, {})
        # REAL corpus match (no more always-"yes"/"maybe" stub)
        ingested = is_ingested(stem, src["file"], corpus_stems)

        rows.append({
            "file": src["file"],
            "pages": src["pages"],
            "pages_with_text": src["pages_with_text"],
            "text_coverage_pct": src["text_coverage_pct"],
            "ocr_applied": src.get("ocr_applied", 0),
            "tables_detected": src.get("tables_detected", 0),
            "tables_extracted": ext.get("tables", 0),
            "figures_found": ext.get("figures", 0),
            "ingested": ingested,
            "total_chars": src.get("total_chars", 0),
        })

    # write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)

    # aggregate
    n = len(rows)
    if n:
        text_pages = sum(r.get("pages_with_text", 0) for r in rows)
        total_pages = sum(r.get("pages", 0) for r in rows)
        tables_det = sum(r.get("tables_detected", 0) for r in rows)
        tables_ext = sum(r.get("tables_extracted", 0) for r in rows)
        figs = sum(r.get("figures_found", 0) for r in rows)
        ocr_count = sum(r.get("ocr_applied", 0) for r in rows)
        ingested_yes = sum(1 for r in rows if r.get("ingested") == "yes")
        ingested_no = sum(1 for r in rows if r.get("ingested") == "no")
        print("\n" + "=" * 60)
        print("COVERAGE REPORT (sir's mandate)")
        print("=" * 60)
        print(f"Source PDFs scanned      : {n}")
        print(f"Total pages              : {total_pages}")
        print(f"Pages with text          : {text_pages}  ({100.0*text_pages/total_pages:.0f}% of pages)")
        print(f"  (of which OCR'd)       : {ocr_count}")
        print(f"Tables detected          : {tables_det}")
        print(f"Tables extracted         : {tables_ext}  ({100.0*tables_ext/max(tables_det,1):.0f}% of tables)")
        print(f"Figures found (diagrams) : {figs}  (logos/icons filtered out)")
        print(f"INGESTED (in corpus)     : {ingested_yes}/{n}  ({100.0*ingested_yes/max(n,1):.0f}%)")
        print(f"  (not in corpus)        : {ingested_no}")
        print(f"CSV report               : {out_path.resolve()}")
        print("=" * 60)
        print("Text coverage = % of PDF pages that yielded extractable text")
        print("               (scanned PDFs count as covered via OCR .txt).")
        print("Tables extracted = captured into tables.json (vs detected by layout).")
        print("Figures = real embedded diagrams/photos (size-filtered); save with")
        print("          --save-images and/or describe with a multimodal model")
        print("          for full content coverage.")
    else:
        print("No PDFs found.")


if __name__ == "__main__":
    main()
