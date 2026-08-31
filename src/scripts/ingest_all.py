"""
One-command ingestion: convert ALL of sir's knowledge formats + public
INCOIS reports + parliamentary corpus -> ONE merged JSONL -> (optional) index.

Classification: OPERATOR TOOL — explicit bulk corpus writer
(workspace cleanup, audit §4). This is the "make everything ingest-ready"
batch conversion entry point for machine migrations and deliberate corpus
(re)builds. It is NOT the day-to-day ingestion path — that is
`python -m src.scripts.ingest` (config-driven registry, incremental,
append-only canonical corpus). The two coexist with a hard boundary:

  * ingest_all WRITES the path given by --out (full overwrite semantics by
    design — that is what an operator bulk rebuild means);
  * the canonical CLI/engine APPEND to data/corpus_reports.jsonl and never
    rewrite foreign records;
  * sync_sources.py uses ingest_all with a SCRATCH --out + a safe id-union
    merge (Phase-3 F.1 fix): even there the canonical corpus is never
    rewritten in place;
  * pointing --out directly AT the canonical corpus stays possible as
    explicit operator replacement intent, and now prints a loud warning
    naming exactly what is about to be replaced.

It reuses the per-format converters from convert_sirs_knowledge.py and the
crawl scripts, then merges with the existing parliamentary processed/
corpus, dedupes, and optionally rebuilds the Hybrid RAG index.

Usage (run on the local dev PC that has the data / internet):
    # Convert everything from sir's machine + public reports:
    python -m src.scripts.ingest_all \
      --qa       "FINAL_audit_qa_dataset.json" \
      --knowledge "Knowledge_Base" \
      --documents "UserKnowledge" \
      --documents "KnowledgeBase(UserAdded)" \
      --scanned   "KnowledgeBase(Scanned)" \
      --annual    data/incois_reports/AnnualReports \
      --reports   data/incois_reports/Others \
      --parliament data/processed \
      --out       data/corpus_merged.jsonl

    # Optional: crawl public reports first (needs internet):
    python -m src.scripts.ingest_all --crawl --out data/corpus_merged.jsonl

    # Optional: build the index right after (needs the ML env + models):
    python -m src.scripts.ingest_all ... --build

Anything you don't pass is skipped. Output is validated QARecord JSONL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.models.qa_record import QARecord
from src.scripts.detect_doc_type import detect_doc_type
from src.scripts.convert_sirs_knowledge import (
    convert_qa_dataset,
    convert_knowledge_json,
    convert_document_json,
    convert_text_file,
    convert_pdf_file,
    convert_annual_pdf,
    convert_report_pdf,
)


def _convert_folder(
    folder: str,
    out: list[QARecord],
    seen: set[str],
    mode: str,
) -> int:
    """Convert a whole folder with a given mode. mode in:
    knowledge | document | scanned | annual | report"""
    p = Path(folder)
    if not p.exists():
        print(f"  [missing] {folder} — skipped")
        return 0
    n = 0
    files = sorted(p.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        try:
            if mode == "knowledge" and f.suffix.lower() == ".json":
                n += convert_knowledge_json(f, out, seen)
            elif mode == "document" and f.suffix.lower() == ".json":
                n += convert_document_json(f, out, seen)
            elif mode == "scanned":
                if f.suffix.lower() in (".txt", ".md"):
                    # detect the type (AR_* -> annual_report, TR_* -> technical,
                    # Report_* -> general) instead of defaulting to "document"
                    text_peek = ""
                    try:
                        text_peek = f.read_text(encoding="utf-8", errors="ignore")[:800]
                    except Exception:  # noqa: BLE001
                        pass
                    dt = detect_doc_type(f, text_peek)
                    n += convert_text_file(f, out, seen, doc_type=dt)
                elif f.suffix.lower() == ".pdf":
                    n += convert_pdf_file(f, out, seen)
            elif mode == "annual" and f.suffix.lower() == ".pdf":
                if f.name.lower().startswith("report_"):
                    n += convert_report_pdf(f, out, seen)
                else:
                    n += convert_annual_pdf(f, out, seen)
            elif mode == "report" and f.suffix.lower() == ".pdf":
                n += convert_report_pdf(f, out, seen)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {f.name}: {e}")
    return n


def merge_parliament(parliament_dir: str, out: list[QARecord], seen: set[str]) -> int:
    """Load existing parliamentary processed JSONL files into the corpus."""
    p = Path(parliament_dir)
    if not p.exists():
        print(f"  [missing] parliament {parliament_dir} — skipped")
        return 0
    n = 0
    for f in sorted(p.glob("*.jsonl")):
        try:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = QARecord.model_validate_json(line)
                if rec.question_id not in seen:
                    seen.add(rec.question_id)
                    out.append(rec)
                    n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {f.name}: {e}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", default=None, help="FINAL_audit_qa_dataset.json")
    ap.add_argument("--knowledge", default=None, help="Knowledge_Base folder")
    ap.add_argument("--documents", action="append", default=[], help="UserKnowledge / UserAdded folder (repeatable)")
    ap.add_argument("--scanned", default=None, help="KnowledgeBase(Scanned) folder")
    ap.add_argument("--annual", default=None, help="Annual reports PDF folder")
    ap.add_argument("--reports", action="append", default=[], help="Other INCOIS report PDF folders (repeatable)")
    ap.add_argument("--parliament", default="data/processed", help="Existing parliamentary JSONL dir (merge)")
    ap.add_argument("--out", default="data/corpus_merged.jsonl")
    ap.add_argument("--crawl", action="store_true", help="Run public crawlers first (needs internet)")
    ap.add_argument("--build", action="store_true", help="Rebuild Hybrid RAG index after writing merged corpus")
    args = ap.parse_args()

    # ── Optional crawl ─────────────────────────────────────────────────────
    if args.crawl:
        print("Running public crawlers (internet required)...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "src.scripts.crawl_incois_reports",
             "--sections", "all", "--out", "data/incois_reports"],
            check=False,
        )
        subprocess.run(
            [sys.executable, "-m", "src.scripts.crawl_moes_reports",
             "--out", "data/moes_reports"],
            check=False,
        )
        print("Crawl done.\n")

    out: list[QARecord] = []
    seen: set[str] = set()
    print("Ingesting sir's knowledge + public INCOIS reports -> merged corpus\n")

    if args.qa:
        n = convert_qa_dataset(Path(args.qa), out, seen)
        print(f"  [QA dataset] {n} records")
    if args.knowledge:
        n = _convert_folder(args.knowledge, out, seen, "knowledge")
        print(f"  [Knowledge_Base] {n} records")
    for d in args.documents:
        n = _convert_folder(d, out, seen, "document")
        print(f"  [Documents {Path(d).name}] {n} records")
    if args.scanned:
        n = _convert_folder(args.scanned, out, seen, "scanned")
        print(f"  [Scanned] {n} records")
    if args.annual:
        n = _convert_folder(args.annual, out, seen, "annual")
        print(f"  [AnnualReports] {n} records")
    for r in args.reports:
        n = _convert_folder(r, out, seen, "report")
        print(f"  [Reports {Path(r).name}] {n} records")

    # ── Merge existing parliamentary corpus ────────────────────────────────
    n = merge_parliament(args.parliament, out, seen)
    print(f"  [Parliament {args.parliament}] {n} records merged")

    if not out:
        print("Nothing ingested — check the --paths. Exiting.")
        sys.exit(1)

    out_path = Path(args.out)
    # Explicit-replacement warning (workspace cleanup, audit §4): pointing
    # this OPERATOR TOOL at the canonical corpus is lawful only as
    # deliberate rebuild intent. sync_sources never does this (scratch+merge).
    try:
        from src.utils.app_paths import corpus_path

        canonical = corpus_path()
        if out_path.resolve() == canonical.resolve():
            print("=" * 72)
            print("WARNING: --out IS THE CANONICAL CORPUS "
                  f"({canonical}).\nThis file will be REWRITTEN from the "
                  "converted sources above only — every record not produced "
                  "by THIS run (parliament merges, inbox uploads, "
                  "hierarchical trees, RS staging) will be REPLACED.\nIf you "
                  "wanted day-to-day incremental ingestion, press Ctrl-C and "
                  "use: python -m src.scripts.ingest all")
            print("=" * 72)
    except Exception:  # noqa: BLE001 — the warning must never break the tool
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(rec.model_dump_json() + "\n")

    types = {}
    for rec in out:
        t = rec.metadata.document_type or "unknown"
        types[t] = types.get(t, 0) + 1
    print(f"\nWrote {len(out)} records -> {out_path}")
    print("  by type:", types)

    # ── Optional index build ───────────────────────────────────────────────
    if args.build:
        print("\nBuilding Hybrid RAG index...")
        from src.retrieval.cli import build as cli_build
        import click
        from click.testing import CliRunner
        runner = CliRunner()
        res = runner.invoke(cli_build, ["--data", str(out_path), "--rebuild"])
        print(res.output)
        if res.exit_code != 0:
            print(res.exception)


if __name__ == "__main__":
    main()
