"""Reproducible corpus-quality report (Phase 1 / WS1).

Generates a DETERMINISTIC JSON report from a corpus JSONL file — the data
foundation a future graph build (WS2/Neo4j) will be validated against:

  * record counts by id-class / source / document type
  * ministry raw variants → canonical identities (+ the sansad-code
    cross-check against the corpus-observed (house, ministry) mapping)
  * member variants (raw vs canonical keys, honorific-led tokens, the
    multi-variant keys — REVIEW material, never auto-merged)
  * date coverage (shape distribution, spans, invalid dates)
  * nulls / missing metadata coverage per field
  * duplicate indicators (id collisions, (session, qno) collisions,
    identical content hashes, identical question text)
  * malformed rows (JSON or QARecord validation failures)
  * controlled-concept mention counts (text mining — creates no entities)
  * graph-useful field coverage per id-class

No write access to the corpus; no side effects beyond the report file.
Deterministic: same corpus bytes → same report bytes (sorted, no timestamps).

CLI:
    python -m src.vocabulary.corpus_quality --corpus <file> --out <report.json>
    python -m src.vocabulary.corpus_quality --corpus <file> --out r.json --md r.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.vocabulary import normalize as nz
from src.vocabulary.vocab import Vocabulary, load_vocabulary

__all__ = ["generate_corpus_quality_report", "main"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _class_of(doc_id: str) -> str:
    if doc_id.startswith("ls-"):
        return "ls"
    if doc_id.startswith("rs-"):
        return "rs"
    if doc_id.startswith("incdoc-"):
        return "incdoc"
    return "other"


def generate_corpus_quality_report(
    corpus_path: Path | str,
    vocabulary: Vocabulary | None = None,
) -> dict[str, Any]:
    """Build the report dict (pure; write it to disk by the caller/CLI)."""
    corpus_path = Path(corpus_path)
    voc = vocabulary or load_vocabulary()
    vocab_version = voc.version

    rows: list[dict[str, Any]] = []
    json_errors: list[str] = []
    validation_errors: list[str] = []

    # QARecord validation (import lazily — it pulls pydantic but no models)
    try:
        from src.models.qa_record import QARecord
    except Exception:  # noqa: BLE001 — never let report generation die here
        QARecord = None  # type: ignore[assignment]

    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                json_errors.append(f"line {i}: {e.msg}")
                continue
            rows.append(r)
            if QARecord is not None:
                try:
                    QARecord.model_validate_json(line)
                except Exception as e:  # noqa: BLE001
                    if len(validation_errors) < 20:
                        validation_errors.append(f"line {i}: {str(e)[:200]}")

    n = len(rows)
    meta = [r.get("metadata") or {} for r in rows]

    # ── id classes ────────────────────────────────────────────────────────────
    id_classes = Counter(_class_of(r.get("question_id") or "") for r in rows)

    # ── metadata coverage (per field, across all rows) ───────────────────────
    all_fields = Counter()
    for m in meta:
        all_fields.update(m.keys())
    coverage = {}
    for fld in sorted(all_fields):
        present = sum(1 for m in meta if m.get(fld) not in (None, ""))
        coverage[fld] = {
            "rows_with_value": present,
            "rows_missing_or_blank": n - present,
            "coverage": round(present / n, 4) if n else 0.0,
        }

    # ── ministry ──────────────────────────────────────────────────────────────
    ministry_raw = Counter()
    ministry_canon = Counter()
    ministry_unknown: list[str] = []
    sansad_checked = 0
    sansad_mismatch: list[str] = []
    for r, m in zip(rows, meta):
        raw = m.get("ministry")
        if raw is None or str(raw).strip() == "":
            ministry_raw["<missing>"] += 1
            continue
        ministry_raw[str(raw)] += 1
        norm = nz.normalize_ministry(voc, raw)
        if norm.status == "canonical":
            ministry_canon[norm.canonical] += 1
        else:
            ministry_unknown.append(str(raw))
        # sansad-code cross-check (parliamentary rows only)
        qid = r.get("question_id") or ""
        if qid.startswith(("ls-", "rs-")) and m.get("ministry_code") is not None:
            sansad_checked += 1
            expected = nz.expected_sansad_code(voc, m.get("house"), m.get("ministry"))
            if expected is not None and expected != m.get("ministry_code"):
                if len(sansad_mismatch) < 20:
                    sansad_mismatch.append(
                        f"{qid}: code={m.get('ministry_code')} "
                        f"expected={expected} (house={m.get('house')}, "
                        f"ministry={m.get('ministry')})")

    # ── org / source / house ─────────────────────────────────────────────────
    def _controlled(field_name: str, match, default_rule: str) -> dict[str, Any]:
        raw = Counter()
        canon = Counter()
        unknown = Counter()
        for m in meta:
            v = m.get(field_name)
            key = "<missing>" if v is None or str(v).strip() == "" else str(v)
            raw[key] += 1
            e = match(v)
            if e is None and key != "<missing>":
                unknown[str(v)] += 1
            elif e is not None:
                canon[e.canonical] += 1
        return {
            "raw_variants": dict(sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))),
            "canonical_counts": dict(sorted(canon.items(), key=lambda kv: (-kv[1], kv[0]))),
            "unknown_values": dict(sorted(unknown.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    org_block = _controlled("org", voc.match_org, "org")
    source_block = _controlled("source", voc.match_source, "source")
    house_block = _controlled("house", voc.match_house, "house")

    # house vs doc-id-prefix consistency
    house_id_mismatches = 0
    for r, m in zip(rows, meta):
        qid = r.get("question_id") or ""
        by_prefix = voc.house_by_id_prefix(qid)
        h_field = m.get("house")
        if by_prefix is not None:
            if h_field is None or by_prefix.canonical != nz.normalize_house(voc, h_field).canonical:
                house_id_mismatches += 1
        else:
            if h_field is not None and str(h_field).strip():
                house_id_mismatches += 1

    # ── LS terms / RS sessions ───────────────────────────────────────────────
    ls_terms: dict[str, Any] = {}
    ls_term_years: dict[str, list[int]] = {}
    ls_term_uncontrolled = 0
    for r, m in zip(rows, meta):
        qid = r.get("question_id") or ""
        if not qid.startswith("ls-"):
            continue
        term = nz.ls_term_from_doc_id(qid)
        key = f"{term}" if term is not None else "<unparseable>"
        ls_terms[key] = ls_terms.get(key, 0) + 1
        if term is None or voc.ls_term(term) is None:
            ls_term_uncontrolled += 1
        d = m.get("date") or ""
        if re.match(r"^\d{4}", str(d)):
            year = int(str(d)[:4])
            ys = ls_term_years.setdefault(key, [])
            ys.append(year)
    ls_terms_out = {
        "row_counts_by_term": dict(sorted(ls_terms.items(), key=lambda kv: (kv[0] == "<unparseable>", kv[0]))),
        "observed_year_spans": {
            k: [min(v), max(v)] for k, v in sorted(ls_term_years.items())
        },
        "terms_not_in_vocabulary": ls_term_uncontrolled,
        "vocabulary_terms": [
            {"term": t["term"], "years": t.get("years")} for t in voc.ls_terms
        ],
    }

    rs_sessions = set()
    for m in meta:
        if m.get("session") is not None:
            try:
                rs_sessions.add(int(m["session"]))
            except (TypeError, ValueError):
                pass
    rs_block = {
        "distinct_sessions": len(rs_sessions),
        "min": min(rs_sessions) if rs_sessions else None,
        "max": max(rs_sessions) if rs_sessions else None,
        "vocabulary_observed": voc.rs_sessions,
    }

    # ── document_type / question_type ────────────────────────────────────────
    dt_block = _controlled("document_type", voc.match_document_type, "document_type")
    qt_block = _controlled("question_type", voc.match_question_type, "question_type")

    # ── members ──────────────────────────────────────────────────────────────
    member_shape = Counter()
    raw_tokens = Counter()
    canon_tokens = Counter()
    honorific_led = 0
    key_variants: dict[str, Counter] = {}
    for m in meta:
        mem = m.get("member")
        if mem is None or not str(mem).strip():
            member_shape["null"] += 1
            continue
        ms = str(mem)
        member_shape["clubbed" if ";" in ms else "single"] += 1
        norm = nz.normalize_member(voc, ms)
        for t in norm.tokens:
            raw_tokens[t.raw] += 1
            canon_tokens[t.canonical_key] += 1
            if t.honorific:
                honorific_led += 1
            key_variants.setdefault(t.canonical_key, Counter())[t.raw] += 1
    multi_variant = [
        {
            "canonical_key": k,
            "total": sum(c.values()),
            "variants": {raw: cnt for raw, cnt in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))},
        }
        for k, c in sorted(key_variants.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
        if len(c) > 1
    ]
    member_block = {
        "shapes": dict(member_shape),
        "distinct_raw_tokens": len(raw_tokens),
        "distinct_canonical_keys": len(canon_tokens),
        "honorific_led_tokens": honorific_led,
        "keys_with_multiple_raw_variants": len(multi_variant),
        "multi_variant_keys_top100": multi_variant[:100],
        "note": (
            "Variants are case/honorific differences only in the current "
            "corpus; they are NOT merged in the corpus — this list is review "
            "material for a future graph identity step."
        ),
    }

    # ── dates ────────────────────────────────────────────────────────────────
    date_shapes = Counter()
    invalid_dates: list[str] = []
    year_span: list[int] = []
    year_only_by_class = Counter()
    for r, m in zip(rows, meta):
        d = m.get("date")
        norm = nz.normalize_date(voc, d)
        if norm.status == "empty":
            date_shapes["null"] += 1
        elif norm.precision == "day":
            date_shapes["iso_date"] += 1
            year_span.append(int(str(m.get("date"))[:4]))
        elif norm.precision == "year":
            date_shapes["year_only"] += 1
            year_span.append(int(norm.canonical))
            year_only_by_class[_class_of(r.get("question_id") or "")] += 1
        else:
            date_shapes["unrecognized"] += 1
            if norm.rule == "calendar-invalid-date":
                invalid_dates.append(str(d))
    date_block = {
        "shape_counts": dict(sorted(date_shapes.items())),
        "invalid_calendar_dates": invalid_dates[:20],
        "year_span": [min(year_span), max(year_span)] if year_span else None,
        "year_only_by_id_class": dict(sorted(year_only_by_class.items())),
    }

    # ── duplicates ───────────────────────────────────────────────────────────
    id_counts = Counter(r.get("question_id") for r in rows)
    id_collisions = {k: v for k, v in id_counts.items() if v > 1}
    ls_keys = Counter(
        (m.get("session"), m.get("question_number"))
        for r, m in zip(rows, meta)
        if (r.get("question_id") or "").startswith("ls-")
        and m.get("session") is not None and m.get("question_number") is not None
    )
    ls_key_collisions = {f"{s}/{q}": c for (s, q), c in ls_keys.items() if c > 1}
    ch_counts = Counter(r.get("content_hash") for r in rows if r.get("content_hash"))
    content_hash_dups = {k: v for k, v in ch_counts.items() if v > 1}
    qt_counts = Counter((r.get("question_text") or "")[:200] for r in rows)
    question_text_collisions = sum(c - 1 for c in qt_counts.values() if c > 1)
    duplicate_block = {
        "question_id_collisions": dict(sorted(id_collisions.items(), key=lambda kv: str(kv[0]))),
        "ls_session_question_no_collisions": dict(sorted(ls_key_collisions.items())),
        "identical_content_hash_groups": dict(sorted(content_hash_dups.items())),
        "duplicate_question_text_rows": question_text_collisions,
    }

    # ── concepts (text mining; creates nothing) ─────────────────────────────
    concept_docs: dict[str, set[str]] = {}
    concept_mentions: dict[str, int] = Counter()
    surface_hits: dict[str, Counter] = Counter()
    for r in rows:
        qid = r.get("question_id") or ""
        text = f"{r.get('question_text') or ''}\n{r.get('answer_text') or ''}"
        counts = nz.concept_mentions(voc, text)
        for canon, cnt in counts.items():
            concept_docs.setdefault(canon, set()).add(qid)
            concept_mentions[canon] += cnt
    concept_block = {
        "documents_mentioning": {
            c: len(docs) for c, docs in sorted(concept_docs.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        },
        "total_mentions": dict(sorted(concept_mentions.items(), key=lambda kv: (-kv[1], kv[0]))),
        "note": "Counts are text-mining over question+answer for review; they create no entities.",
    }

    # ── graph-useful field coverage per id-class (candidate graph properties)
    class_fields: dict[str, dict[str, int]] = {}
    class_counts = Counter()
    for r, m in zip(rows, meta):
        cls = _class_of(r.get("question_id") or "")
        class_counts[cls] += 1
        cf = class_fields.setdefault(cls, Counter())
        for k, v in m.items():
            if v not in (None, ""):
                cf[k] += 1
    graph_useful = {}
    for cls in sorted(class_fields):
        total = class_counts[cls]
        graph_useful[cls] = {
            "rows": total,
            "fields": {
                f: {"non_null": c, "coverage": round(c / total, 4) if total else 0.0}
                for f, c in sorted(class_fields[cls].items(), key=lambda kv: (-kv[1], kv[0]))
            },
        }

    return {
        "report": "corpus-quality",
        "vocabulary_version": vocab_version,
        "corpus": {
            "path": str(corpus_path),
            "sha256": _sha256(corpus_path),
            "bytes": corpus_path.stat().st_size,
            "rows": n,
        },
        "id_classes": dict(sorted(id_classes.items(), key=lambda kv: (-kv[1], kv[0]))),
        "metadata_coverage": coverage,
        "ministry": {
            "raw_variants": dict(sorted(ministry_raw.items(), key=lambda kv: (-kv[1], kv[0]))),
            "canonical_counts": dict(sorted(ministry_canon.items(), key=lambda kv: (-kv[1], kv[0]))),
            "unknown_values": dict(sorted(Counter(ministry_unknown).items(),
                                          key=lambda kv: (-kv[1], kv[0]))),
            "sansad_code_cross_check": {
                "rows_checked": sansad_checked,
                "mismatches": len(sansad_mismatch),
                "examples": sansad_mismatch,
            },
        },
        "org": org_block,
        "source": source_block,
        "house": {**house_block, "doc_id_prefix_mismatches": house_id_mismatches},
        "ls_terms": ls_terms_out,
        "rs_sessions": rs_block,
        "document_type": dt_block,
        "question_type": qt_block,
        "members": member_block,
        "dates": date_block,
        "duplicates": duplicate_block,
        "malformed": {
            "json_parse_errors": len(json_errors),
            "json_errors_sample": json_errors[:10],
            "qa_record_validation_errors": len(validation_errors),
            "validation_errors_sample": validation_errors,
        },
        "concepts": concept_block,
        "graph_useful_fields": graph_useful,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    c = report["corpus"]
    lines = [
        "# Corpus Quality Report",
        "",
        f"- **rows**: {c['rows']}",
        f"- **sha256**: `{c['sha256']}`",
        f"- **vocabulary version**: {report['vocabulary_version']}",
        "",
        "## id classes",
        "",
        "| class | rows |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in report["id_classes"].items()],
        "",
        "## ministry",
        "",
        "| raw variant | rows |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in report["ministry"]["raw_variants"].items()],
        "",
        "canonical: " + ", ".join(f"`{k}`={v}" for k, v in report["ministry"]["canonical_counts"].items()),
        "",
        f"sansad-code cross-check: {report['ministry']['sansad_code_cross_check']['rows_checked']} rows, "
        f"{report['ministry']['sansad_code_cross_check']['mismatches']} mismatches",
        "",
        "## dates",
        "",
        "| shape | rows |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in report["dates"]["shape_counts"].items()],
        "",
        f"year span: {report['dates']['year_span']}",
        "",
        "## members",
        "",
        f"- shapes: {report['members']['shapes']}",
        f"- distinct raw tokens: {report['members']['distinct_raw_tokens']}",
        f"- distinct canonical keys: {report['members']['distinct_canonical_keys']}",
        f"- honorific-led tokens: {report['members']['honorific_led_tokens']}",
        f"- keys with multiple raw variants: {report['members']['keys_with_multiple_raw_variants']}",
        "",
        "## duplicates",
        "",
        f"- question_id collisions: {len(report['duplicates']['question_id_collisions'])}",
        f"- (session, question_no) collisions: {len(report['duplicates']['ls_session_question_no_collisions'])}",
        f"- identical content-hash groups: {len(report['duplicates']['identical_content_hash_groups'])}",
        f"- duplicate question-text rows: {report['duplicates']['duplicate_question_text_rows']}",
        "",
        "## concepts (text mining — review material)",
        "",
        "| concept | documents | mentions |", "|---|---|---|",
        *[f"| {k} | {report['concepts']['documents_mentioning'].get(k, 0)} | {v} |"
          for k, v in report["concepts"]["total_mentions"].items()],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, type=Path,
                    help="path to corpus_reports.jsonl")
    ap.add_argument("--out", required=True, type=Path,
                    help="output JSON report path")
    ap.add_argument("--md", type=Path, default=None,
                    help="optional human-readable markdown summary path")
    args = ap.parse_args(argv)

    report = generate_corpus_quality_report(args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"[corpus-quality] wrote {args.out} ({report['corpus']['rows']} rows, "
          f"sha256 {report['corpus']['sha256'][:16]}…)")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(_render_markdown(report), encoding="utf-8")
        print(f"[corpus-quality] wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
