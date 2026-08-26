"""Cross-source overlap validation: MoES website PQ documents vs Parliamentary corpus.

READ-ONLY analysis tool. Compares the PQ-titled documents of the MoES website
staging corpus (``data/.moes-website/`` or ``data/moes-website/``) against the
Rajya Sabha Q&A staging corpus (``data/parliamentary-qa/``) and classifies each
MoES PQ document into exactly one of:

  A. EXACT_SHA                  — byte-identical file exists in the parliamentary corpus
  B. TEXTUALLY_NEAR_IDENTICAL   — different bytes, substantially identical extracted text
  C. POTENTIALLY_CORRESPONDING  — different files, text indicates the same PQ
  D. POTENTIALLY_UNIQUE         — no meaningful parliamentary-side counterpart found
  U. UNCOMPARABLE               — comparison impossible, with explicit cause

Design notes (deterministic, no LLM, no network, no corpus writes):

* Stage A (SHA-256) preserves the pre-existing exact-match validation and its
  headline accounting (per MoES document; SHA-uncomparable docs counted).
* Text comes from existing sources first: parliamentary side reuses the INLINE
  ``question_text``/``answer_text`` already stored in ``qa.jsonl`` by the frozen
  RS crawler (no RS PDFs are parsed). MoES side stores only ``text_chars`` in
  its manifest, so MoES PQ PDFs are text-extracted at analysis time via the
  shared lazy PyMuPDF wrapper — in-memory, read-only.
* Parliamentary inline text is English; MoES Hindi documents must not be scored
  against it. Classification therefore happens per PQ RECORD using the English
  document, and the record verdict is projected onto its Hindi sibling(s).
  Hindi-only records are UNCOMPARABLE (cause: no same-language reference).
* Normalization: NFC, casefold, page-number-only line removal, corpus-adaptive
  boilerplate-line removal (lines occurring in >= --boilerplate-fraction of the
  extracted PQ texts), punctuation collapsing, whitespace folding.
* Similarity: word 5-gram shingle sets; primary score is DIRECTIONAL
  containment |MoES ∩ RS| / |MoES| (a PIB "PARLIAMENT QUESTION" release is
  expected to be largely contained in the fuller RS question+answer record);
  Jaccard is reported alongside. Candidate records are prefiltered cheaply by
  title-token coverage (top-k), then scored exactly.
* Thresholds are CLI-configurable. The defaults (0.90 / 0.50) are PROVISIONAL
  placeholders: run with --calibrate on the real corpora to see the score
  distribution and set data-derived values before citing counts.

Usage:
    python -m src.scripts.check_moes_pq_overlap \
        [--moes-root P] [--parliamentary-root P] \
        [--near-identical-threshold 0.90] [--related-threshold 0.50] \
        [--top-k 8] [--boilerplate-fraction 0.5] [--shingle-size 5] \
        [--calibrate] [--output moes_pq_overlap_report.md]

Exit codes: 0 analysis completed · 2 usage / missing roots · 3 internal error.
Always ends stdout with the read-only DONE marker on success.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.scraping.manifest import load_manifest
from src.scraping.moes.normalize import PQ_TITLE_RE  # single source of PQ-title truth
from src.scraping.records import load_jsonl
from src.utils.app_paths import data_dir

MOES_CATEGORIES = ("reports", "press-release")
DONE_LINE = "DONE — read-only comparison; no crawler/corpus files modified."
VOLATILE_NOTE = "generated deterministically from corpus inputs; no timestamps"

# Trivial function words excluded from title prefilter tokens (documented, fixed).
STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
        "will", "with",
    ]
)

CLASS_EXACT = "EXACT_SHA"
CLASS_NEAR = "TEXTUALLY_NEAR_IDENTICAL"
CLASS_RELATED = "POTENTIALLY_CORRESPONDING"
CLASS_UNIQUE = "POTENTIALLY_UNIQUE"
CLASS_UNCOMPARABLE = "UNCOMPARABLE"
CLASS_ORDER = (CLASS_EXACT, CLASS_NEAR, CLASS_RELATED, CLASS_UNIQUE, CLASS_UNCOMPARABLE)


# --------------------------------------------------------------------------- data shapes


@dataclass
class MoesDoc:
    key: str
    record_id: str
    record_title: str
    post_date: str  # YYYY-MM-DD (from post_modified) or ""
    path: Path  # absolute file location inside the MoES staging root
    rel_path: str  # location as recorded in the manifest (display)
    sha256: str | None
    lang: str  # eng | hin | both
    doc_class: str
    text_chars: int | None
    duplicate_of: str | None


@dataclass
class MoesRecord:
    record_id: str
    title: str
    post_date: str
    docs: list[MoesDoc] = field(default_factory=list)


@dataclass
class ParlDoc:
    session: str
    key: str
    record_id: str
    rel_path: str
    sha256: str | None


@dataclass
class ParlRecord:
    qid: str
    session: str
    date: str
    question_text: str
    full_text: str  # question + answer, for scoring
    q_tokens: frozenset[str] = frozenset()
    shingles: frozenset[tuple[str, ...]] = frozenset()


@dataclass
class Verdict:
    doc: MoesDoc
    klass: str
    score: float | None  # containment vs best parliamentary record (0..1) or None
    jaccard: float | None
    candidate: str  # parliamentary record/doc reference or ""
    reason: str


# --------------------------------------------------------------------------- text utils

_PAGE_NUM_LINE = re.compile(r"^\s*(?:page\s*)?[-—–]?\s*\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.I)
_TOKEN_SPLIT = re.compile(r"[^0-9a-z\u0900-\u097f]+")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def raw_lines(text: str) -> list[str]:
    return [ln.strip() for ln in _nfc(text).splitlines() if ln.strip()]


def norm_line(line: str) -> str:
    """Normalization used for boilerplate detection (whitespace/case folding)."""
    return " ".join(line.casefold().split())


def extract_pdf_text(path: Path) -> tuple[str | None, str | None]:
    """(full text, None) or (None, cause). Lazy PyMuPDF via the shared wrapper."""
    try:
        from src.data import pdf_table_extract

        fitz = pdf_table_extract._import_fitz()  # noqa: SLF001 — shared lazy import
    except ImportError:
        return None, "pymupdf-unavailable"
    try:
        doc = fitz.open(stream=path.read_bytes(), filetype="pdf")
    except Exception:  # noqa: BLE001
        return None, "pdf-open-failed"
    try:
        parts = [page.get_text() for page in doc]
    except Exception:  # noqa: BLE001
        return None, "pdf-text-extract-failed"
    return "".join(parts), None


def compute_boilerplate(texts: list[str], fraction: float) -> frozenset[str]:
    """Normalized lines appearing in >= fraction of the texts (doc-frequency)."""
    if not texts:
        return frozenset()
    df: dict[str, int] = {}
    for text in texts:
        for ln in {norm_line(x) for x in raw_lines(text) if len(x) >= 20}:
            df[ln] = df.get(ln, 0) + 1
    cutoff = max(2, math.ceil(fraction * len(texts)))
    return frozenset(ln for ln, n in df.items() if n >= cutoff)


def normalize_text(text: str, boilerplate: Iterable[str] = frozenset()) -> str:
    """Deterministic noise reduction: NFC, drop page-number + boilerplate lines,
    casefold, collapse punctuation/runs of whitespace, keep word tokens only
    (Latin + Devanagari)."""
    bp = set(boilerplate)
    kept: list[str] = []
    for line in raw_lines(text):
        if _PAGE_NUM_LINE.match(line):
            continue
        if norm_line(line) in bp:
            continue
        kept.append(line)
    joined = _nfc(" ".join(kept)).casefold()
    tokens = [t for t in _TOKEN_SPLIT.split(joined) if t]
    return " ".join(tokens)


def word_tokens(normalized: str) -> list[str]:
    return normalized.split()


def shingles(tokens: list[str], n: int) -> frozenset[tuple[str, ...]]:
    """Word n-gram set; unigram fallback for very short texts (flagged upstream)."""
    if len(tokens) >= n:
        return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return frozenset((t,) for t in tokens)


def containment(a: frozenset, b: frozenset) -> float:
    """Directional: fraction of a covered by b."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- loaders


def resolve_moes_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser()
        if root.is_dir():
            return root.resolve()
        raise SystemExit(f"exit 2: --moes-root not a directory: {root}")
    base = data_dir()
    for name in (".moes-website", "moes-website"):  # dot-dir default, operator fallback
        cand = base / name
        if cand.is_dir():
            return cand.resolve()
    raise SystemExit(
        f"exit 2: no MoES staging root under {base} (tried .moes-website, moes-website)"
    )


def resolve_parliamentary_root(explicit: str | None) -> Path:
    root = Path(explicit).expanduser() if explicit else data_dir() / "parliamentary-qa"
    if (root / "rajya-sabha").is_dir():
        return (root / "rajya-sabha").resolve()
    if list(root.glob("session-*")):  # root may point directly at the house dir
        return root.resolve()
    raise SystemExit(f"exit 2: no Rajya Sabha sessions found under {root}")


def load_moes_pq(root: Path) -> dict[str, MoesRecord]:
    """PQ-titled records + their documents from every MoES category manifest."""
    records: dict[str, MoesRecord] = {}
    for category in MOES_CATEGORIES:
        manifest = load_manifest(root / category)
        if manifest is None:
            continue
        recs = {
            r["id"]: r
            for r in manifest.get("records") or []
            if isinstance(r, dict) and PQ_TITLE_RE.match(str(r.get("title") or ""))
        }
        for d in manifest.get("documents") or []:
            rid = d.get("record_id")
            if rid not in recs:
                continue
            src = recs[rid]
            rec = records.setdefault(
                rid,
                MoesRecord(
                    record_id=rid,
                    title=str(src.get("title") or ""),
                    post_date=str(src.get("post_modified") or "")[:10],
                ),
            )
            rel = str(d.get("path") or "")
            rec.docs.append(
                MoesDoc(
                    key=str(d.get("key") or ""),
                    record_id=rid,
                    record_title=rec.title,
                    post_date=rec.post_date,
                    path=(root / category / slug_dir(recs[rid]) / rel) if rel else root,
                    rel_path=f"{category}/{slug_dir(recs[rid])}/{rel}",
                    sha256=d.get("sha256") or None,
                    lang=str(d.get("lang") or ""),
                    doc_class=str(d.get("class") or ""),
                    text_chars=d.get("text_chars"),
                    duplicate_of=d.get("duplicate_of") or None,
                )
            )
    return records


def slug_dir(record: dict[str, Any]) -> str:
    return str(record.get("slug") or "")


def load_parliamentary(root: Path) -> tuple[dict[str, ParlRecord], dict[str, list[ParlDoc]]]:
    """RS records (id -> ParlRecord) and a sha256 -> docs index across sessions."""
    records: dict[str, ParlRecord] = {}
    sha_index: dict[str, list[ParlDoc]] = {}
    for session_dir in sorted(p for p in root.glob("session-*") if p.is_dir()):
        manifest = load_manifest(session_dir)
        if manifest:
            for d in manifest.get("documents") or []:
                sha = d.get("sha256")
                if not sha:
                    continue
                sha_index.setdefault(sha, []).append(
                    ParlDoc(
                        session=str(manifest.get("session") or session_dir.name),
                        key=str(d.get("key") or ""),
                        record_id=str(d.get("id") or ""),
                        rel_path=f"{session_dir.name}/{d.get('path')}",
                        sha256=sha,
                    )
                )
        for row in load_jsonl(session_dir / "qa.jsonl"):
            qid = str(row.get("question_id") or "")
            meta = row.get("metadata") or {}
            full = f"{row.get('question_text') or ''}\n{row.get('answer_text') or ''}"
            records[qid] = ParlRecord(
                qid=qid,
                session=str(meta.get("session") or ""),
                date=str(meta.get("date") or ""),
                question_text=str(row.get("question_text") or ""),
                full_text=full,
            )
    return records, sha_index


# --------------------------------------------------------------------------- matching


def title_tokens(title: str) -> list[str]:
    """MoES PQ title minus the 'PARLIAMENT QUESTION(S):' prefix, normalized."""
    stripped = PQ_TITLE_RE.sub("", _nfc(title))
    norm = normalize_text(stripped)
    return [t for t in word_tokens(norm) if t not in STOPWORDS and len(t) > 1]


def prepare_record_index(records: dict[str, ParlRecord], shingle_size: int) -> None:
    for rec in records.values():
        q_norm = normalize_text(rec.question_text)
        rec.q_tokens = frozenset(word_tokens(q_norm))
        full_norm = normalize_text(rec.full_text)
        rec.shingles = shingles(word_tokens(full_norm), shingle_size)


def candidate_records(
    tokens: list[str], records: dict[str, ParlRecord], top_k: int
) -> list[ParlRecord]:
    """Cheap prefilter: title-token coverage against question_text tokens."""
    tset = frozenset(tokens)
    if not tset:
        # Title carried no usable tokens -> honest FULL scan (no top-k cap;
        # capping here would silently hide the true match elsewhere).
        return sorted(records.values(), key=lambda r: r.qid)
    scored: list[tuple[float, str, ParlRecord]] = []
    for rec in records.values():
        if not rec.q_tokens:
            continue
        cov = len(tset & rec.q_tokens) / len(tset)
        if cov > 0:
            scored.append((cov, rec.qid, rec))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [rec for _, _, rec in scored[: max(1, top_k)]]


def best_match(
    moes_shingles: frozenset[tuple[str, ...]], candidates: list[ParlRecord]
) -> tuple[ParlRecord | None, float, float]:
    best: ParlRecord | None = None
    best_c = best_j = 0.0
    for rec in candidates:
        c = containment(moes_shingles, rec.shingles)
        if c > best_c or (c == best_c and best is not None and rec.qid < best.qid):
            j = jaccard(moes_shingles, rec.shingles)
            best, best_c, best_j = rec, c, j
    return best, best_c, best_j


def title_ratio(moes_title: str, rec: ParlRecord | None) -> float:
    if rec is None:
        return 0.0
    a = normalize_text(PQ_TITLE_RE.sub("", moes_title))
    b = normalize_text(rec.question_text)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------- analysis


def classify(
    moes_records: dict[str, MoesRecord],
    parl_records: dict[str, ParlRecord],
    sha_index: dict[str, list[ParlDoc]],
    *,
    near_threshold: float,
    related_threshold: float,
    top_k: int,
    boilerplate_fraction: float,
    shingle_size: int,
) -> tuple[list[Verdict], dict[str, Any]]:
    """Returns per-document verdicts + stats dict (kept for reporting/calibration)."""
    stats: dict[str, Any] = {
        "sha_missing_docs": 0,
        "record_scores": [],  # (record_id, containment) for calibration
        "boilerplate_lines": 0,
        "text_sources": {"eng": 0, "hin_only": 0, "none": 0},
    }

    # --- text extraction for unique payloads (memoized by sha256) ------------
    text_cache: dict[str, tuple[str | None, str | None]] = {}
    for rec in moes_records.values():
        for doc in rec.docs:
            sha = doc.sha256 or f"path:{doc.path}"
            if sha in text_cache:
                continue
            if doc.doc_class not in {"good", "partial"}:
                text_cache[sha] = (None, f"doc-class-{doc.doc_class}")
                continue
            text_cache[sha] = extract_pdf_text(doc.path)

    def doc_text(doc: MoesDoc) -> tuple[str | None, str | None]:
        return text_cache[doc.sha256 or f"path:{doc.path}"]

    # --- corpus-adaptive boilerplate from the successfully extracted texts ---
    extracted = [t for t, cause in text_cache.values() if t]
    boilerplate = compute_boilerplate(extracted, boilerplate_fraction)
    stats["boilerplate_lines"] = len(boilerplate)

    prepare_record_index(parl_records, shingle_size)

    # --- Stage A: exact SHA-256 (document level) -------------------------------
    exact: dict[str, list[ParlDoc]] = {}
    for rec in moes_records.values():
        for doc in rec.docs:
            if doc.sha256 is None:
                stats["sha_missing_docs"] += 1
                continue
            if doc.sha256 in sha_index:
                exact[doc.key] = sha_index[doc.sha256]

    # --- Stage B/C/D: per-record text classification (English speaks) ---------
    verdicts: list[Verdict] = []
    for rec in sorted(moes_records.values(), key=lambda r: r.record_id):
        text_doc: MoesDoc | None = None
        text: str | None = None
        cause: str | None = "no-english-document"
        for want in ("eng", "both"):
            for doc in rec.docs:
                if doc.lang != want:
                    continue
                t, cause = doc_text(doc)
                if t and normalize_text(t, boilerplate):
                    text_doc, text, cause = doc, t, None
                    break
            if text:
                break
        if text_doc is not None:
            stats["text_sources"]["eng"] += 1

        record_verdict: tuple[str, float | None, float | None, ParlRecord | None, str]
        if text is None or text_doc is None:
            hin_docs = [d for d in rec.docs if d.lang in {"hin", "both"}]
            if hin_docs and cause == "no-english-document":
                stats["text_sources"]["hin_only"] += 1
                cause = "no-english-reference (parliamentary inline text is English)"
            else:
                stats["text_sources"]["none"] += 1
            record_verdict = (CLASS_UNCOMPARABLE, None, None, None, cause or "no-text")
        else:
            tokens = word_tokens(normalize_text(text, boilerplate))
            mode_flag = "unigram-fallback" if len(tokens) < shingle_size else "5-gram"
            cands = candidate_records(title_tokens(rec.title), parl_records, top_k)
            best, c, j = best_match(shingles(tokens, shingle_size), cands)
            stats["record_scores"].append((rec.record_id, round(c, 6)))
            ratio = title_ratio(rec.title, best)
            ref = ""
            if best is not None:
                delta = _date_delta(rec.post_date, best.date)
                ref = f"{best.qid} (session {best.session}, date {best.date or '?'}{delta})"
            if c >= near_threshold:
                record_verdict = (
                    CLASS_NEAR,
                    c,
                    j,
                    best,
                    f"containment {c:.3f} >= {near_threshold} vs {ref}; "
                    f"title ratio {ratio:.2f}; shingles {mode_flag}",
                )
            elif c >= related_threshold:
                record_verdict = (
                    CLASS_RELATED,
                    c,
                    j,
                    best,
                    f"containment {c:.3f} in [{related_threshold}, {near_threshold}) vs {ref}; "
                    f"title ratio {ratio:.2f}",
                )
            else:
                record_verdict = (
                    CLASS_UNIQUE,
                    c,
                    j,
                    best,
                    f"best containment {c:.3f} < {related_threshold} "
                    f"(best candidate {ref or 'none'})",
                )

        klass, c, j, best, reason = record_verdict
        cand_ref = (
            f"{best.qid} / {best.session}" if best is not None else ""
        )
        selected_key = text_doc.key if text_doc is not None else None
        for doc in sorted(rec.docs, key=lambda d: d.key):
            if doc.key in exact:
                parl = exact[doc.key][0]
                verdicts.append(
                    Verdict(
                        doc=doc,
                        klass=CLASS_EXACT,
                        score=1.0,
                        jaccard=1.0,
                        candidate=f"{parl.rel_path} (record {parl.record_id})",
                        reason=f"sha256 {doc.sha256[:16]}… == parliamentary '{parl.key}'",
                    )
                )
                continue
            note = ""
            if klass in {CLASS_NEAR, CLASS_RELATED, CLASS_UNIQUE} and doc.key != selected_key:
                note = " (record verdict via sibling document)"
            verdicts.append(
                Verdict(
                    doc=doc,
                    klass=klass,
                    score=c,
                    jaccard=j,
                    candidate=cand_ref,
                    reason=reason + note,
                )
            )
    return verdicts, stats


def _date_delta(moes_date: str, parl_date: str) -> str:
    try:
        from datetime import date

        a = date.fromisoformat(moes_date)
        b = date.fromisoformat(parl_date)
        return f", Δ{(a - b).days}d"
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- reporting


def render_report(
    verdicts: list[Verdict],
    moes_records: dict[str, MoesRecord],
    stats: dict[str, Any],
    args: argparse.Namespace,
    moes_root: Path,
    parl_root: Path,
) -> str:
    counts = dict.fromkeys(CLASS_ORDER, 0)
    for v in verdicts:
        counts[v.klass] += 1
    record_counts: dict[str, str] = {}
    for v in verdicts:
        record_counts.setdefault(v.doc.record_id, v.klass)
    lines: list[str] = []
    add = lines.append
    add("# MoES PQ ↔ Parliamentary overlap validation")
    add("")
    add("SUMMARY")
    add("")
    add(f"MoES corpus root            : {moes_root}")
    add(f"Parliamentary corpus root   : {parl_root}")
    add(f"PQ-titled MoES records      : {len(moes_records)}")
    add(f"MoES PQ documents compared  : {len(verdicts)}")
    add(f"Exact SHA-256 matches       : {counts[CLASS_EXACT]}")
    add(f"Textually near-identical    : {counts[CLASS_NEAR]}")
    add(f"Potentially corresponding   : {counts[CLASS_RELATED]}")
    add(f"Potentially unique          : {counts[CLASS_UNIQUE]}")
    add(f"Uncomparable                : {counts[CLASS_UNCOMPARABLE]}")
    add(f"Total accounted             : {sum(counts.values())}")
    add("")
    add("Method (deterministic, read-only, no LLM, no network):")
    add(f"- Stage A exact SHA-256: {counts[CLASS_EXACT]} matches / {len(verdicts)} documents "
        f"(sha-missing on MoES side: {stats['sha_missing_docs']}).")
    add(f"- Text: parliamentary side reuses inline qa.jsonl question+answer text (English); "
        f"MoES PQ PDFs text-extracted in-memory (PyMuPDF). Records classified via their "
        f"English document: {stats['text_sources']['eng']} eng-text records, "
        f"{stats['text_sources']['hin_only']} Hindi-only (uncomparable), "
        f"{stats['text_sources']['none']} without extractable text.")
    add(f"- Normalization: NFC, casefold, page-number lines and {stats['boilerplate_lines']} "
        f"corpus-adaptive boilerplate lines removed (>= {args.boilerplate_fraction} doc-fraction), "
        f"punctuation/whitespace collapsed.")
    add(f"- Similarity: word {args.shingle_size}-gram shingle directional containment "
        f"|MoES∩RS|/|MoES| (Jaccard reported); candidates prefiltered by title-token "
        f"coverage (top {args.top_k}).")
    add(f"- Thresholds: near-identical >= {args.near_identical_threshold}, "
        f"corresponding >= {args.related_threshold} (PROVISIONAL defaults unless tuned via "
        f"--calibrate on this corpus).")
    add(f"- Report identity: {VOLATILE_NOTE}.")

    def section(title: str, klass: str) -> None:
        add("")
        add(f"=== {title} ===")
        add("")
        rows = [v for v in verdicts if v.klass == klass]
        if not rows:
            add("(none)")
        for v in rows:
            score = f"{v.score:.3f}" if v.score is not None else "-"
            jac = f"{v.jaccard:.3f}" if v.jaccard is not None else "-"
            cand = v.candidate or "-"
            add(f"- {v.doc.key} [{v.doc.lang}] «{v.doc.record_title[:80]}»")
            add(f"    parliamentary: {cand}")
            add(f"    containment {score} · jaccard {jac} · {v.reason}")

    section("EXACT SHA-256 MATCHES", CLASS_EXACT)
    section("TEXTUALLY NEAR-IDENTICAL", CLASS_NEAR)
    section("POTENTIALLY CORRESPONDING", CLASS_RELATED)
    section("POTENTIALLY UNIQUE MoES PQ DOCUMENTS", CLASS_UNIQUE)
    section("UNCOMPARABLE", CLASS_UNCOMPARABLE)
    add("")
    add(DONE_LINE)
    return "\n".join(lines) + "\n"


def recommended_thresholds(scores: list[float]) -> dict[str, float]:
    """Data-derived near/related thresholds from a descending best-containment
    distribution. Deterministic heuristic:

    * near_identical_threshold — midpoint of the LARGEST gap whose upper score
      is in the "high" region (>= 0.60). This separates a tight high cluster of
      true duplicates (near-identical) from the next cluster down. Falls back
      to 0.90 when no high-region gap exists.
    * related_threshold — midpoint of the LARGEST gap below the near cut
      (separates potentially-corresponding from unique); falls back to 0.50.

    Falls back to the provisional defaults when there are too few scores to
    draw a gap (n < 3). Always returns near >= related."""
    sorted_scores = sorted((s for s in scores if s is not None), reverse=True)
    if len(sorted_scores) < 3:
        return {"near_identical_threshold": 0.90, "related_threshold": 0.50}
    def _clamp(v: float) -> float:
        return max(0.01, min(1.0, round(v, 3)))

    gaps = [
        (sorted_scores[i] - sorted_scores[i + 1], sorted_scores[i], sorted_scores[i + 1])
        for i in range(len(sorted_scores) - 1)
    ]
    if not gaps:
        return {"near_identical_threshold": 0.90, "related_threshold": 0.50}

    # near separates the tight TOP cluster (true duplicates) from the next
    # cluster down: use the gap whose UPPER score is the highest.
    top_gap = max(gaps, key=lambda g: g[1])
    _, near_above, near_below = top_gap
    near = _clamp((near_above + near_below) / 2)

    # related separates the mid "corresponding" cluster from the unique tail:
    # largest remaining gap below the near cut.
    lower = [g for g in gaps if g != top_gap and g[2] <= near_below]
    if lower:
        _, rel_above, rel_below = max(lower, key=lambda g: g[0])
        related = _clamp((rel_above + rel_below) / 2)
    else:
        related = 0.50
    related = min(related, near)
    return {"near_identical_threshold": near, "related_threshold": related}


def write_calibrated_thresholds(path: Path, scores: list[float]) -> str:
    """Write a YAML thresholds file (config/moes_pq_dedup.yaml shape) derived
    from a calibration score distribution. Returns the YAML text written."""
    import yaml

    th = recommended_thresholds(scores)
    data = {
        "near_identical_threshold": th["near_identical_threshold"],
        "related_threshold": th["related_threshold"],
    }
    text = (
        "# MoES \u2194 Parliamentary Q&A cross-source dedup thresholds "
        "(calibrated via\n"
        "# check_moes_pq_overlap --calibrate --write-calibrated-thresholds).\n"
        "# near_identical: EXACT_SHA + TEXTUALLY_NEAR_IDENTICAL are excluded at\n"
        "# ingestion; related/unique/uncertain are preserved.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )
    path.write_text(text, encoding="utf-8")
    return text


def render_calibration(stats: dict[str, Any]) -> str:
    scores = sorted((s for _, s in stats["record_scores"]), reverse=True)
    lines = ["CALIBRATION — per-record best containment scores (descending)", ""]
    lines.append(f"records scored: {len(scores)}")
    if scores:
        dec = [
            f"p{q}: {scores[min(len(scores) - 1, int(len(scores) * q / 10))]:.3f}"
            for q in range(0, 11, 1)
        ]
        lines.append("deciles: " + "  ".join(dec))
        gaps = sorted(
            ((scores[i] - scores[i + 1], scores[i], scores[i + 1]) for i in range(len(scores) - 1)),
            reverse=True,
        )[:5]
        lines.append("largest consecutive gaps (gap, above, below) — candidate threshold cuts:")
        for g, a, b in gaps:
            lines.append(f"  {g:.3f}  between {a:.3f} and {b:.3f}")
    lines.append("")
    lines.append("all scores:")
    lines.append("  " + " ".join(f"{s:.3f}" for s in scores))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--moes-root", default=None,
        help="MoES staging root (default: $APP_DATA_DIR/.moes-website then moes-website)",
    )
    p.add_argument(
        "--parliamentary-root", default=None,
        help="parliamentary-qa root or its rajya-sabha dir "
             "(default: $APP_DATA_DIR/parliamentary-qa)",
    )
    p.add_argument(
        "--near-identical-threshold", type=float, default=0.90,
        help="containment at/above => textually near-identical "
             "(default 0.90; PROVISIONAL — tune via --calibrate)",
    )
    p.add_argument(
        "--related-threshold", type=float, default=0.50,
        help="containment at/above => potentially corresponding "
             "(default 0.50; PROVISIONAL — tune via --calibrate)",
    )
    p.add_argument(
        "--top-k", type=int, default=8,
        help="candidate parliamentary records scored per MoES record (default 8)",
    )
    p.add_argument(
        "--boilerplate-fraction", type=float, default=0.5,
        help="doc-frequency at/above which a line is boilerplate (default 0.5)",
    )
    p.add_argument(
        "--shingle-size", type=int, default=5, help="word shingle size (default 5)",
    )
    p.add_argument(
        "--calibrate", action="store_true",
        help="print score distribution + gap suggestions; no classification/report",
    )
    p.add_argument(
        "--write-calibrated-thresholds", default=None, metavar="PATH",
        help="with --calibrate: write recommended near/related thresholds as YAML "
             "(e.g. config/moes_pq_dedup.yaml) instead of/alongside the text",
    )
    p.add_argument(
        "--output", default=None,
        help="markdown report path (default: moes_pq_overlap_report.md in CWD; "
             "never inside a corpus root)",
    )
    args = p.parse_args(argv)
    if not (0.0 < args.related_threshold <= args.near_identical_threshold <= 1.0):
        p.error("require 0 < related-threshold <= near-identical-threshold <= 1")
    # Deliberately fail loudly on accidental hash-bending inputs (defense in depth).
    bad = [x for x in (argv or sys.argv[1:]) if x.startswith("--thresholds-file")]
    if bad:
        p.error("thresholds must be passed as explicit CLI values, not files")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    moes_root = resolve_moes_root(args.moes_root)
    parl_root = resolve_parliamentary_root(args.parliamentary_root)
    output = Path(args.output or "moes_pq_overlap_report.md").expanduser().resolve()

    for root in (moes_root, parl_root):
        try:
            output.relative_to(root)
        except ValueError:
            continue
        print(f"exit 2: --output must not be inside a corpus root: {output}", file=sys.stderr)
        return 2

    moes_records = load_moes_pq(moes_root)
    parl_records, sha_index = load_parliamentary(parl_root)
    if not moes_records:
        print("exit 2: no PQ-titled MoES records found (check --moes-root)", file=sys.stderr)
        return 2
    if not parl_records:
        print(
            "exit 2: no parliamentary records found (check --parliamentary-root)",
            file=sys.stderr,
        )
        return 2

    verdicts, stats = classify(
        moes_records,
        parl_records,
        sha_index,
        near_threshold=args.near_identical_threshold,
        related_threshold=args.related_threshold,
        top_k=args.top_k,
        boilerplate_fraction=args.boilerplate_fraction,
        shingle_size=args.shingle_size,
    )

    if args.calibrate:
        scores = [s for _, s in stats["record_scores"]]
        if args.write_calibrated_thresholds:
            dest = Path(args.write_calibrated_thresholds).expanduser().resolve()
            for root in (moes_root, parl_root):
                try:
                    dest.relative_to(root)
                except ValueError:
                    continue
                print(
                    f"exit 2: --write-calibrated-thresholds must not be inside a corpus "
                    f"root: {dest}",
                    file=sys.stderr,
                )
                return 2
            write_calibrated_thresholds(dest, scores)
            print(f"(thresholds written: {dest})")
        print(render_calibration(stats), end="")
        print(DONE_LINE)
        return 0

    report = render_report(verdicts, moes_records, stats, args, moes_root, parl_root)
    print(report, end="")
    output.write_text(report, encoding="utf-8")
    print(f"(report written: {output})")
    print(DONE_LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
