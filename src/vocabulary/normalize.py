"""Deterministic domain normalization (Phase 1 / WS1).

Every normalizer:
  * is a PURE function of (raw value, Vocabulary) — no I/O, no LLM, no
    randomness; the same input always gives the same output;
  * PRESERVES the raw value on the result (provenance — the canonical form
    is an annotation, never a replacement of the corpus);
  * is IDEMPOTENT (normalizing an already-normalized value is a no-op);
  * NEVER invents: values with no corpus-justified identity come back as
    ``status="unknown"`` with ``canonical=None`` (two different unknown
    values are never merged).

Status values:
  "canonical"  a controlled identity was found (or derived by a documented
               rule — e.g. member case-fold/honorific-strip)
  "deferred"   a controlled mapping exists in principle but is NOT applied
               today (subject — see normalize_subject)
  "unknown"    no controlled identity; raw preserved, canonical is None
  "empty"      raw was None/blank

The corpus row schema is untouched: normalizers consume metadata fields and
return annotations. Nothing here feeds retrieval or generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.vocabulary.vocab import Vocabulary

__all__ = [
    "Normalization",
    "MemberToken",
    "MemberNormalization",
    "normalize_ministry",
    "normalize_org",
    "normalize_source",
    "normalize_house",
    "normalize_house_from_doc_id",
    "normalize_document_type",
    "normalize_question_type",
    "normalize_member",
    "normalize_date",
    "normalize_subject",
    "ls_term_from_doc_id",
    "expected_sansad_code",
    "concept_mentions",
]


@dataclass(frozen=True)
class Normalization:
    field: str
    raw: Optional[str]
    canonical: Optional[str]
    status: str          # canonical | deferred | unknown | empty
    rule: str            # which rule produced the result (auditability)

    def to_dict(self) -> dict:
        return {
            "field": self.field, "raw": self.raw, "canonical": self.canonical,
            "status": self.status, "rule": self.rule,
        }


def _empty(field: str) -> Normalization:
    return Normalization(field=field, raw=None, canonical=None,
                         status="empty", rule="absent-value")


def _match(field: str, raw: Optional[str], entry, rule: str) -> Normalization:
    if raw is None or not str(raw).strip():
        return _empty(field)
    raw_s = str(raw)
    if entry is None:
        return Normalization(field=field, raw=raw_s, canonical=None,
                             status="unknown", rule="no-controlled-identity")
    return Normalization(field=field, raw=raw_s, canonical=entry.canonical,
                         status="canonical", rule=rule)


# ─────────────────────────────────────────────────────────────────────────────
# Simple controlled vocabularies
# ─────────────────────────────────────────────────────────────────────────────

def normalize_ministry(voc: Vocabulary, raw: Optional[str]) -> Normalization:
    return _match("ministry", raw, voc.match_ministry(raw), "ministry-alias-table")


def normalize_org(voc: Vocabulary, raw: Optional[str]) -> Normalization:
    return _match("org", raw, voc.match_org(raw), "org-alias-table")


def normalize_source(voc: Vocabulary, raw: Optional[str]) -> Normalization:
    return _match("source", raw, voc.match_source(raw), "source-alias-table")


def normalize_document_type(voc: Vocabulary, raw: Optional[str]) -> Normalization:
    return _match("document_type", raw, voc.match_document_type(raw),
                  "document-type-alias-table")


def normalize_question_type(voc: Vocabulary, raw: Optional[str]) -> Normalization:
    return _match("question_type", raw, voc.match_question_type(raw),
                  "question-type-alias-table")


def normalize_house(voc: Vocabulary, raw: Optional[str]) -> Normalization:
    return _match("house", raw, voc.match_house(raw), "house-alias-table")


def normalize_house_from_doc_id(voc: Vocabulary, doc_id: str) -> Normalization:
    """House identity from the doc-id prefix (ls-/rs-) — the corpus shows
    100% agreement between the prefix and metadata.house (0 mismatches)."""
    raw = doc_id if doc_id else None
    entry = voc.house_by_id_prefix(doc_id or "")
    return _match("house", raw, entry, "doc-id-prefix")


# ─────────────────────────────────────────────────────────────────────────────
# Members — the only person rule: case-fold + honorific strip.
# No similarity merging, ever.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemberToken:
    raw: str                      # the token exactly as stored
    canonical_key: str            # casefolded, honorific-stripped, ws-collapsed
    honorific: Optional[str]      # leading honorific stripped (or None)


@dataclass(frozen=True)
class MemberNormalization:
    raw: Optional[str]
    status: str                   # canonical | empty | unknown
    tokens: tuple[MemberToken, ...]
    rule: str

    @property
    def canonical_keys(self) -> tuple[str, ...]:
        return tuple(t.canonical_key for t in self.tokens)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw, "status": self.status, "rule": self.rule,
            "tokens": [
                {"raw": t.raw, "canonical_key": t.canonical_key,
                 "honorific": t.honorific}
                for t in self.tokens
            ],
        }


def _strip_honorific(voc: Vocabulary, text: str) -> tuple[str, Optional[str]]:
    # Case-insensitive: the corpus carries both "Shri X" and "SHRI X" (and
    # "DR."/"Dr."). Longest honorific first so "Shrimati" wins over "Shri".
    for h in sorted(voc.member_honorifics, key=len, reverse=True):
        pat = re.escape(h) + r"[\s.]+(?=[A-Z\"'])"
        m = re.match(pat, text, re.IGNORECASE)
        if m:
            return text[m.end():].strip(), h
    return text, None


def normalize_member(voc: Vocabulary, raw: Optional[str]) -> MemberNormalization:
    """Split clubbed ``;`` lists; per token: trim, collapse whitespace, strip
    ONE leading honorific (strong rule — the only member-merge the corpus
    justifies), casefold the key. Raw token always preserved."""
    if raw is None or not str(raw).strip():
        return MemberNormalization(raw=None, status="empty", tokens=(),
                                   rule="absent-value")
    raw_s = str(raw)
    tokens: list[MemberToken] = []
    for part in raw_s.split(";"):
        part = " ".join(part.split())
        if not part:
            continue
        stripped, honor = _strip_honorific(voc, part)
        key = " ".join(stripped.casefold().split())
        if not key:
            continue
        tokens.append(MemberToken(raw=part, canonical_key=key, honorific=honor))
    if not tokens:
        return MemberNormalization(raw=raw_s, status="unknown", tokens=(),
                                   rule="no-parseable-name")
    return MemberNormalization(raw=raw_s, status="canonical",
                               tokens=tuple(tokens),
                               rule="split-semicolon+casefold+honorific-strip")


# ─────────────────────────────────────────────────────────────────────────────
# Dates — canonical ISO form preserving precision (day vs year)
# ─────────────────────────────────────────────────────────────────────────────

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_YEAR_RE = re.compile(r"^(\d{4})$")


@dataclass(frozen=True)
class DateNormalization(Normalization):
    precision: str = ""           # "day" | "year" | ""
    iso: Optional[str] = None     # full ISO date (day precision only)

    def to_dict(self) -> dict:  # type: ignore[override]
        d = super().to_dict()
        d["precision"] = self.precision
        d["iso"] = self.iso
        return d


def normalize_date(voc: Vocabulary, raw: Optional[str]) -> DateNormalization:
    """Corpus shapes: YYYY-MM-DD (2,575 rows), YYYY (70), null (3). Anything
    else is unknown — no guessing. A calendar-invalid date (e.g. 2024-02-30)
    is unknown, not silently corrected."""
    if raw is None or not str(raw).strip():
        return DateNormalization(field="date", raw=None, canonical=None,
                                 status="empty", rule="absent-value")
    s = str(raw).strip()
    m = _ISO_RE.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = date(y, mo, d)
        except ValueError:
            return DateNormalization(field="date", raw=s, canonical=None,
                                     status="unknown",
                                     rule="calendar-invalid-date",
                                     precision="", iso=None)
        return DateNormalization(field="date", raw=s, canonical=s,
                                 status="canonical", rule="iso-date-validated",
                                 precision="day", iso=dt.isoformat())
    m = _YEAR_RE.match(s)
    if m:
        return DateNormalization(field="date", raw=s, canonical=s,
                                 status="canonical", rule="year-only",
                                 precision="year", iso=None)
    return DateNormalization(field="date", raw=s, canonical=None,
                             status="unknown", rule="no-controlled-date-shape",
                             precision="", iso=None)


# ─────────────────────────────────────────────────────────────────────────────
# Subject — EXPLICITLY deferred (no controlled mapping today)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_subject(voc: Vocabulary, raw: Optional[str]) -> Normalization:
    """Subject is free text (1,438 distinct values; 100% null for RS rows;
    incdoc rows use the document title). NO arbitrary free-text Subject
    entities may be created — the value is preserved raw and the controlled
    mapping is explicitly deferred (a future phase may map high-confidence
    subjects, with a separate review)."""
    if raw is None or not str(raw).strip():
        return _empty("subject")
    return Normalization(field="subject", raw=str(raw), canonical=None,
                         status="deferred", rule="deferred-free-text")


# ─────────────────────────────────────────────────────────────────────────────
# Session/term identities
# ─────────────────────────────────────────────────────────────────────────────

def ls_term_from_doc_id(doc_id: str) -> Optional[int]:
    """`ls-<lok>-<session>-<qno>` → the Lok Sabha term number (identity;
    controlled against voc.ls_terms by the caller)."""
    m = re.match(r"^ls-(\d+)-", str(doc_id or ""))
    return int(m.group(1)) if m else None


def expected_sansad_code(
    voc: Vocabulary, house: Optional[str], ministry: Optional[str]
) -> Optional[int]:
    """The sansad elibrary ministry_code the corpus shows for a
    (house, ministry) pair — 100% coverage on the 2,345 parliamentary rows.
    Returns None when the corpus has no such pair (e.g. LS ocean-development)
    or for non-parliamentary rows — never invents a code."""
    m_entry = voc.match_ministry(ministry)
    if m_entry is None:
        return None
    codes = m_entry.extra.get("sansad_codes") or {}
    return codes.get(house) if house else None


# ─────────────────────────────────────────────────────────────────────────────
# Concept mentions (text mining — counts, NOT entity creation)
# ─────────────────────────────────────────────────────────────────────────────

def _build_surface_index(voc: Vocabulary) -> list[tuple[re.Pattern, str]]:
    """Longest surface first, so longer phrases win span claims over their
    abbreviations (e.g. 'India Meteorological Department' before 'IMD')."""
    pairs: list[tuple[str, str]] = []
    for c in voc.concepts:
        for s in c.get("surfaces") or []:
            pairs.append((str(s), str(c["canonical"])))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return [
        (re.compile(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])",
                    re.IGNORECASE), canon)
        for s, canon in pairs
    ]


def concept_mentions(voc: Vocabulary, text: str) -> dict[str, int]:
    """Count controlled-concept mentions in free text. Each character span
    belongs to the LONGEST surface that matches it (no double counting).
    Pure text mining for the quality report — creates nothing."""
    if not text:
        return {}
    low = text.lower()
    claimed: list[bool] = [False] * len(low)
    counts: dict[str, int] = {}
    for pat, canon in _build_surface_index(voc):
        for m in pat.finditer(low):
            a, b = m.span()
            if any(claimed[a:b]):
                continue  # a longer surface already owns this span
            claimed[a:b] = [True] * (b - a)
            counts[canon] = counts.get(canon, 0) + 1
    return counts
