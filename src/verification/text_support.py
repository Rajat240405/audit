"""Deterministic claim-support core (Phase 3 / WS3).

The authoritative TEXT-level verification engine, shared by every retrieval
mode. This is the server's existing grounding logic, MOVED here unchanged so
that one implementation serves Hybrid RAG, GraphRAG, the /api/verify engine
and the verification tests (spec: one verification contract — no per-mode
verifier duplication).

Strict by design: a claim is supported only when a normalized surface form
(optionally alias-expanded, with number+word swap handling) appears in a
cited source. Weak/partial overlap is NEVER sufficient (no fuzzy matching,
no embedding similarity).
"""
from __future__ import annotations

import json
import re

from src.utils.app_paths import project_root

__all__ = [
    "ALIAS_GROUPS",
    "ACRONYM_STOPWORDS",
    "extract_claims",
    "normalize",
    "singularize",
    "claim_candidates",
    "claim_supported",
    "grounding_report",
    "apply_citation_filter",
    "remove_rejected_sentences",
]

# ─────────────────────────────────────────────────────────────────────────────
# Claim extraction
# ─────────────────────────────────────────────────────────────────────────────

ACRONYM_STOPWORDS = {
    "THE", "AND", "FOR", "NOT", "ARE", "WAS", "WERE", "BUT", "HAS", "HAVE",
    "HAD", "ITS", "YOU", "OUR", "OUT", "OFF", "CAN", "MAY", "THIS", "THAT",
    "WITH", "FROM", "INTO", "WHEN", "WHAT", "WHY", "HOW", "THAN", "THEN",
    "INDIA", "GOVERNMENT", "MINISTRY", "ANSWER", "QUESTION", "STATE",
}

_FIGURE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:%|mm|cm|km|m\b|MW|GW|KW|sq\.?\s?km|crore|lakh|"
    r"million|billion|hrs?|hours?|years?|deg(?:ree)?s?|₹|rs\.?)\b",
    re.IGNORECASE,
)
# "48 Doppler Weather Radars", "32 Water Quality Buoys", "675 AWS" — number +
# a capitalized noun phrase (up to 4 words). Catches list-number swaps that a
# bare figure+unit regex misses ("32 Water Quality Buoys" vs the source's
# "2 Water Quality Buoys").
_NUM_WORD_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,3}\b"
)
_QUOTE_RE = re.compile(r'"([^"\\]{6,80})"')
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,8}\b")
_ACRONYM_PLURAL_RE = re.compile(r"\b[A-Z]{2,7}[a-z]{1,2}\b")
_NAMED_ABBR_RE = re.compile(r"\b[A-Z][A-Za-z&.\- ]{2,60}\s*\([A-Z]{2,10}\)")


def extract_claims(answer: str, max_claims: int = 12) -> list[str]:
    """Extract a bounded set of checkable claims from the answer."""
    claims: list[str] = []
    for m in _FIGURE_RE.finditer(answer):
        claims.append(m.group(0).strip())
    for m in _NUM_WORD_RE.finditer(answer):
        claims.append(m.group(0).strip())
    for m in _QUOTE_RE.finditer(answer):
        claims.append(m.group(1).strip())
    for m in _NAMED_ABBR_RE.finditer(answer):
        claims.append(m.group(0).strip())
    for m in _ACRONYM_RE.finditer(answer):
        tok = m.group(0)
        if tok in ACRONYM_STOPWORDS or len(tok) < 3:
            continue
        claims.append(tok)
    for m in _ACRONYM_PLURAL_RE.finditer(answer):
        tok = m.group(0)
        if tok in ACRONYM_STOPWORDS or len(tok) < 3:
            continue
        claims.append(tok)
    # de-dup, keep order, cap
    seen: set[str] = set()
    out: list[str] = []
    for c in claims:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_claims:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Normalized + alias-aware support matching
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_ALIASES = {
    "rs": "rupee", "inr": "rupee", "₹": "rupee",
    "&": "and", "ltd": "limited", "dept": "department",
    "govt": "government", "yr": "year", "hrs": "hours", "hr": "hour",
}

# Equivalent surface forms of the same entity — SINGLE SOURCE of truth:
# frontend/src/utils/grounding_aliases.json. Editing the JSON is the ONLY way
# to change aliases; this list is loaded from it so backend and frontend can
# never drift again (was: two hardcoded copies, frontend missing 7 terms).
ALIAS_GROUPS: list[list[str]] = json.loads(
    (project_root() / "frontend" / "src" / "utils"
     / "grounding_aliases.json").read_text(encoding="utf-8")
)["groups"]


def normalize(text: str) -> str:
    """Lowercase, unify currency/abbrev tokens, strip punctuation, collapse."""
    t = text.lower()
    # remove digit-grouping commas first: "2,000" -> "2000" so it matches "2000"
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    # token aliases with word boundaries, space-padded so they never glue to
    # neighbours ("₹2,000" -> "rupee 2000", "rs." -> "rupee")
    for k, v in _TOKEN_ALIASES.items():
        t = re.sub(rf"\b{re.escape(k)}\b", f" {v} ", t)
    # replace anything non-alphanumeric with a space
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def singularize(norm: str) -> str:
    """Strip a trailing plural 's' if it leaves a meaningful token."""
    if len(norm) > 4 and norm.endswith("s") and not norm.endswith("ss"):
        return norm[:-1]
    return norm


def claim_candidates(claim: str) -> list[str]:
    """All normalized surface forms that represent the same concept as the
    claim — the claim itself, its singular form, and every alias-group member
    if the claim names a known entity. Number+entity claims ("48 DWRs",
    "675 AWS") additionally expand the entity across its alias group and are
    matched with the number BEFORE or AFTER the entity (tables often list
    "AWS 675"), so a swapped figure ("32 Water Quality Buoys" vs the source's
    "2 Water Quality Buoys") still fails every candidate.
    """
    c = normalize(claim)
    cands = [c, singularize(c)]
    m = re.match(r"^(\d+)\s+(.+)$", c)
    if m:
        num, phrase = m.group(1), m.group(2)
        phrases = {phrase, singularize(phrase)}
        for group in ALIAS_GROUPS:
            if any(mem in phrase or phrase in mem for mem in group if len(mem) > 2):
                phrases.update(group)
        for p in phrases:
            if p:
                cands.append(f"{num} {p}")
                cands.append(f"{p} {num}")
    else:
        for group in ALIAS_GROUPS:
            if c in group or any(mm in c for mm in group if len(mm) > 4):
                cands.extend(group)
    # de-dup
    seen: set[str] = set()
    out: list[str] = []
    for x in cands:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def claim_supported(claim: str, src_normalized: str) -> bool:
    """True if any surface form of the claim appears in the normalized source."""
    for cand in claim_candidates(claim):
        if cand and cand in src_normalized:
            return True
    return False


def grounding_report(answer: str, sources: list[dict]) -> list[dict]:
    """Verify each extracted claim against the sources using normalized +
    alias-aware matching. Returns [{text, found, source}]."""
    if not sources:
        return []
    claims = extract_claims(answer)
    src_texts = [
        {
            "doc_id": s["doc_id"],
            "text": normalize(f"{s.get('question','')} {s.get('answer','')}"),
        }
        for s in sources
    ]
    report: list[dict] = []
    for c in claims:
        found = False
        src = None
        for st in src_texts:
            if claim_supported(c, st["text"]):
                found = True
                src = st["doc_id"]
                break
        report.append({"text": c, "found": found, "source": src})
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Citation-aware filtering (non-destructive)
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[\s*[Ss]ource\s*(\d+)\s*\]")


def apply_citation_filter(
    answer: str,
    sources: list[dict],
    max_drop: int = 3,
) -> tuple[str, list[str]]:
    """Drop only sentences carrying UNVERIFIED claims. Returns
    (filtered_answer, dropped_sentences). Never empties the answer."""
    if not answer.strip() or not sources:
        return answer, []

    # Build a source haystack for per-sentence grounding.
    haystack = " ".join(
        f"{s.get('question','')} {s.get('answer','')}" for s in sources
    ).lower()

    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    kept: list[str] = []
    dropped: list[str] = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # A citation token always lets a sentence through.
        if _CITATION_RE.search(s):
            kept.append(s)
            continue
        # No claims to verify -> plain prose/connective -> keep.
        claims = extract_claims(s)
        if not claims:
            kept.append(s)
            continue
        # Has claims but none are grounded -> hallucination risk -> drop.
        ungrounded = [c for c in claims if c.lower() not in haystack]
        if ungrounded and len(dropped) < max_drop:
            dropped.append(s)
            continue
        kept.append(s)

    if not kept:
        # never return an empty answer
        return answer, dropped
    return "\n\n".join(kept), dropped


def remove_rejected_sentences(
    answer: str, rejected_claims: list[str]
) -> tuple[str, list[str]]:
    """Remove sentences containing judge-rejected claims from the answer.

    Returns (cleaned_answer, removed_sentences). Only sentences that carry a
    claim the LLM judge explicitly rejected are removed — never plain prose,
    never grounded claims. This is the targeted enforcement that makes the
    visible answer correct, not just flagged.
    """
    if not answer.strip() or not rejected_claims:
        return answer, []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    kept: list[str] = []
    removed: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        sl = s.lower()
        if any(rc.lower() in sl for rc in rejected_claims):
            removed.append(s)
        else:
            kept.append(s)
    if not kept:
        return answer, removed  # never empty the answer
    return "\n\n".join(kept), removed
