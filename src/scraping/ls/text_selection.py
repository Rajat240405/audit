"""Deterministic per-field source selection for Lok Sabha inline-vs-document text.

WHY THIS EXISTS
---------------
The LS ladder used to be *inline-first*: if the upstream API returned
``answerText``, the official English PDF was downloaded but never parsed
(``pipeline.py`` only called ``apply_extraction`` when ``answer_text`` was
empty). That was correct under the original assumption, still stated in
``extract.py``/``normalize.py``/``discovery.py``, that LS inline text is
"rare / legacy api rows / modern rows are null".

Upstream coverage widened. 649 records silently switched from PDF-derived to
API-derived text and lost annexure tables (``ls-16-9-1611`` state-wise rainfall
departures, ``ls-17-11-5336`` and ``ls-17-15-0756`` Doppler-radar lists).

DESIGN
------
Selection is **per field**, because neither source is uniformly better:

- ``answer_text``  → **document** primary. Annexures and tables exist only in
  the official PDF.
- ``question_text`` → **inline** primary. PDF-derived questions carry document
  furniture ("GOVERNMENT OF INDIA / LOK SABHA / UNSTARRED QUESTION No. 1611 /
  TO BE ANSWERED ON …") and, when the ANSWER/REPLY boundary regex misses,
  ``_split_question_answer`` falls back to an arbitrary one-third length split
  (``src/data/scraper.py``) — in which case the "question" is not a question.

Arbitration is **cheap and deterministic** — no model, no network. It compares
length, numeric tokens, annexure markers, ``(a)``-style sub-part labels and
truncation signals, and it never lets a poorer inline answer displace a richer
extracted one.

PROVENANCE POLICY (deliberate)
------------------------------
``question_text_source`` / ``answer_text_source`` / ``text_selection_reason``
are written **only when a real choice was made between two non-empty
candidates**. Single-candidate records keep exactly the metadata they have
today, so the ~2,000 unaffected LS records stay byte-stable and do not churn
``qa_content_hash``. Flip ``ALWAYS_RECORD_PROVENANCE`` to change this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "INLINE",
    "DOCUMENT",
    "TextMetrics",
    "Selection",
    "measure",
    "bleed_through_offset",
    "in_scope_text",
    "structural_signals",
    "text_equivalent",
    "split_was_boundary_based",
    "select_answer",
    "select_question",
    "is_richer",
]

#: ``metadata.answer_source`` vocabulary (unchanged from the existing model)
INLINE = "inline"
DOCUMENT = "document-extract"

#: Write per-field provenance even when only one candidate existed. Kept False
#: so unaffected records stay byte-stable (see module docstring).
ALWAYS_RECORD_PROVENANCE = False

# ── thresholds (all cheap, all deterministic) ────────────────────────────────
#: below this an inline candidate is not trusted as a real question
MIN_SUBSTANTIVE_CHARS = 15
#: a candidate this much longer than the other is "materially" longer
MATERIAL_LENGTH_RATIO = 1.15
#: trailing fragment shorter than this, without terminal punctuation → truncated
TRUNCATION_TAIL_CHARS = 60

_NUMERIC = re.compile(r"\d+(?:[.,]\d+)*")
_ANNEXURE = re.compile(r"(?i)\bannex(?:ure|ures)?\b|\bappendix\b")
_SUBPART = re.compile(r"\(\s*([a-zA-Z])\s*\)")
# A line restarting the document furniture marks the next record's content.
_SECOND_RECORD_HEAD = re.compile(
    r"(?m)^(?:GOVERNMENT OF INDIA|LOK SABHA|(?:UN)?STARRED QUESTION\s*(?:No\.?|NO:?)\s*\d+)"
)
_BOUNDARY_HEAD = re.compile(r"(?is)^(?:ANSWER|REPLY|A\s*N\s*S\s*W\s*E\s*R)\b")
# A candidate counts as complete when it ends in terminal punctuation OR in a
# digit — annexure tables legitimately end on a data row ("2022-03-18"), and
# flagging those as truncated would wrongly favour the inline summary.
_SENTENCE_END = re.compile(r"[.!?;:)\]%\"']\s*$|\d\s*$")
#: document furniture that legitimately belongs to the PDF, not to a question
_FURNITURE = re.compile(
    r"(?i)^\s*GOVERNMENT OF INDIA\b|\bLOK SABHA\b|\bUNSTARRED QUESTION\b"
    r"|\bSTARRED QUESTION\b|\bTO BE ANSWERED ON\b|\bANSWERED ON\b"
)


@dataclass(frozen=True)
class TextMetrics:
    """Cheap structural fingerprint of one candidate text."""

    chars: int = 0
    numeric_tokens: int = 0
    annexure_markers: int = 0
    subpart_labels: int = 0
    truncated: bool = False

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "chars": self.chars,
            "numeric_tokens": self.numeric_tokens,
            "annexure_markers": self.annexure_markers,
            "subpart_labels": self.subpart_labels,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class Selection:
    """Outcome of one per-field selection."""

    source: str | None          # INLINE | DOCUMENT | None
    text: str                   # the text to store ('' when nothing usable)
    reason: str                 # stable machine-readable code
    had_choice: bool = False    # both candidates were non-empty
    signals: tuple[str, ...] = field(default_factory=tuple)
    inline_metrics: TextMetrics | None = None
    document_metrics: TextMetrics | None = None

    @property
    def provenance(self) -> bool:
        """True when the per-field source fields should be written."""
        return ALWAYS_RECORD_PROVENANCE or self.had_choice


def _collapse(text: str) -> str:
    """Single-space form. Keeps word boundaries intact for the ``\\b`` regexes."""
    return re.sub(r"\s+", " ", text).strip()


def _densify(text: str) -> str:
    """Whitespace-free form.

    PDF text extraction is not whitespace-stable: the same glyph run can come
    back as ``26.20 0 C`` or ``26.200C``, ``18 th JULY`` or ``18thJULY``.
    Comparing length or numeric-token counts on the raw text therefore reports
    a formatting-only difference as a content difference — which is exactly
    what a re-extraction looks like. Length and numeric counts are taken on
    this form so they are formatting-invariant.
    """
    return re.sub(r"\s+", "", text)


def text_equivalent(a: str | None, b: str | None) -> bool:
    """True when two texts differ only by whitespace/case — i.e. formatting.

    Used by tests and diagnostics so a normalization-only re-extraction is not
    mistaken for data loss.
    """
    return _densify(a or "").casefold() == _densify(b or "").casefold()


def measure(text: str | None) -> TextMetrics:
    """Fingerprint a candidate. Empty/None degrades to an all-zero metric.

    ``chars`` is a NON-WHITESPACE character count and ``numeric_tokens`` is
    counted on the whitespace-free form, so both are invariant under the
    spacing artifacts PDF extraction introduces. ``annexure_markers`` and
    ``subpart_labels`` run on the single-space form because their patterns
    rely on word boundaries. Trade-off, accepted deliberately: densifying
    merges adjacent numbers ("16.53 80.80" counts as one token), so numeric
    counts are coarser but still directionally correct — losing an annexure
    row always removes digits.
    """
    s = (text or "").strip()
    if not s:
        return TextMetrics()
    collapsed = _collapse(s)
    dense = _densify(s)
    # _SENTENCE_END is $-anchored, so scanning only the tail is equivalent to
    # scanning the whole text and bounds the work on large annexures.
    tail = collapsed[-TRUNCATION_TAIL_CHARS:]
    return TextMetrics(
        chars=len(dense),
        numeric_tokens=len(_NUMERIC.findall(dense)),
        annexure_markers=len(_ANNEXURE.findall(collapsed)),
        # distinct labels, so "(a) … (b) … (a)" counts 2 not 3
        subpart_labels=len({m.lower() for m in _SUBPART.findall(collapsed)}),
        truncated=not bool(_SENTENCE_END.search(tail)),
    )


def bleed_through_offset(text: str | None) -> int | None:
    """Offset where a SECOND record's content begins inside one answer.

    ``_split_question_answer`` splits at the FIRST ANSWER/REPLY match, so when
    an official PDF carries several questions, the tail of the NEXT question —
    its furniture, question body and its own ANSWER block — is swept into this
    record's ``answer_text``. Measured on 4 / 1,377 LS records; e.g.
    ``ls-16-14-4262``, where question 4247's entire Q&A is appended at offset
    1866 of a 2,902-char answer.

    Returns None when the text is single-record. Used so the staged-row guard
    never mistakes that out-of-scope tail for information the fresh extraction
    lost.
    """
    if not text:
        return None
    m = _SECOND_RECORD_HEAD.search(text, 1)   # 1: never match the opening line
    return m.start() if m else None


def in_scope_text(text: str | None) -> str:
    """The portion of an extracted answer that actually belongs to this record."""
    off = bleed_through_offset(text)
    return (text or "")[:off] if off is not None else (text or "")


def split_was_boundary_based(answer_text: str | None) -> bool:
    """Did ``_split_question_answer`` find a real ANSWER/REPLY boundary?

    ``_split_question_answer`` returns ``text[idx:]`` where ``idx`` is the start
    of the boundary match, so a boundary split always yields an answer that
    *begins* with ANSWER/REPLY. The ratio fallback slices at ``len(text)//3``
    and therefore does not. This lets the LS layer flag unreliable splits
    without touching the canonical legacy scraper.
    """
    return bool(_BOUNDARY_HEAD.match((answer_text or "").strip()))


def structural_signals(doc: TextMetrics, inl: TextMetrics) -> list[str]:
    """Information-density reasons ``doc`` carries more than ``inl``.

    Numeric tokens, annexure markers and sub-part labels only. Deliberately
    EXCLUDES length and truncation: those shift with re-pagination and
    extraction spacing, so they must not on their own justify discarding a
    freshly extracted document in favour of an older staged copy.
    """
    out: list[str] = []
    if doc.numeric_tokens > inl.numeric_tokens:
        out.append(f"numeric+{doc.numeric_tokens - inl.numeric_tokens}")
    if doc.annexure_markers > inl.annexure_markers:
        out.append(f"annexure+{doc.annexure_markers - inl.annexure_markers}")
    if doc.subpart_labels > inl.subpart_labels:
        out.append(f"subparts+{doc.subpart_labels - inl.subpart_labels}")
    return out


def _richness_signals(doc: TextMetrics, inl: TextMetrics) -> list[str]:
    """All reasons the document candidate carries more information."""
    out = structural_signals(doc, inl)
    if inl.truncated and not doc.truncated:
        out.append("inline-truncated")
    if doc.chars >= inl.chars * MATERIAL_LENGTH_RATIO:
        out.append(f"length+{doc.chars - inl.chars}")
    return out


def is_richer(candidate: str | None, incumbent: str | None) -> bool:
    """True when ``candidate`` carries materially more information.

    Used by the staged-row guard so a documents-disabled re-crawl cannot
    overwrite a richer already-staged record with a poorer inline one.
    """
    c, i = measure(candidate), measure(incumbent)
    if c.chars == 0:
        return False
    if i.chars == 0:
        return True
    return bool(_richness_signals(c, i))


def select_answer(
    inline: str | None,
    document: str | None,
    *,
    boundary_split: bool = True,
    unavailable_cause: str | None = None,
) -> Selection:
    """Choose ``answer_text``. Document primary; inline never displaces richer.

    Comparison runs on the stripped forms, but the WINNING candidate is
    returned verbatim — the previous ladder stored extraction output untouched,
    and re-stripping here would churn bytes (and ``qa_content_hash``) for
    every record.
    """
    inl_raw = inline or ""
    doc_raw = document or ""
    inl, doc = inl_raw.strip(), doc_raw.strip()
    im, dm = measure(inl), measure(doc)

    if not doc and not inl:
        return Selection(None, "", unavailable_cause or "both-empty",
                         inline_metrics=im, document_metrics=dm)
    if not doc:
        return Selection(INLINE, inl_raw,
                         f"inline-only:{unavailable_cause}" if unavailable_cause
                         else "inline-only:document-empty",
                         inline_metrics=im, document_metrics=dm)
    if not inl:
        return Selection(DOCUMENT, doc_raw, "document-only:inline-absent",
                         had_choice=False,
                         inline_metrics=im, document_metrics=dm)

    # both present — arbitrate
    signals = _richness_signals(dm, im)
    if signals:
        return Selection(DOCUMENT, doc_raw, "document-richer", had_choice=True,
                         signals=tuple(signals),
                         inline_metrics=im, document_metrics=dm)
    if not boundary_split:
        # the document split is an arbitrary 1/3 slice: its "answer" may be
        # missing its own beginning, so inline is the safer representation
        # unless the document still carries structure inline lacks.
        return Selection(INLINE, inl_raw, "inline-preferred:document-ratio-split",
                         had_choice=True, signals=("document-ratio-split",),
                         inline_metrics=im, document_metrics=dm)
    if im.chars >= dm.chars * MATERIAL_LENGTH_RATIO:
        return Selection(INLINE, inl_raw, "inline-richer", had_choice=True,
                         signals=(f"length+{im.chars - dm.chars}",),
                         inline_metrics=im, document_metrics=dm)
    # tie, and the document split is trustworthy → official document wins
    return Selection(DOCUMENT, doc_raw, "document-primary", had_choice=True,
                     inline_metrics=im, document_metrics=dm)


def select_question(
    inline: str | None,
    document: str | None,
    *,
    boundary_split: bool = True,
) -> Selection:
    """Choose ``question_text``. Inline primary; document is the fallback."""
    inl_raw = inline or ""
    doc_raw = document or ""
    inl, doc = inl_raw.strip(), doc_raw.strip()
    im, dm = measure(inl), measure(doc)
    substantive = im.chars >= MIN_SUBSTANTIVE_CHARS

    if not doc and not inl:
        return Selection(None, "", "both-empty",
                         inline_metrics=im, document_metrics=dm)
    if not inl:
        reason = "document-only:inline-absent"
        if not boundary_split:
            reason += ":document-ratio-split"
        return Selection(DOCUMENT, doc_raw, reason,
                         inline_metrics=im, document_metrics=dm)
    if not doc:
        return Selection(INLINE, inl_raw, "inline-only:document-empty",
                         inline_metrics=im, document_metrics=dm)

    # both present — inline is cleaner unless it is not a usable question
    if not substantive:
        return Selection(DOCUMENT, doc_raw, "document-primary:inline-not-substantive",
                         had_choice=True, signals=("inline-not-substantive",),
                         inline_metrics=im, document_metrics=dm)
    if not boundary_split:
        return Selection(INLINE, inl_raw, "inline-preferred:document-ratio-split",
                         had_choice=True, signals=("document-ratio-split",),
                         inline_metrics=im, document_metrics=dm)
    if _FURNITURE.search(doc):
        return Selection(INLINE, inl_raw, "inline-preferred:document-furniture",
                         had_choice=True, signals=("document-furniture",),
                         inline_metrics=im, document_metrics=dm)
    return Selection(INLINE, inl_raw, "inline-primary", had_choice=True,
                     inline_metrics=im, document_metrics=dm)
