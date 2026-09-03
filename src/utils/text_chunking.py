"""Deterministic, structure-aware long-document chunking (shared, stdlib-only).

Why this module exists
----------------------
``HybridRAGPipeline._split_long_doc`` and ``src.generation.evidence`` both
used to split text on ``"\\n"`` only. Corpus inspection (Aug 2026,
``corpus_reports.jsonl``) showed that **every non-parliamentary document
type has ~0 newlines/KB** — OCR/scrape normalization flattened them into one
line (e.g. an annual report of 127,769 chars with zero newlines; a
``general_report`` "chunk" of 316,021 chars). Paragraph-only splitting
therefore degenerated to one mega-chunk per report, poisoning dense
embeddings (bge-m3's 8,192-token window), BM25 length normalization, the
cross-encoder's 512-token window, and evidence segmentation alike.

Design (deterministic; no LLM)
------------------------------
Atomic units are built first, then packed:

1. Page markers (``--- Page N (OCR) ---`` and close variants) are HARD
   boundaries and are dropped from chunk text (layout noise).
2. Real newlines are honored: a line <= max_chars stays one unit.
3. A line longer than max_chars is sentence-split (with a small
   abbreviation guard so ``Dr.`` / ``Rs.`` / decimals do not tear); a
   single sentence longer than max_chars is word-wrapped as a last resort.
4. Units are packed greedily toward ``target`` chars; a chunk body never
   exceeds ``max_chars`` (the bonded overlap prefix is extra, documented
   below).
5. Heading-ish units (short colon-terminated, ALL-CAPS, or structural
   headings like ``ANNEXURE-II``) are never left as the LAST unit of a
   chunk — a heading stays bonded to the content it labels.
6. Overlap: the tail of a closed chunk (<= ``overlap`` chars, at a word
   boundary) is prepended to the next chunk so boundary context (a torn
   sentence/table row) survives. At most ``overlap`` chars — local context,
   never whole-neighbor duplication. Skipped when the next chunk starts
   with a heading (a heading carries its own context).

Guarantees (pinned by tests/test_long_doc_chunking.py)
------------------------------------------------------
- No empty chunks; after dropping page markers and normalizing whitespace,
  the chunk texts cover the input COMPLETELY (exact reconstruction with
  ``overlap=0``; full containment of every unit with overlap>0).
- Every chunk is <= ``max_chars + overlap`` (~880 chars at the defaults).
- Chunk order is document order; output is deterministic.
- Known corner: a heading immediately followed by a unit of (almost)
  ``max_chars`` may emit a tiny heading-only chunk — bounded, deterministic,
  and vanishingly rare in this corpus (units are far under max_chars).

The retrieval pipeline keys chunks ``{parent_id}_L{idx}``; this module only
produces the ordered strings — ID assignment stays with the caller so the
Deep-sibling (``_L{n±1}``) machinery keeps working unchanged.
"""

from __future__ import annotations

import re

CHUNKER_VERSION = "split-v2"

DEFAULT_TARGET_CHARS = 500
DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP_CHARS = 80  # ~16% of target

# --- Page 1 (OCR) --- / --- Page 1 --- / ---- PAGE 12  ---- etc.
PAGE_MARKER_RE = re.compile(
    r"-{2,}\s*page\s+\d+(?:\s*\([^)]{0,24}\))?\s*-{2,}",
    re.IGNORECASE,
)

# Sentence boundary: sentence-final punctuation followed by whitespace.
# Decimals ("2.5") never match — there is no whitespace after the period.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# Abbreviations whose trailing period is NOT a sentence end. Small and
# corpus-relevant (Indian parliamentary + scientific-report English).
_ABBREV_RE = re.compile(
    r"\b(?:Dr|Shri|Smt|Sh|Km|Mr|Mrs|Ms|Prof|Capt|Col|Gen|No|Nos|Rs|St|"
    r"Fig|Eq|Sec|Sch|pp|vs|viz|approx|resp|inc|ltd|i\.e|e\.g)\.$",
    re.IGNORECASE,
)

# Structural report headings (ANNEXURE-II, Section 4, TABLE 2:, ...).
_STRUCT_HEAD_RE = re.compile(
    r"^(ANNEXURE|APPENDIX|SECTION|CHAPTER|PART|EXHIBIT|SCHEDULE|TABLE|FIGURE)\b",
    re.IGNORECASE,
)


def _is_heading_unit(unit: str) -> bool:
    """Heading-ish unit: labels following content rather than standing alone.

    Mirrors the heading detection used by evidence segmentation, kept here
    so chunking bonds headings to their content even when evidence is not
    involved.
    """
    s = unit.strip()
    if not s:
        return False
    if _STRUCT_HEAD_RE.match(s) and s[-1:] not in ".!?":
        return True
    if len(s) <= 90 and s.rstrip().endswith(":") and s.rstrip()[-1:] != ".":
        return True
    if len(s) < 60 and s.upper() == s and any(c.isalpha() for c in s):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split into sentences, protecting abbreviations ("Dr. Rao" stays one)
    and decimals ("2.5" never splits). Whitespace between sentences is
    normalized to single spaces."""
    if not text or not text.strip():
        return []
    pieces = [p for p in _SENT_BOUNDARY_RE.split(text.strip()) if p.strip()]
    if not pieces:
        return []
    out: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        if _ABBREV_RE.search(out[-1].rstrip()):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _word_wrap(text: str, max_chars: int) -> list[str]:
    """Last-resort splitter for a single over-long sentence: cut at the last
    space inside the budget; hard-cut only if there is no space at all."""
    out: list[str] = []
    rest = text.strip()
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        out.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


def units_from_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Flatten arbitrary text into ordered atomic units, each <= max_chars.

    Page markers are hard boundaries and dropped. Real newlines are honored;
    over-long lines become sentence lists; over-long sentences are
    word-wrapped. Returns [] for empty input.
    """
    if not text:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    units: list[str] = []
    # Page markers split first — a chunk never straddles a page boundary.
    for piece in PAGE_MARKER_RE.split(normalized):
        for line in piece.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) <= max_chars:
                units.append(line)
                continue
            for sent in split_sentences(line):
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) <= max_chars:
                    units.append(sent)
                else:
                    units.extend(_word_wrap(sent, max_chars))
    return units


def wrap_lines(text: str, max_line: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Line list for evidence segmentation: identical to the old
    ``[l for l in text.split("\\n") if l.strip()]`` behavior for structured
    text, and sentence-wrapped pseudo-lines for flattened (newline-less)
    walls of text. Used by ``src.generation.evidence.segment_blocks`` (and
    the legacy evidence extractor) so a ~300k-char report can never again
    become one giant Block."""
    return units_from_text(text, max_chars=max_line)


def split_long_text(
    text: str,
    target: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Pack atomic units into chunks of ~target chars with bounded overlap.

    ``target`` is the packing goal; ``max_chars`` is the hard cap for new
    chunk content; ``overlap`` is the max size of the context prefix carried
    from the previous chunk's tail (not counted against the body cap).
    """
    if not text or not text.strip():
        return []
    target = max(1, int(target))
    max_chars = max(target, int(max_chars))
    overlap = max(0, int(overlap))

    units = units_from_text(text, max_chars=max_chars)
    if not units:
        return []

    # Fast path: everything fits one chunk (units already marker-free).
    if sum(len(u) + 1 for u in units) - 1 <= max_chars:
        return ["\n".join(units).strip()]

    chunks: list[str] = []
    prefix = ""  # overlap carried INTO the current chunk from the previous one
    i, n = 0, len(units)

    while i < n:
        body: list[str] = []
        used = 0

        # Pack units toward target. Progress is guaranteed: the first unit
        # of a chunk is always accepted (units are <= max_chars).
        while i < n:
            sep = 1 if body else 0
            if body and used + sep + len(units[i]) > target:
                # Heading bond exception: if the chunk so far is ONLY a
                # heading, bond the next (content) unit despite exceeding
                # target, as long as the pair stays under the hard cap.
                if not (
                    len(body) == 1
                    and _is_heading_unit(body[0])
                    and used + sep + len(units[i]) <= max_chars
                ):
                    break
            body.append(units[i])
            used += sep + len(units[i])
            i += 1

        # Heading bond: never leave trailing heading(s) at a chunk end —
        # carry them to the next chunk so they stay with their content.
        while len(body) > 1 and _is_heading_unit(body[-1]):
            i -= 1
            body.pop()

        # Emit this chunk WITH the overlap prefix carried from the previous
        # chunk (the prefix is context FROM before — it belongs up front).
        emitted = "\n".join(([prefix] if prefix else []) + body).strip()
        if emitted:
            chunks.append(emitted)

        # Compute the overlap prefix for the NEXT chunk: tail of the chunk
        # just emitted, <= overlap chars, at a word boundary. Skipped when
        # the next chunk starts with a heading (a heading carries its own
        # context), and when the tail would duplicate the next unit (rare
        # repeated content).
        prefix = ""
        if overlap > 0 and i < n:
            nxt = units[i]
            if not _is_heading_unit(nxt):
                tail = body[-1]
                if len(tail) <= overlap:
                    prefix = tail
                else:
                    cut = tail.rfind(" ", len(tail) - overlap)
                    prefix = tail[cut + 1 :] if cut > 0 else tail[-overlap:]
                if not prefix.strip() or prefix == nxt:
                    prefix = ""

    return [c for c in chunks if c]
