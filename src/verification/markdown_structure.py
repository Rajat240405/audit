"""Structure-aware Markdown safety for verification rewrites.

The verification cleanup was a WHOLE-ANSWER LLM regeneration
(``VerificationAuthority.rewrite_answer``): the model re-emits the entire
answer and is merely *asked* to "preserve markdown formatting". With no
positional anchor for rejected claims and no post-condition check, a rejected
token that happens to live in a table header made the model regenerate — and
corrupt — the whole table.

This module provides the pieces that make cleanup safe:

  1. ``remove_table_rows_for_claims`` — DETERMINISTIC removal of table data
     rows, used only when a rejected claim maps unambiguously to exactly one
     row. Header and delimiter rows are never touched.
  2. ``remove_list_and_heading_for_claims`` — the same idea for list items and
     headings. A unit is removed only when a rejected claim maps to EXACTLY
     ONE candidate unit in the whole document AND the claim accounts for that
     unit's content (exact match, or a contiguous whole-token fragment covering
     >= ``_DOMINANCE_RATIO`` of it). Everything else is left unmapped so the
     caller falls back to the LLM path and its post-condition check.
  3. ``structure_signature`` / ``validate_rewrite`` — a post-condition check.
     If an LLM rewrite changed the block structure in any way other than
     dropping whole blocks, the rewrite is rejected and the caller keeps the
     original answer. ``validate_rewrite`` accepts an optional
     ``allowed_removals`` set naming block signatures a DETERMINISTIC step
     deliberately dropped; with the default ``None`` the check is byte-for-byte
     the original strict behaviour, so an LLM rewrite can never drop a heading.

The invariant the caller gets:

    table -> valid table   bullets -> bullets   ordered -> ordered
    headings -> headings   paragraphs -> paragraphs
    fenced code -> byte-identical, never rewritten
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

__all__ = [
    "Block",
    "BlockKind",
    "TargetedRemoval",
    "split_blocks",
    "structure_signature",
    "validate_rewrite",
    "remove_table_rows_for_claims",
    "remove_list_and_heading_for_claims",
    "code_block_spans",
]

BlockKind = Literal["code", "table", "heading", "ulist", "olist", "para", "blank"]

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DELIM_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
_ULIST_RE = re.compile(r"^\s*[-*+]\s+")
_OLIST_RE = re.compile(r"^\s*\d+[.)]\s+")

# Marker-capturing variants (used only by the list/heading remover, which needs
# the marker and the content separately). Deliberately separate from the
# detection regexes above so block classification stays untouched.
_ULIST_MARKER_RE = re.compile(r"^(\s*)([-*+])(\s+)(.*)$")
_OLIST_MARKER_RE = re.compile(r"^(\s*)(\d+)([.)])(\s+)(.*)$")
_HEADING_LINE_RE = re.compile(r"^(\s{0,3})(#{1,6})(\s+)(.*?)\s*#*\s*$")

# ── Safety thresholds for deterministic list/heading removal ─────────────────
# A claim must be at least this many normalized tokens to justify deleting a
# whole unit. Single-token claims ("IMD", "9", "AWS") are far too generic: they
# are exactly the fragments `extract_claims` produces from acronyms and bare
# table-cell numbers, and matching one against a list item would delete a whole
# supported statement over one incidental token.
_MIN_CLAIM_TOKENS = 2
# The claim must account for at least this fraction of the unit's tokens. This
# is what makes the removal *targeted*: we only delete a unit that is
# essentially the rejected claim, never a unit that merely mentions it.
_DOMINANCE_RATIO = 0.6


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class TargetedRemoval:
    """Result of a deterministic list/heading removal pass.

    ``allowed`` carries the ``structure_signature`` entries this pass
    deliberately dropped, so the caller can hand them to ``validate_rewrite``
    and distinguish an intentional heading removal from an LLM silently
    flattening one. Empty when nothing was removed.
    """

    text: str
    applied: tuple[str, ...]
    unmapped: tuple[str, ...]
    allowed: frozenset[tuple]


def _classify(line: str) -> BlockKind:
    if not line.strip():
        return "blank"
    if _HEADING_RE.match(line):
        return "heading"
    if _TABLE_ROW_RE.match(line):
        return "table"
    if _ULIST_RE.match(line):
        return "ulist"
    if _OLIST_RE.match(line):
        return "olist"
    return "para"


def split_blocks(text: str) -> list[Block]:
    """Segment markdown into typed blocks.

    Fenced code is captured verbatim (including the fences) and never
    inspected further, so nothing inside a code block can be mistaken for a
    table row, list item or heading.
    """
    out: list[Block] = []
    buf: list[str] = []
    buf_kind: BlockKind | None = None
    in_code = False
    code: list[str] = []

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf and buf_kind is not None:
            out.append(Block(buf_kind, tuple(buf)))
        buf, buf_kind = [], None

    for line in text.split("\n"):
        if in_code:
            code.append(line)
            if _FENCE_RE.match(line):
                out.append(Block("code", tuple(code)))
                code, in_code = [], False
            continue
        if _FENCE_RE.match(line):
            flush()
            in_code = True
            code = [line]
            continue
        kind = _classify(line)
        if kind != buf_kind:
            flush()
            buf_kind = kind
        buf.append(line)
    if in_code and code:  # unterminated fence — keep it verbatim
        out.append(Block("code", tuple(code)))
    flush()
    return out


def code_block_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of fenced code blocks."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for b in split_blocks(text):
        seg = b.text
        idx = text.find(seg, pos)
        if idx < 0:
            continue
        if b.kind == "code":
            spans.append((idx, idx + len(seg)))
        pos = idx + len(seg)
    return spans


def structure_signature(text: str) -> list[tuple]:
    """Comparable fingerprint of the document's block structure.

    Tables additionally carry their header row and column count, so a rewrite
    that silently reshapes a table is detected even when the block sequence is
    otherwise unchanged. Blank blocks are ignored (pure whitespace churn is
    not a structural change).
    """
    sig: list[tuple] = []
    for b in split_blocks(text):
        if b.kind == "blank":
            continue
        if b.kind == "table":
            rows = [l for l in b.lines if l.strip()]
            header = rows[0].strip() if rows else ""
            cols = header.count("|")
            # A table's identity includes whether it still HAS a delimiter row
            # and how many data rows remain. Without these a rewrite could drop
            # the delimiter (making it not a table any more) or flatten the
            # whole thing and still look "unchanged" to the validator.
            has_delim = any(_TABLE_DELIM_RE.match(r) for r in rows)
            # NOTE: data-row COUNT is deliberately excluded — removing an
            # unsupported row is legitimate cleanup. Header text, column count
            # and delimiter presence are what must stay stable.
            sig.append(("table", header, cols, has_delim))
        elif b.kind == "code":
            sig.append(("code", b.text))
        elif b.kind == "heading":
            sig.append(("heading", b.lines[0].strip()))
        else:
            sig.append((b.kind,))
    return sig


def validate_rewrite(
    original: str,
    rewritten: str,
    allowed_removals: Iterable[tuple] | None = None,
) -> tuple[bool, str]:
    """Is ``rewritten`` a structurally safe cleanup of ``original``?

    Cleanup may REMOVE whole blocks (a fully-unsupported paragraph) but may
    never introduce blocks, reorder them, alter a table's header/shape, or
    modify fenced code. Returns ``(ok, reason)``; ``reason`` is empty when ok.

    ``allowed_removals`` names block signatures a DETERMINISTIC step has
    already proven safe to drop (see ``remove_list_and_heading_for_claims``).
    It only ever relaxes the "structural blocks may not vanish" count check,
    and only for the exact signatures listed. The default ``None`` preserves
    the original strict behaviour exactly — an LLM rewrite is still never
    permitted to drop a table, a heading or a code block.
    """
    if not rewritten.strip():
        return False, "rewrite is empty"

    orig = structure_signature(original)
    new = structure_signature(rewritten)

    if len(new) > len(orig):
        return False, f"rewrite added blocks ({len(orig)} -> {len(new)})"

    # `new` must be an in-order subsequence of `orig`
    i = 0
    for item in new:
        while i < len(orig) and orig[i] != item:
            i += 1
        if i == len(orig):
            return False, f"rewrite altered block structure near {item[0]!r}"
        i += 1

    # Structural blocks may not VANISH. Dropping a paragraph is legitimate
    # cleanup; dropping a table/heading/code block means the rewrite flattened
    # or swallowed structure, which is exactly the corruption we guard against.
    permitted = [s for s in (allowed_removals or ()) if s]
    for kind in ("table", "code", "heading"):
        n_orig = sum(1 for x in orig if x[0] == kind)
        n_new = sum(1 for x in new if x[0] == kind)
        if n_new < n_orig:
            allowance = sum(1 for s in permitted if s[0] == kind)
            if n_orig - n_new > allowance:
                return False, f"rewrite dropped a {kind} block ({n_orig} -> {n_new})"

    # code blocks are inviolate
    orig_code = [b.text for b in split_blocks(original) if b.kind == "code"]
    new_code = [b.text for b in split_blocks(rewritten) if b.kind == "code"]
    if new_code != orig_code:
        return False, "rewrite modified a fenced code block"

    return True, ""


def _row_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def remove_table_rows_for_claims(
    text: str, rejected: Iterable[str]
) -> tuple[str, list[str], list[str]]:
    """Deterministically drop table DATA rows matched by rejected claims.

    A claim is applied only when it maps to EXACTLY ONE data row across the
    whole document — an ambiguous claim (matching several rows, or none) is
    left for the caller to handle, because guessing risks deleting good data.

    Header and delimiter rows are never candidates. Returns
    ``(new_text, applied_claims, unmapped_claims)``.
    """
    blocks = split_blocks(text)
    claims = [c.strip() for c in rejected if c and c.strip()]
    applied: list[str] = []
    unmapped: list[str] = []

    # index every data row once: (block_index, line_index) -> cells
    data_rows: list[tuple[int, int, list[str]]] = []
    for bi, b in enumerate(blocks):
        if b.kind != "table":
            continue
        seen_delim = False
        for li, line in enumerate(b.lines):
            if not line.strip():
                continue
            if _TABLE_DELIM_RE.match(line):
                seen_delim = True
                continue
            if not seen_delim:
                continue  # header row(s)
            data_rows.append((bi, li, _row_cells(line)))

    drop: set[tuple[int, int]] = set()
    for claim in claims:
        needle = claim.casefold()
        hits = [
            (bi, li) for bi, li, cells in data_rows
            if any(c.casefold() == needle for c in cells)
        ]
        if len(hits) == 1:
            drop.add(hits[0])
            applied.append(claim)
        else:
            unmapped.append(claim)  # 0 = not a table claim, >1 = ambiguous

    if not drop:
        return text, [], unmapped

    rebuilt: list[str] = []
    for bi, b in enumerate(blocks):
        if b.kind != "table":
            rebuilt.append(b.text)
            continue
        kept = [l for li, l in enumerate(b.lines) if (bi, li) not in drop]
        rebuilt.append("\n".join(kept))
    return "\n".join(rebuilt), applied, unmapped


# ── Deterministic list / heading removal ─────────────────────────────────────


def _norm_tokens(text: str) -> list[str]:
    """Lowercase, strip every non-alphanumeric run, split on whitespace.

    Deliberately local (not ``text_support.normalize``): this compares a claim
    against a unit taken from the SAME document, so surface forms already agree
    and the alias/number-swap machinery is not needed. Keeping it local also
    avoids pulling text_support's import-time alias-file read into this module.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip().split()


def _claim_matches_unit(claim_tokens: list[str], unit_tokens: list[str]) -> bool:
    """Does this rejected claim justify deleting this whole unit?

    Two accepted shapes, both strict:
      * EXACT — the claim is the unit's entire content.
      * DOMINANT FRAGMENT — the claim is a contiguous whole-token run inside
        the unit, has >= ``_MIN_CLAIM_TOKENS`` tokens, and covers at least
        ``_DOMINANCE_RATIO`` of the unit.

    Token-boundary matching (not substring) is what stops a rejected "9" from
    deleting every item containing a 9. The dominance floor is what stops a
    rejected acronym from deleting a long, mostly-supported item that happens
    to mention it.
    """
    if not unit_tokens or not claim_tokens:
        return False
    if claim_tokens == unit_tokens:
        return True
    if len(claim_tokens) < _MIN_CLAIM_TOKENS:
        return False
    n, m = len(claim_tokens), len(unit_tokens)
    if n / m < _DOMINANCE_RATIO:
        return False
    return any(unit_tokens[i:i + n] == claim_tokens for i in range(m - n + 1))


def _next_block_is_para(blocks: list[Block], bi: int) -> bool:
    nxt = blocks[bi + 1] if bi + 1 < len(blocks) else None
    return nxt is not None and nxt.kind == "para"


def _section_has_body(blocks: list[Block], heading_bi: int) -> bool:
    """Does the section introduced by ``blocks[heading_bi]`` carry content?

    Scans forward to the next heading block (any level) or end of document.
    Any non-blank block in between is body content — removing the heading
    would orphan it, which is a semantic corruption we refuse to perform.
    """
    for b in blocks[heading_bi + 1:]:
        if b.kind == "heading":
            return False
        if b.kind != "blank":
            return True
    return False


def _list_candidates(blocks: list[Block]) -> list[tuple[int, int, list[str]]]:
    """Every single-line list item that may be considered for removal."""
    out: list[tuple[int, int, list[str]]] = []
    for bi, b in enumerate(blocks):
        if b.kind not in ("ulist", "olist"):
            continue
        for li, line in enumerate(b.lines):
            m = (_ULIST_MARKER_RE if b.kind == "ulist" else _OLIST_MARKER_RE).match(line)
            if not m:
                continue
            content = m.group(4) if b.kind == "ulist" else m.group(5)
            if not content.strip():
                continue
            # A trailing item can own a continuation paragraph that
            # split_blocks classified as the NEXT block. Deleting the marker
            # line would orphan that text, so the item is not a candidate.
            if li == len(b.lines) - 1 and _next_block_is_para(blocks, bi):
                continue
            out.append((bi, li, _norm_tokens(content)))
    return out


def _heading_candidates(blocks: list[Block]) -> list[tuple[int, int, list[str]]]:
    """Every heading that may be considered for removal.

    A heading is a candidate only when its block is a single heading line (so
    removal drops a whole, individually-identified block rather than reshaping
    a multi-heading block) and its section is empty (so no content is
    orphaned).
    """
    out: list[tuple[int, int, list[str]]] = []
    for bi, b in enumerate(blocks):
        if b.kind != "heading" or len(b.lines) != 1:
            continue
        m = _HEADING_LINE_RE.match(b.lines[0])
        if not m or not m.group(4).strip():
            continue
        if _section_has_body(blocks, bi):
            continue
        out.append((bi, 0, _norm_tokens(m.group(4))))
    return out


def _renumber_olist(lines: list[str]) -> list[str]:
    """Renumber an ordered list after item removal.

    Keeps the original start number and increments by one. Markdown renderers
    renumber from the first marker anyway; this keeps the raw text honest
    instead of leaving "1. 3. 4." behind. Non-marker lines pass through.
    """
    out: list[str] = []
    n: int | None = None
    for line in lines:
        m = _OLIST_MARKER_RE.match(line)
        if not m:
            out.append(line)
            continue
        n = int(m.group(2)) if n is None else n + 1
        out.append(f"{m.group(1)}{n}{m.group(3)}{m.group(4)}{m.group(5)}")
    return out


def remove_list_and_heading_for_claims(
    text: str, rejected: Iterable[str]
) -> TargetedRemoval:
    """Deterministically drop list items / headings matched by rejected claims.

    The list-and-heading counterpart of ``remove_table_rows_for_claims``, with
    the same ambiguity rule: a claim is applied only when it maps to EXACTLY
    ONE candidate unit across the whole document. Zero hits (not a list/heading
    claim) or several hits (ambiguous) leave the claim unmapped for the caller
    to handle, because guessing risks deleting supported content.

    Guarantees:
      * fenced code is never a candidate (``split_blocks`` isolates it);
      * a unit is removed only when the claim IS essentially that unit
        (``_claim_matches_unit``);
      * an item with a continuation paragraph is never removed;
      * a heading with section content is never removed;
      * remaining ordered-list markers are renumbered, never left gapped.

    Returns a ``TargetedRemoval``; ``allowed`` must be passed to
    ``validate_rewrite`` for the heading removals to be accepted.
    """
    blocks = split_blocks(text)
    claims = [c.strip() for c in rejected if c and c.strip()]
    if not claims:
        return TargetedRemoval(text, (), (), frozenset())

    heading_blocks = {bi for bi, b in enumerate(blocks) if b.kind == "heading"}
    candidates = _list_candidates(blocks) + _heading_candidates(blocks)
    if not candidates:
        return TargetedRemoval(text, (), tuple(claims), frozenset())

    applied: list[str] = []
    unmapped: list[str] = []
    drop: set[tuple[int, int]] = set()
    allowed: set[tuple] = set()

    for claim in claims:
        ct = _norm_tokens(claim)
        if not ct:
            unmapped.append(claim)
            continue
        hits = [(bi, li) for bi, li, ut in candidates if _claim_matches_unit(ct, ut)]
        if len(hits) != 1:
            unmapped.append(claim)
            continue
        bi, li = hits[0]
        drop.add((bi, li))
        applied.append(claim)
        if bi in heading_blocks:
            allowed.add(("heading", blocks[bi].lines[0].strip()))

    if not drop:
        return TargetedRemoval(text, (), tuple(unmapped), frozenset())

    rebuilt: list[str] = []
    for bi, b in enumerate(blocks):
        dropped_here = {li for (bbi, li) in drop if bbi == bi}
        if not dropped_here:
            rebuilt.append(b.text)
            continue
        kept = [l for li, l in enumerate(b.lines) if li not in dropped_here]
        if b.kind == "heading":
            # Single-line heading blocks are the only candidates, so `kept` is
            # empty here and the whole block disappears — recorded in `allowed`.
            if kept:
                rebuilt.append("\n".join(kept))
            continue
        if not kept:
            continue  # every item removed: the list block itself goes away
        if b.kind == "olist":
            kept = _renumber_olist(kept)
        rebuilt.append("\n".join(kept))

    return TargetedRemoval(
        "\n".join(rebuilt), tuple(applied), tuple(unmapped), frozenset(allowed)
    )
