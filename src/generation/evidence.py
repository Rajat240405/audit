"""Dynamic evidence budgeting & semantic-boundary-preserving context assembly
(Task 3).

Replaces the fixed-count + fixed-char-cap evidence selection with a
budget-driven admission system driven by the Task-1 ExecutionPlan:

    ExecutionPlan.evidence_budget_tokens      (reserve-based, model-agnostic)
      − actual system-prompt / question tokens (measured here, per request)
      ↓
    allocate_evidence(results, question, budget)
      ↓  greedy, relevance-ordered; whole docs first; block selection second;
      ↓  at most ONE sentence-boundary truncation of the most relevant block
    Admission list
      ↓
    render_user_prompt / assemble_budgeted_prompt
      ↓  identical visual format to the legacy build_user_prompt
    LLM request (never over budget)

Semantic units (Blocks) are atomic: plain lines (paragraphs), grouped list
items, table-like runs, and headings — a detected heading line is BONDED to
the following block so a heading is never separated from its content by
budgeting. We never cut mid-sentence; when the budget is exhausted the
chosen degradation is: drop low-relevance candidates/blocks → marked gaps
("[… N passage(s) omitted …]") → single marked sentence truncation of the
most relevant block. Everything the model gives up is visible in the prompt.

This module also owns the text helpers moved out of generator.py
(clean_parliament_text, truncate_at_sentence, extract_relevant_evidence);
generator.py re-exports them for backward compatibility.

No LLM calls happen anywhere in this module. No model names are read —
budgets arrive as numbers from the ExecutionPlan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.retrieval.result import RetrievedResult

# ─────────────────────────────────────────────────────────────────────────────
# Token estimation — consistent with the rest of the stack (chars // 4).
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """chars//4 estimate, matching generator/providers. Floored at 1 for any
    non-empty text so tiny evidence still counts against the budget."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Parliamentary boilerplate cleaning (moved from generator.py, one tweak —
# DECLARED: structural heading lines (ANNEXURE/APPENDIX/SECTION/CHAPTER/PART/
# EXHIBIT/SCHEDULE/TABLE …) are now KEPT so the segmenter can bond them to
# their content. Subject-title-style caps lines are still stripped.
# ─────────────────────────────────────────────────────────────────────────────

_BOILER_LINE_PREFIXES = (
    "GOVERNMENT OF INDIA",
    "MINISTRY OF EARTH SCIENCES",
    "LOK SABHA",
    "RAJYA SABHA",
    "UNSTARRED QUESTION",
    "STARRED QUESTION",
    "QUESTION NO.",
    "TO BE ANSWERED ON",
    "WILL THE MINISTER",
    "THE MINISTER OF STATE",
    "THE MINISTER FOR STATE",
    "MINISTRY OF SCIENCE AND TECHNOLOGY",
    "AND EARTH SCIENCES",
    "ANSWER",
    "(DR.",
    "DR. ",
    "PROF. ",
    "********",
    "*****",
)

_MEMBER_NAME_RE = re.compile(r"^(SHRI|SMT|SMT\.|MS|MRS|DR|PROF|KUMARI|MR)\.?\s+[A-Z]", re.IGNORECASE)
_QUESTION_NUM_RE = re.compile(r"^\d{3,4}\.\s*$")
_QUESTION_NUM_NAME_RE = re.compile(r"^\d{3,4}\.\s+(SHRI|SMT|DR|PROF)", re.IGNORECASE)
_STRUCT_HEAD_RE = re.compile(
    r"^(ANNEXURE|APPENDIX|SECTION|CHAPTER|PART|EXHIBIT|SCHEDULE|TABLE|FIGURE)\b",
    re.IGNORECASE,
)
_STRUCT_TAIL_RE = re.compile(r"(\d|:|[-–](?=[IVX0-9])|\b[IVX]{1,4}\b\s*$)")


def _is_structural_heading(line: str) -> bool:
    """Structural document heading — ANNEXURE-II, SECTION 4, TABLE 2:,
    APPENDIX, FIGURE 3 — NOT merely a sentence that happens to start with a
    common word ("Section findings indicate…." is prose, not a heading).
    Requires a numeric/roman/colon/annexure-dash tail and no sentence-final
    punctuation."""
    l = line.strip()
    if not _STRUCT_HEAD_RE.match(l):
        return False
    if l[-1:] in ".!?":
        return False
    return bool(_STRUCT_TAIL_RE.search(l))


def clean_parliament_text(text: str) -> str:
    """Strip parliamentary boilerplate lines, keep substantive content."""
    if not text:
        return text
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        u = l.upper()
        # skip all-star separators
        if set(u) <= {"*", " "}:
            continue
        # skip boilerplate prefixes
        if any(u.startswith(p) for p in _BOILER_LINE_PREFIXES):
            continue
        # skip member-name lines ("SHRI YOGENDER CHANDOLIA:")
        if _MEMBER_NAME_RE.match(l) and l.rstrip().endswith(":"):
            continue
        # skip subject-title lines (short all-caps, not (a)/(b)/(c), no colon)
        # — but KEEP structural report headings so they stay bonded to the
        # content they label (Task-3 boundary preservation).
        if (
            u == l
            and len(l) < 60
            and not l.startswith("(")
            and ":" not in l
            and not _is_structural_heading(l)
        ):
            continue
        # skip standalone question numbers ("3035.")
        if _QUESTION_NUM_RE.match(l):
            continue
        # skip "3035. SHRI X" question-number+member lines
        if _QUESTION_NUM_NAME_RE.match(l):
            continue
        cleaned.append(l)
    return "\n".join(cleaned)


def truncate_at_sentence(text: str, max_chars: int, marker: str = " ... [Truncated to fit context budget]") -> str:
    """Truncate at a SENTENCE boundary — never mid-number/unit.

    GLM critique #3: hard char-slicing (text[:max_chars]) can cut a figure
    ("₹2,00,000 crore over 2024-2") or sever a [Source N] citation. This cuts
    at the last sentence boundary before the limit instead.
    """
    if not text or len(text) <= max_chars:
        return text
    limit = max_chars - len(marker)
    if limit <= 0:
        return text[:max_chars]
    # find the last sentence-ending punctuation before the limit
    cut = -1
    for end in (".", "!", "?", "\n"):
        idx = text.rfind(end, 0, limit)
        if idx > cut:
            cut = idx
    if cut > 0:
        # keep the sentence-ending char + a bit of breathing room
        return text[: cut + 1] + marker
    # no sentence boundary found before limit — fall back to word boundary
    space = text.rfind(" ", 0, limit)
    if space > 0:
        return text[:space] + marker
    return text[:max_chars] + marker


def extract_relevant_evidence(text: str, query: str, max_chars: int = 1500) -> str:
    """
    LEGACY evidence extraction (kept for the no-plan fallback path and the
    retrieval-time long-doc pinch): keyword-matched lines, sentence-safe cap.

    The budgeted Task-3 path uses segment_blocks + allocate_evidence instead.
    """
    if len(text) <= max_chars:
        return text

    keywords = query_keywords(query)
    if not keywords:
        return truncate_at_sentence(text, max_chars)

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    matched_paragraphs = []

    for p in paragraphs:
        p_lower = p.lower()
        if any(kw in p_lower for kw in keywords):
            matched_paragraphs.append(p)

    if matched_paragraphs:
        assembled = ""
        for p in matched_paragraphs:
            if len(assembled) + len(p) + 2 <= max_chars:
                assembled += p + "\n\n"
            else:
                remaining = max_chars - len(assembled)
                if remaining > 100:
                    assembled += truncate_at_sentence(p, remaining)
                break
        return assembled.strip() or truncate_at_sentence(text, max_chars)

    return truncate_at_sentence(text, max_chars)


# ─────────────────────────────────────────────────────────────────────────────
# Relevance vocabulary (shared by extraction and block scoring)
# ─────────────────────────────────────────────────────────────────────────────

def query_keywords(query: str) -> list[str]:
    """Same keyword extraction the legacy extractor used (>3-char words)."""
    return [w.lower() for w in re.sub(r"[^\w\s]", " ", query or "").split() if len(w) > 3]


def block_relevance(text: str, keywords: list[str]) -> int:
    """Count of DISTINCT query keywords present in the block (0 = no signal)."""
    if not keywords:
        return 0
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic segmentation — atomic blocks; headings bond to their content
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Block:
    text: str
    kind: str  # "paragraph" | "list" | "table" | "heading"


_LIST_ITEM_RE = re.compile(r"^\s*(\((?:[a-z]|[ivxIVX]+|\d+)\)|\d{1,2}[.)]\s|[-•–*]\s)\s*")
_TABLE_HINT_RE = re.compile(r"\S\s{2,}\S")


def _is_heading_line(line: str) -> bool:
    """A line that labels following content rather than standing alone:
    structural keyword, colon-terminated, or ALL-CAPS-ish and short."""
    l = line.strip()
    if not l:
        return False
    if _is_structural_heading(l):
        return True
    if len(l) <= 90 and l.rstrip().endswith(":") and not l.rstrip()[-2:-1].isdigit():
        return True
    if len(l) < 60 and l.upper() == l and any(c.isalpha() for c in l):
        return True
    return False


def _is_table_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return False
    if l.startswith("|") and l.count("|") >= 2:
        return True
    if "│" in l or "┼" in l or "─" in l:
        return True
    # tabular columns separated by 2+ spaces (figures table rows)
    if _TABLE_HINT_RE.search(l) and not _LIST_ITEM_RE.match(l):
        # avoid classifying ordinary prose with an accidental double space
        return len(re.findall(r"\s{2,}", l)) >= 2
    return False


def segment_blocks(text: str) -> list[Block]:
    """Split cleaned evidence into ATOMIC semantic blocks.

    Rules:
      * a run of table-like lines          -> one "table" block
      * a run of list items                -> one "list" block
      * any other non-empty line           -> one "paragraph" block
      * a heading line is never its own surviving unit — it PREFIXES the
        next block (kind preserved), so budgeting can never detach a heading
        from its content. A heading with no following content is dropped.
      * an oversized single line stays whole (never force-split here; only
        the allocator's single last-resort sentence truncation may cut it).
    """
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    blocks: list[Block] = []
    pending_heading: list[str] = []
    i = 0

    def take_heading() -> str:
        nonlocal pending_heading
        if not pending_heading:
            return ""
        h = "\n".join(pending_heading)
        pending_heading = []
        return h

    while i < len(lines):
        line = lines[i]
        if _is_heading_line(line) and not _LIST_ITEM_RE.match(line):
            pending_heading.append(line)
            i += 1
            continue
        head = take_heading()
        if _is_table_line(line):
            buf = [line]
            i += 1
            while i < len(lines) and _is_table_line(lines[i]):
                buf.append(lines[i])
                i += 1
            text = (head + "\n" if head else "") + "\n".join(buf)
            blocks.append(Block(text=text, kind="table"))
            continue
        if _LIST_ITEM_RE.match(line):
            buf = [line]
            i += 1
            while i < len(lines) and (
                _LIST_ITEM_RE.match(lines[i])
                or (lines[i].startswith((" ", "\t")) and not _is_heading_line(lines[i]))
            ):
                # swallow continuation/indented lines of the same list
                buf.append(lines[i])
                i += 1
            text = (head + "\n" if head else "") + "\n".join(buf)
            blocks.append(Block(text=text, kind="list"))
            continue
        text = (head + "\n" if head else "") + line
        blocks.append(Block(text=text, kind="paragraph"))
        i += 1
    # trailing heading with no content: drop (it labels nothing).
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Allocation
# ─────────────────────────────────────────────────────────────────────────────

# Rough footprint of one "[… N passage(s) omitted …]" gap marker.
_OMISSION_MARKER_TOKENS = 16
# The evidence floor: below this a candidate is not worth its source header —
# a truncation/select shorter than ~40 tokens (≈1-2 sentences) helps no one.
# This is the ONLY count gate: anything above it gets the full block
# selection treatment (verified by test_block_selection_marks_internal_gaps).
_MIN_TRUNCATED_TOKENS = 40

_OMISSION_TEMPLATE = "[… {n} passage(s) omitted …]"


@dataclass
class Admission:
    result: RetrievedResult
    question_text: str     # cleaned
    evidence_text: str     # final emitted answer text (whole or block-selected)
    whole: bool            # True → zero information loss for this candidate
    omitted_units: int     # block/line count given up (transparency)
    truncated: bool        # the single sentence-boundary truncation fired


@dataclass
class Allocation:
    admissions: list[Admission] = field(default_factory=list)
    budget_tokens: int = 0
    used_tokens: int = 0
    skipped_doc_ids: list[str] = field(default_factory=list)
    truncated_doc_ids: list[str] = field(default_factory=list)

    @property
    def admitted_ids(self) -> list[str]:
        return [a.result.doc_id for a in self.admissions]


def allocate_evidence(
    results: list[RetrievedResult],
    question: str,
    budget_tokens: int,
) -> Allocation:
    """Budget-driven admission of retrieved candidates.

    Candidates arrive RELEVANCE-ORDERED (rerank/desc). Order of operations
    per candidate: admit WHOLE if it fits → else block-select within the rest
    of the budget → else drop. Low-relevance candidates are therefore always
    the first to be dropped. Never exceeds ``budget_tokens``. Pure function —
    no I/O, no LLM calls, no model names.
    """
    keywords = query_keywords(question)
    remaining = max(0, int(budget_tokens))
    alloc = Allocation(budget_tokens=max(0, int(budget_tokens)))
    seen_evidence: set[str] = set()

    for r in results:
        q_text = clean_parliament_text(r.question or "")
        a_text = clean_parliament_text(r.answer or "")
        if not q_text and not a_text:
            alloc.skipped_doc_ids.append(r.doc_id)
            continue
        header = _source_header(r, len(alloc.admissions) + 1, q_text)
        header_cost = estimate_tokens(header)

        # exact-duplicate evidence across candidates buys nothing
        if a_text in seen_evidence:
            alloc.skipped_doc_ids.append(r.doc_id)
            continue

        # ── preferred path: admit the WHOLE candidate (zero loss) ──
        whole_cost = header_cost + estimate_tokens(a_text)
        if a_text and whole_cost <= remaining:
            alloc.admissions.append(Admission(r, q_text, a_text, True, 0, False))
            seen_evidence.add(a_text)
            remaining -= whole_cost
            alloc.used_tokens += whole_cost
            continue

        # ── second path: block-select within what remains ──
        if remaining < header_cost + _MIN_TRUNCATED_TOKENS:
            # not enough room for meaningful evidence from this candidate —
            # drop it (order = relevance, so the lowest-relevance go first)
            alloc.skipped_doc_ids.append(r.doc_id)
            continue
        avail = remaining - header_cost
        selection = _select_blocks(segment_blocks(a_text), keywords, avail)
        if selection is None:
            alloc.skipped_doc_ids.append(r.doc_id)
            continue
        emitted, omitted, truncated, evidence_cost = selection
        # header cost is part of the candidate's footprint — charge it too,
        # or successive block-selected candidates silently overdraw.
        total_cost = header_cost + evidence_cost
        alloc.admissions.append(
            Admission(r, q_text, emitted, False, omitted, truncated)
        )
        seen_evidence.add(a_text)
        remaining -= total_cost
        alloc.used_tokens += total_cost
        if truncated:
            alloc.truncated_doc_ids.append(r.doc_id)

    # ── degraded path: nothing fit, but there ARE candidates ──
    # Admit ONE sentence-truncated block from the top candidate, or nothing
    # at all if even that does not leave a meaningful span. Never over budget.
    if not alloc.admissions and results:
        r = results[0]
        q_text = clean_parliament_text(r.question or "")
        a_text = clean_parliament_text(r.answer or "")
        header = _source_header(r, 1, q_text)
        avail = remaining - estimate_tokens(header)
        if avail >= _MIN_TRUNCATED_TOKENS and a_text:
            blocks = segment_blocks(a_text)
            best = _best_single_block(blocks, keywords)
            truncated_text = _truncate_tokens(best.text, avail)
            cost = estimate_tokens(header) + estimate_tokens(truncated_text)
            if cost <= alloc.budget_tokens:
                alloc.admissions.append(
                    Admission(r, q_text, truncated_text, False,
                              max(0, len(blocks) - 1), True)
                )
                alloc.used_tokens += cost
                alloc.truncated_doc_ids.append(r.doc_id)

    return alloc


def _truncate_tokens(text: str, avail_tokens: int) -> str:
    """Sentence-boundary truncation that provably fits ``avail_tokens``.
    A 4-char cushion absorbs the //4 estimator's rounding and the marker."""
    avail_chars = max(0, int(avail_tokens) * 4 - 4)
    out = truncate_at_sentence(text, avail_chars)
    # absolute guarantee (all truncate_at_sentence branches ≤ max_chars+marker)
    while estimate_tokens(out) > avail_tokens and len(out) > 0:
        out = truncate_at_sentence(text, len(out) - 96)
    return out


def _best_single_block(blocks: list[Block], keywords: list[str]) -> Block:
    if not blocks:
        return Block(text="", kind="paragraph")
    scored = [(block_relevance(b.text, keywords), i, b) for i, b in enumerate(blocks)]
    scored.sort(key=lambda s: (-s[0], s[1]))
    return scored[0][2]


def _select_blocks(
    blocks: list[Block], keywords: list[str], avail_tokens: int
) -> Optional[tuple[str, int, bool, int]]:
    """Pick blocks within ``avail_tokens``.

    Keyword-bearing blocks first (relevance desc, position asc); with no
    keyword signal at all, plain document order — never a blind char cut of
    the tail. Emission keeps ORIGINAL order with explicit gap markers; if the
    marked render overflows, lowest-relevance selections are shed until it
    fits. As an absolute last resort, ONE block is sentence-truncated.

    Returns (emitted_text, omitted_count, truncated, total_cost_tokens) or
    None when nothing meaningful fits.
    """
    if not blocks or avail_tokens <= 0:
        return None
    scored = [(block_relevance(b.text, keywords), i, b) for i, b in enumerate(blocks)]
    hits = [s for s in scored if s[0] > 0]
    order = sorted(hits, key=lambda s: (-s[0], s[1])) if hits else list(scored)

    selected: set[int] = set()
    for rel, i, b in order:
        if estimate_tokens(b.text) <= avail_tokens - _OMISSION_MARKER_TOKENS:
            selected.add(i)

    if not selected:
        # nothing fits whole — the single last-resort sentence truncation
        best = _best_single_block(blocks, keywords)
        if avail_tokens < _MIN_TRUNCATED_TOKENS:
            return None
        emitted = _truncate_tokens(best.text, avail_tokens)
        cost = estimate_tokens(emitted)
        if cost > avail_tokens:
            return None
        omitted = len(blocks) - 1
        return emitted, max(0, omitted), True, cost

    dropped_note: list[tuple[int, int]] = []  # shed order bookkeeping
    while True:
        emitted, omitted = _render_selected(blocks, selected)
        cost = estimate_tokens(emitted)
        if cost <= avail_tokens:
            return emitted, omitted, False, cost
        # shed the lowest-relevance (ties: latest position) selection
        worst = min(selected, key=lambda i: (block_relevance(blocks[i].text, keywords), -i))
        selected.remove(worst)
        dropped_note.append((worst, cost))
        if not selected:
            return None


def _render_selected(blocks: list[Block], selected: set[int]) -> tuple[str, int]:
    """Emit selected blocks in ORIGINAL order; non-contiguous gaps are marked
    so the prompt never pretends the evidence was contiguous."""
    parts: list[str] = []
    omitted = 0
    i = 0
    while i < len(blocks):
        if i in selected:
            parts.append(blocks[i].text)
            i += 1
            continue
        j = i
        while j < len(blocks) and j not in selected:
            j += 1
        gap = j - i
        if parts and j < len(blocks):  # internal gap only (not head/tail)
            parts.append(_OMISSION_TEMPLATE.format(n=gap))
            omitted += gap
        elif gap:
            omitted += gap  # head/tail omissions counted, not marked
        i = j
    return "\n\n".join(parts), omitted


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Document dating (R1/R2, temporal-awareness remediation)
#
# metadata.date rides along on every RetrievedResult (pipeline.py:552,620) but
# historically reached NEITHER the prompt header NOR the UI. R1 renders it in
# the per-source header; R2 adds ONE temporal-conflict note when the rendered
# sources span more than a year, so 2008/2011/2015/2022/2026 duplicates of the
# same parliamentary answer no longer compete anonymously. Neither mechanism
# touches the SYSTEM_PROMPT — this is user-prompt furniture, same class as the
# document-type hints. Dates are RAW stamps (formats vary by ingest path:
# ISO, "2023-24", raw scrape strings, None); we only parse years for the note.
# ─────────────────────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def doc_signal_year(stamp: Optional[str]) -> Optional[int]:
    """Representative year of a raw date stamp: the MAX 4-digit year found.

    ISO stamps ("2026-07-29") yield their year; a FY stamp like "2023-24"
    yields 2023 (its coverage year); dotted parliament stamps ("29.07.2026")
    yield 2026; unparseable/empty stamps yield None (caller must degrade).
    """
    years = [int(m.group(1)) for m in _YEAR_RE.finditer(stamp or "")]
    return max(years) if years else None


def _temporal_note(results: list[RetrievedResult]) -> Optional[str]:
    """One-line conflict warning when rendered sources span > 1 year.

    Pure furniture: advisory wording, deterministic trigger, never raises.
    Fires only when >= 2 DISTINCT parseable years exist and span > 1 year —
    consecutive years (2025 vs 2026) and undated corpora stay silent.
    """
    years = sorted({
        y for r in results
        if (y := doc_signal_year((r.metadata or {}).get("date"))) is not None
    })
    if len(years) < 2 or years[-1] - years[0] <= 1:
        return None
    return (
        f"NOTE: These sources are dated {years[0]} to {years[-1]}. Where "
        f"figures conflict across them, prefer the newest dated source(s) "
        f"for current values; treat older sources as historical."
    )


# Rendering — ONE visual format shared by legacy and budgeted assembly
# ─────────────────────────────────────────────────────────────────────────────

_DOC_TYPE_HINTS = {
    "technical_report": "Type: INCOIS Technical Report \u2014 scientific methodology, model results, data. Prioritize figures, dates, and quantitative claims.",
    "annual_report": "Type: INCOIS Annual Report \u2014 year-in-review of activities. Cite the report year and section names where relevant.",
    "research_publication": "Type: Research publication \u2014 academic paper. Prioritize findings, dates, and data.",
    "general_report": "Type: INCOIS general report.",
    "parliamentary_qa": "Type: Parliamentary Q&A \u2014 verbatim question-answer record.",
    "audit_qa": "Type: Audit Q&A \u2014 verbatim question-answer record.",
    "document": "Type: Audit document.",
}


def _date_line(result: RetrievedResult) -> Optional[str]:
    """R1 header furniture: the raw date stamp, only when non-blank.
    Whitespace-only stamps would produce a phantom 'Date:   ' line."""
    stamp = (result.metadata.get("date") or "").strip()
    return f"Date: {stamp}" if stamp else None


def _source_header(result: RetrievedResult, index: int, q_text: str) -> str:
    """The per-source block BEFORE the answer body (budgeted separately from
    evidence). Must match the renderer's format exactly."""
    lines = [f"[Source {index}] (ID: {result.doc_id})"]
    if result.metadata.get("ministry"):
        lines.append(f"Ministry: {result.metadata['ministry']}")
    if result.metadata.get("subject"):
        lines.append(f"Subject: {result.metadata['subject']}")
    dl = _date_line(result)
    if dl:
        lines.append(dl)
    hint = _DOC_TYPE_HINTS.get((result.metadata.get("document_type") or "").lower())
    if hint:
        lines.append(hint)
    lines.append("")
    lines.append(f"QUESTION: {q_text}")
    lines.append("ANSWER: ")
    return "\n".join(lines)


def render_user_prompt(
    question: str,
    items: list[tuple[RetrievedResult, str, str]],
) -> str:
    """The canonical user-prompt format (was build_user_prompt's layout).

    ``items`` = (result, cleaned_question_text, final_answer_text) triples —
    evidence decision has already happened upstream; this function never
    truncates. Kept byte-compatible with the legacy prompt furniture so the
    SYSTEM_PROMPT contract ([Source N] citations) holds unchanged.
    """
    parts = [
        "Below is the most relevant parliamentary Question & Answer context retrieved for your question.",
        "",
        "=" * 70,
        f"RETRIEVED CONTEXT ({len(items)} records):",
        "=" * 70,
        "",
    ]

    # R2: temporal-conflict note — one furniture line when the rendered
    # sources span > 1 year (2008 vs 2026 must not compete anonymously).
    note = _temporal_note([r for r, _, _ in items])
    if note:
        parts.append(note)
        parts.append("")

    for i, (result, q_text, a_text) in enumerate(items, start=1):
        parts.append(f"[Source {i}] (ID: {result.doc_id})")
        if result.metadata.get("ministry"):
            parts.append(f"Ministry: {result.metadata['ministry']}")
        if result.metadata.get("subject"):
            parts.append(f"Subject: {result.metadata['subject']}")
        dl = _date_line(result)
        if dl:
            parts.append(dl)
        hint = _DOC_TYPE_HINTS.get((result.metadata.get("document_type") or "").lower())
        if hint:
            parts.append(hint)
        parts.append("")
        parts.append(f"QUESTION: {q_text}")
        parts.append(f"ANSWER: {a_text}")
        parts.append("")
        parts.append("-" * 70)

    parts.extend([
        "",
        "=" * 70,
        "USER QUESTION:",
        "=" * 70,
        question,
        "",
        "=" * 70,
        "ANSWER:",
        "=" * 70,
    ])

    return "\n".join(parts)


def assemble_budgeted_prompt(question: str, admissions: list[Admission]) -> str:
    """Render an Allocation for the wire."""
    items = [(a.result, a.question_text, a.evidence_text) for a in admissions]
    return render_user_prompt(question, items)


# ─────────────────────────────────────────────────────────────────────────────
# Deep-mode neighbor-chunk pull-in (retrieval-side enrichment)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_seq(chunk_id: str) -> Optional[tuple[str, int]]:
    """Parse ``{parent}_L{n}`` → (parent, n); None for other id forms."""
    head, sep, tail = (chunk_id or "").rpartition("_L")
    if not sep or not tail.isdigit():
        return None
    return head, int(tail)


def _neighbor_is_contextual(chunk_text: str) -> bool:
    """Only pull a neighbor that plausibly labels the matched chunk (short
    heading-ish lines), never a big body blob."""
    lines = [l.strip() for l in (chunk_text or "").split("\n") if l.strip()]
    if not lines or len(lines) > 3:
        return False
    if _is_heading_line(lines[0]):
        return True
    return all(len(l) <= 90 and not l.endswith(".") for l in lines)


def enrich_deep_neighbors(results: list[RetrievedResult], long_chunk_map) -> int:
    """DEEP profile only: when a result's evidence came from middle-of-document
    long chunks, pre-bond the IMMEDIATELY PRECEDING chunk if it is heading-like
    — the matched 500-char chunk then keeps its section label (metadata chain
    ``{parent}_L{n}`` already persisted in the index maps; no reindexing).

    Mutates ``r.answer`` by prefixing at most ONE neighbor chunk (dedup-safe);
    the downstream budget accounting simply sees the enlarged evidence, so the
    overall context budget is still enforced. Returns enrichment count."""
    added = 0
    for r in results:
        ids = (r.metadata or {}).get("chunk_ids") or []
        seqs = sorted(n for n in (_chunk_seq(cid) for cid in ids) if n)
        if not seqs:
            continue
        parent, n = seqs[0]
        if n <= 0:
            continue
        neighbor = getattr(long_chunk_map, "get", lambda _k: None)(f"{parent}_L{n - 1}")
        if neighbor is None:
            continue
        text = getattr(neighbor, "chunk_text", "") or ""
        if not text or text in (r.answer or ""):
            continue
        if _neighbor_is_contextual(text):
            r.answer = text.rstrip() + "\n" + (r.answer or "")
            added += 1
    return added


# ─────────────────────────────────────────────────────────────────────────────
# 413 self-heal aggressiveness (budget-conform rebuild replacing the legacy
# fixed 2-docs × 500-chars heuristic)
# ─────────────────────────────────────────────────────────────────────────────

AGGRESSIVE_HEAL_POOL = 2        # candidates
AGGRESSIVE_HEAL_TOKENS = 1500   # evidence budget cap for the retry
