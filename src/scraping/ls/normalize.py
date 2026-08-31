"""RawLsQuestion → QARecord-shaped normalization (LS vocabulary).

Mirrors ``src/scraping/rs/normalize.py`` conventions (house packages stay
self-contained by precedent — the small shared helpers are re-declared here,
not imported across houses):

- ``question_id`` = ``ls-<lok>-<ses>-<qno:04d>`` (RS-style prefixed namespace;
  never co-lives with the legacy phase-1 ``<lok>-<ses>-<qno>`` ids — the
  cutover is a one-step source swap, see REPORT.md §7).
- qtype whitespace/casing variants (``"UNSTARRED "`` …) → lowercase starred/
  unstarred, like RS ``norm_qtype``.
- dates: ``dd.mm.yyyy`` (api_ls) and ISO datetimes (DSpace
  ``dc.date.issued``) normalize to ``YYYY-MM-DD``.
- ministry stamped PER RECORD from the configured ministry the row routed to
  (never a folder default — same decision as RS D5).
- inline ``questionText``/``answerText`` pass through HTML-stripped (populated
  only by legacy api_ls rows; modern rows are null — extraction fills from
  the official documents, and never overwrites upstream inline text).
- ``scraped_at`` NOT stamped here (merge-by-id keeps re-runs byte-stable).
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from typing import Any

from src.scraping.ls.discovery import RawLsQuestion

_QTYPE_MAP = {"STARRED": "starred", "UNSTARRED": "unstarred"}

HOUSE = "lok-sabha"


def strip_html(value: str | None) -> str:
    """HTML → plain text (tags, entities incl. double-escapes, line hygiene)."""
    if not value:
        return ""
    s = value
    for _ in range(2):
        unescaped = html.unescape(s)
        if unescaped == s:
            break
        s = unescaped
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[^\S\n]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def norm_qtype(raw: str | None) -> str:
    return _QTYPE_MAP.get((raw or "").strip().upper(), "unknown")


def parse_date(raw: str | None) -> str | None:
    """'dd.mm.yyyy' | 'dd/mm/yyyy' | ISO datetime → 'YYYY-MM-DD' (or None)."""
    s = (raw or "").strip()
    if not s:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    for fmt in ("%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def record_id(loksabha: int, session: int, qno: int) -> str:
    """Display id: ls-<loksabha>-<session>-<qno, zero-padded>."""
    return f"ls-{loksabha}-{session}-{qno:04d}"


def member_name(q: RawLsQuestion) -> str | None:
    return "; ".join(q.members) if q.members else None


def utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_record(q: RawLsQuestion, ministry: dict[str, Any]) -> dict[str, Any]:
    """One RawLsQuestion → QARecord-shaped dict.

    ``answer_source``/``documents`` are filled later by the pipeline (after
    document download + extraction); ``question_text``/``answer_text`` may
    still be empty here — that is the normal LS case (the frozen workbook's
    text columns are entirely empty; content comes from the documents).
    """
    return {
        "question_id": record_id(q.loksabha, q.session, q.ques_no),
        "question_text": strip_html(q.question_text),
        "answer_text": strip_html(q.answer_text),
        "metadata": {
            "ministry": ministry["slug"],
            "ministry_label": ministry["label"],
            "ministry_code": int(ministry["api_ministry_code"]),
            "document_type": "parliamentary_qa",
            "org": "sansad",
            "source": "parliamentary-qa",
            "house": HOUSE,
            "loksabha": int(q.loksabha),
            "session": int(q.session),
            "question_number": int(q.ques_no),
            "question_type": norm_qtype(q.qtype_raw),
            "answer_status": "answered",
            "subject": q.subjects,
            "member": member_name(q),
            "date": parse_date(q.date_raw),
            "record_source": q.source,               # api_ls | dspace
            "source_url": q.source_url,               # dspace handle; None api era
            "answer_source": None,
            "eng_doc_url": q.eng_url,
            "hin_doc_url": q.hin_url,
            "documents": {},
        },
    }


def sort_key(row: dict[str, Any]) -> tuple:
    meta = row.get("metadata") or {}
    return (meta.get("question_number") or 0, row.get("question_id") or "")
