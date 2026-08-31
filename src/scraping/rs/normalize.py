"""Rsdoc question-row → QARecord-shaped normalization (design §5 step 3).

Handles the upstream defects frozen by validation:

- ``qtype`` casing variants (``STARRED   `` / ``UNSTARRED  `` / ``UnStarred``)
- trailing whitespace in every string field (``"EARTH SCIENCES "``)
- HTML payloads in ``qn_text`` / ``ans_text`` (tags, ``<br/>``, doubly-escaped
  entities like ``&lt;br/&gt;`` seen in Ocean Development rows)
- ``qno`` as float (always integral — verified 866+103/866+103)
- dates duplicated as ``adate`` (ISO) and ``ans_date`` (dd.mm.yyyy)

Ministry is stamped PER RECORD from the configured ministry the row was
fetched under (decision D5) — never a global/folder default.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any

from src.scraping.rs.client import RsClient

_QTYPE_MAP = {"STARRED": "starred", "UNSTARRED": "unstarred"}


def strip_html(value: str | None) -> str:
    """HTML → plain text (tags, entities incl. double-escapes, line hygiene)."""
    if not value:
        return ""
    s = value
    for _ in range(2):  # handle "&lt;br/&gt;" double-escapes
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


def parse_date(row: dict[str, Any]) -> str | None:
    adate = (row.get("adate") or "").strip()
    if len(adate) >= 10:
        return adate[:10]
    ans_date = (row.get("ans_date") or "").strip()
    try:
        return datetime.strptime(ans_date, "%d.%m.%Y").date().isoformat()
    except ValueError:
        return None


def record_id(ses_no: int, qno: int) -> str:
    """Display id (decision D4): rs-<session>-<qno, zero-padded>."""
    return f"rs-{ses_no}-{qno:04d}"


def qno_of(row: dict[str, Any]) -> int:
    return int(float(row.get("qno")))


def member_name(row: dict[str, Any]) -> str:
    parts = [(row.get("shri") or "").strip(), (row.get("name") or "").strip()]
    return " ".join(p for p in parts if p)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_record(raw: dict[str, Any], ministry: dict[str, Any]) -> dict[str, Any]:
    """One rsdoc row → QARecord-shaped dict.

    ``answer_source``/``documents`` are filled later by the pipeline (after
    document download + optional PDF-extraction fallback). ``scraped_at`` is
    NOT set here; it is stamped at first serialization so unchanged re-runs
    stay byte-stable.
    """
    ses = int(row_get(raw, "ses_no"))
    qno = qno_of(raw)
    qslno = int(row_get(raw, "qslno"))
    return {
        "question_id": record_id(ses, qno),
        "question_text": strip_html(raw.get("qn_text")),
        "answer_text": strip_html(raw.get("ans_text")),
        "metadata": {
            "ministry": ministry["slug"],
            "ministry_label": ministry["label"],
            "ministry_code": int(ministry["code"]),
            "document_type": "parliamentary_qa",
            "org": "sansad",
            "source": "parliamentary-qa",
            "house": "rajya-sabha",
            "session": ses,
            "question_number": qno,
            "question_type": norm_qtype(raw.get("qtype")),
            "answer_status": "answered",
            "member": member_name(raw) or None,
            "mp_code": raw.get("mp_code"),
            "date": parse_date(raw),
            "qslno": qslno,
            "source_url": RsClient.record_permalink(qslno),
            "answer_source": None,
            "eng_doc_url": raw.get("files") or None,
            "hin_doc_url": raw.get("hindifiles") or None,
            "documents": {},
        },
    }


def row_get(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        raise KeyError(f"rsdoc row missing required field {key!r}: {row!r:.200}")
    return value


def sort_key(row: dict[str, Any]) -> tuple:
    meta = row.get("metadata") or {}
    return (meta.get("question_number") or 0, meta.get("qslno") or 0)
