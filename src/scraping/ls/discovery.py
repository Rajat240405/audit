"""Raw Lok Sabha question model + era-routed inventory building.

One dataclass — :class:`RawLsQuestion` — is the interchange between the two
eras and normalization/documents:

- **api_ls rows** (modern era, LS ≥ boundary): carry quesNo/sessionNo/type/
  date/member/subjects + annex document links; ministry routed per row via
  ``row_labels`` (one api ministry code spans ministry renames across eras).
  Rows claiming no configured ministry are counted as out-of-scope and
  skipped (honest, never silently re-labelled).
- **DSpace metadata** (legacy era, LS ≤ 15): ``dc.identifier.*`` fields;
  ``sessionnumber`` is ROMAN upstream (strict subtractive parsing — IV/IX/
  XIV/XIX pinned by tests); item metadata has NO member name (kept None —
  never fabricated); the ``dc.identifier.uri`` handle is both source_url and
  the eng document slot.

Dedupe key is (loksabha, session, ques_no) — the frozen workbook contains
exactly one duplicated trio (lok 15 / session 6 / quesNo 320), so the guard
is real, not theoretical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.scraping.ls.config import ERA_API_LS, LsConfigError, route_ministry

# ── roman numerals (DSpace sessionnumber era) ────────────────────────────────

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_ROMAN_RE = re.compile(r"^[IVXLC]+$")


def roman_to_int(value: str | None) -> int | None:
    """Strict subtractive roman→int ("VI"→6, "VIII"→8, "IV"→4, "IX"→9,
    "XIV"→14, "XIX"→19). Returns None for unparseable input — never guesses.
    Arabic strings pass through as their value (defensive tolerance)."""
    s = (value or "").strip().upper()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if not _ROMAN_RE.match(s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN_VALUES[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total or None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _first_metadata(md: dict[str, Any], key: str) -> str | None:
    """First value of a DSpace metadata field (values are [{value: ...}])."""
    values = md.get(key)
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, dict):
            return _text(first.get("value"))
        return _text(first)
    return None


# ── the raw interchange model ────────────────────────────────────────────────

@dataclass
class RawLsQuestion:
    loksabha: int
    session: int
    ques_no: int
    ministry_slug: str
    source: str                          # "api_ls" | "dspace"
    qtype_raw: str | None = None
    date_raw: str | None = None
    members: list[str] = field(default_factory=list)
    subjects: str | None = None
    eng_url: str | None = None
    hin_url: str | None = None
    source_url: str | None = None        # dspace handle; None for api_ls rows
    question_text: str | None = None     # inline upstream text (rare; legacy api rows)
    answer_text: str | None = None


@dataclass
class Inventory:
    """Per-loksabha discovery result: session → routed raw questions."""

    loksabha: int
    era: str
    sessions: dict[int, list[RawLsQuestion]] = field(default_factory=dict)
    ministries: dict[str, int] = field(default_factory=dict)     # slug → rows kept
    out_of_scope: list[dict[str, Any]] = field(default_factory=list)

    def add(self, q: RawLsQuestion) -> None:
        self.sessions.setdefault(q.session, []).append(q)
        self.ministries[q.ministry_slug] = self.ministries.get(q.ministry_slug, 0) + 1

    def sorted_rows(self, session: int) -> list[RawLsQuestion]:
        return sorted(self.sessions.get(session, []),
                      key=lambda q: (q.ques_no, q.ministry_slug))


def from_api_row(
    row: dict[str, Any], ministries: list[dict[str, Any]], loksabha: int
) -> RawLsQuestion | None:
    """One qetFilteredQuestionsAns row → RawLsQuestion; None when out of scope
    or missing its identity fields (never guessed)."""
    qno = _int(row.get("quesNo"))
    ses = _int(row.get("sessionNo"))
    if qno is None or ses is None:
        return None
    ministry = route_ministry(ministries, row.get("ministry"))
    if ministry is None:
        return None
    members: list[str] = []
    raw_members = row.get("member")
    if isinstance(raw_members, list):
        members = [m for m in (_text(x) for x in raw_members) if m]
    elif _text(raw_members):
        members = [_text(raw_members)]
    return RawLsQuestion(
        loksabha=loksabha,
        session=ses,
        ques_no=qno,
        ministry_slug=ministry["slug"],
        source=ERA_API_LS,
        qtype_raw=row.get("type"),
        date_raw=row.get("date"),
        members=members,
        subjects=_text(row.get("subjects")) or None,
        eng_url=_text(row.get("questionsFilePath")) or None,
        hin_url=_text(row.get("questionsFilePathHindi")) or None,
        source_url=None,
        question_text=row.get("questionText") or None,
        answer_text=row.get("answerText") or None,
    )


def from_dspace_metadata(
    md: dict[str, Any], ministry: dict[str, Any], loksabha: int
) -> RawLsQuestion | None:
    """One DSpace item metadata dict → RawLsQuestion; None when identity
    fields fail to parse (counted by the caller — never fabricated)."""
    handle = _first_metadata(md, "dc.identifier.uri")
    qno = _int(_first_metadata(md, "dc.identifier.questionnumber"))
    ses = roman_to_int(_first_metadata(md, "dc.identifier.sessionnumber"))
    if qno is None or ses is None or not handle:
        return None
    return RawLsQuestion(
        loksabha=loksabha,
        session=ses,
        ques_no=qno,
        ministry_slug=ministry["slug"],
        source="dspace",
        qtype_raw=_first_metadata(md, "dc.identifier.questiontype"),
        date_raw=_first_metadata(md, "dc.date.issued"),
        members=[],                         # not exposed anonymously — never fabricate
        subjects=_first_metadata(md, "dc.title") or None,
        eng_url=handle,
        hin_url=None,                       # no per-language handles exposed
        source_url=handle,
    )


def build_inventory_api(
    client, loksabha: int, ministries: list[dict[str, Any]], era: str
) -> Inventory:
    """api_ls era inventory: one paged walk per ministry code; rows routed by
    their own ministry label (codes span renames)."""
    inv = Inventory(loksabha=loksabha, era=era)
    seen_codes: set[int] = set()
    for mcfg in ministries:
        code = int(mcfg["api_ministry_code"])
        if code in seen_codes:
            continue  # renamed-era ministries share one code — walk it once
        seen_codes.add(code)
        rows = client.questions_for_loksabha(loksabha, code)
        for raw in rows:
            q = from_api_row(raw, ministries, loksabha)
            if q is None:
                inv.out_of_scope.append({
                    "loksabha": loksabha,
                    "ministry_label": _text(raw.get("ministry")) or None,
                    "quesNo": raw.get("quesNo"),
                    "reason": "no configured ministry claims this row label",
                })
                continue
            inv.add(q)
    return inv


def build_inventory_dspace(
    client, loksabha: int, ministries: list[dict[str, Any]], era: str
) -> Inventory:
    """DSpace era inventory: one paged discover search per (ministry ×
    elibrary facet label); items converted from their metadata."""
    inv = Inventory(loksabha=loksabha, era=era)
    for mcfg in ministries:
        for label in mcfg["elibrary_labels"]:
            label = str(label)
            if not label.strip():
                continue
            for md in client.dspace_search(loksabha, label):
                q = from_dspace_metadata(md, mcfg, loksabha)
                if q is None:
                    inv.out_of_scope.append({
                        "loksabha": loksabha,
                        "ministry_label": label,
                        "handle": _first_metadata(md, "dc.identifier.uri"),
                        "reason": "unparseable dspace metadata "
                                  "(handle/questionnumber/sessionnumber)",
                    })
                    continue
                inv.add(q)
    return inv


def dedupe_rows(rows: list[RawLsQuestion]) -> list[RawLsQuestion]:
    """First occurrence wins on (loksabha, session, ques_no) — deterministic.

    Guards the one true duplicate in the frozen workbook (lok 15/6/320) and
    any upstream re-listing; duplicates are dropped, never merged.
    """
    seen: set[tuple[int, int, int]] = set()
    out: list[RawLsQuestion] = []
    for q in rows:
        key = (q.loksabha, q.session, q.ques_no)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def build_inventory(client, loksabha: int, ministries: list[dict[str, Any]],
                    era: str) -> Inventory:
    """Era-routed inventory entry point (config-driven, never guessed)."""
    if era == ERA_API_LS:
        return build_inventory_api(client, loksabha, ministries, era)
    if era == "dspace":
        return build_inventory_dspace(client, loksabha, ministries, era)
    raise LsConfigError(f"unknown discovery era: {era!r}")
