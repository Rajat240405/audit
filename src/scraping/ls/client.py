"""sansad.in api_ls + elibrary.sansad.in DSpace client (frozen endpoint contracts).

Live-validated 2026-08-25 (see ls_scraper_plan/REPORT.md §3/§10.1):

- Per-record listing (THE discovery API, modern era):
  ``GET <api>/question/qetFilteredQuestionsAns?loksabhaNo={N}&ministryCode={C}
  &pageNo={1-based}&pageSize={n}`` — pagination is 1-based (pageNo=0 → 404);
  the response envelope is ``[{"totalRecordSize": M, "listOfQuestions": [...]}]``.
  A ``sessionNo`` query param is IGNORED upstream → callers group rows by the
  row's own ``sessionNo`` field.
- Session calendar: ``GET <api>/business/getAllLoksabhaAndSession?locale=en``
  → one entry per Lok Sabha with its session list (drives the crawl walk).
- Both api_ls endpoints need browser-identity headers (UA + Referer) — the
  framework client is constructed with them by the pipeline.
- Legacy-era inventory: ``GET <elib>/server/api/discover/search/objects
  ?f.ministry={LABEL},equals&f.loksabhanumber={N},equals&size={≤100}&page={0-based}``
  — anonymous 200, paginated via the ``page`` envelope block.

Values placed into query strings come from our own config — never from user
input — so both URL builders live in exactly one place here.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from src.scraping.http import CrawlHttpClient, HttpApiError  # noqa: F401  (re-export)


class LsClient:
    def __init__(
        self,
        http: CrawlHttpClient,
        *,
        api_base_url: str,
        api_page_size: int,
        elib_base_url: str,
        elib_page_size: int,
    ) -> None:
        self.http = http
        self.api_base = api_base_url.rstrip("/")
        self.api_page_size = max(1, int(api_page_size))
        self.elib_base = elib_base_url.rstrip("/")
        self.elib_page_size = max(1, int(elib_page_size))

    # ── api_ls: session calendar ────────────────────────────────────────────

    def loksabha_calendar(self) -> dict[int, list[int]]:
        """{loksabha_no: sorted session numbers} from the official calendar."""
        data = self.http.get_json(
            f"{self.api_base}/business/getAllLoksabhaAndSession?locale=en"
        )
        if not isinstance(data, list):
            raise HttpApiError("getAllLoksabhaAndSession: expected JSON list")
        out: dict[int, list[int]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            lok = _first_int(item, ("loksabhaNo", "lokSabhaNo", "loksabha", "lokNo"))
            if lok is None:
                continue
            sessions: set[int] = set()
            for s in item.get("sessions") or []:
                if not isinstance(s, dict):
                    continue
                ses = _first_int(s, ("sessionNo", "sessionNumber", "session"))
                if ses is not None:
                    sessions.add(ses)
            out[lok] = sorted(sessions)
        return out

    # ── api_ls: per-record listing (paged) ──────────────────────────────────

    def questions_page_url(self, loksabha: int, ministry_code: int, page_no: int) -> str:
        qs = urllib.parse.urlencode({
            "loksabhaNo": int(loksabha),
            "ministryCode": int(ministry_code),
            "pageNo": int(page_no),          # 1-based — 0 answers 404 (validated)
            "pageSize": self.api_page_size,
        })
        return f"{self.api_base}/question/qetFilteredQuestionsAns?{qs}"

    def questions_for_loksabha(self, loksabha: int, ministry_code: int) -> list[dict]:
        """ALL listing rows for one (loksabha, ministry-code), paging until the
        upstream-declared ``totalRecordSize`` is collected."""
        rows: list[dict] = []
        page_no = 1
        total: int | None = None
        while True:
            data = self.http.get_json(
                self.questions_page_url(loksabha, ministry_code, page_no)
            )
            envelope = _question_envelope(data)
            batch = [x for x in envelope["listOfQuestions"] if isinstance(x, dict)]
            total = envelope["totalRecordSize"]
            rows.extend(batch)
            if not batch or len(rows) >= total:
                return rows
            page_no += 1
            if page_no > 10000:  # termination guard: upstream change/loop
                raise HttpApiError(
                    f"qetFilteredQuestionsAns: aborted after 10000 pages "
                    f"(loksabha={loksabha}, ministry={ministry_code})"
                )

    # ── elibrary DSpace: discover search (paged) ────────────────────────────

    def dspace_search_page_url(self, loksabha: int, ministry_label: str, page: int) -> str:
        qs = urllib.parse.urlencode({
            "f.ministry": f"{ministry_label},equals",
            "f.loksabhanumber": f"{int(loksabha)},equals",
            "size": self.elib_page_size,
            "page": int(page),               # 0-based (validated)
        })
        return f"{self.elib_base}/server/api/discover/search/objects?{qs}"

    def dspace_search(self, loksabha: int, ministry_label: str) -> list[dict]:
        """All item metadata dicts for one (loksabha, ministry-facet-label)."""
        items: list[dict] = []
        page = 0
        total_pages: int | None = None
        while True:
            data = self.http.get_json(
                self.dspace_search_page_url(loksabha, ministry_label, page)
            )
            if not isinstance(data, dict):
                raise HttpApiError("discover/search/objects: expected JSON object")
            search_result = (data.get("_embedded") or {}).get("searchResult") or {}
            objects = ((search_result.get("_embedded") or {}).get("objects")) or []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                md = ((obj.get("_embedded") or {}).get("indexableObject") or {}) \
                    .get("metadata")
                if isinstance(md, dict):
                    items.append(md)
            pageinfo = search_result.get("page") or {}
            try:
                total_pages = int(pageinfo.get("totalPages"))
            except (TypeError, ValueError):
                raise HttpApiError(
                    "discover/search/objects: missing page.totalPages envelope"
                ) from None
            page += 1
            if not objects or page >= max(1, total_pages):
                return items


def _first_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    """First recognized key parsed as an int (tolerant of float/str payloads)."""
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            continue
    return None


def _question_envelope(data: Any) -> dict[str, Any]:
    """Normalize the qetFilteredQuestionsAns envelope to
    {totalRecordSize: int, listOfQuestions: list} (fail-closed)."""
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise HttpApiError("qetFilteredQuestionsAns: unexpected envelope shape")
    questions = data.get("listOfQuestions")
    if not isinstance(questions, list):
        raise HttpApiError("qetFilteredQuestionsAns: missing listOfQuestions")
    try:
        total = int(data.get("totalRecordSize"))
    except (TypeError, ValueError):
        raise HttpApiError(
            "qetFilteredQuestionsAns: missing totalRecordSize"
        ) from None
    return {"totalRecordSize": total, "listOfQuestions": questions}
