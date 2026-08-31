"""rsdoc.nic.in API client (frozen endpoint contracts, design §3).

- Sessions:    GET <base>/question/Get_sessionforQuestionSearch
- Ministries:  GET <base>/question/GetAllministary           (sic: upstream spelling)
- Questions:   GET <base>/Question/Search_Questions?whereclause=<clause>

The whereclause grammar is restricted to the verified predicate fields
(``ses_no``, ``min_code``). Values come from our own config — never from
user input — so the SQL-shaped clause is built in exactly one place here.
httpx percent-encodes the query string (spaces etc.); the server accepts
that (verified live 2026-08-23).
"""

from __future__ import annotations

from src.scraping.http import CrawlHttpClient, HttpApiError  # noqa: F401  (re-export)


class RsClient:
    def __init__(self, http: CrawlHttpClient, base_url: str = "https://rsdoc.nic.in") -> None:
        self.http = http
        self.base = base_url.rstrip("/")

    def sessions(self) -> list[int]:
        """All session numbers listed by the server (e.g. 174..271)."""
        data = self.http.get_json(f"{self.base}/question/Get_sessionforQuestionSearch")
        if not isinstance(data, list):
            raise HttpApiError("Get_sessionforQuestionSearch: expected JSON list")
        return sorted(int(x["ssn_no"]) for x in data if isinstance(x, dict) and x.get("ssn_no"))

    def ministries(self) -> list[dict]:
        data = self.http.get_json(f"{self.base}/question/GetAllministary")
        if not isinstance(data, list):
            raise HttpApiError("GetAllministary: expected JSON list")
        return data

    def search(self, *, ses_no: int, min_code: int | str) -> list[dict]:
        """All question rows for one (session, ministry-code).

        Per-session-per-ministry queries are unpaginated and small (worst
        observed MoES session: 34 rows); the API offers no paging params.
        Returns [] when the session has no rows for the ministry (verified:
        empty JSON list, HTTP 200).
        """
        clause = f"ses_no={int(ses_no)} and min_code='{min_code}'"
        url = f"{self.base}/Question/Search_Questions?whereclause={clause}"
        data = self.http.get_json(url)
        if not isinstance(data, list):
            raise HttpApiError(f"Search_Questions: expected JSON list for {clause!r}")
        return [x for x in data if isinstance(x, dict)]

    @staticmethod
    def record_permalink(qslno: int) -> str:
        """Stable one-row API query for a record (verified to return exactly 1 row)."""
        return f"https://rsdoc.nic.in/Question/Search_Questions?whereclause=qslno={int(qslno)}"
