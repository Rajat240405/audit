"""Scope rules for INCOIS tender documents (internal category name: ``budget``).

``budget`` is an INTERNAL CATEGORY NAME ONLY. These are tender documents —
notices inviting tender (NIT), RFPs and EoIs published on
``https://incois.gov.in/site/tenders.jsp``. Nothing here claims they are
budget/expenditure records.

This module is the single source of truth for three rules that would
otherwise be duplicated across the crawler, the extractor and the ingest
path, drifting apart over time:

1. **Which URLs are in scope** — only the *main* tender document,
   ``/documents/Tenders/tender_<digits>.pdf``. Corrigenda, pre-bid minutes and
   portal logos live in sibling sub-directories and are deliberately excluded.
2. **What a document's canonical name is** — so a manually supplied copy and
   a scraped copy of the same tender do not both land on disk.
3. **Whether a path belongs to the budget category** — which selects the
   DOTS-only extraction route.

Why the URL pattern is safe
---------------------------
``tender_`` must follow ``/documents/Tenders/`` *immediately*, so the
sub-directory forms cannot match::

    /documents/Tenders/tender_20260902153951.pdf                 MATCH
    /documents/Tenders/Corrigendum/corrigendum_96_2026...pdf     no match
    /documents/Tenders/PreBid/prebid_64_20260617120209.pdf       no match
    /documents/Tenders/logos/GeMPortal.png                       no match (.pdf anchor)

Observed naming is ``tender_<YYYYMMDDHHMMSS>.pdf``. The digit run is matched
as ``{8,}`` rather than exactly 14 so that a future INCOIS change to a plain
``YYYYMMDD`` stamp still gets collected — under an append-only policy, silently
skipping a new document is worse than collecting one extra.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── category identity ────────────────────────────────────────────────────
BUDGET_LABEL = "budget"
BUDGET_FOLDER = "incois_reports/budget"
BUDGET_DOC_TYPE = "tender"
TENDERS_PAGE = "/site/tenders.jsp"

# ── scope pattern ────────────────────────────────────────────────────────
# Consumed verbatim by crawl_incois_reports.SECTIONS["budget"]["pat"], so the
# crawler and every test assert against the same expression.
TENDER_URL_PATTERN = r"/documents/Tenders/tender_[0-9]{8,}\.pdf"

_TENDER_URL_RE = re.compile(TENDER_URL_PATTERN, re.IGNORECASE)
_TENDER_NAME_RE = re.compile(r"(tender_[0-9]{8,})", re.IGNORECASE)


def is_tender_url(url: str) -> bool:
    """Is this URL a main tender PDF (not a corrigendum / pre-bid / logo)?"""
    return bool(_TENDER_URL_RE.search(url or ""))


def canonical_tender_filename(name: str) -> str | None:
    """Canonical ``tender_<stamp>.pdf`` for a file, or None if out of scope.

    Normalises the real-world variants that would otherwise create duplicate
    copies of one tender — most importantly the ``" (1)"`` suffix a browser or
    Google Drive appends when the same file is downloaded twice::

        "tender_20260908191745 (1).pdf"  -> "tender_20260908191745.pdf"
        "TENDER_20260908191745.PDF"      -> "tender_20260908191745.pdf"
        "corrigendum_96_2026...pdf"      -> None
    """
    stem = Path(name or "").name
    if not stem.lower().endswith(".pdf"):
        return None
    # Reject sub-directory document families outright: a corrigendum filename
    # contains "tender" nowhere, but guard explicitly so a renamed copy of one
    # cannot sneak in through the substring search below.
    low = stem.lower()
    if low.startswith(("corrigendum_", "prebid_", "pre_bid_")):
        return None
    m = _TENDER_NAME_RE.search(stem)
    if not m:
        return None
    return f"{m.group(1).lower()}.pdf"


def is_budget_path(path: str | Path) -> bool:
    """Does this file live in the budget tender collection?

    Matches the *consecutive* parts ``incois_reports/budget`` so an unrelated
    directory that merely happens to be called ``budget`` does not silently
    acquire the DOTS-only extraction route.
    """
    parts = [p.lower() for p in Path(path).parts]
    for i in range(len(parts) - 1):
        if parts[i] == "incois_reports" and parts[i + 1] == BUDGET_LABEL:
            return True
    return False
