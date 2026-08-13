"""
Ministry -> org hierarchy + metadata derivation helpers.

Single source of truth for the source-filter tree (design doc:
HPC_MULTI_MINISTRY_DOC_HANDLING.md, section 2). Kept as a Python dict (not
YAML) so there is no runtime dependency and it is typed/importable everywhere.

TREE RULE
---------
- Selecting a MINISTRY includes ALL orgs under it (like a folder tree).
- Org slugs are globally unique across ministries, so the retrieval filter
  only ever needs a FLAT set of org slugs -> no hierarchical logic in the
  pipeline, no index changes.
- "sansad" is a special top-level source: parliamentary Q&A is a separate
  bucket (questions ABOUT ministries, not documents PRODUCED by them).

DERIVATION
----------
Existing records may lack an `org` / `doc_category` field. derive_org() /
derive_category() fill the gap deterministically from metadata already present
(document_type, subject, source_url, session, ...) so the WHOLE corpus becomes
filterable WITHOUT a data rewrite or index rebuild. Backfill scripts can later
stamp explicit values; the derivation stays as a safe fallback.
"""

from __future__ import annotations

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# The tree. Adding a ministry / org = editing this dict only.
#   categories: doc_category values that org can produce (used for UI hints)
# ─────────────────────────────────────────────────────────────────────────────

ORG_TREE: dict[str, dict] = {
    "sansad": {
        "name": "Sansad (Parliament Q&A)",
        "orgs": [
            {"slug": "sansad", "name": "Parliamentary Questions", "categories": ["parliamentary"]},
        ],
    },
    "moes": {
        "name": "Ministry of Earth Sciences",
        "orgs": [
            {"slug": "incois", "name": "INCOIS", "categories": ["annual", "scientific", "technical", "general"]},
            {"slug": "imd", "name": "IMD", "categories": ["annual", "monthly", "scientific"]},
            {"slug": "iitm", "name": "IITM", "categories": ["annual", "scientific"]},
            {"slug": "niot", "name": "NIOT", "categories": ["annual", "research"]},
            {"slug": "moes_hq", "name": "MoES HQ", "categories": ["annual", "monthly", "budget", "policy", "gazette", "news", "scientific", "misc"]},
        ],
    },
    "moa": {
        "name": "Ministry of Agriculture & Farmers Welfare",
        "orgs": [
            {"slug": "moa_hq", "name": "MoA HQ", "categories": ["annual", "monthly", "budget", "policy"]},
        ],
    },
    "mof": {
        "name": "Ministry of Finance",
        "orgs": [
            {"slug": "mof_hq", "name": "MoF HQ", "categories": ["annual", "budget", "policy"]},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# document_type -> doc_category mapping (the cadence axis: annual / monthly /
# budget / parliamentary / ...). Unknown types fall back to "misc".
# ─────────────────────────────────────────────────────────────────────────────

_DT_CATEGORY: dict[str, str] = {
    "parliamentary_qa": "parliamentary",
    "annual_report": "annual",
    "monthly_report": "monthly",
    "research_publication": "scientific",
    "publication": "scientific",
    "bibliometrics": "scientific",
    "technical_report": "technical",
    "general_report": "general",
    "demands_for_grants": "budget",
    "performance_budget": "budget",
    "gazette_notification": "gazette",
    "order_notice": "gazette",
    "newsletter": "news",
    "download": "misc",
    "document": "misc",
}

# Org tokens recognized in subject/source_url text (longest first).
_ORG_SIGNALS: list[tuple[str, str]] = [
    ("INDIA METEOROLOGICAL", "imd"),
    ("/IMD/", "imd"),
    ("INCOIS", "incois"),
    ("INDIAN NATIONAL CENTRE FOR OCEAN INFORMATION SERVICES", "incois"),
]


def derive_org(meta: Optional[dict]) -> str:
    """Determine which org produced a record, given its metadata dict."""
    if not meta:
        return "moes_hq"
    explicit = meta.get("org")
    if explicit:
        return str(explicit)

    dt = str(meta.get("document_type") or "").lower()
    if dt == "parliamentary_qa":
        return "sansad"

    # Parliamentary records always carry session / question_number / member.
    if (
        meta.get("session") is not None
        or meta.get("question_number") is not None
        or meta.get("member") is not None
    ):
        return "sansad"

    # IDENTITY signals only — org = WHO PRODUCED the doc, not who's mentioned
    # in its content. answer_text is deliberately EXCLUDED: INCOIS annual
    # reports repeatedly mention "India Meteorological Department", which
    # would misclassify them as IMD.
    blob = " ".join(
        str(meta.get(k) or "") for k in (
            "subject", "source_url", "collection", "question_text", "title",
        )
    ).upper()
    for token, slug in _ORG_SIGNALS:
        if token in blob:
            return slug

    return "moes_hq"


def derive_category(meta: Optional[dict]) -> str:
    """Determine doc_category (cadence axis) from metadata (fallback mapping)."""
    if not meta:
        return "misc"
    explicit = meta.get("doc_category")
    if explicit:
        return str(explicit)
    dt = str(meta.get("document_type") or "").lower()
    return _DT_CATEGORY.get(dt, "misc")


def resolve_orgs(ministry: Optional[str] = None, orgs: Optional[list[str]] = None) -> set[str]:
    """
    Tree-rule expansion: ministry -> all its orgs; explicit orgs are unioned.
    Returns an EMPTY set when no filter applies (means: no org restriction).
    """
    selected: set[str] = set()
    if ministry and ministry != "all":
        m = ORG_TREE.get(ministry)
        if m:
            selected.update(o["slug"] for o in m["orgs"])
    if orgs:
        selected.update(orgs)
    return selected
