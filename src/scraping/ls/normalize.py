"""Listing JSON → normalized MoES records (real API shapes, live-verified).

Scope is derived FROM THE LIVE TAXONOMY, never hardcoded: reports families
(annual-reports / monthly-reports / demands-for-grants) are resolved by
config-declared substring patterns against the child terms of the `reports`
parent. A family matching zero terms (upstream rename) fails closed instead
of silently widening scope. `press-release` is a whole-category scope (no
family subdivision): every press-release post is normalized as-is.

`central_documents` (underscore) is NOT a category: it is the backend's
internal attachment post_type. It appears only in the attachment-resolution
path (client/documents) and is never normalized or scoped here.
"""

from __future__ import annotations

import re
from typing import Any

from src.scraping.moes.config import MoesConfigError

#: titles like "PARLIAMENT QUESTION: MISSION MAUSAM" (stats only — such
#: documents are NEVER excluded; cross-source dedupe is a later concern)
#: singular AND plural observed live ("PARLIAMENT QUESTIONS: MAJOR
#: IMPROVEMENTS…", post 28112) — accepted via `questions?`
PQ_TITLE_RE = re.compile(r"^\s*parliament\s+questions?\b", re.IGNORECASE)

REPORTS_PARENT_SLUG = "reports"

# ── Demand for Grants document family ────────────────────────────────────────
# DfG is a distinct MoES document family with its own extraction pipeline
# (src/data/extract_dfg.py). Two things are decided HERE, before any bytes are
# downloaded, so the policy is cheap and auditable:
#
#   * ``extraction_profile`` routes the document to the DfG extractor at
#     ingest time instead of the generic pdf_table_extract path.
#   * ``skip_slots`` drops the ``pdf_hindi`` slot before download. DfG PDFs are
#     bilingual *inside one file* (all 5 live docs carry English AND Devanagari
#     in the same PDF, including the one published on the ``-eng`` slot), so the
#     Hindi-only slot is redundant. ``pdf_both`` and ``pdf``/eng are NEVER
#     skipped — a bilingual document is not purged just because Hindi exists.
#
# Mirrored in config/crawlers/moes_website.yaml under
# categories.reports.families.demands-for-grants; a test asserts the two agree
# so the declared and enforced policy cannot drift.
DFG_PROFILE = "demands-for-grants"
DFG_SKIP_SLOTS: tuple[str, ...] = ("hin",)
#: live taxonomy term for "Demand for Grants" (child of reports/107)
DFG_TERM_ID = 385

DFG_TITLE_RE = re.compile(r"^\s*(?:detailed\s+)?demands?\s+for\s+grants?\b",
                          re.IGNORECASE)
DFG_SLUG_RE = re.compile(r"(?:^|[-_])demands?[-_]for[-_]grants?(?:[-_]|$)",
                         re.IGNORECASE)


def is_dfg_post(
    title: str | None = None,
    slug: str | None = None,
    category_slugs: list[str] | None = None,
    category_term_ids: list[int] | None = None,
) -> bool:
    """True when a post belongs to the Demand for Grants family.

    Three independent signals, any one sufficient — the live post 24108 is
    titled "Demands for Grants", slugged ``demands-for-grants``, and carries
    taxonomy term 385. Content-based on purpose: it must work before the
    attachment (and therefore the document) is fetched.
    """
    if title and DFG_TITLE_RE.match(title):
        return True
    if slug and DFG_SLUG_RE.search(slug):
        return True
    for s in category_slugs or []:
        if DFG_SLUG_RE.search(str(s) or ""):
            return True
    return DFG_TERM_ID in {int(t) for t in (category_term_ids or []) if t}


def is_parliament_question(title: str | None) -> bool:
    return bool(PQ_TITLE_RE.match(title or ""))


def parent_term_id(terms: list[dict[str, Any]], slug: str) -> int:
    tid = top_term_id(terms, slug)
    if tid is None:
        raise MoesConfigError(
            f"taxonomy has no top-level term {slug!r} (upstream schema change?)")
    return tid


def top_term_id(terms: list[dict[str, Any]], slug: str) -> int | None:
    """Non-raising top-level parent lookup; None when absent."""
    for t in terms:
        if t.get("slug") == slug and int(t.get("parent") or 0) == 0:
            return int(t["term_id"])
    return None


def child_terms_of(post: dict[str, Any], parent_id: int | None) -> list[dict[str, Any]]:
    """A post's child-category terms (terms whose parent == the category
    parent). None parent (category absent from the tree) → []."""
    out = []
    for t in post.get("documents_category") or []:
        if isinstance(t, dict) and int(t.get("parent") or 0) == parent_id:
            out.append(t)
    return sorted(out, key=lambda t: int(t.get("term_id") or 0))


def resolve_family_terms(
    terms: list[dict[str, Any]],
    families_cfg: dict[str, Any],
) -> dict[str, list[int]]:
    """Map each configured reports family → matching live child-term ids.

    Fail-closed: zero matches for a family, or one term matching two
    families, is a config/scope error (never silently widened/drifted).
    """
    rid = parent_term_id(terms, REPORTS_PARENT_SLUG)
    children = [t for t in terms if int(t.get("parent") or 0) == rid]
    resolved: dict[str, list[int]] = {}
    owner: dict[int, str] = {}
    for family, fcfg in families_cfg.items():
        patterns = [str(m).strip().lower() for m in (fcfg or {}).get("match") or []]
        matched = [
            t for t in children
            if any(p and (p in str(t.get("slug", "")).lower()
                          or p in str(t.get("name", "")).lower()) for p in patterns)
        ]
        if not matched:
            raise MoesConfigError(
                f"reports family {family!r} matched ZERO live taxonomy terms "
                f"(patterns {patterns}); upstream renamed the tree — refusing to guess"
            )
        for t in matched:
            tid = int(t["term_id"])
            if tid in owner and owner[tid] != family:
                raise MoesConfigError(
                    f"taxonomy term {t.get('slug')!r} matches two families "
                    f"({owner[tid]!r}, {family!r}) — ambiguous scope"
                )
            owner[tid] = family
        resolved[family] = sorted(int(t["term_id"]) for t in matched)
    return resolved


def post_family(child_ids: list[int], family_terms: dict[str, list[int]]) -> str | None:
    """Family for a post's child-term ids (config order); None = scoped out."""
    wanted = set(child_ids)
    for family, tids in family_terms.items():
        if wanted.intersection(tids):
            return family
    return None


def normalize_file_rows(post: dict[str, Any]) -> list[dict[str, Any]]:
    """ACF `file` rows → normalized slots (attachment | external | empty)."""
    rows = []
    for i, row in enumerate((post.get("acf_data") or {}).get("file") or []):
        row = row if isinstance(row, dict) else {}
        fid = row.get("file")
        attachment_id = None
        if isinstance(fid, list) and fid:
            try:
                attachment_id = int(fid[0])
            except (TypeError, ValueError):
                attachment_id = None
        elif isinstance(fid, int):
            attachment_id = fid
        external = str(row.get("external_link") or "").strip() or None
        rows.append({
            "row": i,
            "type": (str(row.get("type")).strip() if row.get("type") else None),
            "title": str(row.get("title") or "").strip(),
            "attachment_id": attachment_id,
            "external_url": external,
            "content_ref": bool(row.get("content")),  # observed empty in v1 census
        })
    return rows


def normalize_post(
    post: dict[str, Any],
    *,
    category: str,
    family: str | None,
    child_terms: list[dict[str, Any]],
    listing_url: str,
    scraped_at: str,
) -> dict[str, Any]:
    """One `documents` post → one deterministic record (manifest/hash material).

    Volatile ``scraped_at`` is excluded from ``records.row_sha256`` by the
    frozen records module (its VOLATILE_KEYS), so re-runs are byte-stable.
    """
    acf = post.get("acf_data") or {}
    title = str(acf.get("title") or post.get("post_title") or "").strip()
    post_date = str(post.get("post_date") or "")
    description = None
    for key in ("post_content", "post_excerpt"):
        val = str(post.get(key) or "").strip()
        if val:
            description = val
            break
    category_path = [category] + ([family] if family else [])
    slug = str(post.get("post_name") or f"post-{int(post['ID'])}")
    dfg = is_dfg_post(
        title,
        slug,
        category_slugs=[str(t.get("slug") or "") for t in child_terms],
        category_term_ids=[int(t.get("term_id") or 0) for t in child_terms],
    )
    record = {
        "id": f"moes-web-{int(post['ID'])}",
        "source": "moes-website",
        "site": "www.moes.gov.in",
        "wp_id": int(post["ID"]),
        "slug": slug,
        "title": title,
        "category": category,
        "family": family,
        "category_path": category_path,
        "child_terms": [
            {"term_id": int(t.get("term_id") or 0),
             "slug": str(t.get("slug") or ""),
             "name": str(t.get("name") or "")}
            for t in child_terms
        ],
        "date": post_date[:10] or None,                       # ISO day (authoritative)
        "acf_date": (str(acf.get("date") or "").strip() or None),  # dd/mm/yyyy verbatim
        "post_modified": str(post.get("post_modified") or "") or None,
        "ministry": "earth-sciences",                          # record-level provenance
        "org": "moes_hq",                                      # (staging only; no engine yet)
        "persona": [str(p) for p in (acf.get("persona") or [])],
        "language": None,              # not reliably exposed for `documents` posts
        "description": description,
        "is_parliament_question": is_parliament_question(title),
        "guid": str(post.get("guid") or "") or None,           # cms-moes host (never fetched)
        "api_refs": {"listing_url": listing_url},
        "files": normalize_file_rows(post),
        "scraped_at": scraped_at,                              # volatile (hash-excluded)
    }
    # DfG-only keys. Added conditionally so every other family's record stays
    # byte-identical (and its records.row_sha256 unchanged).
    if dfg:
        record["extraction_profile"] = DFG_PROFILE
        record["skip_slots"] = list(DFG_SKIP_SLOTS)
    return record


_SLUG_SAFE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 60) -> str:
    """Deterministic ascii slug for document filenames."""
    out = _SLUG_SAFE.sub("-", text.strip().lower()).strip("-")
    out = re.sub(r"-{2,}", "-", out)
    return out[:max_len].strip("-") or "document"
