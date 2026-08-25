"""Listing JSON → normalized MoES records (real API shapes, live-verified).

Scope is derived FROM THE LIVE TAXONOMY, never hardcoded: reports families
(annual-reports / monthly-reports / demands-for-grants) are resolved by
config-declared substring patterns against the child terms of the `reports`
parent. A family matching zero terms (upstream rename) fails closed instead
of silently widening scope.
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


def is_parliament_question(title: str | None) -> bool:
    return bool(PQ_TITLE_RE.match(title or ""))


def parent_term_id(terms: list[dict[str, Any]], slug: str) -> int:
    for t in terms:
        if t.get("slug") == slug and int(t.get("parent") or 0) == 0:
            return int(t["term_id"])
    raise MoesConfigError(f"taxonomy has no top-level term {slug!r} (upstream schema change?)")


def child_terms_of(post: dict[str, Any], parent_id: int) -> list[dict[str, Any]]:
    """A post's child-category terms (terms whose parent == the category parent)."""
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
    return {
        "id": f"moes-web-{int(post['ID'])}",
        "source": "moes-website",
        "site": "www.moes.gov.in",
        "wp_id": int(post["ID"]),
        "slug": str(post.get("post_name") or f"post-{int(post['ID'])}"),
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


_SLUG_SAFE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 60) -> str:
    """Deterministic ascii slug for document filenames."""
    out = _SLUG_SAFE.sub("-", text.strip().lower()).strip("-")
    out = re.sub(r"-{2,}", "-", out)
    return out[:max_len].strip("-") or "document"
