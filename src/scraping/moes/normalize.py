"""Listing JSON → normalized MoES records (real API shapes, live-verified).

Scope is derived FROM THE LIVE TAXONOMY, never hardcoded: reports families
(annual-reports / monthly-reports / demands-for-grants) are resolved by
config-declared substring patterns against the child terms of the `reports`
parent. A family matching zero terms (upstream rename) fails closed instead
of silently widening scope.

central-documents is deliberately different (see the central section at the
bottom): its upstream taxonomy is known-INCOMPLETE, so the approved scopes
(central_documents.scopes) gate per RECORD — an approved child term, else a
central_documents attachment with a provable PDF whose title/terms the scope
`covers`. Everything else in the mixed bucket stays excluded.
"""

from __future__ import annotations

import re
from typing import Any

from src.scraping.moes.config import MoesConfigError
from src.scraping.moes.documents import extract_pdf_links  # no cycle (documents ⟂ normalize)

#: titles like "PARLIAMENT QUESTION: MISSION MAUSAM" (stats only — such
#: documents are NEVER excluded; cross-source dedupe is a later concern)
#: singular AND plural observed live ("PARLIAMENT QUESTIONS: MAJOR
#: IMPROVEMENTS…", post 28112) — accepted via `questions?`
PQ_TITLE_RE = re.compile(r"^\s*parliament\s+questions?\b", re.IGNORECASE)

REPORTS_PARENT_SLUG = "reports"


def is_parliament_question(title: str | None) -> bool:
    return bool(PQ_TITLE_RE.match(title or ""))


def parent_term_id(terms: list[dict[str, Any]], slug: str) -> int:
    tid = top_term_id(terms, slug)
    if tid is None:
        raise MoesConfigError(
            f"taxonomy has no top-level term {slug!r} (upstream schema change?)")
    return tid


def top_term_id(terms: list[dict[str, Any]], slug: str) -> int | None:
    """Non-raising parent lookup for scopes whose taxonomy is OPTIONAL
    upstream (central-documents: live-verified ABSENT from the tree — the
    listing category axis exists without any taxonomy node)."""
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


# ── central-documents scoping (fail-closed) ─────────────────────────────────
# ONLY the approved central_documents.scopes families pass; everything else
# under the central-documents category is excluded at scope — deliberately
# NOT simply "enabled": live probing (2026-08-24/25) found the category is a
# mixed bucket (2,471 posts: guidelines, acts, policies, notifications, … —
# plus Annual Reports) and the taxonomy tree has NO central-documents node
# AT ALL (the listing category axis exists without any tree entry). Two
# deliberately different semantics vs reports:
#   * reports families are taxonomy-COMPLETE upstream (zero live term match
#     = scope drift = abort); for central, the taxonomy is advisory-only —
#     zero/absent terms are tolerated and acceptance falls through to
#     per-record content evidence below (the ONLY live-viable arbiter).
#   * fail-closed holds per RECORD either way: only (1) an approved child
#     term, or (2) a central_documents attachment with a provable PDF whose
#     title/terms the scope `covers`, accepts a post.

CENTRAL_DOCUMENTS_PARENT_SLUG = "central-documents"


def resolve_central_family_terms(
    terms: list[dict[str, Any]],
    scope_cfg: dict[str, Any],
) -> dict[str, list[int]]:
    """Each approved central family → matching live child-term ids of the
    central-documents parent (config order; EMPTY list when no live child
    term matches — TOLERATED here, unlike reports, because the upstream
    central taxonomy is known-incomplete: live probing 2026-08-25 found no
    central-documents node in the tree AT ALL — Annual Reports exist only
    as ``central_documents`` attachments, so per-record content evidence is
    the arbiter). Fail-closed guard that remains: one term matching two
    approved families aborts (ambiguous scope — never silently widened)."""
    pid = top_term_id(terms, CENTRAL_DOCUMENTS_PARENT_SLUG)
    children = [t for t in terms
                if pid is not None and int(t.get("parent") or 0) == pid]
    resolved: dict[str, list[int]] = {}
    owner: dict[int, str] = {}
    for family, fcfg in scope_cfg.items():
        patterns = [str(c).strip().lower() for c in (fcfg or {}).get("covers") or []]
        matched = [
            t for t in children
            if any(p and (p in str(t.get("slug", "")).lower()
                          or p in str(t.get("name", "")).lower()) for p in patterns)
        ]
        for t in matched:
            tid = int(t["term_id"])
            if tid in owner and owner[tid] != family:
                raise MoesConfigError(
                    f"central taxonomy term {t.get('slug')!r} matches two scopes "
                    f"({owner[tid]!r}, {family!r}) — ambiguous scope")
            owner[tid] = family
        resolved[family] = sorted(int(t["term_id"]) for t in matched)
    return resolved


def covers(post_like: dict[str, Any], scope_cfg: dict[str, Any]) -> str | None:
    """First approved central scope family (config order) whose ``covers``
    patterns substring-match the post-like object's title or one of its
    documents_category term slugs/names. ``None`` = NOT covered — unrelated
    central families never match an approved scope's patterns."""
    haystacks: list[str] = []
    acf = post_like.get("acf_data") or {}
    title = str(acf.get("title") or post_like.get("post_title") or "").strip().lower()
    if title:
        haystacks.append(title)
    for t in post_like.get("documents_category") or []:
        if isinstance(t, dict):
            haystacks.append(str(t.get("slug", "")).lower())
            haystacks.append(str(t.get("name", "")).lower())
    for family, fcfg in scope_cfg.items():
        patterns = [str(c).strip().lower() for c in (fcfg or {}).get("covers") or []]
        if any(p and any(p in h for h in haystacks) for p in patterns):
            return family
    return None


def post_central_family(
    post: dict[str, Any],
    family_terms: dict[str, list[int]],
    scope_cfg: dict[str, Any],
    *,
    resolved_attachment: dict[str, Any] | None = None,
) -> str | None:
    """Approved central family (e.g. 'annual-reports') for a post, or None
    (fail-closed: unrelated central-documents categories stay excluded).

    Evidence, in order — taxonomy is authoritative first, mirroring how the
    reports category treats its families:
      1. one of the post's documents_category terms matches an approved
         central family (``resolve_central_family_terms``);
      2. attachment-content fallback (uncategorized uploads): the post's
         embedded attachment — else the caller-supplied REST-resolved
         attachment (post-page/post?id=) — must a) be a ``central_documents``
         post, b) provably carry a PDF object (``extract_pdf_links``), and
         c) have a title/terms the scope ``covers``. Title/pattern matches
         alone never suffice without the PDF proof, and any lookup failure
         degrades to None (never guessed, never widened).
    """
    term_ids = [int(t.get("term_id") or 0)
                for t in (post.get("documents_category") or [])
                if isinstance(t, dict)]
    family = post_family(term_ids, family_terms)
    if family is not None:
        return family
    att = attachment_inline_of(post) or resolved_attachment
    if not att or str(att.get("post_type") or "") != "central_documents":
        return None
    if not (extract_pdf_links(att) or extract_pdf_links(post)):
        return None
    media_title = central_media_title(att)
    att_like = {
        "post_title": media_title,
        "acf_data": {"title": media_title},
        "documents_category": list(att.get("documents_category") or []),
    }
    return covers(att_like, scope_cfg)


def attachment_inline_of(post: dict[str, Any]) -> dict[str, Any] | None:
    """The post's embedded attachment document (first ACF file row), or
    None. Live listings frequently carry the FULL central_documents object
    inline; int-id rows carry no inline object (REST resolution handles
    those — the existing post-page/post?id= contract resolves every id)."""
    files = (post.get("acf_data") or {}).get("file") or []
    if not files or not isinstance(files[0], dict):
        return None
    fid = files[0].get("file")
    if isinstance(fid, list) and fid and isinstance(fid[0], dict):
        return fid[0]
    if isinstance(fid, dict):
        return fid
    return None


def central_media_title(att: dict[str, Any]) -> str:
    """Media title of a central attachment: ACF title → post_title →
    post_name → guid. Empty when none (never guessed)."""
    acf = att.get("acf_data") or {}
    for key in (("acf_data", "title"), ("", "title"), ("", "media_title"),
                ("", "post_title"), ("", "post_name"), ("", "guid")):
        scope, name = key
        src = acf if scope else att
        val = str(src.get(name) or "").strip()
        if val:
            return val
    return ""


def terms_of(obj: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in obj.get("documents_category") or [] if isinstance(t, dict)]


def _attachment_id_of(file_entry: Any) -> int | None:
    """Media id of one ACF file entry: int | [entry] | media dict
    (ID/id/media_id/attachment_id). None = no resolvable id (never
    fabricated)."""
    if isinstance(file_entry, list):
        if not file_entry:
            return None
        file_entry = file_entry[0]
    if isinstance(file_entry, (int, float)) and not isinstance(file_entry, bool):
        return int(file_entry)
    if isinstance(file_entry, dict):
        for key in ("ID", "id", "media_id", "attachment_id"):
            mid = file_entry.get(key)
            if mid is not None:
                try:
                    return int(mid)
                except (TypeError, ValueError):
                    continue
    return None


def first_file_attachment_id(post: dict[str, Any]) -> int | None:
    """Media id of the post's FIRST ACF file row (plain id or embedded media
    object), or None. Single source for the row-unwrapping rules (pipeline
    REST probe + row normalization share it)."""
    rows = (post.get("acf_data") or {}).get("file") or []
    if rows and isinstance(rows[0], dict):
        return _attachment_id_of(rows[0].get("file"))
    return None


def normalize_central_file_rows(post: dict[str, Any]) -> list[dict[str, Any]]:
    """ACF ``file`` rows of a central_documents post → the SAME row shape
    the existing documents pipeline consumes (attachment | external |
    empty; see ``normalize_file_rows``) plus ``attachment_title`` provenance
    when the row embeds the full media object. ACF ids and inline objects
    both collapse to their media id — the existing REST attachment resolver
    handles every id, so no new endpoint or downloader path is needed."""
    rows = []
    for i, row in enumerate((post.get("acf_data") or {}).get("file") or []):
        row = row if isinstance(row, dict) else {}
        fid = row.get("file")
        att = fid[0] if isinstance(fid, list) and fid else fid
        rows.append({
            "row": i,
            "type": (str(row.get("type")).strip() if row.get("type") else None),
            "title": str(row.get("title") or "").strip(),
            "attachment_id": _attachment_id_of(fid),
            "attachment_title": (central_media_title(att)
                                 if isinstance(att, dict) else "") or None,
            "external_url": str(row.get("external_link") or "").strip() or None,
            "content_ref": bool(row.get("content")),
        })
    return rows


def normalize_central_record(
    post: dict[str, Any],
    *,
    category_slug: str,
    family: str,
    listing_url: str,
    scraped_at: str,
    scope_cfg: dict[str, Any],
) -> dict[str, Any]:
    """One accepted central_documents post → one record in the canonical
    record.json vocabulary of the other MoES documents (``normalize_post``
    keys, plus ``record_path_from`` / ``media_title`` provenance — accepted
    posts are central attachments rescued from the mixed bucket, never
    mislabeled as first-class category documents)."""
    acf = post.get("acf_data") or {}
    att = attachment_inline_of(post)
    media_title = central_media_title(att) if att else ""
    title = str(acf.get("title") or post.get("post_title") or "").strip()
    # claim the media title ONLY when it is itself scope-covered — else keep
    # the post title (media titles are arbitrary upstream; never mislabeled)
    if media_title and media_title.strip().lower() != title.strip().lower() and covers(
        {"post_title": media_title, "acf_data": {"title": media_title},
         "documents_category": []}, scope_cfg,
    ):
        title = media_title.strip()
    post_date = str(post.get("post_date") or "")
    description = None
    for key in ("post_content", "post_excerpt"):
        val = str(post.get(key) or "").strip()
        if val:
            description = val
            break
    post_name = str(post.get("post_name") or "").strip()
    slug = str(acf.get("slug") or "").strip() or (post_name if post_name != "0" else "") \
        or slugify(title)
    category_path = [category_slug, family or "uncategorized"]
    return {
        "id": f"moes-web-{int(post['ID'])}",
        "source": "moes-website",
        "site": "www.moes.gov.in",
        "wp_id": int(post["ID"]),
        "slug": slug,
        "title": title,
        "category": category_slug,
        "family": family,
        "category_path": category_path,
        "record_path_from": category_slug,      # provenance: the mixed bucket
        "child_terms": [
            {"term_id": int(t.get("term_id") or 0),
             "slug": str(t.get("slug") or ""),
             "name": str(t.get("name") or "")}
            for t in terms_of(post)
        ],
        "date": post_date[:10] or None,
        "acf_date": (str(acf.get("date") or "").strip() or None),
        "post_modified": str(post.get("post_modified") or "") or None,
        "post_type": post.get("post_type"),
        "ministry": "earth-sciences",
        "org": "moes_hq",                        # staging only; no engine yet
        "persona": [str(p) for p in (acf.get("persona") or [])],
        "language": None,                        # not reliably exposed
        "description": description,
        "media_title": media_title or None,      # provenance only (not canonical)
        "is_parliament_question": is_parliament_question(title),
        "guid": str(post.get("guid") or "") or None,
        "api_refs": {"listing_url": listing_url},
        "files": normalize_central_file_rows(post),
        "scraped_at": scraped_at,                # volatile (hash-excluded)
    }
