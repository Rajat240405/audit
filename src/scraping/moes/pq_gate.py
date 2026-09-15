"""Discovery-time parliamentary-question gate for the MoES website crawler.

WHY THIS EXISTS
---------------
``moes_website.yaml`` deliberately ingests the whole ``press-release``
category, which on the live site includes a large family of posts titled
``PARLIAMENT QUESTION: …``. Those are PIB-format *reply summaries* — they are
not the authoritative parliamentary record, and the authoritative LS/RS answers
already exist elsewhere in the corpus. Crawling them costs one attachment
resolve call plus every PDF byte per post, for content we then have to
deduplicate away at ingestion.

The signal needed to identify them is available BEFORE any download: the
listing response already carries the post title. So the gate runs after
normalisation and before ``AttachmentMap.ensure()``.

This is layer 1 of a two-layer design. Layer 2 is the existing
content-containment net (``moes/dedup.py`` + ``config/moes_pq_dedup.yaml``),
which is left completely untouched and continues to catch the PQ-derived
documents that carry no PQ title (backfill batches, unprefixed summaries,
news-styled same-day reports). Neither layer replaces the other: this one
cannot see untitled documents, that one cannot run before the bytes exist.

THE TIER POLICY, AND WHY IT IS TIER-1 ONLY
------------------------------------------
``normalize.py::PQ_TITLE_RE`` (``^\\s*parliament\\s+questions?\\b``) requires no
separator. It therefore matches adversarial *news* headlines such as
``"Parliament Questions Ministry's climate claims"`` — a genuine press release
about a minister being questioned, not a PQ reply summary. That regex is the
equivalent of tier-2 here.

Exclusion is therefore restricted to **tier-1: a separator is mandatory**.
Tier-2 and slug-only hits are recorded as ``audit`` alarms and are NEVER
sufficient to exclude. This makes the false-positive class structurally
unreachable rather than tuned down.

HASH STABILITY
--------------
Nothing in this module is written into the crawler record row. The verdict is
computed from ``record["title"]``/``record["slug"]``, which are already keys of
the hashed row, so ``records.row_sha256`` is unchanged for every record. That
matters: adding a key would flip every press-release record to "changed" once
and trigger a full re-download of the category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TIER1_RE",
    "TIER2_RE",
    "SLUG_RE",
    "DETECTOR_VERSION",
    "PqVerdict",
    "pq_title_tier",
    "pq_slug_signal",
    "pq_gate_verdict",
    "title_is_pq_tier1",
]

#: Bumped whenever the patterns or the tier policy change. Persisted in the
#: cleanup manifest so a removal list can be tied to the rule that produced it.
DETECTOR_VERSION = "1.0.0"

#: Tier-1 — the ONLY signal permitted to exclude. A separator is mandatory.
#: ``parliam[a-z]*`` tolerates the upstream typos observed live
#: (``PARLIAMNENT``, ``PARLIAMET``) while still matching ``PARLIAMENT``.
#: The separator class covers ASCII colon, ASCII hyphen, en dash and em dash;
#: a space before the separator is allowed (``QUESTION :SEISMIC``).
TIER1_RE = re.compile(r"^\s*parliam[a-z]*\s+questions?\s*[:\-–—]", re.IGNORECASE)

#: Tier-2 — no separator required. Diagnostic only: this is what
#: ``normalize.py::PQ_TITLE_RE`` already computes, and it admits adversarial
#: headlines. Never sufficient for exclusion.
TIER2_RE = re.compile(r"^\s*parliam[a-z]*\s+questions?\b", re.IGNORECASE)

#: Slug corroboration. Diagnostic only: the live corpus shows the slug missing
#: real PQ posts (legacy ``pib-N`` slugs, typo concatenations such as
#: ``parliament-questionautomated…``), so it is neither sound nor complete.
SLUG_RE = re.compile(r"parliam[a-z]*[-_]?questions?", re.IGNORECASE)


@dataclass(frozen=True)
class PqVerdict:
    """Outcome of the gate for one normalized MoES record.

    ``is_pq`` mirrors the legacy tier-1 ∪ tier-2 notion and is informational;
    ``exclude`` is the actual policy and is tier-1 only.
    """

    is_pq: bool
    exclude: bool
    audit: bool
    tier: int
    signal: str
    reason: str

    def as_manifest_entry(self, record: dict[str, Any]) -> dict[str, Any]:
        """Audit row for the ``pq_excluded`` manifest bucket.

        Deliberately carries the verbatim title and the attachment ids that
        were NOT fetched, so an operator can see exactly what was skipped and
        what it would have cost.
        """
        return {
            "record_id": record.get("id"),
            "wp_id": record.get("wp_id"),
            "slug": record.get("slug"),
            "title": record.get("title"),
            "tier": self.tier,
            "signal": self.signal,
            "excluded": self.exclude,
            "audit": self.audit,
            "reason": self.reason,
            "attachment_ids": sorted(
                int(f["attachment_id"])
                for f in (record.get("files") or [])
                if f.get("attachment_id") is not None
            ),
        }


_KEEP = PqVerdict(is_pq=False, exclude=False, audit=False, tier=0,
                  signal="none", reason="no parliamentary-question signal")


def pq_title_tier(title: str | None) -> int:
    """1 for a separator-anchored PQ title, 2 for an unanchored one, else 0.

    Tier-1 is checked first so a title qualifying for both reports its
    strongest (and only actionable) classification. Non-string input is
    coerced rather than raising: the gate runs inside a crawl and must never
    be able to break one over a malformed title.
    """
    if not isinstance(title, str):
        title = "" if title is None else str(title)
    if TIER1_RE.match(title):
        return 1
    if TIER2_RE.match(title):
        return 2
    return 0


def pq_slug_signal(slug: str | None) -> bool:
    """True when the slug looks PQ-derived. Corroborating only — see module docstring."""
    if not slug:
        return False
    try:
        return bool(SLUG_RE.search(str(slug)))
    except TypeError:
        return False


def pq_gate_verdict(record: dict[str, Any]) -> PqVerdict:
    """Classify one normalized MoES record. Pure; never raises.

    Reads only ``title`` and ``slug``, both already present in the hashed
    record row, so calling this cannot change ``row_sha256``.
    """
    try:
        title = record.get("title")
        slug = record.get("slug")
    except AttributeError:  # defensive: a non-mapping never excludes
        return _KEEP
    except Exception:  # noqa: BLE001 — the gate must never break a crawl
        return _KEEP

    tier = pq_title_tier(title)
    slug_hit = pq_slug_signal(slug)

    if tier == 1:
        return PqVerdict(
            is_pq=True, exclude=True, audit=False, tier=1,
            signal="title-tier1",
            reason="tier-1: separator-anchored PQ title "
                   "(auto-excluded before attachment resolution)",
        )
    if tier == 2:
        # Unanchored PQ-looking title. Could be a news headline ABOUT a
        # question. Never exclude; raise an alarm instead.
        return PqVerdict(
            is_pq=True, exclude=False, audit=True, tier=2,
            signal="title-tier2",
            reason="tier-2 only: no separator after 'question' — ambiguous, "
                   "kept and flagged for review",
        )
    if slug_hit:
        return PqVerdict(
            is_pq=False, exclude=False, audit=True, tier=0,
            signal="slug",
            reason="PQ-looking slug with an unprefixed title — kept and "
                   "flagged (slug alone is neither sound nor complete)",
        )
    return _KEEP


def title_is_pq_tier1(title: str | None) -> bool:
    """Convenience predicate for the ingestion seam.

    Used by ``ingest._moes_website_dedup_excludes`` to recognise already-staged
    PQ bytes by their ``record.json`` title, so a historical corpus cleanup
    cannot be undone by the next ingestion run.
    """
    return pq_title_tier(title) == 1
