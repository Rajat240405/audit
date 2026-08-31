"""
Query expansion for the Hybrid RAG pipeline.

Fixes the "Q3-class" retrieval gap: when a query names an entity (e.g. "INCOIS
and NIOT"), the raw query only contains the words the user typed. If those
words are semantically one-sided (INCOIS-side terms), documents about the OTHER
entity (NIOT) never rank. We append a curated set of role/keyword synonyms for
known entities before BM25 (and optionally as an extra dense query), so both
sides of a comparison surface in the top-k.

This is a lightweight, deterministic expansion (no LLM call needed): a curated
map of entity -> role keywords mined from the corpus itself.
"""

from __future__ import annotations

import re
from typing import Optional

# ── curated entity → role-keywords map (mined from the corpus) ──────────────
# Keys are the canonical entity names/acronyms; values are keywords that appear
# in documents about that entity but may be absent from the user's query.
ENTITY_KEYWORDS: dict[str, list[str]] = {
    "INCOIS": [
        "ocean information", "forecast", "early warning", "tsunami warning",
        "potential fishing zone", "advisory", "SAS", "ocean state",
        "Tarang", "forecasting system", "coral bleaching", "wave",
    ],
    "NIOT": [
        "ocean technology", "deep sea", "submersible", "underwater vehicle",
        "polymetallic", "manganese", "cobalt", "hydrothermal", "research vessel",
        "ballast", "matsya", "mining", "LTTD", "desalination",
    ],
    "IMD": [
        "india meteorological department", "forecast", "doppler", "radar",
        "agromet", "weather", "warning", "cyclone", "monsoon", "nowcast",
    ],
    "DWR": [
        "doppler weather radar", "radar network", "S-band", "C-band", "X-band",
        "forecast accuracy", "Mission Mausam",
    ],
    "MISSION MAUSAM": [
        "radar", "dwr", "wind profiler", "radiometer", "forecast",
        "weather observation", "2000 crore",
    ],
    "ISRO": ["space", "satellite", "VSSC", "remote sensing", "rocket"],
    "DRDO": ["defence", "bio-vest", "inertial navigation", "submersible"],
    "CSIR-NIO": [
        "national institute of oceanography", "environmental impact",
        "marine biodiversity", "survey",
    ],
    "GRSE": ["garden reach", "research vessel", "shipbuilders"],
    "CMLRE": ["marine living", "seamount", "biodiversity", "ecology"],
    "CWC": ["central water commission", "flood forecast", "coastal management"],
    "NDMA": ["disaster management", "national disaster", "CAP", "sachet"],
    "INCOIS AND NIOT": [
        "ocean technology", "ocean information", "deep sea", "forecast",
        "research vessel", "submersible", "advisory",
    ],
}

# Normalize an entity key for matching (strip spaces, lower, sort tokens)
def _key(text: str) -> str:
    return " ".join(sorted(re.sub(r"[^a-z0-9]", " ", text.lower()).split()))

_NORMALIZED: dict[str, str] = {}
for _k in ENTITY_KEYWORDS:
    _NORMALIZED[_key(_k)] = _k


def expand_query(query: str) -> str:
    """Append role keywords for any known entity mentioned in the query.

    Returns the expanded query string (original + trailing keywords). If no
    known entity is detected, returns the original query unchanged.
    """
    q = query.strip()
    if not q:
        return q
    q_lower = q.lower()
    extra: list[str] = []

    # exact/contains match on each entity name
    for name, kws in ENTITY_KEYWORDS.items():
        if name.lower() in q_lower:
            extra.extend(kws)

    # token-boundary match for acronyms and base entity names, so both sides
    # of a comparison ("INCOIS and NIOT") contribute their role keywords.
    for name, kws in ENTITY_KEYWORDS.items():
        base = re.sub(r"\s+(AND|&)\s+.*$", "", name, flags=re.I).strip()
        if len(base) >= 3 and re.search(rf"\b{re.escape(base)}\b", q_lower, re.IGNORECASE):
            extra.extend(kws)

    if not extra:
        return q
    # de-dup preserving order
    seen: set[str] = set()
    dedup = []
    for e in extra:
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    # Cap the EXPANSION tail (not the whole query) so both sides of a
    # comparison keep their role keywords. 24 expansion terms is enough for
    # two entities; BM25 + dense handle the rest.
    return q + " " + " ".join(dedup[:24])
