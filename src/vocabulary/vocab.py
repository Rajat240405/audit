"""Domain vocabulary loading (Phase 1 / WS1).

Single source of truth: ``config/vocabulary.yaml``. There is deliberately NO
built-in fallback copy (a mirrored fallback is how drift happens — see
docs/phase1/BASELINE.md B5): a missing or malformed vocabulary file is a loud
error, not a silent default.

This module is pure data plumbing — no retrieval, no models, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.app_paths import config_path

__all__ = [
    "VocabularyError",
    "VocabularyEntry",
    "Vocabulary",
    "load_vocabulary",
]


class VocabularyError(Exception):
    """Vocabulary file missing/invalid — fail loudly, never fall back."""


@dataclass(frozen=True)
class VocabularyEntry:
    """One controlled vocabulary value: canonical identity + label + aliases."""

    canonical: str
    label: str
    aliases: tuple[str, ...]
    # optional structured extras (e.g. sansad codes, id prefixes, year spans)
    extra: dict[str, Any] = field(default_factory=dict)

    def surface_forms(self) -> tuple[str, ...]:
        """All accepted raw surfaces (canonical + aliases), de-duplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for s in (self.canonical, *self.aliases):
            key = _fold(s)
            if key not in seen:
                seen.add(key)
                out.append(s)
        return tuple(out)


def _fold(s: str) -> str:
    """Deterministic comparison key: casefold + collapsed whitespace."""
    return " ".join(str(s).casefold().split())


@dataclass(frozen=True)
class Vocabulary:
    version: int
    ministry: tuple[VocabularyEntry, ...]
    org: tuple[VocabularyEntry, ...]
    source: tuple[VocabularyEntry, ...]
    house: tuple[VocabularyEntry, ...]
    document_type: tuple[VocabularyEntry, ...]
    question_type: tuple[VocabularyEntry, ...]
    ls_terms: tuple[dict[str, Any], ...]
    rs_sessions: dict[str, Any]
    member_honorifics: tuple[str, ...]
    concepts: tuple[dict[str, Any], ...]

    # ── lookup indexes (folded-surface → entry), built once at load ──
    _indexes: dict[str, dict[str, VocabularyEntry]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def _index(self, name: str) -> dict[str, VocabularyEntry]:
        if name not in self._indexes:
            self._indexes[name] = {
                _fold(s): e for e in getattr(self, name) for s in e.surface_forms()
            }
        return self._indexes[name]

    def match_ministry(self, raw: Optional[str]) -> Optional[VocabularyEntry]:
        return self._entry("ministry", raw)

    def match_org(self, raw: Optional[str]) -> Optional[VocabularyEntry]:
        return self._entry("org", raw)

    def match_source(self, raw: Optional[str]) -> Optional[VocabularyEntry]:
        return self._entry("source", raw)

    def match_house(self, raw: Optional[str]) -> Optional[VocabularyEntry]:
        return self._entry("house", raw)

    def match_document_type(self, raw: Optional[str]) -> Optional[VocabularyEntry]:
        return self._entry("document_type", raw)

    def match_question_type(self, raw: Optional[str]) -> Optional[VocabularyEntry]:
        return self._entry("question_type", raw)

    def _entry(self, name: str, raw: Optional[str]) -> Optional[VocabularyEntry]:
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        return self._index(name).get(_fold(s))

    def house_by_id_prefix(self, doc_id: str) -> Optional[VocabularyEntry]:
        for e in self.house:
            pref = e.extra.get("id_prefix")
            if pref and str(doc_id).startswith(pref):
                return e
        return None

    def ls_term(self, term: int) -> Optional[dict[str, Any]]:
        for t in self.ls_terms:
            if t["term"] == term:
                return t
        return None


def load_vocabulary(path: Optional[Path | str] = None) -> Vocabulary:
    """Load and validate the vocabulary file. Raises VocabularyError on any
    structural problem — the caller must treat that as a configuration fault."""
    p = Path(path) if path else config_path("vocabulary.yaml")
    if not p.exists():
        raise VocabularyError(
            f"vocabulary file not found: {p} — it is the single source of "
            "truth for the domain vocabularies (no built-in fallback exists)"
        )
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise VocabularyError(f"vocabulary file is not valid YAML ({p}): {e}") from e
    if not isinstance(raw, dict) or "version" not in raw:
        raise VocabularyError(f"vocabulary file has no 'version' field: {p}")

    def entries(key: str) -> tuple[VocabularyEntry, ...]:
        rows = raw.get(key)
        if not isinstance(rows, list) or not rows:
            raise VocabularyError(f"vocabulary '{key}' must be a non-empty list")
        out = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or "canonical" not in row:
                raise VocabularyError(f"vocabulary '{key}[{i}]' missing 'canonical'")
            aliases = row.get("aliases") or []
            if not isinstance(aliases, list):
                raise VocabularyError(f"vocabulary '{key}[{i}].aliases' must be a list")
            extra = {k: v for k, v in row.items()
                     if k not in ("canonical", "label", "aliases")}
            out.append(VocabularyEntry(
                canonical=str(row["canonical"]),
                label=str(row.get("label") or row["canonical"]),
                aliases=tuple(str(a) for a in aliases),
                extra=extra,
            ))
        # canonicals must be unique per vocabulary
        cands = [e.canonical for e in out]
        if len(cands) != len(set(cands)):
            raise VocabularyError(f"vocabulary '{key}' has duplicate canonicals")
        return tuple(out)

    concepts = raw.get("concepts") or []
    if not isinstance(concepts, list):
        raise VocabularyError("vocabulary 'concepts' must be a list")
    for c in concepts:
        if not isinstance(c, dict) or "canonical" not in c or not c.get("surfaces"):
            raise VocabularyError("each concept needs 'canonical' and 'surfaces'")

    ls_terms = raw.get("ls_terms") or []
    if not isinstance(ls_terms, list):
        raise VocabularyError("vocabulary 'ls_terms' must be a list")

    honorifics = raw.get("member_honorifics") or []
    if not isinstance(honorifics, list):
        raise VocabularyError("vocabulary 'member_honorifics' must be a list")

    return Vocabulary(
        version=int(raw["version"]),
        ministry=entries("ministry"),
        org=entries("org"),
        source=entries("source"),
        house=entries("house"),
        document_type=entries("document_type"),
        question_type=entries("question_type"),
        ls_terms=tuple(ls_terms),
        rs_sessions=dict(raw.get("rs_sessions") or {}),
        member_honorifics=tuple(str(h) for h in honorifics),
        concepts=tuple(concepts),
    )
