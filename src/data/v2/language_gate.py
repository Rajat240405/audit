"""Stage 11 — language gate.

INCOIS publishes bilingual material and the corpus excludes ``*-hin.*`` files
(``config/sources.yaml`` ``exclude_globs``), but Hindi still reaches the index
through mixed-language pages and through OCR output that recovers Devanagari.
An English-first retrieval that silently returns Devanagari-only passages
produces unusable evidence, so the gate classifies rather than drops: text is
labelled, and the caller decides.

Nothing here imports a language model or a tokenizer. The signal is the
Devanagari codepoint ratio — the same cheap measurement Stage 1 already uses for
its mojibake gate — so it is deterministic and free.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.data.v2.config import V2Config

#: Devanagari block U+0900-U+097F (plus the extended-A block used by Marathi).
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f\ua8e0-\ua8ff]")
#: Latin letters, for the denominator.
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class LanguageProfile:
    """A deterministic language measurement for one text."""

    devanagari: int
    latin: int
    ratio: float
    label: str

    @property
    def is_hindi_dominant(self) -> bool:
        return self.label == "hindi"

    def as_dict(self) -> dict[str, object]:
        return {
            "devanagari": self.devanagari,
            "latin": self.latin,
            "ratio": self.ratio,
            "label": self.label,
        }


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def language_profile(text: str, config: V2Config | None = None) -> LanguageProfile:
    """Measure a text's script composition.

    ``ratio`` is Devanagari *letters* over all letters, bounded to [0, 1], so a
    page that is 40% Devanagari and 60% Latin reads as mixed rather than Hindi. Empty or
    letter-free text (a scanned page with no OCR, a table of numbers) is
    ``unknown`` and never counted as Hindi — mislabeling an empty page as
    Hindi would silently drop it.
    """
    cfg = config if config is not None else V2Config()
    source = text or ""

    # Only *letters* in the Devanagari block count. Vowel signs (matras) and the
    # virama are category Mn, not L; counting them would let the numerator
    # exceed the denominator and push the ratio above 1.0 (measured: 1.55 on a
    # plain Hindi sentence). The ratio must stay a proportion.
    devanagari = sum(
        1 for char in source if _DEVANAGARI_RE.match(char) and _is_letter(char)
    )
    latin = len(_LATIN_RE.findall(source))
    letters = sum(1 for char in source if _is_letter(char))

    if letters == 0:
        return LanguageProfile(devanagari, latin, 0.0, "unknown")

    ratio = round(devanagari / letters, 4)
    if ratio >= cfg.hindi_gate_ratio:
        label = "hindi"
    elif devanagari > 0:
        label = "mixed"
    else:
        label = "english"
    return LanguageProfile(devanagari, latin, ratio, label)


def passes_english_gate(text: str, config: V2Config | None = None) -> bool:
    """True when the text is safe to serve as English-language evidence.

    Hindi-dominant text fails; mixed and unknown pass, because a mixed page
    still carries usable English and an unknown page is far more likely to be a
    numeric table than Hindi prose. Dropping on ``unknown`` would discard every
    table-only page in the corpus.
    """
    return not language_profile(text, config).is_hindi_dominant


def gate_reason(text: str, config: V2Config | None = None) -> str | None:
    """Why a text failed the gate, or ``None`` if it passed.

    Returned instead of a bare boolean so a filtered passage can explain itself
    in a log or in evidence provenance.
    """
    profile = language_profile(text, config)
    if not profile.is_hindi_dominant:
        return None
    cfg = config if config is not None else V2Config()
    return (
        f"hindi-dominant (devanagari ratio {profile.ratio} >= "
        f"{cfg.hindi_gate_ratio})"
    )


__all__ = [
    "LanguageProfile",
    "gate_reason",
    "language_profile",
    "passes_english_gate",
]
