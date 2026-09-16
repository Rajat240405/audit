"""Stage 3 — selective OCR for the experimental INCOIS V2 layer.

OCR runs **only** on pages the router classified as unusable
(``scan`` / ``vector`` / ``mojibake``), never on native pages, and never merely
because V2 is enabled. A per-document page budget is enforced and pages skipped
for budget are marked honestly rather than presented as recovered.

Every dependency here is optional and imported **lazily**. A missing
``pytesseract``, ``Pillow``, the tesseract binary, or the ``hin`` language pack
degrades this module to "unavailable" — it can never break the legacy path,
which does not import this module at all when ``V2_ENABLED=false``.
"""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING, Any

from src.data.v2.config import V2Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Languages required by the default ``hin+eng`` configuration.
_DEFAULT_LANGS = ("hin", "eng")

#: Cached availability probe: (available, reason). ``None`` = not probed yet.
_availability: tuple[bool, str] | None = None


def ocr_available(required_langs: str = "hin+eng") -> tuple[bool, str]:
    """Probe whether OCR can actually run here. Cached after the first call.

    Checks the ``pytesseract`` wrapper, the tesseract binary, and every requested
    language pack. Returns ``(False, reason)`` instead of raising, so a caller
    can skip OCR and say why in the sidecar.
    """
    global _availability  # noqa: PLW0603 - single cached probe
    if _availability is not None:
        return _availability

    try:
        import pytesseract
    except ImportError as exc:
        _availability = (False, f"pytesseract not installed: {exc}")
        return _availability

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001 - binary missing or not runnable
        _availability = (False, f"tesseract binary unavailable: {exc}")
        return _availability

    try:
        installed = pytesseract.get_languages(config="")
    except Exception as exc:  # noqa: BLE001 - cannot enumerate languages
        _availability = (False, f"cannot list tesseract languages: {exc}")
        return _availability

    wanted = [lang.strip() for lang in required_langs.split("+") if lang.strip()]
    missing = [lang for lang in wanted if lang not in installed]
    if missing:
        _availability = (
            False,
            f"missing tesseract language pack(s): {','.join(missing)} "
            f"(installed: {','.join(sorted(installed)) or 'none'})",
        )
        return _availability

    _availability = (True, "ok")
    return _availability


def reset_availability_cache() -> None:
    """Clear the probe cache. Tests and post-install checks use this."""
    global _availability  # noqa: PLW0603
    _availability = None


def _prepare_image(page: Any, config: V2Config) -> Any:
    """Render to grayscale, invert dark pages, autocontrast."""
    from PIL import Image, ImageOps, ImageStat

    pixmap = page.get_pixmap(dpi=config.ocr_dpi)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")
    # Dark-navy annual-report scans yield ~0 characters without inversion;
    # measured on AR_2018 (spec §3.7).
    if ImageStat.Stat(image).mean[0] < config.ocr_invert_lum:
        image = ImageOps.invert(image)
    return ImageOps.autocontrast(image)


def ocr_page(page: Any, config: V2Config | None = None) -> tuple[str, str]:
    """OCR one page. Returns ``(text, status)``.

    ``status`` is one of ``"ok"``, ``"unavailable:<reason>"``, or
    ``"error:<reason>"`` — recorded in the sidecar so a failed page is never
    silently indistinguishable from a page that legitimately has no text.

    Never raises.
    """
    cfg = config if config is not None else V2Config()
    if not cfg.ocr_active:
        return "", "disabled"

    available, reason = ocr_available(cfg.ocr_lang)
    if not available:
        return "", f"unavailable:{reason}"

    try:
        import pytesseract

        image = _prepare_image(page, cfg)
        text = pytesseract.image_to_string(image, lang=cfg.ocr_lang)
    except Exception as exc:  # noqa: BLE001 - OCR must never abort ingestion
        return "", f"error:{exc}"

    return text or "", "ok"


class OcrBudget:
    """Per-document OCR resource accounting.

    **Both limits default to unlimited.** The intended behaviour is page-by-page
    OCR for every page classified as needing it; a limit exists only as an
    explicit operator-imposed safety valve for pathological documents.

    The critical property is honesty: when a limit does stop OCR, every skipped
    page is labelled with the reason and the document is marked incomplete. The
    previous fixed cap of 14 left 108 of AR_2018's 122 scan pages un-OCR'd while
    the record still looked like a complete extraction.
    """

    def __init__(self, page_cap: int = 0, wall_seconds: float = 0.0) -> None:
        #: 0 = unlimited.
        self.page_cap = max(0, page_cap)
        #: 0 = unlimited.
        self.wall_seconds = max(0.0, wall_seconds)
        self.used = 0
        self.skipped = 0
        self.started = time.perf_counter()
        self.limit_reason: str | None = None

    def check(self) -> str | None:
        """``None`` if OCR may run now, else the explicit reason it may not."""
        if self.page_cap and self.used >= self.page_cap:
            return "page-cap"
        if self.wall_seconds and self.elapsed >= self.wall_seconds:
            return "wall-clock"
        return None

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def may_run(self) -> bool:
        return self.check() is None

    def record_used(self) -> None:
        self.used += 1

    def record_skipped(self, reason: str) -> None:
        self.skipped += 1
        if self.limit_reason is None:
            self.limit_reason = reason

    @property
    def complete(self) -> bool:
        """True when no page was left un-OCR'd for want of budget."""
        return self.skipped == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "page_cap": self.page_cap,
            "wall_seconds": self.wall_seconds,
            "used": self.used,
            "skipped": self.skipped,
            "complete": self.complete,
            "limit_reason": self.limit_reason,
            "elapsed_secs": round(self.elapsed, 1),
        }


def route_for_class(page_class: str) -> str:
    """Base route label for a page class.

    The vocabulary is pinned to the strings found in the measured
    ``pages.jsonl`` artifacts (asserted by
    ``tests/test_v2_page_router.py::test_observed_route_vocabulary_is_recorded_for_stage_3``):
    ``native``, ``native_low``, ``empty``, ``scan+OCR``,
    ``scan+OCR-skipped(budget)``.
    """
    if page_class in ("native", "native_low", "empty"):
        return page_class
    return page_class  # caller appends "+OCR" or "+OCR-skipped(budget)"


def ocr_suffix(ocr_used: bool, budget_limited: bool) -> str:
    """The ``+OCR`` / ``+OCR-skipped(budget)`` route suffix."""
    if ocr_used:
        return "+OCR"
    if budget_limited:
        return "+OCR-skipped(budget)"
    return ""


__all__ = [
    "OcrBudget",
    "ocr_available",
    "ocr_page",
    "ocr_suffix",
    "reset_availability_cache",
    "route_for_class",
]
