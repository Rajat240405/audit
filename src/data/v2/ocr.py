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

import os
import time
from typing import TYPE_CHECKING, Any

from src.data.v2.config import V2Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Languages required by the default ``hin+eng`` configuration.
_DEFAULT_LANGS = ("hin", "eng")

#: Environment variable that caps Tesseract's OpenMP thread count.
ENV_OMP_THREAD_LIMIT = "OMP_THREAD_LIMIT"
#: Value we pin it to when the operator has not set it explicitly.
DEFAULT_OMP_THREAD_LIMIT = "1"


def apply_omp_thread_limit() -> None:
    """Pin Tesseract to one OpenMP thread unless the operator said otherwise.

    **Why this exists.** Tesseract links ``libgomp`` and, left unconstrained,
    opens one thread per visible CPU *inside each single-page call*. For LSTM
    page segmentation that intra-call parallelism is counter-productive rather
    than helpful: measured on a 2-vCPU host over real INCOIS scan pages, one page
    took **8,692 ms** with the default and **1,508 ms** with
    ``OMP_THREAD_LIMIT=1`` — the default is 5.76x *slower* — while producing
    **byte-identical text** on every page compared. A full 48-page pipeline run
    went from 457.6 s to 106.4 s (4.30x) with identical output.
    See ``INCOIS_V2_PERFORMANCE_OPTIMIZATION_RESEARCH.md`` §C and §L.

    Concurrency is instead obtained at the *page* level (``ocr_pages``), which
    scales properly because each page is an independent Tesseract process.

    Uses ``setdefault`` so an explicit ``OMP_THREAD_LIMIT`` already present in
    the environment always wins: the operator stays in control, and the value can
    be raised or unset without a code change. Called at OCR-invocation scope, not
    at module import, so importing this module never mutates the environment.
    """
    os.environ.setdefault(ENV_OMP_THREAD_LIMIT, DEFAULT_OMP_THREAD_LIMIT)

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
    # Read the raster straight out of the pixmap instead of encoding it to PNG
    # and decoding it again. The round-trip cost 208.5 ms per page (162.6 ms
    # encode + 45.9 ms decode) against 8.5 ms for this, and the pixels are
    # identical — verified, and asserted by test_prepare_image_matches_png_path.
    # The grayscale/invert/autocontrast chain below is unchanged: AR_2018's
    # dark-navy scans still need the inversion to OCR at all.
    mode = "RGBA" if pixmap.alpha else "RGB"
    image = Image.frombytes(
        mode, (pixmap.width, pixmap.height), pixmap.samples
    ).convert("L")
    # Dark-navy annual-report scans yield ~0 characters without inversion;
    # measured on AR_2018 (spec §3.7).
    if ImageStat.Stat(image).mean[0] < config.ocr_invert_lum:
        image = ImageOps.invert(image)
    return ImageOps.autocontrast(image)


def prepare_image(
    page: Any, config: V2Config | None = None
) -> tuple[Any, str | None]:
    """Render and preprocess one page for OCR **on the calling thread**.

    Returns ``(image, None)`` on success, or ``(None, status)`` when the page
    cannot be prepared — with exactly the status strings ``ocr_page`` has always
    produced (``"disabled"``, ``"unavailable:<reason>"``, ``"error:<reason>"``).

    This is the only function that touches a PyMuPDF ``Page``. Splitting it out
    is what makes page-level OCR parallelism safe: the caller renders every page
    serially on the main thread and hands workers plain PIL images, so no worker
    can ever reach a ``Document``, ``Page`` or ``Pixmap``. That is the same
    structural rule ``src/data/extract_dfg.py`` uses to stay clear of the
    uncontainable MuPDF SIGSEGV failure class.
    """
    cfg = config if config is not None else V2Config()
    if not cfg.ocr_active:
        return None, "disabled"

    available, reason = ocr_available(cfg.ocr_lang)
    if not available:
        return None, f"unavailable:{reason}"

    try:
        # Pin Tesseract's OpenMP before it is first used in this process.
        apply_omp_thread_limit()
        return _prepare_image(page, cfg), None
    except Exception as exc:  # noqa: BLE001 - OCR must never abort ingestion
        return None, f"error:{exc}"


def ocr_image(image: Any, config: V2Config | None = None) -> tuple[str, str]:
    """OCR an already-materialised PIL image. Returns ``(text, status)``.

    Touches no PyMuPDF object, so it is safe to call from a worker thread.
    Never raises.
    """
    cfg = config if config is not None else V2Config()
    try:
        import pytesseract

        apply_omp_thread_limit()
        text = pytesseract.image_to_string(image, lang=cfg.ocr_lang)
    except Exception as exc:  # noqa: BLE001 - OCR must never abort ingestion
        return "", f"error:{exc}"

    return text or "", "ok"


def ocr_page(page: Any, config: V2Config | None = None) -> tuple[str, str]:
    """OCR one page. Returns ``(text, status)``.

    ``status`` is one of ``"ok"``, ``"unavailable:<reason>"``, or
    ``"error:<reason>"`` — recorded in the sidecar so a failed page is never
    silently indistinguishable from a page that legitimately has no text.

    Never raises. Behaviour is unchanged by the render/OCR split: this is still
    the serial render-then-OCR path.
    """
    image, status = prepare_image(page, config)
    if image is None:
        return "", status or "error:unknown"
    return ocr_image(image, config)


#: Assumed RSS per OCR worker (MB): the page image plus the Tesseract process.
#: Used to keep the pool inside available RAM. See the research report §G.
_MEM_PER_WORKER_MB = 150


def _available_mb() -> int | None:
    """Currently available RAM in MB, or ``None`` when it cannot be determined."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def resolve_workers(requested: int, n_pages: int) -> int:
    """How many OCR threads to actually run.

    ``min(requested, vcpu_count, available_mb // 150, n_pages)``, never below 1.
    ``requested <= 0`` means the serial path. The RAM term only applies where
    available memory can be read, so the formula degrades gracefully on
    platforms without ``/proc/meminfo``.
    """
    if requested <= 0 or n_pages <= 1:
        return 1
    limit = min(requested, n_pages, _cpu_count())
    available = _available_mb()
    if available is not None:
        limit = min(limit, max(1, available // _MEM_PER_WORKER_MB))
    return max(1, limit)


def ocr_prepared(
    pending: list[tuple[int, Any]],
    config: V2Config | None = None,
    *,
    workers: int | None = None,
) -> dict[int, tuple[str, str]]:
    """OCR already-rendered pages and return results **keyed by page number**.

    ``pending`` is a list of ``(page_number, pil_image)`` produced by
    :func:`prepare_image` on the main thread. Workers receive only PIL images —
    never a PyMuPDF object — so MuPDF is never entered concurrently and the
    uncontainable-SIGSEGV failure class cannot arise.

    Results are keyed by page number (and ``ThreadPoolExecutor.map`` also
    preserves input order), so the caller restores the original page sequence
    deterministically regardless of completion order.

    ``workers <= 1`` runs the identical work serially, which is the safe
    fallback and the default.
    """
    if not pending:
        return {}
    cfg = config if config is not None else V2Config()
    n_workers = (
        resolve_workers(cfg.ocr_workers, len(pending)) if workers is None else workers
    )

    if n_workers <= 1:
        return {pno: ocr_image(image, cfg) for pno, image in pending}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(lambda item: ocr_image(item[1], cfg), pending))
    return {pno: result for (pno, _image), result in zip(pending, results)}


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
