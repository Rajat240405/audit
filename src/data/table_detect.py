"""PicoDet page-level table detection — the ONLY table detector in this project.

Role in the unified extraction architecture
-------------------------------------------
``pdf_table_extract`` renders each page and asks this module one question:

    does this page contain a table?

* **yes** -> the page is handed to DOTS OCR (``src.data.dots_client``), which is
  the ONLY table extractor. Every legacy table path is bypassed.
* **no**  -> the page stays on the existing non-table extraction path.

Policy (fixed project decisions)
--------------------------------
1. PicoDet is the only detector. No TATR, no DocLayout-YOLO, no secondary model.
2. Detection is **page-by-page**. Multi-page montages are never built.
3. Inference runs on **CPU**. The GPU on UN003 is reserved exclusively for DOTS;
   a 7.4 MB PicoDet layout model costs ~10-30 ms/page on CPU and must never
   contend for VRAM. ``PICODET_DEVICE`` exists as an escape hatch but defaults
   to ``cpu`` and is forced to ``cpu`` unless explicitly overridden.
4. The score threshold is environment-configurable and defaults to the
   validated production value **0.25**.

Recall bias
-----------
A false negative silently degrades a page to today's legacy behaviour (a
non-regression). A false positive costs one DOTS call. The default threshold is
therefore deliberately recall-biased; see ``PICODET_SCORE_THRESHOLD``.

Availability
------------
Paddle is imported **lazily**. Importing this module adds no heavy dependency,
so ``pdf_table_extract`` keeps importing cleanly on machines without Paddle and
the legacy path stays usable. A missing backend raises
:class:`TableDetectorUnavailable` only when detection is actually requested.

Testability
-----------
The inference backend is injectable (:func:`set_backend`). Tests exercise label
handling, thresholding and ordering without installing Paddle.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, NamedTuple, Sequence

__all__ = [
    "DEFAULT_SCORE_THRESHOLD",
    "TABLE_LABELS",
    "TableBox",
    "TableDetectorUnavailable",
    "detect_tables",
    "detector_version",
    "is_available",
    "page_has_table",
    "render_page_png",
    "reset_backend",
    "score_threshold",
    "set_backend",
]

log = logging.getLogger(__name__)

#: Validated production default. Recall-biased on purpose (see module docstring).
DEFAULT_SCORE_THRESHOLD = 0.25

#: Default rasterisation DPI for DETECTION ONLY. Detection tolerates a coarser
#: raster than OCR, so this is deliberately lower than ``DOTS_RENDER_DPI``.
DEFAULT_DETECT_DPI = 150

#: Labels that mean "this region is a table", matched case-insensitively.
#: Covers the PubLayNet vocabulary used by ``picodet_lcnet_x1_0_fgd_layout``
#: (text/title/list/table/figure), the single-class
#: ``picodet_lcnet_x1_0_fgd_layout_table`` model, and the CDLA variant.
TABLE_LABELS: frozenset[str] = frozenset({"table", "tables", "table_caption_free"})

#: PubLayNet class index for ``table`` — used when a backend reports only
#: integer class ids with no label string.
PUBLAYNET_TABLE_CLASS_ID = 3

_ENV_ENABLED = "PICODET_ENABLED"
_ENV_THRESHOLD = "PICODET_SCORE_THRESHOLD"
_ENV_DEVICE = "PICODET_DEVICE"
_ENV_MODEL_DIR = "PICODET_MODEL_DIR"
_ENV_MODEL_NAME = "PICODET_MODEL_NAME"
_ENV_DPI = "PICODET_DPI"
_ENV_VERSION = "PICODET_VERSION"
_ENV_CPU_THREADS = "PICODET_CPU_THREADS"

_DEFAULT_MODEL_NAME = "PicoDet_layout_1x_table"

_backend: Callable[[bytes], Sequence[Any]] | None = None
_backend_name: str = "none"
_lock = threading.Lock()


class TableDetectorUnavailable(RuntimeError):
    """The PicoDet backend could not be loaded or initialised.

    Raised only when detection is actually attempted — never at import time.
    """


class TableBox(NamedTuple):
    """One detected layout region, in detection-raster pixel coordinates."""

    label: str
    score: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1


# ── environment helpers ──────────────────────────────────────────────────────


def _env_float(name: str, default: float) -> float:
    """Tolerant float read. A typo must never break an ingestion run."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def score_threshold() -> float:
    """Current detection threshold. Default :data:`DEFAULT_SCORE_THRESHOLD` (0.25)."""
    return _env_float(_ENV_THRESHOLD, DEFAULT_SCORE_THRESHOLD)


def detect_dpi() -> int:
    """Rasterisation DPI used for detection only."""
    return _env_int(_ENV_DPI, DEFAULT_DETECT_DPI)


def device() -> str:
    """Inference device. Always ``cpu`` unless deliberately overridden."""
    return (os.environ.get(_ENV_DEVICE) or "cpu").strip().lower() or "cpu"


def enabled() -> bool:
    """Is the detector switched on? Defaults to true; the DOTS master switch gates routing."""
    return _env_bool(_ENV_ENABLED, True)


def cpu_threads() -> int:
    """CPU threads PicoDet may use. ``0`` means "leave Paddle's default alone".

    Paddle grabs every visible core by default. During ingestion it is not
    alone: the v2 pipeline runs ``V2_OCR_WORKERS`` Tesseract workers and
    Tesseract itself is multi-threaded, so an unbounded PicoDet oversubscribes
    the box and slows down everything including itself. A 7.4 MB layout model
    gains nothing past a couple of threads, so capping it is close to free.
    """
    return max(0, _env_int(_ENV_CPU_THREADS, 2))


def detector_version() -> str:
    """Identifier for this detector + its threshold, for ``extractor_version``."""
    explicit = (os.environ.get(_ENV_VERSION) or "").strip()
    if explicit:
        return explicit
    name = (os.environ.get(_ENV_MODEL_NAME) or _DEFAULT_MODEL_NAME).strip()
    return f"picodet-{name}@{score_threshold():g}"


# ── backend management ───────────────────────────────────────────────────────


def set_backend(fn: Callable[[bytes], Sequence[Any]] | None, *, name: str = "injected") -> None:
    """Install a detection backend.

    ``fn(image_bytes)`` returns an iterable of raw detections. Each item may be
    a :class:`TableBox`, a mapping, or a sequence — :func:`_normalise_detection`
    accepts all three. Used by tests and by any future detector swap; the
    production backend is built lazily by :func:`_load_backend`.
    """
    global _backend, _backend_name
    with _lock:
        _backend = fn
        _backend_name = name if fn is not None else "none"


def reset_backend() -> None:
    """Drop the cached backend so the next call reloads it."""
    set_backend(None)


def _load_backend() -> Callable[[bytes], Sequence[Any]]:
    """Build the production PaddleOCR/PaddleX PicoDet backend. CPU-pinned."""
    global _backend, _backend_name
    if _backend is not None:
        return _backend

    with _lock:
        if _backend is not None:
            return _backend

        dev = device()
        model_name = (os.environ.get(_ENV_MODEL_NAME) or _DEFAULT_MODEL_NAME).strip()
        model_dir = (os.environ.get(_ENV_MODEL_DIR) or "").strip() or None

        # Cap BLAS/OpenMP threading BEFORE importing paddle — these are read at
        # import time, so setting them afterwards has no effect. Only filled in
        # when unset, so an explicit operator value always wins.
        threads = cpu_threads()
        if threads:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(var, str(threads))

        try:
            import paddle  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise TableDetectorUnavailable(
                "PicoDet table detection requires PaddlePaddle, which is not "
                "installed. Install paddlepaddle + paddleocr, or set "
                "PICODET_ENABLED=false to stay on the legacy extraction path."
            ) from exc

        # Pin to CPU before any predictor is constructed: the GPU belongs to DOTS.
        try:
            paddle.set_device(dev)
        except Exception as exc:  # pragma: no cover - env dependent
            raise TableDetectorUnavailable(
                f"could not select PaddlePaddle device {dev!r}: {exc}"
            ) from exc

        if threads:
            # Paddle's own CPU math threads, separate from the OpenMP vars above.
            try:
                paddle.set_num_threads(threads)
            except Exception:  # noqa: BLE001 - advisory only, never fatal
                log.debug("paddle.set_num_threads(%s) unavailable", threads)

        predictor = None
        try:
            from paddleocr import LayoutDetection  # type: ignore

            kwargs: dict[str, Any] = {"model_name": model_name}
            if model_dir:
                kwargs["model_dir"] = model_dir
            kwargs["device"] = dev
            predictor = LayoutDetection(**kwargs)
            backend_name = f"paddleocr.LayoutDetection:{model_name}"
        except Exception as exc:  # pragma: no cover - env dependent
            raise TableDetectorUnavailable(
                f"could not initialise PicoDet layout detector {model_name!r} "
                f"on device {dev!r}: {exc}"
            ) from exc

        def _predict(image_bytes: bytes) -> Sequence[Any]:
            import io

            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.asarray(img)
            # threshold is applied by _filter_tables so one code path owns it
            results = predictor.predict(arr)
            out: list[Any] = []
            for res in results or []:
                payload = res
                if hasattr(res, "json"):
                    payload = res.json
                if isinstance(payload, dict):
                    payload = payload.get("res", payload)
                    boxes = payload.get("boxes") or []
                else:  # pragma: no cover - backend shape drift
                    boxes = []
                out.extend(boxes)
            return out

        _backend = _predict
        _backend_name = backend_name
        return _backend


def is_available() -> bool:
    """Can detection run right now? Never raises."""
    if not enabled():
        return False
    try:
        _load_backend()
        return True
    except TableDetectorUnavailable:
        return False
    except Exception:  # noqa: BLE001 - availability probe must never raise
        return False


# ── detection ────────────────────────────────────────────────────────────────


def _normalise_detection(item: Any) -> TableBox | None:
    """Coerce one backend detection into a :class:`TableBox`.

    Accepts a ``TableBox``, a mapping (``label``/``cls_id``/``score``/
    ``coordinate``|``bbox``), or a ``(label, score, bbox)`` sequence. Anything
    unparseable returns ``None`` and is skipped rather than raising — a weird
    detection must not abort a nightly run.
    """
    if isinstance(item, TableBox):
        return item

    if isinstance(item, dict):
        label = item.get("label")
        if label is None:
            cls_id = item.get("cls_id", item.get("class_id"))
            if cls_id is None:
                return None
            label = (
                "table"
                if int(cls_id) == PUBLAYNET_TABLE_CLASS_ID
                else f"class_{int(cls_id)}"
            )
        raw_box = item.get("coordinate", item.get("bbox", item.get("box")))
        score = item.get("score", item.get("confidence", 0.0))
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        label, score, raw_box = item[0], item[1], item[2]
    else:
        return None

    try:
        coords = tuple(float(v) for v in list(raw_box)[:4])
    except (TypeError, ValueError):
        return None
    if len(coords) != 4:
        return None

    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0

    x0, y0, x1, y1 = coords
    return TableBox(
        label=str(label),
        score=score_f,
        bbox=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
    )


def is_table_label(label: str) -> bool:
    """Case-insensitive ``Table`` membership test."""
    return str(label or "").strip().lower().replace("-", "_") in TABLE_LABELS


def _filter_tables(raw: Sequence[Any], threshold: float) -> list[TableBox]:
    """Keep table-labelled detections at or above ``threshold``, in reading order."""
    boxes: list[TableBox] = []
    for item in raw or ():
        box = _normalise_detection(item)
        if box is None:
            continue
        if not is_table_label(box.label):
            continue
        if box.score < threshold:
            continue
        boxes.append(box)
    # Deterministic reading order: top-to-bottom, then left-to-right.
    boxes.sort(key=lambda b: (round(b.bbox[1], 2), round(b.bbox[0], 2)))
    return boxes


def detect_tables(image_bytes: bytes, *, threshold: float | None = None) -> list[TableBox]:
    """Detect table regions in one rendered page image.

    Returns table regions only, ordered top-to-bottom. Raises
    :class:`TableDetectorUnavailable` if the backend cannot run — detection
    failure is never silently treated as "no table", because that would
    downgrade every page to the legacy path without anyone noticing.
    """
    if not image_bytes:
        return []
    backend = _load_backend()
    thr = score_threshold() if threshold is None else float(threshold)
    try:
        raw = backend(image_bytes)
    except Exception as exc:  # noqa: BLE001 - surface as a typed failure
        raise TableDetectorUnavailable(f"PicoDet inference failed: {exc}") from exc
    return _filter_tables(raw, thr)


def render_page_png(page: Any, dpi: int | None = None) -> bytes:
    """Rasterise a PyMuPDF page to PNG bytes at the detection DPI."""
    resolution = detect_dpi() if dpi is None else int(dpi)
    pixmap = page.get_pixmap(dpi=resolution)
    return pixmap.tobytes("png")


def page_has_table(
    page: Any,
    *,
    dpi: int | None = None,
    threshold: float | None = None,
    image_bytes: bytes | None = None,
) -> bool:
    """Does this page contain at least one table?

    ``image_bytes`` lets a caller reuse a raster it already produced, so a page
    is never rendered twice.
    """
    png = render_page_png(page, dpi) if image_bytes is None else image_bytes
    return bool(detect_tables(png, threshold=threshold))
