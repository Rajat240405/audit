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
import math
import numbers
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
    "ensure_ready",
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

_backend: Callable[[bytes, float], Sequence[Any]] | None = None
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
    """Tolerant float read. A typo must never break an ingestion run.

    ``nan`` and ``inf`` are rejected too: a threshold of ``inf`` would drop every
    detection, which is a silent "no table" on every page.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if not math.isfinite(value):
        log.warning("%s=%r is not a finite number; using %s", name, raw, default)
        return default
    return value


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
    """Is the detector switched on? Defaults to true.

    Policy: ``DOTS_ENABLED=false`` is the only legacy switch. When routing is on
    (``DOTS_ENABLED=true``) a disabled detector is an infrastructure failure
    (:class:`TableDetectorUnavailable`), never a silent fallback to legacy text.
    """
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


def set_backend(fn: Callable[[bytes, float], Sequence[Any]] | None, *, name: str = "injected") -> None:
    """Install a detection backend.

    ``fn(image_bytes, threshold)`` returns an iterable of raw detections. Each item may be
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


def _load_backend() -> Callable[[bytes, float], Sequence[Any]]:
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

        # Preflight the staged weights before touching Paddle. The production
        # image sets PICODET_MODEL_DIR=/models/PicoDet_layout_1x_table, which
        # is a bind mount: if the operator has not staged the weights there,
        # Paddle's own error is an opaque load failure deep in the predictor.
        # Checking here turns that into one actionable line. Deliberately only
        # when the variable is set — leaving it unset keeps the stock
        # download-to-cache behaviour for local development.
        if model_dir:
            weights = os.path.join(model_dir, "inference.pdiparams")
            if not os.path.isdir(model_dir) or not os.path.isfile(weights):
                raise TableDetectorUnavailable(
                    f"PICODET_MODEL_DIR={model_dir!r} does not contain staged "
                    f"PicoDet weights (expected {weights!r}). The HPC runtime "
                    "must not fall back to a developer-machine cache. Stage "
                    f"the {model_name} model into that directory (7.4 MB: "
                    "inference.pdiparams, inference.json, inference.yml, "
                    "config.json), or unset PICODET_MODEL_DIR to allow a "
                    "download. To run without table routing, set "
                    "DOTS_ENABLED=false."
                )

        try:
            import paddle  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise TableDetectorUnavailable(
                "PicoDet table detection requires PaddlePaddle, which is not "
                "installed. Install paddlepaddle + paddleocr, or set "
                "DOTS_ENABLED=false to run without table routing."
            ) from exc

        # Pin to CPU before any predictor is constructed: the GPU belongs to DOTS.
        try:
            paddle.set_device(dev)
        except Exception as exc:  # pragma: no cover - env dependent
            raise TableDetectorUnavailable(
                f"could not select PaddlePaddle device {dev!r}: {exc}"
            ) from exc

        # NOTE: there is deliberately no paddle.set_num_threads() call here.
        # That API does not exist in Paddle 3.x (verified: hasattr is False),
        # so it silently did nothing. Inference threads are controlled by the
        # predictor's own `cpu_threads` argument, set below.

        predictor = None
        try:
            from paddleocr import LayoutDetection  # type: ignore

            kwargs: dict[str, Any] = {"model_name": model_name}
            if model_dir:
                kwargs["model_dir"] = model_dir
            kwargs["device"] = dev
            if threads:
                # LayoutDetection's own inference-thread count, default 10.
                # Independent of the OpenMP vars above; without this the
                # predictor spawns 10 threads regardless of them.
                kwargs["cpu_threads"] = threads
            predictor = LayoutDetection(**kwargs)
            backend_name = f"paddleocr.LayoutDetection:{model_name}"
        except Exception as exc:  # pragma: no cover - env dependent
            raise TableDetectorUnavailable(
                f"could not initialise PicoDet layout detector {model_name!r} "
                f"on device {dev!r}: {exc}"
            ) from exc

        def _predict(image_bytes: bytes, threshold: float) -> Sequence[Any]:
            import io

            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.asarray(img)
            # The threshold MUST be pushed into predict(). PaddleX otherwise
            # applies the model's own `draw_threshold` (0.5 in this model's
            # inference.yml) and silently discards every box below it, so a
            # configured 0.25 would never widen recall — only a client-side
            # filter above 0.5 would have had any effect.
            results = list(predictor.predict(arr, threshold=threshold) or [])
            if not results:
                # One image in, one result out. Zero results is a broken
                # predictor, not an empty page, so it must not read as "no table".
                raise TableDetectorUnavailable(
                    "PicoDet returned no result for a page image (expected one)"
                )
            out: list[Any] = []
            for res in results:
                payload = res
                if hasattr(res, "json"):
                    payload = res.json
                if isinstance(payload, dict):
                    payload = payload.get("res", payload)
                boxes = payload.get("boxes") if isinstance(payload, dict) else None
                if not isinstance(boxes, list):
                    # Shape drift (e.g. a PaddleX upgrade). An empty page reports
                    # "boxes": []. A missing or non-list key is a malformed result.
                    raise TableDetectorUnavailable(
                        "PicoDet result has an unrecognised shape (no 'boxes' list)"
                    )
                out.extend(boxes)
            return out

        _backend = _predict
        _backend_name = backend_name
        return _backend


def is_available() -> bool:
    """Can detection run right now? Never raises."""
    try:
        ensure_ready()
        return True
    except Exception:  # noqa: BLE001 - availability probe must never raise
        return False


def ensure_ready() -> None:
    """Raise :class:`TableDetectorUnavailable` unless PicoDet can really run.

    Used by the health preflight. It proves two things, not one:

    * the backend initialises (weights present, Paddle importable, device OK);
    * one inference completes on a blank page. A backend can initialise and
      still fail on the first predict (the PIR/oneDNN failure pinned in
      requirements.txt is exactly that kind of fault).

    The probe result is discarded; only success or a typed failure matters.
    """
    if not enabled():
        raise TableDetectorUnavailable(
            f"{_ENV_ENABLED}=false, but table routing (DOTS_ENABLED=true) requires "
            "the PicoDet detector. Enable PicoDet, or set DOTS_ENABLED=false to run "
            "without table routing."
        )
    backend = _load_backend()
    thr = score_threshold()
    try:
        raw = backend(_probe_png(), thr)
    except Exception as exc:  # noqa: BLE001 - surface as a typed failure
        raise TableDetectorUnavailable(f"PicoDet probe inference failed: {exc}") from exc
    # Validate the shape as well as the call: a malformed probe result means the
    # detector cannot be trusted, so the preflight must fail rather than pass.
    _filter_tables(raw, thr)


def _probe_png() -> bytes:
    """A small blank white page. Content is irrelevant; only the predictor path matters."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 320), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ── detection ────────────────────────────────────────────────────────────────


def _parse_cls_id(value: Any) -> int | None:
    """An integer class id, or ``None``.

    Only a real integer (or a float with an integral, finite value) is a class id.
    ``"unknown"``, ``3.7``, ``True`` and ``nan`` are not, and must not be truncated
    into one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            return int(as_float)
    return None


def _parse_number(value: Any) -> float | None:
    """A finite real number, or ``None`` (missing, unreadable, NaN or infinite)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return as_float if math.isfinite(as_float) else None


def _parse_bbox(raw_box: Any) -> tuple[float, float, float, float] | None:
    """Exactly four finite coordinates, or ``None``.

    A longer or shorter sequence is malformed rather than truncated, and a string
    is never read as a sequence of digits.
    """
    if raw_box is None or isinstance(raw_box, (str, bytes, bytearray, dict)):
        return None
    try:
        values = list(raw_box)
    except TypeError:
        return None
    if len(values) != 4:
        return None
    coords = [_parse_number(v) for v in values]
    if any(c is None for c in coords):
        return None
    return (coords[0], coords[1], coords[2], coords[3])  # type: ignore[return-value]


def _normalise_detection(item: Any) -> TableBox | None:
    """Coerce one backend detection into a :class:`TableBox`, or return ``None``.

    Accepts a ``TableBox``, a mapping (``label``/``cls_id``/``score``/
    ``coordinate``|``bbox``), or a ``(label, score, bbox)`` triple. ``None`` means
    the detection is MALFORMED: a field is missing, unreadable, non-finite or
    shaped wrongly. The caller raises :class:`TableDetectorUnavailable` for it (see
    :func:`_filter_tables`). It is never read as "no table": a detector whose output
    cannot be understood has not told us that the page is table-free.
    """
    if isinstance(item, TableBox):
        return item

    if isinstance(item, dict):
        cls_raw = item.get("cls_id", item.get("class_id"))
        cls_id: int | None = None
        if cls_raw is not None:
            cls_id = _parse_cls_id(cls_raw)
            if cls_id is None:
                return None  # a class id that is present but unreadable
        label = item.get("label")
        if label is None:
            if cls_id is None:
                return None
            label = (
                "table"
                if cls_id == PUBLAYNET_TABLE_CLASS_ID
                else f"class_{cls_id}"
            )
        raw_box = item.get("coordinate", item.get("bbox", item.get("box")))
        score_raw = item.get("score", item.get("confidence"))
    elif isinstance(item, (list, tuple)) and len(item) == 3:
        label, score_raw, raw_box = item[0], item[1], item[2]
        if label is None:
            return None
    else:
        return None

    bbox = _parse_bbox(raw_box)
    score = _parse_number(score_raw)
    if bbox is None or score is None:
        return None

    x0, y0, x1, y1 = bbox
    return TableBox(
        label=str(label),
        score=score,
        bbox=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
    )


def is_table_label(label: str) -> bool:
    """Case-insensitive ``Table`` membership test."""
    return str(label or "").strip().lower().replace("-", "_") in TABLE_LABELS


def _filter_tables(raw: Sequence[Any], threshold: float) -> list[TableBox]:
    """Keep table-labelled detections at or above ``threshold``, in reading order.

    Every fault in the result raises :class:`TableDetectorUnavailable`: a result
    that is not a readable sequence, and any detection that is malformed. Only a
    well-formed detection with a non-table label is dropped. An unreadable one is
    not "no table", and no parse error may escape as a raw exception.
    """
    if raw is None or isinstance(raw, (str, bytes, bytearray, dict)):
        raise TableDetectorUnavailable(
            "PicoDet returned a malformed result (expected a sequence of detections, "
            f"got {type(raw).__name__})"
        )
    try:
        items = list(raw)
    except Exception as exc:  # noqa: BLE001 - any failure reading the result
        raise TableDetectorUnavailable(
            f"PicoDet returned an unreadable result ({type(raw).__name__}: {exc})"
        ) from exc

    boxes: list[TableBox] = []
    for item in items:
        try:
            box = _normalise_detection(item)
        except Exception as exc:  # noqa: BLE001 - a parse fault is a malformed detection
            raise TableDetectorUnavailable(
                f"PicoDet returned a malformed detection ({type(exc).__name__}): {str(item)[:200]}"
            ) from exc
        if box is None:
            raise TableDetectorUnavailable(
                f"PicoDet returned a malformed detection: {str(item)[:200]}"
            )
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
    :class:`TableDetectorUnavailable` if the detector is switched off, the backend
    cannot run, or its output is malformed. Detection failure is never silently
    treated as "no table", because that would downgrade every page to the legacy
    path without anyone noticing.
    """
    if not enabled():
        raise TableDetectorUnavailable(
            f"{_ENV_ENABLED}=false, but table routing (DOTS_ENABLED=true) requires "
            "the PicoDet detector. Enable PicoDet, or set DOTS_ENABLED=false to run "
            "without table routing."
        )
    if not image_bytes:
        return []
    try:
        backend = _load_backend()
    except TableDetectorUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - any load fault is a typed failure
        raise TableDetectorUnavailable(f"PicoDet backend could not be loaded: {exc}") from exc
    thr = score_threshold() if threshold is None else float(threshold)
    try:
        raw = backend(image_bytes, thr)
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
