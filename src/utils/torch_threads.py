"""Apply configured CPU thread limits to PyTorch.

Why this module exists
----------------------
``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` are read by
the OpenMP and BLAS layers, and PyTorch's *intra-op* default is derived from
``omp_get_max_threads()``. But PyTorch has **no** environment variable for
*inter-op* threads — ``TORCH_NUM_THREADS`` and ``TORCH_INTEROP_THREADS`` are not
recognised by PyTorch 2.6.0 (verified against the installed package). The
inter-op pool can therefore only be sized programmatically.

So: the environment stays the single source of truth, and this module is the one
place that explicitly hands those values to PyTorch.

Ordering matters. ``torch.set_num_interop_threads()`` raises ``RuntimeError``
once the inter-op thread pool exists, and that pool is created lazily by the
first inter-op parallel region. In this application the first torch work happens
in the background warm-up thread started from ``server.py``'s startup event, so
``apply_torch_thread_limits()`` must run at **module import** time — before the
warm-up thread is spawned.

Nothing here may ever take the server down: a malformed thread configuration is
reported, not raised.
"""
from __future__ import annotations

import os

__all__ = [
    "apply_torch_thread_limits",
    "configured_thread_limits",
    "parse_positive_int",
    "torch_thread_state",
    "TORCH_INTRA_ENV",
    "TORCH_INTER_ENV",
    "TORCH_CONTROL_ENV",
]

#: Dedicated intra-op override. Absent => fall back to ``OMP_NUM_THREADS``.
TORCH_INTRA_ENV = "TORCH_INTRA_OP_THREADS"
#: Inter-op has no PyTorch-native env var; this one is read by us.
TORCH_INTER_ENV = "TORCH_INTER_OP_THREADS"
#: Kill switch. Set to 0/false/no/off to make this module a complete no-op.
TORCH_CONTROL_ENV = "TORCH_THREAD_CONTROL"

#: Set once the limits have been applied; makes the call idempotent and lets a
#: defensive second call (e.g. inside the warm-up path) prove it was already done.
_APPLIED: bool = False
#: The report from the successful call, so later callers can log the same facts.
_LAST_REPORT: dict = {}


def parse_positive_int(raw: object) -> int | None:
    """Parse a strictly positive integer from env-style input.

    Returns ``None`` for anything unusable — missing, empty, non-numeric,
    zero, or negative. Never raises: a bad value must degrade to "unset", not
    crash the process.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _control_enabled() -> bool:
    """Is thread control switched on? Absent => enabled (production default)."""
    raw = os.environ.get(TORCH_CONTROL_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def configured_thread_limits() -> dict:
    """What the environment asks for, WITHOUT touching torch.

    Returns ``{"intra": int|None, "inter": int|None, "intra_source": str|None}``.
    ``intra_source`` names the variable the intra-op value came from, which is
    what makes the startup log diagnosable when the two sources disagree.
    """
    intra = parse_positive_int(os.environ.get(TORCH_INTRA_ENV))
    intra_source = TORCH_INTRA_ENV if intra is not None else None
    if intra is None:
        intra = parse_positive_int(os.environ.get("OMP_NUM_THREADS"))
        intra_source = "OMP_NUM_THREADS" if intra is not None else None
    return {
        "intra": intra,
        "inter": parse_positive_int(os.environ.get(TORCH_INTER_ENV)),
        "intra_source": intra_source,
    }


def torch_thread_state() -> dict:
    """Current ACTUAL torch thread counts, or ``None`` if torch is unavailable.

    Read-only and side-effect free — safe to call from a debug endpoint at any
    time to confirm what the live process is really using.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 — torch is optional at import time
        return {"available": False, "intra": None, "inter": None}
    try:
        return {
            "available": True,
            "intra": int(torch.get_num_threads()),
            "inter": int(torch.get_num_interop_threads()),
        }
    except Exception:  # noqa: BLE001
        return {"available": True, "intra": None, "inter": None}


def apply_torch_thread_limits(*, force: bool = False) -> dict:
    """Apply the configured thread limits to PyTorch. Idempotent.

    Returns a report dict::

        {"applied": bool, "configured": {...}, "actual": {...}, "errors": [...]}

    ``applied`` is True only on the call that actually performed the work.
    Later calls return the cached report with ``applied=False`` so callers can
    log a defensive invocation without pretending it did anything.

    Never raises. Every failure is collected into ``errors``.
    """
    global _APPLIED, _LAST_REPORT

    if not _control_enabled():
        return {
            "applied": False,
            "skipped": f"{TORCH_CONTROL_ENV} disabled thread control",
            "configured": configured_thread_limits(),
            "actual": torch_thread_state(),
            "errors": [],
        }

    if _APPLIED and not force:
        report = dict(_LAST_REPORT)
        report["applied"] = False
        report["already_applied"] = True
        report["actual"] = torch_thread_state()
        return report

    _APPLIED = True
    configured = configured_thread_limits()
    errors: list[str] = []

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        errors.append(f"torch import failed: {type(e).__name__}: {e}")
        _LAST_REPORT = {
            "applied": True, "configured": configured,
            "actual": {"available": False, "intra": None, "inter": None},
            "errors": errors,
        }
        return dict(_LAST_REPORT)

    # Inter-op FIRST: it is the setting with an ordering constraint. If the
    # inter-op pool already exists PyTorch raises, and we want that captured
    # without losing the intra-op call that follows.
    if configured["inter"] is not None:
        try:
            torch.set_num_interop_threads(configured["inter"])
        except Exception as e:  # noqa: BLE001 — too late, or unsupported
            errors.append(
                f"set_num_interop_threads({configured['inter']}) failed: "
                f"{type(e).__name__}: {e}"
            )

    if configured["intra"] is not None:
        try:
            torch.set_num_threads(configured["intra"])
        except Exception as e:  # noqa: BLE001
            errors.append(
                f"set_num_threads({configured['intra']}) failed: "
                f"{type(e).__name__}: {e}"
            )

    _LAST_REPORT = {
        "applied": True,
        "configured": configured,
        "actual": torch_thread_state(),
        "errors": errors,
    }
    return dict(_LAST_REPORT)


def format_thread_report(report: dict) -> str:
    """One-line human-readable summary for the startup banner."""
    cfg = report.get("configured") or {}
    act = report.get("actual") or {}
    intra_src = cfg.get("intra_source") or "unset"
    parts = [
        f"intra={act.get('intra')} (configured {cfg.get('intra')} from {intra_src})",
        f"inter={act.get('inter')} (configured {cfg.get('inter')})",
    ]
    if report.get("skipped"):
        parts.append(f"SKIPPED: {report['skipped']}")
    if report.get("errors"):
        parts.append("ERRORS: " + " | ".join(report["errors"]))
    return "torch threads: " + " | ".join(parts)
