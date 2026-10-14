"""Infrastructure health for table routing: preflight and failure classification.

Two kinds of failure must never be confused:

* **Infrastructure** — PicoDet cannot run, DOTS is unreachable / timing out /
  serving the wrong model, or a required extraction dependency is missing.
  The whole source must stop: no new ingestion, and a non-zero exit code that
  ``crawl_all`` does not accept (see :data:`EXIT_INFRA`). Existing corpus rows
  are left intact because ingestion is skipped.

* **Per-document** — DOTS answered, but its output was rejected by the
  validator, or one PDF timed out. That document fails. Its previous corpus
  row is retained by the staged-answer guards.

Identifiers (grep-able in logs and in the run history):

``PICODET_UNAVAILABLE``   detector cannot initialise, cannot run, or is disabled
``DOTS_UNAVAILABLE``      DOTS endpoint unreachable, not serving the pinned model
``DEPENDENCY_UNAVAILABLE`` a required extraction dependency (PyMuPDF, Tesseract)
"""

from __future__ import annotations

import logging
from typing import Iterable

log = logging.getLogger(__name__)

#: Exit code for an infrastructure failure. Deliberately distinct from 2
#: (config/usage), 3 (partial success) and 1 (crash). It is NOT in any source's
#: ``exit_ok`` list in ``config/schedule.yaml``, so ``crawl_all`` skips ingest.
EXIT_INFRA = 4

CODE_PICODET = "PICODET_UNAVAILABLE"
CODE_DOTS = "DOTS_UNAVAILABLE"
CODE_DEPENDENCY = "DEPENDENCY_UNAVAILABLE"


class TableRoutingPreflightError(RuntimeError):
    """One or more routing dependencies failed the health preflight."""

    def __init__(self, failures: Iterable[tuple[str, str]]) -> None:
        self.failures: list[tuple[str, str]] = [(c, str(d)) for c, d in failures]
        lines = [f"{code}: {detail}" for code, detail in self.failures]
        super().__init__("; ".join(lines) or "table routing preflight failed")

    @property
    def codes(self) -> list[str]:
        return [c for c, _ in self.failures]


def routing_required() -> bool:
    """Is table routing (PicoDet -> DOTS) switched on for this process?"""
    from src.data.pdf_table_extract import dots_routing_enabled

    return dots_routing_enabled()


def preflight_table_routing() -> None:
    """Verify PicoDet initialises and runs, and DOTS serves the configured model.

    No-op when routing is off. Runs at the start of every LS and RS crawl so a
    scheduled run fails fast on an outage, including runs whose documents are
    all served from the extraction cache (zero new PicoDet/DOTS calls there, but
    the health check still runs). Both dependencies are checked even when one
    fails, so the operator sees every broken component in one run.

    Raises :class:`TableRoutingPreflightError`.
    """
    if not routing_required():
        return

    from src.data import dots_client, table_detect

    failures: list[tuple[str, str]] = []
    try:
        table_detect.ensure_ready()
    except Exception as exc:  # noqa: BLE001 - classify every detector fault
        failures.append((CODE_PICODET, f"{type(exc).__name__}: {exc}"))
    try:
        dots_client.get_client().health_check()
    except Exception as exc:  # noqa: BLE001 - classify every DOTS fault
        failures.append((CODE_DOTS, f"{type(exc).__name__}: {exc}"))

    if failures:
        for code, detail in failures:
            log.error("[preflight] %s %s", code, detail)
        raise TableRoutingPreflightError(failures)
    import sys

    print("[preflight] table routing ready: PicoDet initialised and ran a probe inference; "
          "DOTS serving the configured model", file=sys.stdout, flush=True)


def infrastructure_error_code(exc: BaseException) -> str | None:
    """Map an exception to its infrastructure identifier, or None if it is not one.

    Resolved by type with lazy imports so a checkout without the optional
    extraction stack still imports cleanly.
    """
    if isinstance(exc, TableRoutingPreflightError):
        return ",".join(exc.codes) or CODE_PICODET
    try:
        from src.data.dots_client import DotsUnavailable

        if isinstance(exc, DotsUnavailable):
            return CODE_DOTS
    except ImportError:
        pass
    try:
        from src.data.table_detect import TableDetectorUnavailable

        if isinstance(exc, TableDetectorUnavailable):
            return CODE_PICODET
    except ImportError:
        pass
    try:
        from src.data.pdf_table_extract import DependencyMissingError

        if isinstance(exc, DependencyMissingError):
            return CODE_DEPENDENCY
    except ImportError:
        pass
    return None


def is_infrastructure_failure(exc: BaseException) -> bool:
    return infrastructure_error_code(exc) is not None


def report_infrastructure_failure(exc: BaseException, *, source: str) -> int:
    """Print the greppable line for an infrastructure failure and return EXIT_INFRA."""
    import sys

    code = infrastructure_error_code(exc) or "INFRA_UNAVAILABLE"
    print(
        f"[infra] {source}: {code} — stopping with exit {EXIT_INFRA}; "
        f"crawl_all skips ingest for this source: {exc}",
        file=sys.stderr, flush=True,
    )
    return EXIT_INFRA


__all__: list[str] = [
    "CODE_DEPENDENCY",
    "CODE_DOTS",
    "CODE_PICODET",
    "EXIT_INFRA",
    "TableRoutingPreflightError",
    "infrastructure_error_code",
    "is_infrastructure_failure",
    "preflight_table_routing",
    "report_infrastructure_failure",
    "routing_required",
]
