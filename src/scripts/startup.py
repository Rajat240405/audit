"""Phase 4 — application startup modes (ONE launcher, TWO explicit modes).

    # Mode B — startup WITHOUT ingest (read-only serving):
    APP_MODE=serve python -m src.scripts.startup

    # Mode A — startup WITH ingest (incremental), then serve:
    APP_MODE=ingest python -m src.scripts.startup

The mode comes from the EXISTING ``APP_MODE`` environment variable (extended,
not duplicated — Phase 4 adds no competing mechanism):

  APP_MODE=serve            startup without ingest. Boot performs NO ingestion,
     (or unset/unknown)     NO index rebuild, NO embeddings, NO corpus writes and
                            NO model downloads. The existing corpus/indexes are
                            used exactly as they are (the serving layer loads
                            the saved index lazily on first request). With the
                            literal value ``serve`` the ingest/upload API
                            endpoints stay blocked (403) — unchanged pre-Phase-4
                            behavior of src/retrieval/frontend/server.py.

  APP_MODE=ingest           startup WITH ingest: the Phase-2/3 pipeline runs
  APP_MODE=build            BEFORE the server binds a port —

                                discover registered ∪ discovered sources
                                    ↓   (src.scripts.ingest, "all")
                                extract → dedup → append-only corpus write
                                    ↓
                                embed ONLY genuinely new records
                                    ↓   (incremental FAISS add + BM25 refresh)
                                serve normally

                            Nothing new  → zero index work, serve.
                            No usable index → FIRST build (full) — the only
                            implicit full rebuild, same contract as the CLI.
                            A failed incremental update NEVER falls back to a
                            full rebuild: startup exits 3 (loudly), corpus
                            intact (append-only), on-disk index untouched.

The one-shot equivalent of Mode A for HPC job steps stays the CLI::
    python -m src.scripts.ingest all --ingest    (exit 0 / 2 / 3)

Ingestion never starts or talks to vLLM: embeddings are local bge-m3 (CPU).
The serving phase connects to an already-running vLLM via VLLM_BASE_URL,
exactly as before.

Startup/shutdown safety (non-destructive):
  * a PID file (``<storage>/app.pid``) refuses a duplicate application
    process (exit 2) — including an old crashed-but-restarted container case;
  * when Python unwinds normally (Ctrl+C on a terminal, programmatic stop),
    the finally-block removes ONLY our own PID file (it never deletes a file
    naming another process);
  * when the process exits without unwinding (kill -9, or a signal-driven
    server shutdown that hard-exits the interpreter — observed: uvicorn
    0.52 + CPython 3.13 runs NO finally/atexit code after SIGTERM), the
    leftover file is automatically classified as STALE at the next boot and
    replaced — startup after a previous crash is identical either way;
  * no index/corpus cleanup is ever performed here.

Ingestion failure policy: exit 3 and DO NOT serve — the operator sees a loud
failure instead of a silently stale service. Recovery: fix the cause and re-run
the same command (dedup makes re-runs safe), or serve the last-good index with
APP_MODE=serve.

Exit codes: 0 clean shutdown · 2 duplicate process / bad usage · 3 ingest
failed (matches python -m src.scripts.ingest).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.utils.app_paths import storage_dir

# APP_MODE values that mean "run the ingestion job at boot, then serve".
# Unset, "serve", and anything unrecognized are SAFE: no ingest at boot.
_INGEST_MODES = {"ingest", "build"}


def _log(msg: str) -> None:
    print(f"[startup] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mode resolution (config)
# ─────────────────────────────────────────────────────────────────────────────

def startup_mode(env: dict | None = None) -> str:
    """Resolve the startup mode from APP_MODE -> "serve" | "ingest".

    Only the documented job values trigger boot ingestion; anything else
    (unset, "serve", typos, future values) must never mutate the corpus or
    index — safe default is "serve". The literal APP_MODE value is left
    untouched in os.environ: the API gate in the server (``serve`` blocks
    /api/ingest|/api/upload, ingest/build allow them) keeps its exact
    pre-Phase-4 semantics.
    """
    raw = (env if env is not None else os.environ).get("APP_MODE", "")
    return "ingest" if raw.strip().lower() in _INGEST_MODES else "serve"


# ─────────────────────────────────────────────────────────────────────────────
# PID guard — duplicate-process refusal + stale recovery (non-destructive)
# ─────────────────────────────────────────────────────────────────────────────

def pid_path() -> Path:
    """Application PID file. Lives under the storage root (env-movable via
    APP_INDEX_DIR -> /storage on HPC/containers), never the CWD."""
    return storage_dir() / "app.pid"


def _pid_alive(pid: int) -> bool:
    """Cross-platform liveness probe (no third-party deps)."""
    if pid <= 0:
        return False
    if os.name == "nt":  # Windows: OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # ERROR_ACCESS_DENIED (5): process exists but owned by another user.
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)  # POSIX: signal 0 = existence/permission probe
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _PidGuard:
    """Best-effort single-instance guard for the application process.

    Not a kernel lock (PID reuse is theoretically possible on the exact
    boundary); it is the documented startup/shutdown safety net: duplicates
    are refused, stale files (crash, kill -9, or a signal-exit that never
    unwound Python) are replaced at the next boot, and when Python does
    unwind, release() removes only a file that still names this process.
    """

    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else pid_path()
        self._mine: int | None = None

    def acquire(self) -> None:
        """Claim the PID file or refuse to start (SystemExit(2))."""
        if self.path.exists():
            raw = self.path.read_text(encoding="utf-8", errors="replace").strip()
            try:
                old_pid = int(raw)
            except ValueError:
                old_pid = -1
            if old_pid > 0 and _pid_alive(old_pid):
                _log(f"REFUSING to start: another application process is "
                     f"already running (pid {old_pid}, declared in {self.path}). "
                     f"Stop it first; if that PID is stale, delete the file.")
                raise SystemExit(2)
            _log(f"stale PID file {self.path} "
                 f"(pid {raw!r} not running — prior shutdown or crash); replacing")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        self._mine = os.getpid()
        _log(f"pid {self._mine} → {self.path}")

    def release(self) -> None:
        """Remove the PID file iff it still names THIS process (never another's).

        Best-effort: only runs when Python unwinds through run()'s finally.
        If the interpreter hard-exits after a signal, the leftover file is
        handled by the next boot's stale classification — same outcome.
        """
        if self._mine is None:
            return
        try:
            if self.path.exists() and self.path.read_text(
                    encoding="utf-8", errors="replace").strip() == str(self._mine):
                self.path.unlink()
                _log(f"shutdown complete — removed PID file {self.path}")
        except OSError as e:  # never fail shutdown on guard bookkeeping
            _log(f"warning: could not remove PID file {self.path}: {e}")
        self._mine = None


# ─────────────────────────────────────────────────────────────────────────────
# Phase seams (test injection points — production delegates to existing code)
# ─────────────────────────────────────────────────────────────────────────────

def _ingest_all() -> dict:
    """Mode-A ingest phase: the EXISTING Phase-2/3 pipeline, unchanged.

    Same code path as `python -m src.scripts.ingest all --ingest`:
    registry ∪ discovery → engine conversion → atomic append-only corpus →
    embed phase decision (skip / incremental / first-build rebuild). Raises
    RuntimeError when the incremental update fails (no rebuild fallback
    exists anywhere in this chain).
    """
    from src.scripts import ingest as ingest_cli

    specs, _, category_map = ingest_cli.resolve_sources("all", None)
    return ingest_cli.run_sources(specs, category_map=category_map)


def _serve(port: int) -> None:
    """Mode-final phase: start serving exactly as the pre-Phase-4 CMD did."""
    from src.retrieval.frontend import server as _server

    _server.start_server(port=port)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(mode: str, port: int) -> None:
    guard = _PidGuard()
    guard.acquire()
    try:
        if mode == "ingest":
            _log("phase 1/2: source discovery + incremental ingest "
                 "(only genuinely new documents are embedded)")
            try:
                result = _ingest_all()
            except RuntimeError as e:
                # NO fallback, NO half-served state: the corpus append is
                # intact (append-only) and the on-disk index was untouched.
                _log(f"INGEST FAILED — NOT serving. {e}")
                _log("recovery: fix the cause and re-run (dedup makes re-runs "
                     "safe), or start the last-good index with APP_MODE=serve.")
                raise SystemExit(3)
            _log(f"ingest done: {result.get('added', 0)} new record(s); "
                 f"index action: {result.get('embed')}")
            _log("phase 2/2: serving")
        else:
            _log("startup without ingest — no corpus/index/embedding work; "
                 "using the existing index exactly as saved")
        _serve(port)
    finally:
        guard.release()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000,
                    help="port for the FastAPI/Uvicorn app (default 8000)")
    args = ap.parse_args(argv)

    mode = startup_mode()
    raw = os.environ.get("APP_MODE", "") or "(unset)"
    _log(f"APP_MODE={raw} → mode={mode} "
         f"({'startup + ingest' if mode == 'ingest' else 'startup without ingest'})")
    run(mode, args.port)


if __name__ == "__main__":
    main()
