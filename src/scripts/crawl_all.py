"""Crawl orchestrator — runs all configured sources in sequence.

    python -m src.scripts.crawl_all                       # all enabled sources
    python -m src.scripts.crawl_all --sources ls rs       # specific sources only
    python -m src.scripts.crawl_all --force               # ignore interval check
    python -m src.scripts.crawl_all --dry-run             # print plan, no execution
    python -m src.scripts.crawl_all --no-ingest           # crawl only, skip ingest
    python -m src.scripts.crawl_all --status              # show last-run times, exit 0

Schedule-driven: each source block in ``config/schedule.yaml`` declares an
``interval_hours`` field.  The orchestrator skips a source if
``src.scraping.run_log.ran_recently()`` returns True for it — meaning a
*successful* crawl completed within the interval.  Pass ``--force`` to bypass.

Maintenance lock: a single-file lock (``data/crawl_all.lock``) prevents two
concurrent invocations from running the same crawlers simultaneously.  The
lock is advisory: if the process holding it crashes without cleanup, the
stale lock is detected by PID liveness and removed automatically at the next
invocation.  The lock file contains only the PID of the holder.

Failure ladder:
  - A config/usage error in a crawler (exit code 2) is recorded as a failed
    run and the source is skipped for ingest.  The orchestrator continues with
    the next source.
  - A partial-success exit (exit code 3) IS treated as ok for ingest — the
    staging data is partially populated and incrementally ingesting it is
    better than skipping.
  - A crawler that crashes (exit code not in exit_ok and not 2) is recorded
    as failed; ingest is skipped for that source.
  - An ingest failure (python -m src.scripts.ingest returns non-zero) is
    recorded in the run history.  The corpus append-only guarantee means a
    failed ingest leaves the corpus intact.
  - One source failing does NOT stop the other sources.

HPC compatibility: all paths go through ``app_paths.*()`` which respect
``APP_DATA_DIR`` / ``APP_INDEX_DIR`` env overrides.  The orchestrator is
designed to be called from cron, sbatch, or ``scripts/run_maintenance.sh``
without modification.

Exit codes:
  0  — all enabled (non-skipped) sources completed without error
  1  — one or more sources had failures (partial success)
  2  — bad usage or schedule config error
  3  — maintenance lock is already held by another process

Do NOT add APScheduler/Celery/cron here.  Scheduling is external.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.scraping.run_log import (
    KIND_CRAWL,
    KIND_INGEST,
    append_run,
    make_record,
    ran_recently,
    last_run,
)
from src.utils.app_paths import config_path, data_dir

# ─── constants ────────────────────────────────────────────────────────────────

_SCHEDULE_ENV = "CRAWL_SCHEDULE_CONFIG"
_DEFAULT_SCHEDULE = config_path("schedule.yaml")
_LOCK_FILENAME = "crawl_all.lock"

# Short names accepted by --sources (mapped to schedule.yaml keys)
_SOURCE_ALIASES: dict[str, str] = {
    "ls": "lok_sabha",
    "rs": "rajya_sabha",
    "moes": "moes_website",
    "incois": "incois",
    # full names also accepted
    "lok_sabha": "lok_sabha",
    "rajya_sabha": "rajya_sabha",
    "moes_website": "moes_website",
}


# ─── schedule loading ─────────────────────────────────────────────────────────

class ScheduleError(ValueError):
    pass


def load_schedule(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate config/schedule.yaml.

    Returns the ``sources`` sub-dict keyed by source name.
    """
    cfg_path = Path(path) if path else _schedule_path()
    if not cfg_path.exists():
        raise ScheduleError(f"schedule config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScheduleError(f"schedule config is not a mapping: {cfg_path}")
    sources = raw.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ScheduleError(f"schedule config has no 'sources' block: {cfg_path}")
    # Minimal field validation
    for name, cfg in sources.items():
        for field in ("enabled", "interval_hours", "crawler_module", "exit_ok"):
            if field not in cfg:
                raise ScheduleError(
                    f"schedule source {name!r} missing required field {field!r}"
                )
    return sources


def _schedule_path() -> Path:
    env = os.environ.get(_SCHEDULE_ENV, "").strip()
    return Path(env) if env else _DEFAULT_SCHEDULE


# ─── maintenance lock ─────────────────────────────────────────────────────────

class MaintenanceLock:
    """Advisory PID-based single-instance lock for the crawl_all run.

    Prevents two concurrent crawl_all invocations from running the same
    crawlers simultaneously (which would produce conflicting staging writes).
    Stale locks (from crashes) are detected via os.kill(pid, 0) and replaced.
    """

    def __init__(self, lock_dir: Path | None = None) -> None:
        root = lock_dir if lock_dir is not None else data_dir()
        self._path = root / _LOCK_FILENAME
        self._held = False

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> bool:
        """Try to acquire the lock.  Returns True on success, False if busy."""
        if self._path.exists():
            raw = self._path.read_text(encoding="utf-8", errors="replace").strip()
            try:
                old_pid = int(raw)
            except ValueError:
                old_pid = -1
            if old_pid > 0 and _pid_alive(old_pid):
                return False  # another live process holds the lock
            # Stale lock — remove and proceed
            _log(f"removing stale maintenance lock (pid {raw!r} not running)")
            self._path.unlink(missing_ok=True)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        self._held = True
        return True

    def release(self) -> None:
        """Remove the lock iff it still names this process."""
        if not self._held:
            return
        try:
            if self._path.exists():
                raw = self._path.read_text(encoding="utf-8", errors="replace").strip()
                if raw == str(os.getpid()):
                    self._path.unlink(missing_ok=True)
        except OSError:
            pass
        self._held = False


def _pid_alive(pid: int) -> bool:
    """Cross-platform process liveness probe (no deps)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ─── logging ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Timestamped stdout line (flush immediately for cron/sbatch capture)."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


# ─── subprocess helpers ───────────────────────────────────────────────────────

def _build_crawler_cmd(cfg: dict[str, Any], project_root: Path) -> list[str]:
    """Build the crawler subprocess command list from a schedule source block."""
    cmd = [sys.executable, "-m", cfg["crawler_module"]]
    extra_args = cfg.get("crawler_args") or []
    cmd.extend(str(a) for a in extra_args)
    config_rel = cfg.get("crawler_config")
    if config_rel:
        cmd += ["--config", str(project_root / config_rel)]
    return cmd


def _run_subprocess(cmd: list[str], label: str) -> tuple[int, float]:
    """Run a subprocess, stream its output, return (exit_code, duration_s)."""
    _log(f"  running: {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        exit_code = result.returncode
    except Exception as exc:  # noqa: BLE001
        _log(f"  [error] subprocess launch failed for {label}: {exc}")
        exit_code = -1
    duration = time.monotonic() - t0
    return exit_code, duration


# ─── status display ───────────────────────────────────────────────────────────

def _print_status(sources: dict[str, Any], data_root: Path | None) -> None:
    """Print a table of last-run times for all sources."""
    print(f"\n{'Source':<16} {'Last crawl (UTC)':<24} {'ok':<5} {'Next crawl (if overdue)'}")
    print("-" * 75)
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    for name, cfg in sources.items():
        rec = last_run(name, KIND_CRAWL, data_root=data_root)
        if rec is None:
            last = "never"
            ok_str = "-"
            next_str = "NOW"
        else:
            last = rec.get("ts", "?")
            ok_str = "yes" if rec.get("ok") else "no"
            if rec.get("ok"):
                try:
                    run_ts = datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc
                    )
                    interval = timedelta(hours=float(cfg.get("interval_hours", 168)))
                    next_run = run_ts + interval
                    if next_run > now:
                        remaining = next_run - now
                        h = int(remaining.total_seconds() // 3600)
                        m = int((remaining.total_seconds() % 3600) // 60)
                        next_str = f"in {h}h {m}m"
                    else:
                        next_str = "NOW (overdue)"
                except ValueError:
                    next_str = "?"
            else:
                next_str = "NOW (last failed)"
        enabled = cfg.get("enabled", True)
        flag = "" if enabled else " [disabled]"
        print(f"{name:<16} {last:<24} {ok_str:<5} {next_str}{flag}")
    print()


# ─── single source run ────────────────────────────────────────────────────────

def run_source(
    name: str,
    cfg: dict[str, Any],
    *,
    project_root: Path,
    dry_run: bool,
    no_ingest: bool,
    data_root: Path | None,
) -> bool:
    """Run crawl (and optionally ingest) for one source.  Returns True on full success."""
    _log(f"=== source: {name} ===")
    exit_ok: list[int] = list(cfg.get("exit_ok") or [0])
    ingest_source: str | None = cfg.get("ingest_source")
    crawl_ok = False

    # ── crawl step ────────────────────────────────────────────────────────────
    cmd = _build_crawler_cmd(cfg, project_root)
    if dry_run:
        cmd_preview = " ".join(cmd)
        _log(f"  [dry-run] would run: {cmd_preview}")
        crawl_ok = True
    else:
        exit_code, duration_s = _run_subprocess(cmd, label=f"{name} crawler")
        crawl_ok = exit_code in exit_ok
        _log(
            f"  crawler exit_code={exit_code} "
            f"({'ok' if crawl_ok else 'FAILED'}) "
            f"duration={duration_s:.1f}s"
        )
        rec = make_record(
            source=name,
            kind=KIND_CRAWL,
            exit_code=exit_code,
            duration_s=duration_s,
            ok=crawl_ok,
        )
        append_run(rec, data_root=data_root)

    if not crawl_ok:
        _log(f"  crawler failed — skipping ingest for {name}")
        return False

    # ── ingest step ───────────────────────────────────────────────────────────
    if no_ingest or not ingest_source:
        if not ingest_source:
            _log(f"  no ingest_source configured — crawl-only for {name}")
        else:
            _log(f"  [--no-ingest] skipping ingest for {name}")
        return True

    ingest_cmd = [
        sys.executable, "-m", "src.scripts.ingest",
        ingest_source, "--ingest",
    ]
    if dry_run:
        _log(f"  [dry-run] would run: {' '.join(ingest_cmd)}")
        return True

    ingest_exit, ingest_dur = _run_subprocess(ingest_cmd, label=f"{name} ingest")
    ingest_ok = ingest_exit == 0
    _log(
        f"  ingest exit_code={ingest_exit} "
        f"({'ok' if ingest_ok else 'FAILED'}) "
        f"duration={ingest_dur:.1f}s"
    )
    ingest_rec = make_record(
        source=name,
        kind=KIND_INGEST,
        exit_code=ingest_exit,
        duration_s=ingest_dur,
        ok=ingest_ok,
    )
    append_run(ingest_rec, data_root=data_root)
    return ingest_ok


# ─── main orchestration ───────────────────────────────────────────────────────

def orchestrate(
    schedule_path: Path | None,
    *,
    requested_sources: list[str] | None,
    force: bool,
    dry_run: bool,
    no_ingest: bool,
    data_root: Path | None,
    project_root: Path,
) -> int:
    """Run all enabled sources per the schedule.  Returns an exit code."""
    try:
        all_sources = load_schedule(schedule_path)
    except ScheduleError as exc:
        _log(f"[error] {exc}")
        return 2

    # Resolve requested sources (alias expansion + validation)
    if requested_sources:
        resolved: list[str] = []
        for alias in requested_sources:
            canonical = _SOURCE_ALIASES.get(alias)
            if canonical is None:
                # Accept any key directly from the schedule
                canonical = alias if alias in all_sources else None
            if canonical is None or canonical not in all_sources:
                _log(f"[error] unknown source {alias!r}. "
                     f"Valid: {sorted(all_sources)} or aliases: {sorted(_SOURCE_ALIASES)}")
                return 2
            if canonical not in resolved:
                resolved.append(canonical)
        sources_to_run = {k: all_sources[k] for k in resolved}
    else:
        sources_to_run = all_sources

    # Filter disabled sources
    sources_to_run = {
        k: v for k, v in sources_to_run.items() if v.get("enabled", True)
    }
    if not sources_to_run:
        _log("No enabled sources to run.")
        return 0

    # Plan: decide which sources to run vs skip
    plan: list[tuple[str, dict, bool]] = []  # (name, cfg, will_skip)
    for name, cfg in sources_to_run.items():
        interval = float(cfg.get("interval_hours", 168))
        if not force and ran_recently(name, interval, data_root=data_root):
            plan.append((name, cfg, True))
        else:
            plan.append((name, cfg, False))

    _log(f"Plan: {len(plan)} source(s) — "
         f"{sum(1 for _, _, skip in plan if not skip)} to run, "
         f"{sum(1 for _, _, skip in plan if skip)} to skip")
    for name, cfg, skip in plan:
        action = "SKIP (ran recently)" if skip else "RUN"
        _log(f"  {name:<16} {action}")

    if dry_run:
        _log("[dry-run] no execution — exiting")

    failures = 0
    for name, cfg, skip in plan:
        if skip:
            continue
        ok = run_source(
            name, cfg,
            project_root=project_root,
            dry_run=dry_run,
            no_ingest=no_ingest,
            data_root=data_root,
        )
        if not ok:
            failures += 1

    if failures:
        _log(f"Done — {failures} source(s) had failures (exit 1)")
        return 1
    _log("Done — all sources completed successfully (exit 0)")
    return 0


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sources", "-s", nargs="+", metavar="SOURCE",
        help="Run only these sources (accepts short aliases: ls, rs, moes, incois "
             "or full schedule.yaml keys). Default: all enabled sources.",
    )
    ap.add_argument(
        "--force", "-f", action="store_true",
        help="Ignore interval check — run even if the source ran recently.",
    )
    ap.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print what would run without executing any subprocess.",
    )
    ap.add_argument(
        "--no-ingest", action="store_true",
        help="Run crawlers only; skip the post-crawl ingest step.",
    )
    ap.add_argument(
        "--status", action="store_true",
        help="Print last-run status table for all sources and exit.",
    )
    ap.add_argument(
        "--schedule", default=None, type=Path,
        help=f"Override schedule config path (default: config/schedule.yaml "
             f"or ${_SCHEDULE_ENV}).",
    )
    ap.add_argument(
        "--no-lock", action="store_true",
        help="Skip the maintenance lock (useful when lock is held by parent process "
             "or in testing). USE WITH CAUTION.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]

    # Determine data root for run_history and lock (respects APP_DATA_DIR)
    data_root: Path | None = None  # None → data_dir() inside run_log/lock

    # ── status mode ───────────────────────────────────────────────────────────
    if args.status:
        try:
            sources = load_schedule(args.schedule)
        except ScheduleError as exc:
            _log(f"[error] {exc}")
            return 2
        _print_status(sources, data_root)
        return 0

    _log(f"crawl_all starting (force={args.force} dry_run={args.dry_run} "
         f"no_ingest={args.no_ingest})")

    # ── maintenance lock ──────────────────────────────────────────────────────
    lock = MaintenanceLock()
    if not args.no_lock:
        if not lock.acquire():
            _log(f"[error] maintenance lock is held by another process "
                 f"(see {lock.path}). Aborting to prevent concurrent crawler runs.")
            return 3
        _log(f"maintenance lock acquired: {lock.path}")

    try:
        return orchestrate(
            args.schedule,
            requested_sources=args.sources,
            force=args.force,
            dry_run=args.dry_run,
            no_ingest=args.no_ingest,
            data_root=data_root,
            project_root=project_root,
        )
    finally:
        if not args.no_lock:
            lock.release()
            _log("maintenance lock released")


if __name__ == "__main__":
    sys.exit(main())
