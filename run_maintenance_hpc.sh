#!/bin/bash
# run_maintenance_hpc.sh — scheduled (daily) source-maintenance for HPC/Singularity
#
# Invoked by crond (daily cron job).  Runs the full crawl → ingest → embed
# cycle inside the same Singularity container used by start_hpc.sh / stop_hpc.sh,
# with the same bind mounts and env-file.  Does NOT touch the running app
# process or its PID file.
#
# Usage (manual / test):
#   ./run_maintenance_hpc.sh                   # respect schedule intervals
#   ./run_maintenance_hpc.sh --force           # ignore interval check, run all
#   ./run_maintenance_hpc.sh --dry-run         # print plan, no crawl/ingest
#   ./run_maintenance_hpc.sh --sources rs ls   # specific sources only
#   ./run_maintenance_hpc.sh --status          # show last-run table, exit 0
#   ./run_maintenance_hpc.sh --no-ingest       # crawl only, skip ingest+embed
#
# Cron schedule (edit with: crontab -e):
#   # Daily 02:00 — crawl all sources, ingest, rebuild index if needed
#   0 2 * * * /path/to/project/run_maintenance_hpc.sh >> /path/to/project/runtime/logs/cron.log 2>&1
#
# Exit codes (mirrored from crawl_all.py):
#   0  all sources completed without failures
#   1  one or more sources had partial failures (crawl still ingested what it got)
#   2  bad invocation / schedule config error
#   3  maintenance lock already held by another process (concurrent run protection)
#  >3  unexpected / singularity failure

set -euo pipefail

# ── locate project root (always the directory containing this script) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── config ────────────────────────────────────────────────────────────────────
SIF="incois-audit-app-hpc16.sif"
ENV_FILE=".env.hpc"

TMP_DIR="runtime/tmp"
LOG_BASE="runtime/logs"
LOG_DIR="${LOG_BASE}/maintenance"
LATEST_LINK="${LOG_BASE}/maintenance.log"

# ── validate required files ───────────────────────────────────────────────────
if [ ! -f "$SIF" ]; then
    echo "[maintenance] ERROR: $SIF not found in $SCRIPT_DIR" >&2
    exit 2
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "[maintenance] ERROR: $ENV_FILE not found in $SCRIPT_DIR" >&2
    exit 2
fi

# ── ensure singularity is available ───────────────────────────────────────────
# cron runs with a minimal PATH; add the most common install locations.
# Hardcoded first: this HPC keeps singularity outside the system module
# paths the fallback below probes (verified: /home/apps/singularity_ce_4.0.0,
# module name "singularity/4.0.0").
export PATH="/home/apps/singularity_ce_4.0.0/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
if ! command -v singularity >/dev/null 2>&1; then
    # HPC modules system (try common module locations)
    for MODFILE in \
        /etc/profile.d/modules.sh \
        /usr/share/lmod/lmod/init/bash \
        /usr/local/lmod/lmod/init/bash; do
        if [ -f "$MODFILE" ]; then
            # shellcheck source=/dev/null
            source "$MODFILE" 2>/dev/null || true
            module load singularity 2>/dev/null || true
            break
        fi
    done
fi
if ! command -v singularity >/dev/null 2>&1; then
    echo "[maintenance] ERROR: singularity not found on PATH after module load attempt." >&2
    echo "[maintenance] Add singularity to PATH or edit the PATH line above." >&2
    exit 2
fi

# ── runtime directories ───────────────────────────────────────────────────────
mkdir -p "$TMP_DIR"
mkdir -p "$LOG_DIR"

# Temp/cache for the SIF unpack (multi-GB, --writable-tmpfs): never the
# login node's tiny /tmp — same fix as start_hpc.sh (no-space crash).
export SINGULARITY_TMPDIR="$SCRIPT_DIR/$TMP_DIR"
export SINGULARITY_CACHEDIR="$SCRIPT_DIR/$TMP_DIR"
export APPTAINER_TMPDIR="$SCRIPT_DIR/$TMP_DIR"
export APPTAINER_CACHEDIR="$SCRIPT_DIR/$TMP_DIR"

# ── per-run timestamped log ───────────────────────────────────────────────────
RUN_TS="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_LOG="${LOG_DIR}/${RUN_TS}.log"

# Redirect all output from this point to the timestamped log AND stdout/stderr
# (stdout/stderr are inherited from cron, which appends to cron.log above).
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[maintenance] ============================================================"
echo "[maintenance] run started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[maintenance] script:      $SCRIPT_DIR/run_maintenance_hpc.sh"
echo "[maintenance] SIF:         $SCRIPT_DIR/$SIF"
echo "[maintenance] log:         $RUN_LOG"
echo "[maintenance] args:        $*"
echo "[maintenance] ============================================================"

# Update the "latest" symlink so maintenance.log always points at the current run.
ln -sfn "maintenance/${RUN_TS}.log" "$LATEST_LINK"

# ── bind mount paths ──────────────────────────────────────────────────────────
DATA_REAL="$(realpath data)"
STORAGE_REAL="$(realpath storage/hybrid_rag)"
MODELS_REAL="$(realpath models)"
TMP_REAL="$(realpath "$TMP_DIR")"
CONFIG_REAL="$(realpath config)"

# ── run crawl_all inside the container ───────────────────────────────────────
CRAWL_ARGS=("$@")

echo "[maintenance] invoking crawl_all.py with args: ${CRAWL_ARGS[*]:-<none>}"

singularity exec \
    --env-file    "$ENV_FILE" \
    --writable-tmpfs \
    --pwd         /app \
    --bind        "${DATA_REAL}:/data" \
    --bind        "${STORAGE_REAL}:/storage/hybrid_rag" \
    --bind        "${MODELS_REAL}:/models" \
    --bind        "${CONFIG_REAL}:/app/config" \
    --bind        "${TMP_REAL}:/tmp" \
    "$SIF" \
    python -m src.scripts.crawl_all "${CRAWL_ARGS[@]}"

CRAWL_EXIT=$?

echo "[maintenance] ============================================================"
echo "[maintenance] run finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[maintenance] exit code: $CRAWL_EXIT"
echo "[maintenance] ============================================================"

exit $CRAWL_EXIT
