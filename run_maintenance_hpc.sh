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
PID_FILE="runtime/app.pid"

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

# ── remember whether the web app is up ────────────────────────────────────────
# Sampled BEFORE the crawl: the crawl/ingest runs in its own container process
# and writes the index to disk, but the running server keeps its pipeline in
# memory — it only sees the new vectors after a restart (see the restart block
# at the end of this script). If the app was down, we leave it down.
APP_WAS_RUNNING=0
if [ -f "$PID_FILE" ]; then
    APP_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        APP_WAS_RUNNING=1
    fi
fi

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

# ── GPU for embedding (opt-in, detected) ──────────────────────────────────────
# Embedding is GPU-accelerated only when this job actually lands on a node with
# visible GPUs. cron frequently runs on a login/service node without one, and
# forcing CUDA there would crash the run — so fall back to CPU (correct, just
# slower on the rare full rebuild that a changed record triggers).
GPU_ARGS=()
if ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
    GPU_ARGS=(--nv --env EMBED_DEVICE=cuda)
    echo "[maintenance] GPU detected — embedding on CUDA (EMBED_DEVICE=cuda)"
else
    echo "[maintenance] no GPU visible — embedding on CPU"
fi

# Temporarily disable exit-on-error for the crawl command ONLY.
# crawl_all.py exits 1 when any source has a partial failure (it still
# ingests what it got). Under 'set -e' the shell would exit before
# CRAWL_EXIT=$? executes, skipping the logging block and app-restart
# below. 'set +e' is re-enabled immediately after capture so all
# subsequent commands remain under strict error handling.
set +e
singularity exec \
    ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
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
set -e

echo "[maintenance] ============================================================"
echo "[maintenance] run finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[maintenance] exit code: $CRAWL_EXIT"
echo "[maintenance] ============================================================"

# ── make the new vectors live ────────────────────────────────────────────────
# crawl_all runs ingest in a SEPARATE container process: it writes the index to
# /storage/hybrid_rag on disk, but the running server holds its pipeline in
# memory and never re-reads it (live swap only happens on the UI upload path).
# So a successful run that actually touched the index needs an app restart —
# otherwise the freshly crawled documents stay invisible until someone
# restarts by hand. Marker line printed by src/scripts/ingest.py:
#   [ingest] done: N new record(s) appended, M changed (replaced in corpus); index: <action>
#
# NOTE: CRAWL_EXIT == 0 is NOT required for restart.  crawl_all exits 1 when
# any individual source has a partial failure (e.g. INCOIS crashes) but earlier
# sources (MoES, parliament) may have already written a new index to disk.
# Without this change those successful updates would stay invisible in the
# running app's in-memory pipeline until a manual restart.
# CRAWL_EXIT is still preserved as the final exit code of this script.
if [ "$APP_WAS_RUNNING" -eq 1 ] \
   && grep -qE "\[ingest\] done: .*; index: (incremental|rebuild)" "$RUN_LOG"; then
    echo "[maintenance] index was updated — restarting the app to load it"
    ./stop_hpc.sh || true
    # nohup: the app must survive this cron job's session teardown.
    nohup ./start_hpc.sh >> "$RUN_LOG" 2>&1 \
        || echo "[maintenance] WARNING: app restart failed — new records stay invisible until a manual restart"
fi

exit $CRAWL_EXIT
