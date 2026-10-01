#!/bin/bash
# =============================================================================
# start_dots_vllm_un003.sh — serve dots.ocr on UN003 (1x NVIDIA A40, 48 GB)
# =============================================================================
#
# This starts the DOTS OCR **server**. It is a separate process from the
# extraction app: the app is only ever an HTTP client of this endpoint
# (DOTS_BASE_URL), exactly as it is an HTTP client of the generation vLLM.
# Nothing here goes inside the SIF — the SIF stays a stable application
# artifact and every runtime knob lives in env_hpc.sh / the environment.
#
#   ./start_dots_vllm_un003.sh            # start, wait for health, detach
#   ./start_dots_vllm_un003.sh --fg       # run in the foreground
#   ./start_dots_vllm_un003.sh --status   # is it up, and what is it serving?
#   ./start_dots_vllm_un003.sh --stop     # graceful shutdown
#
# ── Why these settings ───────────────────────────────────────────────────────
#
# BF16, not 4-bit.
#   The A40 is Ampere (GA102, CC 8.6). It has no FP8 W8A8 hardware path, so
#   FP8 brings nothing here. bitsandbytes 4-bit measures 43-64% SLOWER than
#   BF16 under batched serving and raises TTFT sharply; its advantage appears
#   only at batch=1 where decode is bandwidth-bound. dots.ocr is ~3B
#   parameters, so BF16 weights are roughly 6 GB of 48 GB — there is no
#   memory pressure to trade quality or throughput against.
#
# --tensor-parallel-size 1, no pipeline parallelism.
#   One GPU. Pipeline parallelism is additionally reported broken for this
#   model family, so it is avoided outright.
#
# --chat-template-content-format string
#   Required by the official dots.ocr serving recipe.
#
# --limit-mm-per-prompt image=1
#   Pages are sent ONE at a time. Whole multi-page PDFs are never submitted.
#
# Concurrency.
#   DOTS_MAX_NUM_SEQS is intentionally modest. vLLM preallocates KV cache for
#   max_num_seqs x max_model_len, and OCR prompts are image-heavy, so raising
#   it buys queueing rather than throughput and invites preemption. Let vLLM
#   own the queue; do not batch client-side.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ── Runtime-tunable settings ─────────────────────────────────────────────────
# Read from the SAME env file the SIF uses (.env.hpc by default), so there is
# one place to tune and no second config to keep in sync.
#
# The file is PARSED, never sourced. It is dotenv, not shell: values are
# unquoted and routinely contain characters bash would interpret —
# `GRAPHRAG_NEO4J_PASSWORD=<secret>` makes `<` a redirect and aborts the
# script, and a password containing a space or `$` corrupts silently. Only
# DOTS_* / PICODET_* keys are imported, so nothing in this file can overwrite
# PATH, HOME or any other shell variable.

ENV_FILE="${DOTS_ENV_FILE:-.env.hpc}"
[ -f "$ENV_FILE" ] || ENV_FILE="env_hpc.sh"

load_env_file() {
    local file="$1" line key value
    [ -f "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|'#'*) continue ;;
        esac
        line="${line#export }"
        case "$line" in
            DOTS_*=*|PICODET_*=*) ;;
            *) continue ;;
        esac
        key="${line%%=*}"
        value="${line#*=}"
        # strip one layer of matching quotes; never evaluate the contents
        case "$value" in
            \"*\") value="${value#\"}"; value="${value%\"}" ;;
            \'*\') value="${value#\'}"; value="${value%\'}" ;;
        esac
        # environment wins over the file, so one-off overrides still work
        if [ -z "$(eval "printf '%s' \"\${$key:-}\"")" ]; then
            printf -v "$key" '%s' "$value"
            export "${key?}"
        fi
    done < "$file"
    echo "[dots] loaded DOTS_*/PICODET_* settings from $file"
}
load_env_file "$ENV_FILE"

DOTS_MODEL_PATH="${DOTS_MODEL_PATH:-/models/dots-ocr}"
DOTS_SERVED_NAME="${DOTS_SERVED_NAME:-dots-ocr}"
DOTS_HOST="${DOTS_HOST:-0.0.0.0}"
DOTS_PORT="${DOTS_PORT:-8200}"
DOTS_GPU_UTIL="${DOTS_GPU_UTIL:-0.85}"
DOTS_MAX_MODEL_LEN="${DOTS_MAX_MODEL_LEN:-24576}"
DOTS_MAX_NUM_SEQS="${DOTS_MAX_NUM_SEQS:-8}"
DOTS_DTYPE="${DOTS_DTYPE:-bfloat16}"
DOTS_TP_SIZE="${DOTS_TP_SIZE:-1}"
DOTS_CUDA_DEVICES="${DOTS_CUDA_DEVICES:-0}"
DOTS_HEALTH_TIMEOUT="${DOTS_HEALTH_TIMEOUT:-600}"
DOTS_EXTRA_ARGS="${DOTS_EXTRA_ARGS:-}"

LOG_DIR="${DOTS_LOG_DIR:-runtime/logs}"
PID_FILE="${DOTS_PID_FILE:-runtime/dots_vllm.pid}"
LOG_FILE="$LOG_DIR/dots_vllm.log"
BASE_URL="http://127.0.0.1:${DOTS_PORT}/v1"

mkdir -p "$LOG_DIR" "$(dirname "$PID_FILE")"

# ── Subcommands ──────────────────────────────────────────────────────────────

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-}" in
    --status)
        if is_running; then
            echo "[dots] running (pid $(cat "$PID_FILE")) on port ${DOTS_PORT}"
        else
            echo "[dots] not running"
        fi
        echo "[dots] querying ${BASE_URL}/models ..."
        curl -fsS --max-time 5 "${BASE_URL}/models" 2>/dev/null \
            || echo "[dots] endpoint not responding"
        exit 0
        ;;
    --stop)
        if is_running; then
            PID="$(cat "$PID_FILE")"
            echo "[dots] stopping pid $PID ..."
            kill -TERM "$PID" 2>/dev/null || true
            for _ in $(seq 1 30); do
                kill -0 "$PID" 2>/dev/null || break
                sleep 1
            done
            kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "[dots] stopped"
        else
            echo "[dots] not running"
        fi
        exit 0
        ;;
esac

# ── Preflight ────────────────────────────────────────────────────────────────

if is_running; then
    echo "❌ ERROR: already running (pid $(cat "$PID_FILE")). Use --stop first."
    exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "❌ ERROR: 'vllm' not found on PATH."
    echo "   dots.ocr is officially integrated from vLLM 0.11.0 onward."
    exit 1
fi

if [ ! -d "$DOTS_MODEL_PATH" ]; then
    echo "❌ ERROR: model directory not found: $DOTS_MODEL_PATH"
    echo "   Set DOTS_MODEL_PATH in $ENV_FILE."
    exit 1
fi

# The upstream loader has historically mishandled a '.' in the model directory
# name, which is why the published weights are usually staged as 'dots-ocr'.
case "$(basename "$DOTS_MODEL_PATH")" in
    *.*) echo "⚠️  WARNING: '$(basename "$DOTS_MODEL_PATH")' contains a '.'."
         echo "   If loading fails, re-stage the weights in a dot-free directory." ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[dots] GPU:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used \
               --format=csv,noheader 2>/dev/null | sed 's/^/        /'
fi

echo "[dots] model      : $DOTS_MODEL_PATH (served as '$DOTS_SERVED_NAME')"
echo "[dots] endpoint   : http://${DOTS_HOST}:${DOTS_PORT}/v1"
echo "[dots] dtype      : $DOTS_DTYPE   tp=$DOTS_TP_SIZE   gpu-util=$DOTS_GPU_UTIL"
echo "[dots] max_len    : $DOTS_MAX_MODEL_LEN   max_num_seqs=$DOTS_MAX_NUM_SEQS"

# ── Launch ───────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="$DOTS_CUDA_DEVICES"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

build_cmd() {
    # shellcheck disable=SC2206
    CMD=(
        vllm serve "$DOTS_MODEL_PATH"
        --served-model-name "$DOTS_SERVED_NAME"
        --host "$DOTS_HOST"
        --port "$DOTS_PORT"
        --tensor-parallel-size "$DOTS_TP_SIZE"
        --gpu-memory-utilization "$DOTS_GPU_UTIL"
        --max-model-len "$DOTS_MAX_MODEL_LEN"
        --max-num-seqs "$DOTS_MAX_NUM_SEQS"
        --dtype "$DOTS_DTYPE"
        --chat-template-content-format string
        --limit-mm-per-prompt image=1
        --trust-remote-code
    )
    if [ -n "$DOTS_EXTRA_ARGS" ]; then
        EXTRA=($DOTS_EXTRA_ARGS)
        CMD+=("${EXTRA[@]}")
    fi
}
build_cmd

if [ "${1:-}" = "--fg" ]; then
    echo "[dots] starting in foreground (Ctrl-C to stop)"
    exec "${CMD[@]}"
fi

echo "[dots] starting in background -> $LOG_FILE"
nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[dots] pid $(cat "$PID_FILE")"

# ── Wait for health ──────────────────────────────────────────────────────────
# Model load plus CUDA graph capture takes minutes on first start; the app
# must not begin extraction until /v1/models answers.

echo "[dots] waiting up to ${DOTS_HEALTH_TIMEOUT}s for the endpoint ..."
ELAPSED=0
while [ "$ELAPSED" -lt "$DOTS_HEALTH_TIMEOUT" ]; do
    if ! is_running; then
        echo "❌ ERROR: server exited during startup. Last 40 log lines:"
        tail -40 "$LOG_FILE" | sed 's/^/        /'
        rm -f "$PID_FILE"
        exit 1
    fi
    if curl -fsS --max-time 3 "${BASE_URL}/models" >/dev/null 2>&1; then
        echo "✅ [dots] ready after ${ELAPSED}s"
        curl -fsS --max-time 5 "${BASE_URL}/models" | sed 's/^/        /'
        echo
        echo "Point the extraction app at it:"
        echo "    export DOTS_ENABLED=true"
        echo "    export DOTS_BASE_URL=http://$(hostname):${DOTS_PORT}/v1"
        echo "    export DOTS_MODEL=${DOTS_SERVED_NAME}"
        exit 0
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo "❌ ERROR: not healthy after ${DOTS_HEALTH_TIMEOUT}s. Last 40 log lines:"
tail -40 "$LOG_FILE" | sed 's/^/        /'
exit 1
