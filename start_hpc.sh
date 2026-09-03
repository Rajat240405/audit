#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

SIF="incois-audit-app-hpc16.sif"
ENV_FILE=".env.hpc"

TMP_DIR="runtime/tmp"
LOG_DIR="runtime/logs"
PID_FILE="runtime/app.pid"

echo "🚀 Starting INCOIS Audit Platform..."

# -----------------------------
# Validation
# -----------------------------

if [ ! -f "$SIF" ]; then
    echo "❌ ERROR: $SIF not found."
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ ERROR: $ENV_FILE not found."
    exit 1
fi

# -----------------------------
# Runtime directories
# -----------------------------

mkdir -p "$TMP_DIR"
mkdir -p "$LOG_DIR"

export SINGULARITY_TMPDIR="$(pwd)/$TMP_DIR"
export SINGULARITY_CACHEDIR="$(pwd)/$TMP_DIR"
export APPTAINER_TMPDIR="$(pwd)/$TMP_DIR"
export APPTAINER_CACHEDIR="$(pwd)/$TMP_DIR"

# -----------------------------
# Prevent duplicate startup
# -----------------------------

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")

    if kill -0 "$PID" 2>/dev/null; then
        echo "❌ ERROR: Application already running (PID $PID)."
        echo "👉 Run ./stop_hpc.sh first."
        exit 1
    fi

    rm -f "$PID_FILE"
fi

# -----------------------------
# Start application
# -----------------------------
# IMPORTANT: The vLLM server (separate process) MUST be launched with:
#   --reasoning-parser qwen3
# Without this flag, reasoning tokens appear as <think> tags inside
# "content" instead of the separate "reasoning_content" field, breaking
# the frontend's reasoning panel and answer separation.
#
# Example vLLM launch command:
#   python -m vllm.entrypoints.openai.api_server \
#     --model Qwen3.6-35B-A3B-FP8 \
#     --port 8100 \
#     --reasoning-parser qwen3

echo "📦 Starting Singularity container..."

singularity exec \
    --nv \
    --env-file "$ENV_FILE" \
    --writable-tmpfs \
    --bind "$(realpath data):/data" \
    --bind "$(realpath storage/hybrid_rag):/storage/hybrid_rag" \
    --bind "$(realpath models):/models" \
    --bind "$(realpath config):/app/config" \
    --bind "$(realpath "$TMP_DIR"):/tmp" \
    "$SIF" \
    python -m uvicorn src.retrieval.frontend.server:app \
    --host 127.0.0.1 \
    --port 18000 \
    > "$LOG_DIR/app.log" 2>&1 &

PID=$!

echo "$PID" > "$PID_FILE"

echo "✅ Application started."
echo "PID: $PID"
echo "Log: $LOG_DIR/app.log"
echo "URL: http://localhost:18000"
