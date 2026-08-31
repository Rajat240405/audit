#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PID_FILE="runtime/app.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "ℹ️ No running application found."
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "🛑 Stopping application (PID $PID)..."
    kill -TERM "$PID"

    for i in {1..30}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if kill -0 "$PID" 2>/dev/null; then
        echo "⚠️ Process did not stop gracefully. Sending SIGKILL..."
        kill -KILL "$PID" 2>/dev/null || true
    fi
else
    echo "⚠️ PID $PID is no longer running."
fi

rm -f "$PID_FILE"

echo "✅ Application stopped."

