#!/bin/bash
# stop_neo4j_hpc.sh — graceful shutdown of the Neo4j SIF.
#
# Companion to start_neo4j_hpc.sh and stop_hpc.sh.
#
# Neo4j MUST flush its page cache and checkpoint the store on exit. An abrupt
# kill leaves the store dirty and forces recovery on the next boot, which on a
# large graph is slow. Hence a 120s grace period (vs 30s for the app in
# stop_hpc.sh) with SIGKILL strictly as a last resort.
#
# Env overrides (all optional):
#   NEO4J_STOP_GRACE   seconds to wait for clean shutdown (default 120)
#   NEO4J_BOLT_PORT    Bolt port, for the orphan check    (default 7687)
#
# Exit codes:
#   0  stopped (cleanly or forced), or nothing was running

set -euo pipefail
cd "$(dirname "$0")"

PID_FILE="runtime/neo4j.pid"
BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
GRACE_SECONDS="${NEO4J_STOP_GRACE:-120}"

if [ ! -f "$PID_FILE" ]; then
    echo "ℹ️  No running Neo4j found (no $PID_FILE)."
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"

if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "⚠️  PID ${PID:-?} is no longer running. Clearing stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "🛑 Stopping Neo4j (PID $PID) — allowing up to ${GRACE_SECONDS}s to checkpoint..."
kill -TERM "$PID"

for i in $(seq 1 "$GRACE_SECONDS"); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "✅ Neo4j stopped cleanly after ${i}s."
        rm -f "$PID_FILE"
        exit 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "   ...still shutting down (${i}s elapsed)"
    fi
    sleep 1
done

echo "⚠️  Did not stop within ${GRACE_SECONDS}s. Sending SIGKILL." >&2
echo "   The next start may run store recovery — expected after a SIGKILL." >&2
kill -KILL "$PID" 2>/dev/null || true
sleep 2
rm -f "$PID_FILE"

# An orphaned child can outlive the supervised PID; surface it rather than
# leaving a silent port conflict for the next start.
if command -v ss >/dev/null 2>&1; then
    if ss -ltn "sport = :$BOLT_PORT" 2>/dev/null | grep -q LISTEN; then
        echo "⚠️  WARNING: port $BOLT_PORT is still listening — an orphaned process may remain." >&2
        echo "   Investigate: ss -ltnp 'sport = :$BOLT_PORT'" >&2
    fi
fi

echo "✅ Neo4j stopped (forced)."
