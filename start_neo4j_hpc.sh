#!/bin/bash
# start_neo4j_hpc.sh — launch the Neo4j SIF on the HPC node (rootless Singularity).
#
# Companion to start_hpc.sh. Neo4j is a SEPARATE SIF: no application code, no
# Python, no neo4j driver inside it. The app SIF reaches it over the host's
# loopback (bolt://127.0.0.1:7687) because Apptainer creates no network
# namespace by default — so NEVER pass --net/--contain/--containall to either
# container, or that topology breaks.
#
# See NEO4J_SIF_PLAN.md (§0 findings, §H storage, §I startup, §M sizing).
#
# Env overrides (all optional):
#   NEO4J_SIF          path to the SIF            (default neo4j-5.26.4-community.sif)
#   NEO4J_PASSWORD     explicit password override (default: read from .env.hpc)
#   NEO4J_BOLT_PORT    Bolt port                  (default 7687)
#   NEO4J_HTTP_PORT    HTTP port                  (default 7474)
#   NEO4J_HEAP_INIT    JVM heap initial           (default 2G)
#   NEO4J_HEAP_MAX     JVM heap max               (default 2G)
#   NEO4J_PAGECACHE    Neo4j page cache           (default 2G)
#
# Exit codes:
#   0  started, or already running (idempotent)
#   1  Neo4j failed to start / Bolt never came up
#   2  bad invocation (missing SIF, singularity, or password)
#   3  directory permission problem
#   4  port conflict with a foreign process

set -euo pipefail
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

SIF="${NEO4J_SIF:-neo4j-5.26.4-community.sif}"
ENV_FILE=".env.hpc"

TMP_DIR="runtime/tmp"
LOG_DIR="runtime/logs/neo4j"
PID_FILE="runtime/neo4j.pid"
DATA_DIR="storage/neo4j"

BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"

# ── memory (NEO4J_SIF_PLAN.md §I/§M) ─────────────────────────────────────────
# PLACEHOLDERS — conservative on purpose, NOT final.
# The graph is well under 1 GB even at the full 2,648-document build (uniqueness
# constraints + range indexes only; no GDS, no fulltext, no vector), and vLLM is
# the memory-dominant tenant on this node. An over-provisioned heap does not
# degrade gracefully here — the kernel OOM killer takes the JVM outright.
# heap initial == heap max avoids runtime resize pauses.
# TO TUNE: run `free -g` with vLLM resident, then inside the SIF:
#   singularity exec neo4j-5.26.4-community.sif neo4j-admin server memory-recommendation
HEAP_INIT="${NEO4J_HEAP_INIT:-2G}"
HEAP_MAX="${NEO4J_HEAP_MAX:-2G}"
PAGECACHE="${NEO4J_PAGECACHE:-2G}"

echo "🚀 Starting Neo4j (GraphRAG backend)..."

# ── 1. singularity on PATH (cron-safe; mirrors run_maintenance_hpc.sh) ───────
export PATH="/home/apps/singularity_ce_4.0.0/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
if ! command -v singularity >/dev/null 2>&1; then
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
    echo "❌ ERROR: singularity not found on PATH after module load attempt." >&2
    exit 2
fi

# ── 2. SIF exists ────────────────────────────────────────────────────────────
if [ ! -f "$SIF" ]; then
    echo "❌ ERROR: $SIF not found in $PROJECT_DIR" >&2
    echo "👉 Build/transfer the Neo4j SIF first (NEO4J_SIF_PLAN.md §C–§G)." >&2
    exit 2
fi

# ── 3. password: single source of truth ──────────────────────────────────────
# GRAPHRAG_NEO4J_PASSWORD in .env.hpc is authoritative, so the value the app
# connects with and the value the server starts with CANNOT drift apart.
# An explicit NEO4J_PASSWORD in the environment overrides it.
# The password is never echoed and never appears on the command line.
if [ -z "${NEO4J_PASSWORD:-}" ] && [ -f "$ENV_FILE" ]; then
    NEO4J_PASSWORD="$(grep -E '^[[:space:]]*GRAPHRAG_NEO4J_PASSWORD=' "$ENV_FILE" \
                      | tail -n1 | cut -d= -f2- | sed 's/^["'\'']//; s/["'\'']$//')"
fi
if [ -z "${NEO4J_PASSWORD:-}" ]; then
    echo "❌ ERROR: no Neo4j password available." >&2
    echo "👉 Set GRAPHRAG_NEO4J_PASSWORD in $ENV_FILE (chmod 600)," >&2
    echo "   or export NEO4J_PASSWORD for an explicit override." >&2
    exit 2
fi
if [ "${#NEO4J_PASSWORD}" -lt 8 ]; then
    echo "❌ ERROR: Neo4j requires a password of at least 8 characters." >&2
    exit 2
fi

# ── 4. directories, created host-side so they carry OUR uid ──────────────────
# Finding 2: the Neo4j entrypoint refuses to run as root and needs write access
# to /data and /logs. Under Singularity there is no --user/--group-add, so
# host-side ownership is the only lever — create these BEFORE first start.
mkdir -p "$TMP_DIR" "$LOG_DIR"
mkdir -p "$DATA_DIR"/{data,logs,import,run}
# storage/neo4j/conf is deliberately NOT created or bound: binding an empty
# directory over /var/lib/neo4j/conf would MASK the image's default
# configuration. All tuning goes through the NEO4J_* env vars below.

# ── 5. permissions: the most likely first-boot failure ───────────────────────
for d in "$DATA_DIR/data" "$DATA_DIR/logs" "$DATA_DIR/import" "$DATA_DIR/run"; do
    if [ ! -w "$d" ]; then
        echo "❌ ERROR: $d is not writable by uid $(id -u)." >&2
        echo "👉 Neo4j will fail with 'Folder /logs is not accessible for user'." >&2
        echo "   Fix host-side ownership:" >&2
        echo "     chown -R $(id -u):$(id -g) $PROJECT_DIR/$DATA_DIR" >&2
        exit 3
    fi
done

# ── 6. already running? (idempotent) ─────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ℹ️  Neo4j already running (PID $OLD_PID). Nothing to do."
        exit 0
    fi
    echo "🧹 Removing stale PID file (PID ${OLD_PID:-?} not alive)."
    rm -f "$PID_FILE"
fi

# ── 7. port checks ───────────────────────────────────────────────────────────
port_in_use() {
    local p="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$p" 2>/dev/null | grep -q LISTEN && return 0
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | grep -qE "[:.]${p}[[:space:]]" && return 0
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null && { exec 3>&-; return 0; }
    fi
    return 1
}

# Reaching here means we hold no live PID file, so anything on Bolt is foreign.
if port_in_use "$BOLT_PORT"; then
    echo "❌ ERROR: port $BOLT_PORT is already in use, but we have no live PID file." >&2
    echo "👉 A foreign process or an orphaned Neo4j holds Bolt. Investigate:" >&2
    echo "     ss -ltnp 'sport = :$BOLT_PORT'" >&2
    exit 4
fi
if port_in_use "$HTTP_PORT"; then
    echo "⚠️  WARNING: port $HTTP_PORT (HTTP) in use — continuing; Bolt is what the app needs."
fi

# ── 8. disk headroom (warn only) ─────────────────────────────────────────────
avail_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -n1 | tr -dc '0-9'; }
for d in "$TMP_DIR" "$DATA_DIR"; do
    A="$(avail_gb "$d")"
    if [ -n "$A" ] && [ "$A" -lt 5 ]; then
        echo "⚠️  WARNING: only ${A}G free on $d."
    fi
done

# ── 9. temp strategy — identical to start_hpc.sh, never the system /tmp ──────
export SINGULARITY_TMPDIR="$PROJECT_DIR/$TMP_DIR"
export SINGULARITY_CACHEDIR="$PROJECT_DIR/$TMP_DIR"
export APPTAINER_TMPDIR="$PROJECT_DIR/$TMP_DIR"
export APPTAINER_CACHEDIR="$PROJECT_DIR/$TMP_DIR"

# ── 10. launch ───────────────────────────────────────────────────────────────
LOG="$LOG_DIR/neo4j-$(date -u '+%Y%m%dT%H%M%SZ').log"

echo "📦 SIF:        $SIF"
echo "📁 Data:       $DATA_DIR/data"
echo "🔌 Bolt:       127.0.0.1:$BOLT_PORT  (loopback only)"
echo "🧠 Memory:     heap ${HEAP_INIT}/${HEAP_MAX}, pagecache ${PAGECACHE}  (placeholders — tune on HPC)"

# Deliberate omissions, each load-bearing:
#   no --writable-tmpfs : every Neo4j write path is bind-mounted to real dirs,
#                         which also sidesteps the Lustre overlay warning
#   no --nv             : Neo4j needs no GPU; the GPU belongs to vLLM
#   no --net/--contain  : they would break the shared-loopback topology
#   no --env-file       : the app's config must not leak into the DB container
#
# Listening on 127.0.0.1 (not 0.0.0.0) is the security boundary: the container
# shares the host network namespace, so 0.0.0.0 would expose an unencrypted,
# password-only database to the whole cluster network.
#
# `neo4j console` (foreground) + nohup, NOT `neo4j start`: the daemon form forks
# and Singularity would lose the child, leaving an unsupervisable PID.
#
# Env var name mangling is the documented Neo4j convention: `_` → `.`, `__` → `_`.
nohup singularity exec \
    --bind "$(realpath "$DATA_DIR/data"):/data" \
    --bind "$(realpath "$DATA_DIR/logs"):/logs" \
    --bind "$(realpath "$DATA_DIR/import"):/import" \
    --bind "$(realpath "$DATA_DIR/run"):/var/lib/neo4j/run" \
    --bind "$(realpath "$TMP_DIR"):/tmp" \
    --env NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" \
    --env NEO4J_server_default__listen__address=127.0.0.1 \
    --env NEO4J_server_bolt_listen__address="127.0.0.1:${BOLT_PORT}" \
    --env NEO4J_server_http_listen__address="127.0.0.1:${HTTP_PORT}" \
    --env NEO4J_server_memory_heap_initial__size="$HEAP_INIT" \
    --env NEO4J_server_memory_heap_max__size="$HEAP_MAX" \
    --env NEO4J_server_memory_pagecache__size="$PAGECACHE" \
    --env TMPDIR=/tmp \
    "$SIF" neo4j console \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/neo4j.log"

# ── 11. wait for Bolt (first boot initialises the store) ─────────────────────
echo -n "⏳ Waiting for Bolt on 127.0.0.1:$BOLT_PORT "
for i in $(seq 1 90); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo
        echo "❌ ERROR: Neo4j exited during startup. Last 30 log lines:" >&2
        tail -n 30 "$LOG" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if port_in_use "$BOLT_PORT"; then
        echo
        echo "✅ Neo4j started."
        echo "PID: $PID"
        echo "Log: $LOG"
        echo "URI: bolt://127.0.0.1:$BOLT_PORT"
        echo
        echo "ℹ️  NEO4J_AUTH only sets the password on FIRST boot (store creation)."
        echo "   To change it later: stop the server and run"
        echo "     neo4j-admin dbms set-initial-password"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo
echo "⚠️  Bolt not listening after 180s — process still alive (PID $PID)." >&2
echo "👉 Check: tail -f $LOG" >&2
exit 1
