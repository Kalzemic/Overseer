#!/usr/bin/env bash
#
# matmul_baseline.sh — one matmul benchmark run WITHOUT overseer.
#
# Drops cache, ensures no overseer state contamination, launches matmul under
# a memory-capped systemd scope, polls /proc/PID/stat, then tears everything
# down and emits a CSV row.
#
# Prints ONE CSV row to stdout. Status goes to stderr.
# Columns: run_id,condition,elapsed_s,throughput_mb_s,majflt,minflt,prefetch_success,prefetch_fail
#
# Usage: ./matmul_baseline.sh <run_id>
#
# Env overrides: MATMUL_BIN, MEMORY_CAP

set -euo pipefail

RUN_ID="${1:?usage: $0 <run_id>}"
SCOPE_NAME="matmul_baseline_${RUN_ID}_$$"
MATMUL_LOG="/tmp/matmul_baseline_${RUN_ID}_$$.log"

MATMUL_BIN="${MATMUL_BIN:-$(pwd)/matmul/matmul}"
MEMORY_CAP="${MEMORY_CAP:-83M}"
THP_PATH="/sys/kernel/mm/transparent_hugepage/enabled"

log() { echo "[baseline $RUN_ID] $*" >&2; }

PID=""
PREV_THP=""
cleanup() {
    [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
    systemctl --user stop "${SCOPE_NAME}.scope" 2>/dev/null || true
    rm -f "$MATMUL_LOG"
    if [ -n "$PREV_THP" ]; then
        echo "$PREV_THP" | sudo tee "$THP_PATH" >/dev/null 2>&1 || true
        log "THP restored to $PREV_THP"
    fi
}
trap cleanup EXIT INT TERM

# ── Sanity check ────────────────────────────────────────────────────────────
[ -x "$MATMUL_BIN" ] || { log "ERROR: matmul binary not found at $MATMUL_BIN"; exit 1; }

# ── Ensure no overseer contamination ───────────────────────────────────────
sudo pkill -f overseer_collector 2>/dev/null || true
sudo pkill -f OverNet 2>/dev/null || true
sleep 0.3
if lsmod | grep -q overseer_enforcer; then
    sudo rmmod overseer_enforcer 2>/dev/null || {
        log "ERROR: overseer_enforcer still loaded and cannot be unloaded"
        log "       baseline measurement would be contaminated; aborting"
        exit 1
    }
    log "unloaded stale overseer_enforcer module"
fi

# Verify cleanup worked
if lsmod | grep -q overseer_enforcer; then
    log "ERROR: overseer_enforcer still present after unload attempt, aborting"
    exit 1
fi
log "pre-flight cleanup done (no overseer state present)"

# ── Normalize THP ────────────────────────────────────────────────────────────
if [ -r "$THP_PATH" ]; then
    PREV_THP=$(grep -oP '\[\K[^\]]+' "$THP_PATH" || echo "")
    echo never | sudo tee "$THP_PATH" >/dev/null
    log "THP set to never (was ${PREV_THP:-unknown})"
else
    log "warning: $THP_PATH not readable; THP not normalized"
fi

# ── Drop caches ──────────────────────────────────────────────────────────────
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
vmtouch -e "$(dirname "$MATMUL_BIN")/data/"*.bin 2>/dev/null || true
log "page cache dropped and files evicted"

# ── Launch matmul detached under a memory-capped scope ──────────────────────
systemd-run --user --scope --quiet --collect \
    --unit="$SCOPE_NAME" \
    -p MemoryMax="$MEMORY_CAP" \
    -p MemorySwapMax=0 \
    --working-directory="$(dirname "$MATMUL_BIN")" \
    "$MATMUL_BIN" > "$MATMUL_LOG" 2>&1 &

sleep 0.3
PID=$(pgrep -f "$MATMUL_BIN" | tail -1)
[ -n "$PID" ] || { log "ERROR: failed to find matmul PID"; exit 1; }
log "matmul pid=$PID (scope=$SCOPE_NAME cap=$MEMORY_CAP)"

# ── Poll /proc/PID until matmul exits ───────────────────────────────────────
LAST_STAT=""
while [ -d "/proc/$PID" ]; do
    STAT=$(cat "/proc/$PID/stat" 2>/dev/null) && LAST_STAT="$STAT"
    sleep 0.5
done

# ── Collect matmul output ──────────────────────────────────────────────────
LOGS=$(cat "$MATMUL_LOG" 2>/dev/null || echo "")

ELAPSED=$(echo "$LOGS" | grep -oP 'Time\s*=\s*\K[0-9.]+' | head -1 || true)
THROUGHPUT=$(echo "$LOGS" | grep -oP 'throughput:\s*\K[0-9.]+' | head -1 || true)

# ── Parse fault counts from last stat snapshot ──────────────────────────────
MINFLT=""
MAJFLT=""
if [ -n "$LAST_STAT" ]; then
    REST="${LAST_STAT##*) }"
    MINFLT=$(echo "$REST" | awk '{print $8}')
    MAJFLT=$(echo "$REST" | awk '{print $10}')
fi

log "elapsed=${ELAPSED}s throughput=${THROUGHPUT}MB/s majflt=$MAJFLT minflt=$MINFLT"

# ── CSV row ─────────────────────────────────────────────────────────────────
echo "$RUN_ID,baseline,${ELAPSED},${THROUGHPUT},${MAJFLT},${MINFLT},,"