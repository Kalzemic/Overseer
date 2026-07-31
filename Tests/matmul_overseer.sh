#!/usr/bin/env bash
#
# matmul_overseer.sh — one matmul benchmark run WITH overseer.
#
# Drops cache, ensures no stale overseer state, loads enforcer module, starts
# OverNet, launches matmul under a memory-capped systemd scope, starts the
# collector, polls /proc/PID/stat, then tears everything down and emits a CSV row.
#
# Prints ONE CSV row to stdout. Status goes to stderr.
# Columns: run_id,condition,elapsed_s,throughput_mb_s,majflt,minflt,prefetch_success,prefetch_fail
#
# Usage: ./matmul_overseer.sh <run_id>
#
# Env overrides: MATMUL_BIN, MEMORY_CAP, OVERNET_IP, OVERNET_PORT,
#                ENFORCER_KO, OVERNET_BIN, COLLECTOR_BIN

set -euo pipefail

RUN_ID="${1:?usage: $0 <run_id>}"
SCOPE_NAME="matmul_overseer_${RUN_ID}_$$"

MATMUL_BIN="${MATMUL_BIN:-$(pwd)/matmul/matmul}"
MEMORY_CAP="${MEMORY_CAP:-83M}"
OVERNET_PORT="${OVERNET_PORT:-9090}"
OVERNET_IP="${OVERNET_IP:-127.0.0.1}"

# Remote OverNet (uncomment to use):
# OVERNET_IP="${OVERNET_IP:-192.168.55.1}"
# REMOTE_HOST="jetson"

THP_PATH="/sys/kernel/mm/transparent_hugepage/enabled"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="${BIN_DIR:-$HOME/Projects/Overseer/Daemon}"
ENFORCER_KO="${ENFORCER_KO:-$BIN_DIR/Enforcer/overseer_enforcer.ko}"
OVERNET_BIN="${OVERNET_BIN:-$BIN_DIR/OverNet/build/OverNet}"
COLLECTOR_BIN="${COLLECTOR_BIN:-$BIN_DIR/Collector/overseer_collector}"

MATMUL_LOG="/tmp/matmul_overseer_${RUN_ID}_$$.log"
DMESG_LOG="/tmp/overseer_dmesg_${RUN_ID}_$$.log"
OVERNET_LOG="/tmp/overseer_overnet_${RUN_ID}_$$.log"
COLLECTOR_LOG="/tmp/overseer_collector_${RUN_ID}_$$.log"

PID=""
OVERNET_PID=""
COLLECTOR_PID=""
DMESG_PID=""
PREV_THP=""

log() { echo "[overseer_local $RUN_ID] $*" >&2; }


cleanup() {
    log "cleanup starting"

    [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
    systemctl --user stop "${SCOPE_NAME}.scope" 2>/dev/null || true

    sudo pkill -f "$(basename "$COLLECTOR_BIN")" 2>/dev/null || true
    [ -n "$OVERNET_PID" ] && kill "$OVERNET_PID" 2>/dev/null || true
    pkill -f "$(basename "$OVERNET_BIN")" 2>/dev/null || true

    sleep 1

    sudo rmmod overseer_enforcer 2>/dev/null || true

    [ -n "$DMESG_PID" ] && sudo kill "$DMESG_PID" 2>/dev/null || true
    rm -f "$MATMUL_LOG"

    if [ -n "$PREV_THP" ]; then
        echo "$PREV_THP" | sudo tee "$THP_PATH" >/dev/null 2>&1 || true
        log "THP restored to $PREV_THP"
    fi

    log "cleanup done"
}
trap cleanup EXIT INT TERM

# ── Sanity checks ───────────────────────────────────────────────────────────
[ -x "$MATMUL_BIN" ]    || { log "missing matmul binary at $MATMUL_BIN";       exit 1; }
[ -x "$OVERNET_BIN" ]   || { log "missing OverNet binary at $OVERNET_BIN";     exit 1; }
[ -x "$COLLECTOR_BIN" ] || { log "missing collector binary at $COLLECTOR_BIN"; exit 1; }
[ -f "$ENFORCER_KO" ] || {
    log "enforcer .ko not found at $ENFORCER_KO"; exit 1;
}


# ── Normalize THP ────────────────────────────────────────────────────────────
if [ -r "$THP_PATH" ]; then
    PREV_THP=$(grep -oP '\[\K[^\]]+' "$THP_PATH" || echo "")
    echo never | sudo tee "$THP_PATH" >/dev/null
    log "THP set to never (was ${PREV_THP:-unknown})"
else
    log "warning: $THP_PATH not readable; THP not normalized"
fi

# ── Drop caches ─────────────────────────────────────────────────────────────
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
vmtouch -e "$(dirname "$MATMUL_BIN")/data/"*.bin 2>/dev/null || true
log "page cache dropped and files evicted"

# ── Load enforcer module ───────────────────────────────────────────────────
if ! sudo insmod "$ENFORCER_KO"; then
    log "ERROR: insmod failed"
    sudo dmesg | tail -5 >&2
    exit 1
fi
log "enforcer module inserted"

# ── Start tailing dmesg in background ───────────────────────────────────────
sudo dmesg -W > "$DMESG_LOG" 2>/dev/null &
DMESG_PID=$!
sleep 0.2

# ── Start OverNet ───────────────────────────────────────────────────────────
( cd "$(dirname "$OVERNET_BIN")" && exec "./$(basename "$OVERNET_BIN")" ) \
    > "$OVERNET_LOG" 2>&1 &
OVERNET_PID=$!
sleep 1
if ! kill -0 "$OVERNET_PID" 2>/dev/null; then
    log "ERROR: OverNet failed to start. Last lines of log:"
    tail -n 20 "$OVERNET_LOG" >&2
    exit 1
fi
log "OverNet pid=$OVERNET_PID"

# Remote OverNet (uncomment to use):
# ssh "$REMOTE_HOST" 'cd ~/OverNet/build && nohup ./OverNet > overnet.log 2>&1 &' \
#     > "$OVERNET_LOG" 2>&1
# OVERNET_PID=""
# sleep 2
# if ! ssh "$REMOTE_HOST" 'pgrep -f OverNet' >/dev/null 2>&1; then
#     log "ERROR: OverNet failed to start on remote."
#     exit 1
# fi
# log "OverNet running on $REMOTE_HOST"

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

CGROUP_PATH=$(cat /proc/$PID/cgroup | sed 's|^0::/||')
CGROUP_ID=$(sudo stat -c '%i' "/sys/fs/cgroup/$CGROUP_PATH")
log "matmul pid=$PID cgroup_id=$CGROUP_ID (scope=$SCOPE_NAME cap=$MEMORY_CAP)"

# ── Start collector ─────────────────────────────────────────────────────────
sudo "$COLLECTOR_BIN" "$CGROUP_ID" "$OVERNET_IP" "$OVERNET_PORT" \
    > "$COLLECTOR_LOG" 2>&1 &
COLLECTOR_PID=$!

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

MINFLT=""
MAJFLT=""
if [ -n "$LAST_STAT" ]; then
    REST="${LAST_STAT##*) }"
    MINFLT=$(echo "$REST" | awk '{print $8}')
    MAJFLT=$(echo "$REST" | awk '{print $10}')
fi

# ── Give dmesg tail a moment to flush, then stop it ────────────────────────
sleep 0.5
sudo kill "$DMESG_PID" 2>/dev/null || true
DMESG_PID=""

PREFETCH_OK=$(grep -c "successfully prefetched" "$DMESG_LOG" 2>/dev/null || echo 0)
PREFETCH_FAIL=$(grep -c "failed to prefetch" "$DMESG_LOG" 2>/dev/null || echo 0)

log "elapsed=${ELAPSED}s throughput=${THROUGHPUT}MB/s majflt=$MAJFLT minflt=$MINFLT ok=$PREFETCH_OK fail=$PREFETCH_FAIL"

# ── CSV row ─────────────────────────────────────────────────────────────────
echo "$RUN_ID,overseer_local,${ELAPSED},${THROUGHPUT},${MAJFLT},${MINFLT},${PREFETCH_OK},${PREFETCH_FAIL}"