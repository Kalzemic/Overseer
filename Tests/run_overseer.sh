#!/usr/bin/env bash
#
# run_overseer.sh — one Hilbert benchmark run WITH overseer.
#
# Drops cache, ensures enforcer module is loaded, starts OverNet, launches
# the container detached, starts the collector, polls /proc/PID/stat, then
# tears everything down and emits a CSV row.
#
# Prints ONE CSV row to stdout. Status goes to stderr.
# Columns: run_id,condition,elapsed_s,throughput_mb_s,majflt,minflt,prefetch_success,prefetch_fail
#
# Usage: ./run_overseer.sh <run_id>
#
# Env overrides: HILBERT_DAT, LOG_N, IMAGE, OVERNET_IP, OVERNET_PORT,
#                ENFORCER_KO, OVERNET_BIN, COLLECTOR_BIN
#
# NOTE: enforcer process_worker.c does pr_info per prefetched page. At
# ~12k faults/s the kernel ring buffer will likely drop messages, so the
# prefetch counts reported here are a FLOOR, not the true number. Worth
# gating those prints behind a debug flag for clean numbers.

set -euo pipefail

RUN_ID="${1:?usage: $0 <run_id>}"
CONTAINER_NAME="hilbert_overseer_${RUN_ID}_$$"
HILBERT_DAT="${HILBERT_DAT:-$HOME/hilbert_bench.dat}"
LOG_N="${LOG_N:-11}"
IMAGE="${IMAGE:-hilbert_bench}"
OVERNET_PORT="${OVERNET_PORT:-9090}"
#Local OverNet IP:
OVERNET_IP="${OVERNET_IP:-127.0.0.1}"

#Remote OverNet IP:
# OVERNET_IP="${OVERNET_IP:-192.168.55.1}"
# REMOTE_HOST="jetson"


BIN_DIR="${BIN_DIR:-$HOME/Projects/Overseer/Daemon}"
#SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENFORCER_KO="${ENFORCER_KO:-$BIN_DIR/Enforcer/overseer_enforcer.ko}"
OVERNET_BIN="${OVERNET_BIN:-$BIN_DIR/OverNet/build/OverNet}"
COLLECTOR_BIN="${COLLECTOR_BIN:-$BIN_DIR/Collector/overseer_collector}"

DMESG_LOG="/tmp/overseer_dmesg_${RUN_ID}_$$.log"
OVERNET_LOG="/tmp/overseer_overnet_${RUN_ID}_$$.log"
COLLECTOR_LOG="/tmp/overseer_collector_${RUN_ID}_$$.log"

OVERNET_PID=""
COLLECTOR_PID=""
DMESG_PID=""

log() { echo "[overseer_local $RUN_ID] $*" >&2; }

cleanup() {
    [ -n "$COLLECTOR_PID" ] && sudo kill "$COLLECTOR_PID" 2>/dev/null || true
    [ -n "$OVERNET_PID" ]   && kill "$OVERNET_PID"        2>/dev/null || true
    # ssh "$REMOTE_HOST" 'pkill -f OverNet' 2>/dev/null || true
    [ -n "$DMESG_PID" ]     && sudo kill "$DMESG_PID"     2>/dev/null || true
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# ── Sanity checks ───────────────────────────────────────────────────────────
[ -x "$OVERNET_BIN" ]   || { log "missing OverNet binary at $OVERNET_BIN";     exit 1; }
[ -x "$COLLECTOR_BIN" ] || { log "missing collector binary at $COLLECTOR_BIN"; exit 1; }
[ -f "$ENFORCER_KO" ] || lsmod | grep -q overseer_enforcer || {
    log "enforcer .ko not at $ENFORCER_KO and module not loaded"; exit 1;
}

# ── Drop caches ─────────────────────────────────────────────────────────────
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
log "page cache dropped"

# ── Ensure enforcer module loaded ───────────────────────────────────────────
if [ -d /sys/module/overseer_enforcer ]; then
    log "enforcer module already loaded"
else
    sudo insmod "$ENFORCER_KO" 2>/dev/null
    if [ -d /sys/module/overseer_enforcer ]; then
        log "enforcer module inserted"
    else
        log "ERROR: failed to load enforcer module"
        exit 1
    fi
fi

# ── Start tailing dmesg in background ───────────────────────────────────────
sudo dmesg -W > "$DMESG_LOG" 2>/dev/null &
DMESG_PID=$!
sleep 0.2

# ── Start OverNet ───────────────────────────────────────────────────────────

# Local OverNet:
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


#Remote OverNet: 
# ssh "$REMOTE_HOST" 'cd ~/OverNet/build && nohup ./OverNet > overnet.log 2>&1 &' \
#     > "$OVERNET_LOG" 2>&1
# OVERNET_PID=""
# sleep 2
# if ! ssh "$REMOTE_HOST" 'pgrep -f OverNet' >/dev/null 2>&1; then
#     log "ERROR: OverNet failed to start on remote. Check ~/OverNet/build/overnet.log on $REMOTE_HOST."
#     exit 1
# fi
# log "OverNet running on $REMOTE_HOST"


# ── Launch container detached ───────────────────────────────────────────────
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
    --name "$CONTAINER_NAME" \
    -v "$HILBERT_DAT:/data/hilbert_bench.dat" \
    "$IMAGE" -n "$LOG_N" /data/hilbert_bench.dat >/dev/null

PID=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Pid}}')
CGROUP_PATH=$(cat /proc/$PID/cgroup | sed 's/0::\///')
CGROUP_ID=$(sudo stat -c '%i' "/sys/fs/cgroup/$CGROUP_PATH")
log "container pid=$PID cgroup_id=$CGROUP_ID"

# ── Start collector ─────────────────────────────────────────────────────────
sudo "$COLLECTOR_BIN" "$CGROUP_ID" "$OVERNET_IP" "$OVERNET_PORT" \
    > "$COLLECTOR_LOG" 2>&1 &
COLLECTOR_PID=$!

# ── Poll /proc/PID/stat until container exits ───────────────────────────────
LAST_STAT=""
while [ -e "/proc/$PID/stat" ]; do
    STAT=$(cat "/proc/$PID/stat" 2>/dev/null) && LAST_STAT="$STAT"
    sleep 0.5
done

# ── Collect container output ────────────────────────────────────────────────
docker wait "$CONTAINER_NAME" >/dev/null
LOGS=$(docker logs "$CONTAINER_NAME" 2>&1)

ELAPSED=$(echo "$LOGS"    | grep -oP 'elapsed\s*:\s*\K[0-9.]+'  | head -1 || true)
THROUGHPUT=$(echo "$LOGS" | grep -oP 'throughput:\s*\K[0-9.]+'  | head -1 || true)

MINFLT=""
MAJFLT=""
if [ -n "$LAST_STAT" ]; then
    REST="${LAST_STAT##*) }"
    MINFLT=$(echo "$REST" | awk '{print $8}')
    MAJFLT=$(echo "$REST" | awk '{print $10}')
fi

# Give dmesg tail a moment to flush, then stop it
sleep 0.5
sudo kill "$DMESG_PID" 2>/dev/null || true
DMESG_PID=""

PREFETCH_OK=$(grep -c "successfully prefetched" "$DMESG_LOG" 2>/dev/null || echo 0)
PREFETCH_FAIL=$(grep -c "failed to prefetch" "$DMESG_LOG" 2>/dev/null || echo 0)

log "elapsed=${ELAPSED}s throughput=${THROUGHPUT}MB/s majflt=$MAJFLT minflt=$MINFLT ok=$PREFETCH_OK fail=$PREFETCH_FAIL"

# Keep tmp logs around for debugging; orchestrate will clean them up if desired.
echo "$RUN_ID,overseer_local,${ELAPSED},${THROUGHPUT},${MAJFLT},${MINFLT},${PREFETCH_OK},${PREFETCH_FAIL}"