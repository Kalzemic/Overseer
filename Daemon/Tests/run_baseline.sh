#!/usr/bin/env bash
#
# run_baseline.sh — one Hilbert benchmark run WITHOUT overseer.
#
# Drops page cache, launches the container detached, polls /proc/PID/stat
# while it runs, then parses the container's stdout for elapsed/throughput
# and the last stat snapshot for major/minor faults.
#
# Prints ONE CSV row to stdout (no header). Status goes to stderr.
# Columns: run_id,condition,elapsed_s,throughput_mb_s,majflt,minflt,prefetch_success,prefetch_fail
#
# Usage: ./run_baseline.sh <run_id>
#
# Env overrides: HILBERT_DAT, LOG_N, IMAGE

set -euo pipefail

RUN_ID="${1:?usage: $0 <run_id>}"
CONTAINER_NAME="hilbert_baseline_${RUN_ID}_$$"
HILBERT_DAT="${HILBERT_DAT:-$HOME/hilbert_bench.dat}"
LOG_N="${LOG_N:-11}"
IMAGE="${IMAGE:-hilbert_bench}"

log() { echo "[baseline $RUN_ID] $*" >&2; }

cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# ── Drop caches ──────────────────────────────────────────────────────────────
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
log "page cache dropped"

# ── Launch container detached ───────────────────────────────────────────────
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
    --name "$CONTAINER_NAME" \
    -v "$HILBERT_DAT:/data/hilbert_bench.dat" \
    "$IMAGE" -n "$LOG_N" /data/hilbert_bench.dat >/dev/null

PID=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Pid}}')
log "container pid=$PID"

# ── Poll /proc/PID/stat until container exits ───────────────────────────────
LAST_STAT=""
while [ -e "/proc/$PID/stat" ]; do
    STAT=$(cat "/proc/$PID/stat" 2>/dev/null) && LAST_STAT="$STAT"
    sleep 0.5
done

# ── Collect container output ────────────────────────────────────────────────
docker wait "$CONTAINER_NAME" >/dev/null
LOGS=$(docker logs "$CONTAINER_NAME" 2>&1)

ELAPSED=$(echo "$LOGS"    | grep -oP 'elapsed\s*:\s*\K[0-9.]+'   | head -1 || true)
THROUGHPUT=$(echo "$LOGS" | grep -oP 'throughput:\s*\K[0-9.]+'   | head -1 || true)

# ── Parse fault counts from last stat snapshot ──────────────────────────────
# /proc/[pid]/stat layout: pid (comm) state ppid ... where (comm) may contain
# spaces and parens. The clean way is to split on the LAST ") ".
#   After stripping through ") ":  field 8 = minflt (orig 10), field 10 = majflt (orig 12)
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