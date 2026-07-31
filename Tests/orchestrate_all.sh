#!/usr/bin/env bash
#
# orchestrate_all.sh — run N trials each of: baseline, overseer-local,
# overseer-remote (jetson), in randomized interleaved order. Writes one
# CSV with rows for all three conditions.
#
# Usage:   ./orchestrate_all.sh <N_per_condition> <output.csv> [--dry-run]
# Example: ./orchestrate_all.sh 30 results.csv
#
# Requires three sibling scripts:
#   run_baseline.sh
#   run_overseer.sh         (local OverNet — runs on host GPU)
#   run_overseer_remote.sh  (remote OverNet — runs on Jetson)
#
# Each sub-script must print exactly one CSV row to stdout.

set -euo pipefail

N="${1:?usage: $0 <N_per_condition> <output.csv> [--dry-run]}"
OUT="${2:?usage: $0 <N_per_condition> <output.csv> [--dry-run]}"
DRY_RUN="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A SCRIPTS=(
    [baseline]="$SCRIPT_DIR/run_baseline.sh"
    [overseer_local]="$SCRIPT_DIR/run_overseer.sh"
    [overseer_remote]="$SCRIPT_DIR/run_overseer_remote.sh"
)

for cond in "${!SCRIPTS[@]}"; do
    [ -x "${SCRIPTS[$cond]}" ] || { echo "missing or non-executable: ${SCRIPTS[$cond]}" >&2; exit 1; }
done

log() { echo "[orchestrate-all] $*" >&2; }

# ── Build randomized interleaved sequence ──────────────────────────────────
SEQ=()
for i in $(seq 1 "$N"); do
    for cond in "${!SCRIPTS[@]}"; do
        SEQ+=("$cond:$i")
    done
done
mapfile -t SHUFFLED < <(printf '%s\n' "${SEQ[@]}" | shuf)

TOTAL=${#SHUFFLED[@]}
log "Plan: $N x ${#SCRIPTS[@]} conditions = $TOTAL runs, randomized order"
log "Output: $OUT"

# Rough ETA: baseline ~30s, local ~22s, remote ~30s after warmup, plus ~5s overhead each
ETA_MIN=$(( N * 2 ))   # ~2 min per triplet
log "Rough ETA: ${ETA_MIN} minutes"

if [ "$DRY_RUN" = "--dry-run" ]; then
    log "Dry run — sequence below, exiting without executing:"
    printf '  %s\n' "${SHUFFLED[@]}" >&2
    exit 0
fi

# Prime sudo and keep alive
sudo -v
( while true; do sudo -n true; sleep 60; kill -0 $$ 2>/dev/null || exit; done ) &
SUDO_KEEPALIVE_PID=$!
trap '[ -n "${SUDO_KEEPALIVE_PID:-}" ] && kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT

# ── CSV header ─────────────────────────────────────────────────────────────
echo "run_id,condition,elapsed_s,throughput_mb_s,majflt,minflt,prefetch_success,prefetch_fail" > "$OUT"

# ── Execute ────────────────────────────────────────────────────────────────
RUN_NUM=0
START_TS=$(date +%s)

for item in "${SHUFFLED[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))
    CONDITION="${item%%:*}"
    RUN_ID="${item#*:}"
    log "── [$RUN_NUM/$TOTAL] $CONDITION run_id=$RUN_ID ──"

    SUB="${SCRIPTS[$CONDITION]:-}"
    if [ -z "$SUB" ]; then
        log "unknown condition: $CONDITION"
        continue
    fi

    set +e
    ROW=$("$SUB" "$RUN_ID")
    RC=$?
    set -e

    if [ $RC -ne 0 ] || [ -z "$ROW" ]; then
        log "WARN: run failed (rc=$RC)"
        ROW="$RUN_ID,$CONDITION,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL"
    fi

    echo "$ROW" >> "$OUT"
    log "→ $ROW"

    sleep 2
done

ELAPSED=$(( $(date +%s) - START_TS ))
log "Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s. Results: $OUT"