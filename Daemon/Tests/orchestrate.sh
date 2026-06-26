#!/usr/bin/env bash
#
# orchestrate.sh — run N baseline + N overseer trials in random interleaved
# order. Writes a CSV with one row per run.
#
# Usage:   ./orchestrate.sh <N_per_condition> <output.csv> [--dry-run]
# Example: ./orchestrate.sh 30 results.csv
#
# Calls run_baseline.sh and run_overseer.sh (must be in the same dir).
# Each sub-script prints exactly one CSV row to stdout.
#
# Tip: run `sudo -v` before invoking so your sudo timestamp is fresh; this
# script also runs sudo -v up front and periodically to keep it alive.

set -euo pipefail

N="${1:?usage: $0 <N_per_condition> <output.csv> [--dry-run]}"
OUT="${2:?usage: $0 <N_per_condition> <output.csv> [--dry-run]}"
DRY_RUN="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE="$SCRIPT_DIR/run_baseline.sh"
OVERSEER="$SCRIPT_DIR/run_overseer.sh"

[ -x "$BASELINE" ] || { echo "missing or non-executable: $BASELINE" >&2; exit 1; }
[ -x "$OVERSEER" ] || { echo "missing or non-executable: $OVERSEER" >&2; exit 1; }

log() { echo "[orchestrate] $*" >&2; }

# ── Build randomized interleaved sequence ──────────────────────────────────
SEQ=()
for i in $(seq 1 "$N"); do
    SEQ+=("baseline:$i")
    SEQ+=("overseer:$i")
done
mapfile -t SHUFFLED < <(printf '%s\n' "${SEQ[@]}" | shuf)

TOTAL=${#SHUFFLED[@]}
log "Plan: $N baseline + $N overseer = $TOTAL runs, randomized order"
log "Output: $OUT"

# Rough ETA: baseline ~6min, overseer ~5min  →  per-pair ~11min
ETA_MIN=$(( N * 11 ))
log "Rough ETA: ${ETA_MIN} minutes (~$(( ETA_MIN / 60 ))h $(( ETA_MIN % 60 ))m)"

if [ "$DRY_RUN" = "--dry-run" ]; then
    log "Dry run — sequence below, exiting without executing:"
    printf '  %s\n' "${SHUFFLED[@]}" >&2
    exit 0
fi

# Prime sudo so sub-scripts don't prompt mid-run
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

    case "$CONDITION" in
        baseline) SUB="$BASELINE" ;;
        overseer) SUB="$OVERSEER" ;;
        *) log "unknown condition: $CONDITION"; exit 1 ;;
    esac

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

    # Brief pause to let kernel/disk settle
    sleep 2
done

ELAPSED=$(( $(date +%s) - START_TS ))
log "Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s. Results: $OUT"