#!/usr/bin/env bash
#
# Collect N runs of fault traces and convert each to binary for OverNet training.
# Run AFTER: make && ./hilbert_bench (to create the backing file without tracing).
#
# Usage: ./overpin_run.sh [n_runs] [log_n] [backing_file]
#   n_runs   number of collection runs (default 5)
#   log_n    grid exponent (default 11 → 16 GB, > RAM → continuous major faults)
#   file     path to backing file (default /home/kalzemic/hilbert_bench.dat)
#
# Each run:
#   1. Drops page caches (requires sudo/root)
#   2. Captures bpftrace fault trace
#   3. Converts trace to binary (ip, page) pairs
#
# Requires: THP set to 'madvise' or 'never' before running (run_bench.sh does this).

set -e
cd "$(dirname "$0")"

N_RUNS=${1:-5}
LOG_N=${2:-11}
BENCH_FILE=${3:-/home/kalzemic/hilbert_bench.dat}

# Ensure THP is not 'always'
THP=$(sed 's/.*\[\(.*\)\].*/\1/' /sys/kernel/mm/transparent_hugepage/enabled)
if [[ "$THP" == "always" ]]; then
    echo "Setting THP to madvise (requires sudo)..."
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null
fi

if [[ ! -x ./hilbert_bench ]]; then make; fi
if [[ ! -f "$BENCH_FILE" ]]; then
    echo "Backing file not found. Creating (no bpftrace)..."
    ./hilbert_bench -n "$LOG_N" "$BENCH_FILE"
fi

echo "Collecting $N_RUNS fault-trace runs (log_n=$LOG_N, file=$BENCH_FILE)"
echo ""

for i in $(seq 1 "$N_RUNS"); do
    echo "=== Run $i / $N_RUNS ==="

    # Drop caches for cold-start (real disk I/O, not simulated eviction)
    sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
    echo "  Caches dropped. Available: $(free -h | awk '/^Mem:/{print $7}')"

    ./overpin.sh "$LOG_N" "$BENCH_FILE"

    echo "  Converting to binary..."
    python3 bin_convert_trace.py trace_faults.txt "hilbert_run_${i}.bin"
    echo "  Saved: hilbert_run_${i}.bin  ($(wc -l < trace_faults.txt) fault events)"
    echo ""
done

echo "Done. Binary files:"
ls -lh hilbert_run_*.bin 2>/dev/null || echo "(none found)"
