#!/usr/bin/env bash
#
# Full benchmark pipeline:
#   1. Build the binary
#   2. Check / warn about THP
#   3. Create the backing file (if not present) WITHOUT bpftrace
#   4. Drop page caches for a cold-start run
#   5. Run bpftrace + benchmark (faults from traversal only)
#   6. Run fault_test.py  (quick stride histogram)
#   7. Run analysis.py    (extended: autocorrelation, multi-run, perf stats)
#
# Usage: ./run_bench.sh [log_n] [backing_file]
#   log_n=9   → 1 GB  (fast sanity check)
#   log_n=11  → 16 GB (> RAM, continuous major faults; ~10 min on NVMe)

set -e
cd "$(dirname "$0")"

LOG_N=${1:-11}
# /tmp is tmpfs on this machine — use real NVMe-backed path for disk I/O faults
BENCH_FILE=${2:-/home/kalzemic/hilbert_bench.dat}

echo "=== Hilbert benchmark  log_n=$LOG_N  file=$BENCH_FILE ==="

# ── 1. Build ──────────────────────────────────────────────────────────────────
if [[ ! -x ./hilbert_bench ]]; then
    echo "[1/7] Building..."
    make
else
    echo "[1/7] Binary already built."
fi

# ── 2. THP check ──────────────────────────────────────────────────────────────
THP=$(sed 's/.*\[\(.*\)\].*/\1/' /sys/kernel/mm/transparent_hugepage/enabled)
echo "[2/7] THP = $THP"
if [[ "$THP" == "always" ]]; then
    echo "      WARNING: THP is 'always'. Setting to 'madvise' (requires sudo)."
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null
    echo "      THP is now: $(sed 's/.*\[\(.*\)\].*/\1/' /sys/kernel/mm/transparent_hugepage/enabled)"
fi

# ── 3. Create backing file (no bpftrace) ──────────────────────────────────────
echo "[3/7] Ensuring backing file exists (faults here are NOT traced)..."
./hilbert_bench -n "$LOG_N" "$BENCH_FILE"

# ── 4. Drop page caches ───────────────────────────────────────────────────────
echo "[4/7] Dropping page caches (requires sudo)..."
sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
echo "      Done. Available RAM: $(free -h | awk '/^Mem:/{print $7}')"

# ── 5. Capture faults + run ───────────────────────────────────────────────────
echo "[5/7] Running benchmark under bpftrace..."
./overpin.sh "$LOG_N" "$BENCH_FILE"

# ── 6. Quick stride histogram ─────────────────────────────────────────────────
echo ""
echo "[6/7] === Stride histogram (fault_test.py) ==="
python3 ../fault_test.py

# ── 7. Extended analysis ──────────────────────────────────────────────────────
echo ""
echo "[7/7] === Extended analysis (analysis.py) ==="
python3 ./analysis.py
