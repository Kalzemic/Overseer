#!/usr/bin/env bash
#
# collect_hilbert_data.sh — run hilbert_bench under bpftrace N times and
# produce hilbert_1.bin … hilbert_N.bin in OUTPUT_DIR for OverNet training.
#
# Schedule:
#   runs 1–8  : log_n=11  (N=2048, 16 GB — real I/O-bound major faults)
#   run  9    : log_n=10  (N=1024,  4 GB — same Hilbert motif, different scale)
#   run  10   : log_n=11  (validation file)
#
# Drop page cache between every run so faults are cold.
# Backing files are created once without bpftrace; the log_n=10 file is
# only created when needed for run 9.
#
# Must be run as a normal user; individual commands that need root use sudo.
#
# Usage: ./collect_hilbert_data.sh [OUTPUT_DIR]
#   OUTPUT_DIR  where to write *.bin (default: ../../training/data)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/../../training/data}"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

BENCH="$SCRIPT_DIR/hilbert_bench"
CONVERT="$SCRIPT_DIR/bin_convert_trace.py"
TRACE="$SCRIPT_DIR/trace_faults.txt"

BENCH_FILE_11="/home/kalzemic/hilbert_bench.dat"     # log_n=11, 16 GB
BENCH_FILE_10="/home/kalzemic/hilbert_bench_10.dat"  # log_n=10,  4 GB

N_TOTAL=10

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }

drop_caches() {
    echo "  [drop_caches] flushing page cache..."
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
    echo "  [drop_caches] done."
}

ensure_file() {
    local log_n="$1" path="$2"
    local N=$((1 << log_n))
    local expected=$(( N * N * 4096 ))
    if [[ -f "$path" ]]; then
        local actual
        actual=$(stat -c '%s' "$path")
        if [[ "$actual" -eq "$expected" ]]; then
            echo "  Reusing existing backing file: $path ($(numfmt --to=iec $actual))"
            return
        fi
        echo "  Existing file size mismatch — recreating."
    fi
    echo "  Creating backing file for log_n=$log_n: $path"
    "$BENCH" -n "$log_n" "$path"
    drop_caches  # evict pages just written so first traced run is cold
}

run_trace() {
    local log_n="$1" bench_file="$2" out_bin="$3" run_num="$4"
    echo
    echo "=== Run $run_num / $N_TOTAL  (log_n=$log_n, $(( 1 << log_n ))×$(( 1 << log_n )) grid) ==="

    drop_caches

    echo "  Starting bpftrace capture..."
    sudo bpftrace \
        -c "$BENCH -n $log_n $bench_file" \
        -e 'tracepoint:exceptions:page_fault_user /pid == cpid/ { printf("ip: %p, addr: %p\n", args->ip, args->address); }' \
        > "$TRACE"

    local lines
    lines=$(wc -l < "$TRACE")
    echo "  Captured $lines fault events."

    echo "  Converting to binary: $out_bin"
    python "$CONVERT" "$TRACE" "$out_bin"

    local sz
    sz=$(stat -c '%s' "$out_bin")
    echo "  Done: $out_bin  ($(numfmt --to=iec $sz))"
}

# ── preflight ─────────────────────────────────────────────────────────────────

[[ -x "$BENCH" ]]    || die "$BENCH not found. Run 'make' first."
[[ -f "$CONVERT" ]]  || die "$CONVERT not found."
mkdir -p "$OUTPUT_DIR"

echo "Output directory : $OUTPUT_DIR"
echo "Benchmark binary : $BENCH"
echo

# ── create backing files (no bpftrace) ───────────────────────────────────────

echo "--- Phase 1: ensure backing files exist ---"
ensure_file 11 "$BENCH_FILE_11"
ensure_file 10 "$BENCH_FILE_10"

# ── collection loop ───────────────────────────────────────────────────────────

echo
echo "--- Phase 2: collect $N_TOTAL traces ---"

for run in $(seq 1 $N_TOTAL); do
    out_bin="$OUTPUT_DIR/hilbert_${run}.bin"

    if [[ $run -eq 9 ]]; then
        # One run at the smaller scale for intra-dataset variety
        run_trace 10 "$BENCH_FILE_10" "$out_bin" "$run"
    else
        run_trace 11 "$BENCH_FILE_11" "$out_bin" "$run"
    fi
done

echo
echo "=== Collection complete ==="
echo "Files written:"
ls -lh "$OUTPUT_DIR"/hilbert_*.bin
echo
echo "hilbert_1 … hilbert_9  → training"
echo "hilbert_10             → validation"

