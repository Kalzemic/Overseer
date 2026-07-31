#!/usr/bin/env bash
#
# Capture user-space page faults during a single Hilbert benchmark traversal.
# The benchmark binary must already exist (run: make); the backing file must
# also already exist (run the binary once first without bpftrace so file
# creation faults are NOT mixed into the traversal trace).
#
# Usage: ./overpin.sh [log_n] [backing_file]
#   log_n        grid exponent (default 11 → N=2048, 16 GB)
#   backing_file path to pre-created file (default /tmp/hilbert_bench.dat)
#
# Output: trace_faults.txt  (one "ip: 0x... addr: 0x..." line per fault)

LOG_N=${1:-11}
BENCH_FILE=${2:-/home/kalzemic/hilbert_bench.dat}
BENCH="$(dirname "$0")/hilbert_bench"

if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: $BENCH not found. Run 'make' first." >&2
    exit 1
fi
if [[ ! -f "$BENCH_FILE" ]]; then
    echo "ERROR: backing file $BENCH_FILE not found." >&2
    echo "  Run: $BENCH -n $LOG_N $BENCH_FILE   (without bpftrace) to create it." >&2
    exit 1
fi

echo "Tracing faults for: $BENCH -n $LOG_N $BENCH_FILE"
echo "Output: $(dirname "$0")/trace_faults.txt"

sudo bpftrace -c "$BENCH -n $LOG_N $BENCH_FILE" -e '
tracepoint:exceptions:page_fault_user
/pid == cpid/
{ printf("ip: %p, addr: %p\n", args->ip, args->address); }
' > "$(dirname "$0")/trace_faults.txt"

echo "Capture done. Lines: $(wc -l < "$(dirname "$0")/trace_faults.txt")"
