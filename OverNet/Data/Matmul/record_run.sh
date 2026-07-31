#!/usr/bin/env bash
set -euo pipefail

CAP="${CAP:-85M}"
MATMUL_BIN="${MATMUL_BIN:-$HOME/Projects/Overseer/OverNet/Data/Matmul/matmul}"
OUT_DIR="${OUT_DIR:-.}"
N_RUNS="${N_RUNS:-10}"

# ── sudo setup ──────────────────────────────────────────────────────────────
sudo -v

while true; do sudo -n -v; sleep 60; done &
SUDO_KEEPALIVE_PID=$!

# ── Save and disable THP ────────────────────────────────────────────────────
THP_ENABLED=$(cat /sys/kernel/mm/transparent_hugepage/enabled | grep -o '\[[^]]*\]' | tr -d '[]')
THP_DEFRAG=$(cat /sys/kernel/mm/transparent_hugepage/defrag | grep -o '\[[^]]*\]' | tr -d '[]')

echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null

cleanup() {
    echo "$THP_ENABLED" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
    echo "$THP_DEFRAG"  | sudo tee /sys/kernel/mm/transparent_hugepage/defrag  >/dev/null
    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Main loop ───────────────────────────────────────────────────────────────
MATMUL_NAME=$(basename "$MATMUL_BIN")
mkdir -p "$OUT_DIR"

echo "Running $N_RUNS iterations of $MATMUL_NAME with MemoryMax=$CAP"

for i in $(seq 1 "$N_RUNS"); do
    echo "=== run $i / $N_RUNS ==="
    sudo sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

    # Launch matmul under the memory cap
    systemd-run --user --scope --quiet \
        -p MemoryMax="$CAP" -p MemorySwapMax=0 \
        "$MATMUL_BIN" &

    # Wait for it to actually start, then find its PID
    sleep 0.3
    MATMUL_PID=$(pgrep -f "$MATMUL_NAME" | tail -1)
    if [ -z "$MATMUL_PID" ]; then
        echo "failed to find matmul PID"
        exit 1
    fi
    echo "matmul pid=$MATMUL_PID"

    # Start bpftrace on that PID
    ./record.sh "$MATMUL_PID" &
    RECORD_PID=$!
    sleep 0.5

    # Poll for matmul exit
    while kill -0 "$MATMUL_PID" 2>/dev/null; do
        sleep 0.5
    done

    # Stop bpftrace
    sudo pkill -P "$RECORD_PID" bpftrace 2>/dev/null || true
    wait "$RECORD_PID" 2>/dev/null || true

    python bin_convert_trace.py trace_faults.txt "$OUT_DIR/matmul_$i.bin"
    echo "wrote $OUT_DIR/matmul_$i.bin"
done

echo "done"