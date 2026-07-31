sudo -v
( while true; do sudo -n true; sleep 60; kill -0 $$ 2>/dev/null || exit; done ) &
KEEPALIVE_PID=$!
trap "kill $KEEPALIVE_PID 2>/dev/null" EXIT

for cap in 87 85 83; do
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    echo "=== cap=${cap}M ==="
    timeout 20m systemd-run --user --scope \
        -p MemoryMax=${cap}M -p MemorySwapMax=0 \
        /usr/bin/time -f "cap=${cap}M time=%e faults=%F" ./matmul 2>&1 | grep cap= \
        || echo "cap=${cap}M TIMEOUT"
done