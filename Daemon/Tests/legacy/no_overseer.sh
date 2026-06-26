#!/usr/bin/env bash

cleanup() {
    for dev in $(lsblk -dno NAME | grep -v loop); do
        echo 128 | sudo tee /sys/block/$dev/queue/read_ahead_kb > /dev/null 2>&1 || true
    done
    echo "restored readahead to 128KB"
}

trap cleanup EXIT INT TERM

for dev in $(lsblk -dno NAME | grep -v loop); do
    echo 0 | sudo tee /sys/block/$dev/queue/read_ahead_kb > /dev/null
    echo "readahead disabled on /dev/$dev"
done

echo 0 | sudo tee /proc/sys/vm/page-cluster > /dev/null

sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo "page cache dropped"

# ── Container launch ───────────────────────────────────────────────────────────

docker rm -f no_overseer 2>/dev/null

docker run -d --name no_overseer --memory=800m --memory-swap=800m bert-trainer

PID=$(docker inspect no_overseer --format '{{.State.Pid}}')

docker logs -f no_overseer

echo "=== NO_OVERSEER major faults ==="
if [[ -f /proc/$PID/stat ]]; then
    read -r stat < /proc/$PID/stat
    echo "majflt: $(echo $stat | awk '{print $12}')"
else
    echo "pid already gone"
fi