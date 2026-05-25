#!/usr/bin/env bash


CONTAINER_NAME=overseer_test

OVERNET_IP=127.0.0.1

#OVERNET_IP=192.168.55.1

OVERNET_PORT=9090



cleanup() {
    echo "cleaning up..."
    pkill -f OverNet 2>/dev/null
    sudo rmmod overseer_enforcer 2>/dev/null
}

trap cleanup EXIT INT TERM

sudo insmod ../Enforcer/overseer_enforcer.ko 2>/dev/null || \
    lsmod | grep -q overseer_enforcer || \
    { echo "failed to load module"; exit 1; }

cd ../OverNet/build/ && ./OverNet &
OVERNET_PID=$!
cd ../Tests

sleep 1


docker rm -f $CONTAINER_NAME 2>/dev/null

#docker run -d --name $CONTAINER_NAME ubuntu bash -c "while true; do dd if=/dev/urandom of=/dev/null bs=1M count=100; sleep 0.1; done"
docker run -d --name $CONTAINER_NAME stride-trainer

PID=$(docker inspect $CONTAINER_NAME --format '{{.State.Pid}}')

CGROUP_PATH=$(cat /proc/$PID/cgroup | sed 's/0::\///')

CGROUP_ID=$(sudo stat -c '%i' /sys/fs/cgroup/$CGROUP_PATH)

echo "PID: $PID, Cgroup: $CGROUP_PATH, ID: $CGROUP_ID"

sudo ../Collector/overseer_collector $CGROUP_ID $OVERNET_IP $OVERNET_PORT &


docker logs -f $CONTAINER_NAME

echo "=== OVERSEER major faults ==="
cat /proc/$PID/stat 2>/dev/null | awk '{print "majflt: " $12}' || echo "pid already gone"