#!/usr/bin/env bash

CONTAINER_NAME=$1

#OVERNET_IP=127.0.0.1

OVERNET_IP=192.168.55.1

OVERNET_PORT=9090



docker rm -f $CONTAINER_NAME 2>/dev/null

#docker run -d --name $CONTAINER_NAME ubuntu bash -c "while true; do dd if=/dev/urandom of=/dev/null bs=1M count=100; sleep 0.1; done"
docker run -d --name $CONTAINER_NAME bert-trainer

PID=$(docker inspect $CONTAINER_NAME --format '{{.State.Pid}}')

CGROUP_PATH=$(cat /proc/$PID/cgroup | sed 's/0::\///')

CGROUP_ID=$(sudo stat -c '%i' /sys/fs/cgroup/$CGROUP_PATH)

echo "PID: $PID, Cgroup: $CGROUP_PATH, ID: $CGROUP_ID"

sudo ./overseer_collector $CGROUP_ID $OVERNET_IP $OVERNET_PORT