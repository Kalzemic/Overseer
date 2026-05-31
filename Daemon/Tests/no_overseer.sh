#!/usr/bin/env bash

docker rm -f no_overseer 2>/dev/null
docker run -d --name no_overseer bert-trainer

PID=$(docker inspect no_overseer --format '{{.State.Pid}}')

docker logs -f no_overseer

echo "=== NO_OVERSEER major faults ==="
cat /proc/$PID/stat 2>/dev/null | awk '{print "majflt: " $12}' || echo "pid already gone"