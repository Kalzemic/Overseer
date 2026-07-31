#!/usr/bin/env bash
# record.sh — trace page faults from an already-running PID
set -euo pipefail

PID="${1:?usage: $0 <pid>}"

# Start bpftrace in background so we can manage it
sudo bpftrace -e "
tracepoint:exceptions:page_fault_user
/pid == $PID/
{
  printf(\"ip: %p, addr: %p\\n\", args->ip, args->address);
}" > trace_faults.txt &
BPFTRACE_PID=$!

# Poll target PID; when it disappears, stop bpftrace
while [ -d "/proc/$PID" ]; do
    sleep 0.5
done

# Target exited — stop bpftrace and let it drain
sudo kill -TERM "$BPFTRACE_PID" 2>/dev/null || true
# bpftrace sometimes needs SIGINT (not SIGTERM) to drain properly
sudo pkill -INT -f "pid == $PID" 2>/dev/null || true
wait "$BPFTRACE_PID" 2>/dev/null || true