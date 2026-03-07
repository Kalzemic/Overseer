import socket
import struct

BATCH_SIZE = 16
PORT = 9080

# 1 int32 + 64 uint64 + 64 uint64
FMT = f'<i{BATCH_SIZE}Q{BATCH_SIZE}Q'
SIZE = struct.calcsize(FMT)  # should be 1028

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', PORT))

print(f"Listening on UDP :{PORT}, expecting {SIZE} bytes per packet")

while True:
    data, ret = sock.recvfrom(2048)
    if len(data) < SIZE:
        print(f"Short packet: {len(data)} bytes")
        continue

    fields = struct.unpack(FMT, data[:SIZE])
    pid    = fields[0]
    addrs  = fields[1:1 + BATCH_SIZE]
    ips    = fields[1 + BATCH_SIZE:]

    print(f"\n--- FaultReport from {ret} ---")
    print(f"  PID: {pid}")
    for i in range(BATCH_SIZE):
        print(f"  [{i:02d}] addr=0x{addrs[i]:016x}  ip=0x{ips[i]:016x}")