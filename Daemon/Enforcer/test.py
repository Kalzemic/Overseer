import socket
import struct

BATCH_SIZE = 64
PORT = 9091

FMT = f'<i{BATCH_SIZE}Q'


pid = 1531
base = 0x00007f3be4687000
addrs = [base + i * 0x1000 for i in range(BATCH_SIZE)]

payload = struct.pack(FMT, pid, *addrs)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(payload, ('127.0.0.1', PORT))
sock.close()

print(f"sent payload for pid {pid}")