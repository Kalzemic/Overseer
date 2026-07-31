"""
Generate a synthetic trace_faults.txt from the Hilbert traversal order.
Simulates what bpftrace would capture if every page access caused a fault
(cold start after drop_caches). Useful for pipeline verification without root.

Usage: python3 gen_synthetic_trace.py [log_n] [output_file]
"""
import sys
import random
from pathlib import Path

PAGE = 4096
FAKE_IP = 0x5555_5555_5000  # fake instruction pointer (Hilbert traversal loop)

def hilbert_d2xy(n, d):
    x = y = 0
    s = 1
    t = d
    while s < n:
        rx = 1 & (t >> 1)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y

def main():
    log_n = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    out_path = sys.argv[2] if len(sys.argv) > 2 else "trace_faults.txt"

    N = 1 << log_n
    total = N * N

    # Simulate ASLR: random base address aligned to 2MB (typical mmap alignment)
    base = random.randint(0x7f00_0000_0000, 0x7fff_0000_0000) & ~0x1FFFFF

    print(f"Generating synthetic trace: N={N}, pages={total}, base=0x{base:016x}")
    print(f"Output: {out_path}")

    with open(out_path, 'w') as f:
        for d in range(total):
            x, y = hilbert_d2xy(N, d)
            page_idx = y * N + x
            addr = base + page_idx * PAGE
            f.write(f"ip: 0x{FAKE_IP:016x}, addr: 0x{addr:016x}\n")

    print(f"Done. {total} lines written.")

if __name__ == "__main__":
    main()
