#!/usr/bin/env python3
"""Generate raw binary matrices for the mmap'd matmul benchmark.

Writes a.bin, b.bin, c.bin as raw little-endian float64 in C row-major order.
Byte-identical to what C reads as double[N][N] on x86.
"""
import numpy as np
import os
import sys

N = int(os.environ.get("N", 3000))
OUT_DIR = os.environ.get("OUT_DIR", "./data")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Generating {N}x{N} matrices in {OUT_DIR}/ "
      f"({N*N*8 / 1024**2:.0f} MB each)", file=sys.stderr)

# Same values as the original init loop: a[i][j] = b[i][j] = i * j
m = np.fromfunction(lambda i, j: i * j, (N, N), dtype=np.float64)

m.tofile(os.path.join(OUT_DIR, "a.bin"))
m.tofile(os.path.join(OUT_DIR, "b.bin"))
np.zeros((N, N), dtype=np.float64).tofile(os.path.join(OUT_DIR, "c.bin"))

print("done", file=sys.stderr)