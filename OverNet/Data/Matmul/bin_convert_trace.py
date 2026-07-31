import numpy as np
import sys
import re

LINE_RE = re.compile(r"ip:\s*(0x[0-9a-fA-F]+),\s*addr:\s*(0x[0-9a-fA-F]+)")

def convert_to_binary(input_path: str, output_path: str, chunk_size: int = 10_000_000):
    pairs = []

    with open(input_path, "r") as infile, open(output_path, "wb") as outfile:
        for lineno, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            m = LINE_RE.match(line)
            if not m:
                print(f"Skipping malformed line {lineno}: {line}")
                continue

            ip = int(m.group(1), 16)
            addr = int(m.group(2), 16)

            # Convert address → page number
            page = addr >> 12

            pairs.append((ip, page))

            if len(pairs) >= chunk_size:
                np.array(pairs, dtype=np.uint64).tofile(outfile)
                pairs = []

        if pairs:
            np.array(pairs, dtype=np.uint64).tofile(outfile)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_trace.txt> <output.bin>")
        sys.exit(1)

    convert_to_binary(sys.argv[1], sys.argv[2])