"""
Dump the raw, ORDERED delta sequence -- no histogram, no aggregation.
This is for directly eyeballing what the model is actually being fed,
position by position, to sanity-check the Markov-chain assumption.

Usage:
    python3 dump_deltas.py trace_faults.txt [n]
        -> parses addr lines, prints first n page-deltas in order (default 300)

    python3 dump_deltas.py --bin hilbert_1.bin [n]
        -> adjust BIN_DTYPE/BIN_FIELDS below to match bin_convert_trace.py's
           actual output layout, then prints the first n page-rank deltas
"""
import sys
import re

PAGE = 4096

def deltas_from_trace(path, n):
    addrs = []
    with open(path) as f:
        for line in f:
            m = re.search(r'addr: (0x[0-9a-f]+)', line)
            if m:
                addrs.append(int(m.group(1), 16))
            if len(addrs) > n + 1:
                break
    return [(addrs[i+1] - addrs[i]) // PAGE for i in range(len(addrs)-1)]


def main():
    args = sys.argv[1:]
    n = 300
    # allow trailing integer arg to set count
    if args and args[-1].isdigit():
        n = int(args[-1])
        args = args[:-1]

    if not args:
        print(__doc__)
        return

    path = args[0]
    deltas = deltas_from_trace(path, n)

    print(f"First {len(deltas)} ordered page-deltas from {path}:\n")
    # print with index, 10 per line, so you can visually scan for the
    # repeating bigram pattern (e.g. does -1 ever immediately precede +1?)
    for i in range(0, len(deltas), 10):
        chunk = deltas[i:i+10]
        idx_str = f"[{i:4d}] "
        vals_str = " ".join(f"{d:+5d}" for d in chunk)
        print(idx_str + vals_str)

    # quick explicit check: does any -1 immediately precede a +1, or vice versa
    # (the "reversal" we predicted should be near-absent)
    reversal_pairs = sum(
        1 for i in range(len(deltas)-1)
        if (deltas[i], deltas[i+1]) in [(1, -1), (-1, 1), (512, -512), (-512, 512)]
    )
    print(f"\nDirect reversal pairs (e.g. +1 then -1) in this window: "
          f"{reversal_pairs} / {len(deltas)-1}")


if __name__ == "__main__":
    main()