"""
Check run-length / repeat structure of consecutive page-delta strides.

fault_test.py and verify_hilbert.py both only compute a flat histogram of
deltas (Counter) -- that destroys ordering, so it can't answer "does the
same stride ever repeat back-to-back?" This script looks at the actual
sequence instead.
"""
import re
from collections import Counter

PAGE = 4096

def load_deltas_from_trace(path="trace_faults.txt"):
    addrs = []
    with open(path) as f:
        for line in f:
            m = re.search(r'addr: (0x[0-9a-f]+)', line)
            if m:
                addrs.append(int(m.group(1), 16))
    return [(addrs[i+1] - addrs[i]) // PAGE for i in range(len(addrs)-1)]


def analyze(deltas, label=""):
    n = len(deltas)
    if n < 2:
        print(f"{label}: not enough deltas to analyze")
        return

    # 1. How often is consecutive delta[i] == delta[i+1]? (immediate repeat)
    repeats = sum(1 for i in range(n-1) if deltas[i] == deltas[i+1])
    print(f"\n=== {label} ===")
    print(f"total transitions: {n}")
    print(f"immediate repeats (delta[i] == delta[i+1]): "
          f"{repeats} ({repeats/(n-1)*100:.1f}% of {n-1} adjacent pairs)")

    # 2. Run-length distribution: how long are runs of the same delta value?
    run_lengths = Counter()
    cur_val = deltas[0]
    cur_len = 1
    for d in deltas[1:]:
        if d == cur_val:
            cur_len += 1
        else:
            run_lengths[cur_len] += 1
            cur_val = d
            cur_len = 1
    run_lengths[cur_len] += 1  # flush last run

    total_runs = sum(run_lengths.values())
    print(f"total runs (maximal same-stride sequences): {total_runs}")
    print("run-length distribution:")
    for length in sorted(run_lengths):
        count = run_lengths[length]
        print(f"  length {length}: {count} runs ({count/total_runs*100:.1f}%)")

    max_run = max(run_lengths)
    print(f"longest run of same stride: {max_run} consecutive steps")

    # 3. Bigram structure: given current stride, what's next?
    print("transition matrix (given delta[i], what is delta[i+1]?):")
    trans = Counter()
    for i in range(n-1):
        trans[(deltas[i], deltas[i+1])] += 1
    from_counts = Counter()
    for (a, b), c in trans.items():
        from_counts[a] += c
    for a in sorted(from_counts):
        print(f"  from {a:+d}:")
        total_from_a = from_counts[a]
        for (x, b), c in sorted(trans.items()):
            if x == a:
                print(f"    -> {b:+d}: {c} ({c/total_from_a*100:.1f}%)")


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1 and sys.argv[1] == "--hilbert-sim":
        # Run directly on a simulated small Hilbert curve (no trace file needed)
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

        log_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        N = 1 << log_n
        pages = [hilbert_d2xy(N, d) for d in range(N * N)]
        pages = [y * N + x for (x, y) in pages]
        deltas = [pages[i+1] - pages[i] for i in range(len(pages)-1)]
        analyze(deltas, label=f"simulated Hilbert N={N}")
    else:
        trace_path = sys.argv[1] if len(sys.argv) > 1 else "trace_faults.txt"
        if not os.path.exists(trace_path):
            print(f"'{trace_path}' not found. Run overpin.sh first, "
                  f"or use --hilbert-sim <log_n> to test on a simulated curve.")
            sys.exit(1)
        deltas = load_deltas_from_trace(trace_path)
        analyze(deltas, label=f"real trace ({trace_path})")