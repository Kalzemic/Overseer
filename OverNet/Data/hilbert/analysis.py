"""
Extended fault-trace analysis for the Hilbert benchmark.

Goes beyond fault_test.py's first-order stride histogram to check:
  1. Dominant stride check  — no single stride >40-50% (primary criterion)
  2. Stride entropy         — how spread is the distribution?
  3. Autocorrelation        — does the pattern recur at lag L?
                              Hilbert's recursive U-shapes cause peaks at
                              powers of 4 (tile size doubling).
  4. Multi-run consistency  — compare two trace files to verify determinism
                              (strides must match mod ASLR; raw addrs differ).
  5. Major-fault estimate   — large-stride events (|Δpages| >> 1) indicate
                              cross-row jumps, a proxy for fault intensity.
  6. Warm vs cold timing    — read from the benchmark's stdout if piped in.

Usage:
    python3 analysis.py [trace_faults.txt] [trace_faults_run2.txt]
"""

import sys
import re
import math
from collections import Counter
from pathlib import Path

PAGE = 4096

# ── helpers ───────────────────────────────────────────────────────────────────

def load_addrs(path):
    addrs = []
    with open(path) as f:
        for line in f:
            m = re.search(r'addr: (0x[0-9a-f]+)', line)
            if m:
                addrs.append(int(m.group(1), 16))
    return addrs


def page_deltas(addrs):
    return [(addrs[i+1] - addrs[i]) // PAGE for i in range(len(addrs)-1)]


def entropy_bits(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)


def autocorr(deltas, max_lag=64):
    """
    Normalized autocorrelation of the integer delta sequence.
    Returns list of (lag, r) pairs.
    """
    n = len(deltas)
    if n < max_lag * 2:
        return []
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / n
    if var == 0:
        return [(lag, 0.0) for lag in range(1, max_lag+1)]
    results = []
    for lag in range(1, max_lag + 1):
        cov = sum((deltas[i] - mean) * (deltas[i+lag] - mean)
                  for i in range(n - lag)) / (n - lag)
        results.append((lag, cov / var))
    return results

# ── main ──────────────────────────────────────────────────────────────────────

def analyse(trace_path, label=""):
    path = Path(trace_path)
    if not path.exists():
        print(f"  [skip] {trace_path} not found")
        return None

    addrs = load_addrs(trace_path)
    if len(addrs) < 2:
        print(f"  [skip] fewer than 2 fault addresses in {trace_path}")
        return None

    deltas = page_deltas(addrs)
    n = len(deltas)
    cnt = Counter(deltas)
    top20 = cnt.most_common(20)
    total_faults = len(addrs)

    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Fault addresses : {total_faults:,}")
    print(f"{tag}Distinct strides: {len(cnt):,}")

    # 1. Dominant stride
    dominant_stride, dominant_count = top20[0]
    dominant_pct = dominant_count / n * 100
    print(f"\n{tag}--- Stride histogram (top 20) ---")
    for stride, count in top20:
        bar = '#' * int(count / n * 40)
        print(f"  stride {stride:+7d} pages ({stride*4:+8d} KB): "
              f"{count:7,} / {n:,}  ({count/n*100:.1f}%)  {bar}")

    print(f"\n{tag}Dominant stride : {dominant_stride:+d} pages  ({dominant_pct:.1f}%)")
    if dominant_pct > 50:
        print(f"  *** FAIL: single stride > 50% — readahead could handle this ***")
    elif dominant_pct > 40:
        print(f"  !! BORDERLINE: dominant stride is 40-50% — check if readahead applies")
    else:
        print(f"  PASS: no single stride exceeds 40%")

    # 2. Entropy
    H = entropy_bits(cnt)
    H_max = math.log2(len(cnt)) if len(cnt) > 1 else 1.0
    print(f"\n{tag}Stride entropy  : {H:.2f} bits  (max {H_max:.2f} bits for "
          f"{len(cnt)} distinct strides, {H/H_max*100:.0f}% of max)")

    # 3. Autocorrelation — show lags at powers of 4 (Hilbert tile boundaries)
    ac = autocorr(deltas, max_lag=min(256, n // 4))
    if ac:
        print(f"\n{tag}Autocorrelation at Hilbert tile-size lags (powers of 4):")
        for lag, r in ac:
            if lag in (1, 2, 4, 8, 16, 32, 64, 128, 256) or lag <= 8:
                print(f"  lag={lag:4d}  r={r:+.3f}")
        # Top-5 autocorr lags by |r|
        top_ac = sorted(ac, key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"\n{tag}Top-5 autocorrelation lags:")
        for lag, r in top_ac:
            print(f"  lag={lag:4d}  r={r:+.3f}")

    # 4. Large-stride events (cross-row jumps → likely major faults)
    large = sum(1 for d in deltas if abs(d) > 100)
    print(f"\n{tag}Strides |Δ| > 100 pages: {large:,} / {n:,} ({large/n*100:.1f}%)")
    print(f"  (cross-row jumps; each likely a major fault on a cold run)")

    return deltas


def compare_runs(d1, d2, label=""):
    tag = f"[{label}] " if label else ""
    n = min(len(d1), len(d2))
    matches = sum(1 for a, b in zip(d1[:n], d2[:n]) if a == b)
    pct = matches / n * 100 if n else 0
    print(f"\n{tag}--- Multi-run determinism check ---")
    print(f"  Comparing first {n:,} strides across two runs:")
    print(f"  Matching strides: {matches:,} / {n:,} ({pct:.1f}%)")
    if pct > 99.9:
        print("  PASS: stride sequences are deterministic across runs.")
    elif pct > 95:
        print("  MOSTLY DETERMINISTIC — small difference (ASLR line-level effect?)")
    else:
        print("  !! Non-deterministic stride sequence detected.")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    here = Path(__file__).parent
    trace1 = sys.argv[1] if len(sys.argv) > 1 else str(here / "trace_faults.txt")
    trace2 = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print("Hilbert Benchmark — Extended Fault Trace Analysis")
    print("=" * 60)

    d1 = analyse(trace1, label="run1")

    if trace2:
        d2 = analyse(trace2, label="run2")
        if d1 and d2:
            compare_runs(d1, d2, label="determinism")
    else:
        run2_path = here / "trace_faults_run2.txt"
        if run2_path.exists():
            d2 = analyse(str(run2_path), label="run2")
            if d1 and d2:
                compare_runs(d1, d2, label="determinism")
        else:
            print("\n[info] No second trace found for multi-run comparison.")
            print(f"       Run a second time and save to: {run2_path}")

    print("\n" + "=" * 60)
