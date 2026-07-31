"""
N-gram oracle accuracy ceiling for the Hilbert benchmark -- .bin format.

Reads the (ip, page) uint64 pairs produced by the trace->binary converter
directly. For a given context order k, answers: "if you always predict the
single most-likely next page-delta given the last k deltas, how often
would you be right?" -- the ceiling for any model conditioning purely on
that much history (a kth-order Markov oracle).

Run across multiple orders to see whether more context raises the ceiling
(real exploitable structure beyond 1-step) or stays flat (1-step history is
already close to the practical limit).

Computed independently per file -- rank/value meaning is instance-specific
by design, so the oracle is also built fresh per file.

Usage:
    python3 bigram_oracle_bin.py hilbert_1.bin hilbert_2.bin ...
    python3 bigram_oracle_bin.py --orders 1,2,4,8 hilbert_1.bin ...
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def load_page_deltas(path):
    data = np.memmap(path, dtype=np.uint64, mode='r').reshape(-1, 2)
    pages = data[:, 1].astype(np.int64)
    deltas = (pages[1:] - pages[:-1]).tolist()
    return deltas


def ngram_oracle_accuracy(deltas, order=1):
    """
    order=1: condition on the single immediately-previous delta (bigram).
    order=k: condition on the tuple of the last k deltas.
    Returns (accuracy, num_distinct_contexts, n_transitions_evaluated).
    """
    n = len(deltas)
    if n < order + 2:
        return None, 0, 0

    trans_counts = defaultdict(Counter)
    for i in range(order - 1, n - 1):
        context = tuple(deltas[i - order + 1: i + 1])
        trans_counts[context][deltas[i + 1]] += 1

    best_guess = {ctx: nxt_counter.most_common(1)[0][0]
                  for ctx, nxt_counter in trans_counts.items()}

    correct = 0
    total = 0
    for i in range(order - 1, n - 1):
        context = tuple(deltas[i - order + 1: i + 1])
        actual_next = deltas[i + 1]
        guess = best_guess.get(context)
        if guess is not None:
            total += 1
            if guess == actual_next:
                correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy, len(trans_counts), total


def coverage_at_k(deltas, order=1, k=4):
    n = len(deltas)
    trans_counts = defaultdict(Counter)
    for i in range(order - 1, n - 1):
        context = tuple(deltas[i - order + 1: i + 1])
        trans_counts[context][deltas[i + 1]] += 1

    top_k = {ctx: {v for v, _ in nxt_counter.most_common(k)}
             for ctx, nxt_counter in trans_counts.items()}

    hits, total = 0, 0
    for i in range(order - 1, n - 1):
        context = tuple(deltas[i - order + 1: i + 1])
        actual_next = deltas[i + 1]
        if context in top_k:
            total += 1
            if actual_next in top_k[context]:
                hits += 1
    return (hits / total) if total else 0.0


def show_bigram_table(deltas):
    """Order-1 transition table, for human-readable inspection."""
    n = len(deltas)
    trans_counts = defaultdict(Counter)
    for i in range(n - 1):
        trans_counts[deltas[i]][deltas[i + 1]] += 1

    from_totals = {cur: sum(c.values()) for cur, c in trans_counts.items()}
    print(f"  transition table (most frequent 'from' deltas, order=1):")
    for cur, _ in sorted(from_totals.items(), key=lambda kv: -kv[1])[:8]:
        total_from = from_totals[cur]
        top3 = trans_counts[cur].most_common(3)
        parts = ", ".join(f"{nxt:+d} ({c/total_from*100:.0f}%)" for nxt, c in top3)
        print(f"    from {cur:+6d}  (n={total_from:6,}):  -> {parts}")


def analyze_file(path, orders):
    deltas = load_page_deltas(path)

    print(f"\n=== {path} ===")
    print(f"  total deltas: {len(deltas):,}")

    results = []
    for order in orders:
        acc, n_contexts, n_eval = ngram_oracle_accuracy(deltas, order=order)
        cov4 = coverage_at_k(deltas, order=order, k=4)
        if acc is None:
            print(f"  order={order:2d}: not enough data")
            continue
        print(f"  order={order:2d}  oracle accuracy: {acc*100:6.2f}%   "
              f"top-4 coverage: {cov4*100:6.2f}%   "
              f"distinct contexts: {n_contexts:,}  (n_eval={n_eval:,})")
        results.append((order, acc, cov4, n_contexts, n_eval))

    show_bigram_table(deltas)
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    orders = [1, 2, 4, 8, 16]

    if args and args[0] == "--orders":
        orders = [int(x) for x in args[1].split(",")]
        args = args[2:]

    if not args:
        print(__doc__)
        sys.exit(1)

    all_results = {}
    for arg in args:
        p = Path(arg)
        if not p.exists():
            print(f"[skip] {arg} not found")
            continue
        all_results[str(p)] = analyze_file(str(p), orders)

    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Summary: oracle accuracy by order, across files")
        print("=" * 60)
        for order in orders:
            accs = [dict((o, a) for o, a, *_ in res).get(order)
                    for res in all_results.values()]
            accs = [a for a in accs if a is not None]
            if accs:
                print(f"  order={order:2d}: min={min(accs)*100:.2f}%  "
                      f"max={max(accs)*100:.2f}%  mean={sum(accs)/len(accs)*100:.2f}%")