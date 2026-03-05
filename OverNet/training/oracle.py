#!/usr/bin/env python3
"""
DLRM Trace Prefetchability Analysis for OverNet
=================================================
Works directly with your binary ndarray files (uint64 memmap),
same format as MemDataset.

Usage:
    python dlrm_trace_analysis.py --trace /path/to/dlrm.bin [--max_pages 1000000]
    
    # Compare multiple workloads:
    python dlrm_trace_analysis.py --trace /path/to/dlrm.bin \
        --compare hpl:/path/to/hpl.bin bert:/path/to/bert.bin
"""

import argparse
import numpy as np
from collections import Counter, defaultdict
import sys


def load_pages(path, max_pages=None):
    """Load pages from binary memmap file (uint64), same as MemDataset."""
    data = np.memmap(path, dtype=np.uint64, mode='r')
    if max_pages:
        data = np.array(data[:max_pages])
    else:
        data = np.array(data)
    print(f"Loaded {len(data):,} pages from {path}")
    return data


# ===========================================================================
# ANALYSIS 1: Phase-level breakdown
# ===========================================================================

def analyze_phases(pages, window_size=10000):
    """
    Segment trace into windows. Per window:
    - Unique pages, entropy, stride regularity, repeat rate
    
    Shows WHERE in the DLRM trace the 8.9% is coming from.
    """
    n = len(pages)
    n_windows = n // window_size
    
    print(f"\n{'='*70}")
    print(f"PHASE-LEVEL ANALYSIS (window={window_size:,})")
    print(f"{'='*70}")
    print(f"Total: {n:,} pages, {n_windows} windows")
    
    results = []
    
    for w in range(n_windows):
        start = w * window_size
        window = pages[start:start + window_size]
        
        unique = len(set(window.tolist()))
        
        # Entropy
        counts = Counter(window.tolist())
        probs = np.array(list(counts.values()), dtype=np.float64) / window_size
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(unique) if unique > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Stride analysis
        deltas = np.diff(window.astype(np.int64))
        delta_counts = Counter(deltas.tolist())
        most_common_stride, mc_count = delta_counts.most_common(1)[0]
        stride_reg = mc_count / (window_size - 1)
        
        top3 = delta_counts.most_common(3)
        top3_cov = sum(c for _, c in top3) / (window_size - 1)
        
        # Repeat rate (page seen in last K=64 accesses)
        K = 64
        window_list = window.tolist()
        repeat_count = 0
        for i in range(K, window_size):
            if window_list[i] in set(window_list[i-K:i]):
                repeat_count += 1
        repeat_rate = repeat_count / (window_size - K)
        
        results.append({
            'window': w, 'start': start, 'unique_pages': unique,
            'entropy': entropy, 'norm_entropy': norm_entropy,
            'stride_reg': stride_reg, 'top3_cov': top3_cov,
            'repeat_rate': repeat_rate, 'top_stride': most_common_stride,
        })
    
    # Print first 20 windows
    print(f"\n{'Win':>5} {'Start':>10} {'Uniq':>7} {'Entropy':>8} "
          f"{'NormEnt':>8} {'StrReg':>7} {'Top3':>7} {'Repeat':>7}")
    print("-" * 70)
    for r in results[:20]:
        print(f"{r['window']:>5} {r['start']:>10,} {r['unique_pages']:>7} "
              f"{r['entropy']:>8.2f} {r['norm_entropy']:>8.3f} "
              f"{r['stride_reg']:>7.3f} {r['top3_cov']:>7.3f} {r['repeat_rate']:>7.3f}")
    if n_windows > 20:
        print(f"  ... ({n_windows - 20} more)")
    
    # Classify
    easy = sum(1 for r in results if r['norm_entropy'] < 0.5 and r['stride_reg'] > 0.3)
    medium = sum(1 for r in results if 0.5 <= r['norm_entropy'] < 0.8)
    hard = sum(1 for r in results if r['norm_entropy'] >= 0.8)
    
    print(f"\n--- Window Classification ---")
    print(f"Easy   (low ent + high stride):  {easy}/{n_windows} ({100*easy/n_windows:.1f}%)")
    print(f"Medium (moderate entropy):        {medium}/{n_windows} ({100*medium/n_windows:.1f}%)")
    print(f"Hard   (high entropy, ~random):   {hard}/{n_windows} ({100*hard/n_windows:.1f}%)")
    
    # Aggregate
    print(f"\n--- Aggregates ---")
    print(f"Unique pages/window:  {np.mean([r['unique_pages'] for r in results]):.0f} "
          f"(min={min(r['unique_pages'] for r in results)}, "
          f"max={max(r['unique_pages'] for r in results)})")
    print(f"Mean norm entropy:    {np.mean([r['norm_entropy'] for r in results]):.3f}")
    print(f"Mean stride reg:      {np.mean([r['stride_reg'] for r in results]):.3f}")
    print(f"Mean repeat rate:     {np.mean([r['repeat_rate'] for r in results]):.3f}")
    
    return results


# ===========================================================================
# ANALYSIS 2: Oracle (Markov upper bound)
# ===========================================================================

def oracle_analysis(pages, context_sizes=[1, 2, 4, 8, 16, 32, 64]):
    """
    Perfect Markov oracle: given last N pages, predict next page as
    the most common successor. This is the theoretical ceiling.
    """
    n = len(pages)
    pages_list = pages.tolist()
    
    print(f"\n{'='*70}")
    print(f"ORACLE ANALYSIS (Markov upper bound, n={n:,})")
    print(f"{'='*70}")
    
    results = []
    
    for ctx in context_sizes:
        if ctx >= n:
            continue
        
        print(f"  Context={ctx}...", end=" ", flush=True)
        
        transitions = defaultdict(Counter)
        for i in range(ctx, n):
            context = tuple(pages_list[i-ctx:i])
            transitions[context][pages_list[i]] += 1
        
        correct = 0
        for i in range(ctx, n):
            context = tuple(pages_list[i-ctx:i])
            predicted = transitions[context].most_common(1)[0][0]
            if predicted == pages_list[i]:
                correct += 1
        
        acc = correct / (n - ctx)
        n_ctx = len(transitions)
        branching = np.mean([len(v) for v in transitions.values()])
        
        print(f"acc={acc:.4f}, contexts={n_ctx:,}, avg_branch={branching:.1f}")
        results.append({
            'ctx': ctx, 'accuracy': acc,
            'n_contexts': n_ctx, 'avg_branching': branching,
        })
    
    print(f"\n{'Ctx':>5} {'Oracle Acc':>12} {'# Contexts':>12} {'Avg Branch':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['ctx']:>5} {r['accuracy']:>12.4f} {r['n_contexts']:>12,} {r['avg_branching']:>12.1f}")
    
    if results:
        best = max(results, key=lambda r: r['accuracy'])
        print(f"\nBest oracle: {best['accuracy']:.1%} at context={best['ctx']}")
        if best['accuracy'] < 0.15:
            print(f"-> LARGELY UNPREFETCHABLE. Even a perfect lookup table can't do much.")
            print(f"   Your 8.9% may be near the theoretical limit for this workload.")
        elif best['accuracy'] < 0.5:
            print(f"-> PARTIALLY PREFETCHABLE. Room to improve but ceiling is modest.")
        else:
            print(f"-> PREFETCHABLE. Model should be able to do much better.")
    
    return results


# ===========================================================================
# ANALYSIS 3: Address space characterization
# ===========================================================================

def address_space_analysis(pages):
    n = len(pages)
    pages_list = pages.tolist()
    
    print(f"\n{'='*70}")
    print(f"ADDRESS SPACE CHARACTERIZATION")
    print(f"{'='*70}")
    
    freq = Counter(pages_list)
    n_unique = len(freq)
    
    print(f"Total accesses:    {n:,}")
    print(f"Unique pages:      {n_unique:,}")
    print(f"Reuse ratio:       {n/n_unique:.1f}x")
    
    # Top-K coverage
    sorted_freq = sorted(freq.values(), reverse=True)
    cumsum = np.cumsum(sorted_freq)
    for k in [1, 5, 10, 50, 100, 500, 1000]:
        if k <= len(sorted_freq):
            print(f"  Top-{k:>4} pages: {cumsum[k-1]/n:.1%} of accesses")
    
    # Stride distribution
    deltas = np.diff(pages.astype(np.int64))
    delta_freq = Counter(deltas.tolist())
    print(f"\nTop 10 strides:")
    for d, c in delta_freq.most_common(10):
        print(f"  delta={d:>12}  ({c:>10,} times, {100*c/(n-1):.1f}%)")
    
    return freq


# ===========================================================================
# ANALYSIS 4: Baselines
# ===========================================================================

def baseline_analysis(pages, model_acc=0.089):
    n = len(pages)
    pages_list = pages.tolist()
    
    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISON (your model: {model_acc:.1%})")
    print(f"{'='*70}")
    
    # Repeat-prev
    repeat = np.sum(pages[1:] == pages[:-1]) / (n - 1)
    
    # Best stride
    deltas = np.diff(pages.astype(np.int64))
    best_delta, best_count = Counter(deltas.tolist()).most_common(1)[0]
    stride_rate = best_count / (n - 1)
    
    # Last-seen predictor
    K = 64
    lastk_correct = 0
    sample_n = min(n, 200000)
    for i in range(K, sample_n):
        predicted = Counter(pages_list[i-K:i]).most_common(1)[0][0]
        if predicted == pages_list[i]:
            lastk_correct += 1
    lastk_rate = lastk_correct / (sample_n - K)
    
    # Delta entropy
    delta_counts = Counter(deltas.tolist())
    probs = np.array(list(delta_counts.values()), dtype=np.float64) / len(deltas)
    delta_entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    print(f"Repeat-prev:               {repeat:.4f}")
    print(f"Best-stride (delta={best_delta}): {stride_rate:.4f}")
    print(f"Last-{K} most-common:       {lastk_rate:.4f}")
    print(f"Delta entropy:             {delta_entropy:.2f} bits")
    print()
    
    baselines = {'repeat': repeat, 'stride': stride_rate, 'lastk': lastk_rate}
    best_name = max(baselines, key=baselines.get)
    best_val = baselines[best_name]
    
    if model_acc > best_val:
        print(f"OK: Your model ({model_acc:.1%}) beats best baseline ({best_name}: {best_val:.1%})")
    else:
        print(f"WARNING: Your model ({model_acc:.1%}) is BELOW best baseline ({best_name}: {best_val:.1%})")


# ===========================================================================
# COMPARISON MODE
# ===========================================================================

def compare_workloads(traces_dict, max_pages):
    print(f"\n{'='*70}")
    print(f"WORKLOAD COMPARISON")
    print(f"{'='*70}")
    
    all_metrics = {}
    
    for name, path in traces_dict.items():
        print(f"\nProcessing {name}...")
        pages = load_pages(path, max_pages=max_pages)
        n = len(pages)
        pages_list = pages.tolist()
        unique = len(set(pages_list))
        
        repeat = np.sum(pages[1:] == pages[:-1]) / (n - 1)
        
        deltas = np.diff(pages.astype(np.int64))
        best_delta, best_count = Counter(deltas.tolist()).most_common(1)[0]
        stride_rate = best_count / (n - 1)
        
        dcounts = Counter(deltas.tolist())
        probs = np.array(list(dcounts.values()), dtype=np.float64) / len(deltas)
        delta_entropy = -np.sum(probs * np.log2(probs + 1e-12))
        
        # Markov-1
        trans = defaultdict(Counter)
        for i in range(1, n):
            trans[pages_list[i-1]][pages_list[i]] += 1
        m1 = sum(1 for i in range(1, n) if trans[pages_list[i-1]].most_common(1)[0][0] == pages_list[i]) / (n-1)
        
        # Markov-4
        sub_n = min(n, 200000)
        sub = pages_list[:sub_n]
        trans4 = defaultdict(Counter)
        for i in range(4, sub_n):
            trans4[tuple(sub[i-4:i])][sub[i]] += 1
        m4 = sum(1 for i in range(4, sub_n) if trans4[tuple(sub[i-4:i])].most_common(1)[0][0] == sub[i]) / (sub_n-4)
        
        all_metrics[name] = {
            'total': n, 'unique': unique, 'reuse': n/unique,
            'repeat': repeat, 'stride': stride_rate,
            'delta_ent': delta_entropy, 'markov1': m1, 'markov4': m4,
        }
    
    names = list(all_metrics.keys())
    col_w = max(14, max(len(n) for n in names) + 2)
    
    header = f"{'Metric':<22}" + "".join(f"{n:>{col_w}}" for n in names)
    print(f"\n{header}")
    print("-" * len(header))
    
    rows = [
        ('Total pages',     'total',     lambda v: f"{v:>,}"),
        ('Unique pages',    'unique',    lambda v: f"{v:>,}"),
        ('Reuse ratio',     'reuse',     lambda v: f"{v:>.1f}x"),
        ('Repeat-prev',     'repeat',    lambda v: f"{v:>.4f}"),
        ('Best stride',     'stride',    lambda v: f"{v:>.4f}"),
        ('Delta entropy',   'delta_ent', lambda v: f"{v:>.2f} bits"),
        ('Markov-1 oracle', 'markov1',   lambda v: f"{v:>.4f}"),
        ('Markov-4 oracle', 'markov4',   lambda v: f"{v:>.4f}"),
    ]
    
    for label, key, fmt in rows:
        row = f"{label:<22}"
        for name in names:
            row += f"{fmt(all_metrics[name][key]):>{col_w}}"
        print(row)
    
    best = max(names, key=lambda n: all_metrics[n]['markov4'])
    worst = min(names, key=lambda n: all_metrics[n]['markov4'])
    print(f"\nMost prefetchable:  {best} (Markov-4: {all_metrics[best]['markov4']:.1%})")
    print(f"Least prefetchable: {worst} (Markov-4: {all_metrics[worst]['markov4']:.1%})")


def main():
    parser = argparse.ArgumentParser(description='OverNet Trace Prefetchability Analysis')
    parser.add_argument('--trace', type=str, required=True,
                        help='Path to binary trace (uint64 memmap)')
    parser.add_argument('--max_pages', type=int, default=None,
                        help='Max pages to load (500000 for quick test)')
    parser.add_argument('--oracle_max', type=int, default=500000,
                        help='Max pages for oracle (default: 500K)')
    parser.add_argument('--window_size', type=int, default=10000)
    parser.add_argument('--model_acc', type=float, default=0.089,
                        help='Your model page match rate (default: 0.089)')
    parser.add_argument('--compare', nargs='*', default=None,
                        help='name:path pairs (e.g. hpl:hpl.bin bert:bert.bin)')
    
    args = parser.parse_args()
    
    pages = load_pages(args.trace, max_pages=args.max_pages)
    
    if len(pages) < 100:
        print("ERROR: Too few pages. Expected uint64 binary file.")
        sys.exit(1)
    
    address_space_analysis(pages)
    baseline_analysis(pages, model_acc=args.model_acc)
    analyze_phases(pages, window_size=args.window_size)
    
    oracle_pages = pages[:min(len(pages), args.oracle_max)]
    if len(oracle_pages) < len(pages):
        print(f"\n[Oracle limited to first {len(oracle_pages):,} pages]")
    oracle_analysis(oracle_pages)
    
    if args.compare:
        traces = {'dlrm': args.trace}
        for spec in args.compare:
            name, path = spec.split(':', 1)
            traces[name] = path
        compare_workloads(traces, max_pages=args.oracle_max)
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()