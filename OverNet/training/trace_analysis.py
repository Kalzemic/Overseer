import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import sys

RESNET_TRAIN = [f'./data/resnet{j}/resnet{j}_train_0{i}.bin'
                for i in range(1, 9) for j in ['', 2, 3]]
RESNET_VAL   = [f'./data/resnet{j}/resnet{j}_val.bin'
                for j in ['', 2, 3]]

BLOCK_SIZE   = 3_000_000
PC_COL, PAGE_COL = 0, 1

def iter_blocks(path, block_size=BLOCK_SIZE):
    data = np.memmap(path, dtype=np.uint64, mode='r').reshape(-1, 2)
    for start in range(0, len(data), block_size):
        block = np.array(data[start:start + block_size])
        yield block[:, PC_COL], block[:, PAGE_COL]

def first_block(path):
    for pc, page in iter_blocks(path):
        return pc, page

def analyze_layout(files):
    print(f"\n{'='*60}")
    print("ADDRESS SPACE LAYOUT (first block per file)")
    print(f"{'='*60}")
    high_sets = {}
    for path in files:
        p = Path(path)
        if not p.exists():
            continue
        _, pages = first_block(path)
        lo, hi   = pages.min(), pages.max()
        unique   = np.unique(pages)
        key      = p.stem
        high_sets[key] = set((pages >> 20).tolist())
        print(f"\n  {key}:")
        print(f"    rows in block  : {len(pages):>10,}")
        print(f"    unique pages   : {len(unique):>10,}")
        print(f"    page range     : [{lo:#014x}, {hi:#014x}]")
        print(f"    span (pages)   : {hi - lo:>10,}")
    keys = list(high_sets.keys())
    if len(keys) >= 2:
        print("\n  Cross-instance high-20-bit Jaccard overlap:")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = high_sets[keys[i]], high_sets[keys[j]]
                j_score = len(a & b) / max(len(a | b), 1)
                print(f"    {keys[i]} ∩ {keys[j]}: {j_score:.1%}")

def analyze_deltas(files, top_n=20):
    print(f"\n{'='*60}")
    print("DELTA DISTRIBUTION (first block per file)")
    print(f"{'='*60}")
    all_deltas = []
    for path in files:
        p = Path(path)
        if not p.exists():
            continue
        _, pages = first_block(path)
        deltas   = np.diff(pages.astype(np.int64))
        all_deltas.append(deltas)
        print(f"\n  {p.stem}:")
        print(f"    mean   : {deltas.mean():>12.1f}")
        print(f"    median : {np.median(deltas):>12.1f}")
        print(f"    std    : {deltas.std():>12.1f}")
        print(f"    zero   : {(deltas == 0).mean():.1%}")
        print(f"    |d|<=1 : {(np.abs(deltas) <= 1).mean():.1%}")
        print(f"    |d|<=8 : {(np.abs(deltas) <= 8).mean():.1%}")
        print(f"    |d|<=64: {(np.abs(deltas) <= 64).mean():.1%}")
    if all_deltas:
        combined = np.concatenate(all_deltas)
        counts   = defaultdict(int)
        for d in combined.tolist():
            counts[d] += 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        print(f"\n  Top {top_n} deltas (combined):")
        for delta, cnt in top:
            print(f"    {delta:+10d}  {cnt:>10,}")
    return all_deltas

def analyze_saturation(files, n_blocks=4):
    print(f"\n{'='*60}")
    print(f"VOCABULARY SATURATION (first {n_blocks} blocks per file)")
    print(f"{'='*60}")
    for path in files:
        p = Path(path)
        if not p.exists():
            continue
        seen, total = set(), 0
        print(f"\n  {p.stem}:")
        print(f"    {'rows seen':>12}  {'unique pages':>14}  {'new this block':>15}")
        for b, (_, pages) in enumerate(iter_blocks(path)):
            if b >= n_blocks:
                break
            new    = set(pages.tolist()) - seen
            seen  |= new
            total += len(pages)
            print(f"    {total:>12,}  {len(seen):>14,}  {len(new):>15,}")

def analyze_split(files, candidates=(8, 12, 16, 20)):
    print(f"\n{'='*60}")
    print("LOW-BIT REUSE BY SPLIT WIDTH (first block per file)")
    print(f"{'='*60}")
    for path in files:
        p = Path(path)
        if not p.exists():
            continue
        _, pages = first_block(path)
        print(f"\n  {p.stem}:")
        print(f"    {'bits':>6}  {'low vocab':>10}  {'coverage':>10}  {'avg highs/low':>14}  {'max highs/low':>14}")
        for bits in candidates:
            low  = pages & ((1 << bits) - 1)
            high = pages >> bits
            n_low    = len(np.unique(low))
            coverage = n_low / (2 ** bits)
            order      = np.argsort(low, stable=True)
            low_s      = low[order]
            high_s     = high[order]
            boundaries = np.where(np.diff(low_s))[0] + 1
            groups     = np.split(high_s, boundaries)
            reuse      = np.array([len(np.unique(g)) for g in groups])
            print(f"    {bits:>6}  {n_low:>10,}  {coverage:>9.1%}    {reuse.mean():>12.1f}  {reuse.max():>14,}")

def plot_deltas(all_deltas, label="train", clip=256):
    if not all_deltas:
        return
    combined = np.concatenate(all_deltas)
    clipped  = combined[(combined >= -clip) & (combined <= clip)]
    plt.figure(figsize=(10, 4))
    plt.hist(clipped, bins=min(512, clip * 2), log=True)
    plt.title(f"Page delta distribution ({label}, |d|<={clip})")
    plt.xlabel("Delta (pages)")
    plt.ylabel("Count (log)")
    plt.tight_layout()
    out = f"delta_hist_{label}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved -> {out}")

if __name__ == '__main__':
    train_files = [f for f in RESNET_TRAIN if Path(f).exists()]
    val_files   = [f for f in RESNET_VAL   if Path(f).exists()]

    if not train_files and not val_files:
        print("No .bin files found. Run from the directory containing data/")
        sys.exit(1)

    print(f"Found {len(train_files)} train file(s), {len(val_files)} val file(s)")

    representative = []
    for j in ['', 2, 3]:
        candidates = [f for f in train_files if f'resnet{j}_train' in f]
        if candidates:
            representative.append(candidates[0])

    all_files = representative + val_files

    analyze_layout(all_files)
    deltas = analyze_deltas(all_files)
    analyze_saturation(representative, n_blocks=4)
    analyze_split(all_files)
    plot_deltas(deltas[:len(representative)], "train")
    plot_deltas(deltas[len(representative):], "val")

    print("\nDone.")