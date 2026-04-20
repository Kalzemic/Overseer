import sys
import numpy as np
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python ip_ea_distribution.py <file1.bin> [file2.bin ...]")
    sys.exit(1)

ROWS_PER_FILE = 1_000_000
OUTLIER_THRESHOLD = 10 * 1024 * 1024

def load(files):
    all_ips, all_pages = [], []
    for f in files:
        data = np.fromfile(f, dtype=np.uint64, count=ROWS_PER_FILE * 2).reshape(-1, 2)
        all_ips.append(data[:, 0].astype(np.int64))
        all_pages.append(data[:, 1].astype(np.int64))
        print(f"Loaded {len(data):,} rows from {f}")
    return np.concatenate(all_ips), np.concatenate(all_pages)

def main():
    ips, pages = load(sys.argv[1:])

    # align ip[t] with ea_delta[t] = page[t] - page[t-1]
    ips_aligned  = ips[1:]
    ea_deltas    = np.diff(pages)

    # filter outliers
    mask = np.abs(ea_deltas) < OUTLIER_THRESHOLD
    ips_aligned = ips_aligned[mask]
    ea_deltas   = ea_deltas[mask]

    # group ea deltas by ip
    ip_to_deltas = defaultdict(list)
    for ip, delta in zip(ips_aligned, ea_deltas):
        ip_to_deltas[ip].append(delta)

    # analyze top 20 most frequent IPs
    top_ips = sorted(ip_to_deltas.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    print(f"\n{'IP':>16} | {'count':>8} | {'mean':>12} | {'std':>12} | {'median':>12} | {'% single val':>12}")
    print("-" * 85)
    for ip, deltas in top_ips:
        d = np.array(deltas)
        unique = len(np.unique(d))
        dominant = (d == np.bincount(np.abs(d).astype(np.int64) % 100000).argmax()).mean() * 100
        print(f"  {ip:16x} | {len(d):>8,} | {d.mean():>12.1f} | {d.std():>12.1f} | {np.median(d):>12.1f} | {unique:>8} unique")

    # summary
    stds = [np.array(v).std() for v in ip_to_deltas.values() if len(v) > 10]
    print(f"\nGlobal EA delta std:      {ea_deltas.std():.1f}")
    print(f"Median per-IP EA std:     {np.median(stds):.1f}")
    print(f"Mean per-IP EA std:       {np.mean(stds):.1f}")
    print(f"% IPs with std < 1000:    {(np.array(stds) < 1000).mean()*100:.1f}%")
    print(f"% IPs with std < 10000:   {(np.array(stds) < 10000).mean()*100:.1f}%")

if __name__ == '__main__':
    main()