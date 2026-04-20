import sys
import numpy as np
import matplotlib.pyplot as plt

RESNET_TRAIN = sys.argv[1:]

if not RESNET_TRAIN:
    print("Usage: python delta_distribution.py <file1.bin> [file2.bin ...]")
    sys.exit(1)
ROWS_PER_FILE = 1_000_000
OUTLIER_THRESHOLD = 10 * 1024 * 1024  # 10MB — anything larger is a context switch

def load_and_diff_per_file(files, rows_per_file=ROWS_PER_FILE):
    all_ip_deltas = []
    all_ea_deltas = []
    for f in files:
        data = np.fromfile(f, dtype=np.uint64, count=rows_per_file * 2)
        data = data.reshape(-1, 2)
        ips   = data[:, 0].astype(np.int64)
        pages = data[:, 1].astype(np.int64)
        all_ip_deltas.append(np.diff(ips))
        all_ea_deltas.append(np.diff(pages))
        print(f"Loaded {len(data):,} rows from {f}")
    return np.concatenate(all_ip_deltas), np.concatenate(all_ea_deltas)

def print_stats(name, deltas, filtered):
    removed = len(deltas) - len(filtered)
    print(f"\n{name}:")
    print(f"  Outliers removed: {removed:,} ({removed/len(deltas)*100:.2f}%)")
    print(f"  Mean:          {filtered.mean():.2f}")
    print(f"  Std:           {filtered.std():.2f}")
    print(f"  Median:        {np.median(filtered):.2f}")
    print(f"  % zero:        {(filtered == 0).mean()*100:.1f}%")
    print(f"  % negative:    {(filtered < 0).mean()*100:.1f}%")
    print(f"  % large >4096: {(np.abs(filtered) > 4096).mean()*100:.1f}%")
    print(f"  Max |delta|:   {np.abs(filtered).max():.2f}")

def plot_distributions(ip_deltas, ea_deltas):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ResNet Delta Distributions (outliers removed)", fontsize=14)

    clipped_ip = np.clip(ip_deltas, -50_000, 50_000)
    axes[0, 0].hist(clipped_ip, bins=300, log=True, color='steelblue')
    axes[0, 0].set_title("IP Delta (clipped ±50k)")
    axes[0, 0].set_xlabel("Delta")
    axes[0, 0].set_ylabel("Count (log)")

    axes[0, 1].hist(np.log1p(np.abs(ip_deltas)), bins=100, log=True, color='steelblue')
    axes[0, 1].set_title("IP log(|delta|)")
    axes[0, 1].set_xlabel("log(|delta|)")

    clipped_ea = np.clip(ea_deltas, -50_000, 50_000)
    axes[1, 0].hist(clipped_ea, bins=300, log=True, color='salmon')
    axes[1, 0].set_title("EA Delta (clipped ±50k)")
    axes[1, 0].set_xlabel("Delta")
    axes[1, 0].set_ylabel("Count (log)")

    axes[1, 1].hist(np.log1p(np.abs(ea_deltas)), bins=100, log=True, color='salmon')
    axes[1, 1].set_title("EA log(|delta|)")
    axes[1, 1].set_xlabel("log(|delta|)")

    # plt.tight_layout()
    # plt.savefig("resnet_delta_dist.png", dpi=150)
    # print("\nSaved resnet_delta_dist.png")
    # plt.show()

def main():
    print("Loading data...")
    ip_deltas, ea_deltas = load_and_diff_per_file(RESNET_TRAIN)

    # Filter outliers
    ip_mask = np.abs(ip_deltas) < OUTLIER_THRESHOLD
    ea_mask = np.abs(ea_deltas) < OUTLIER_THRESHOLD
    ip_filtered = ip_deltas[ip_mask]
    ea_filtered = ea_deltas[ea_mask]

    print("\n--- Delta Statistics (raw) ---")
    print(f"  IP  Max |delta|: {np.abs(ip_deltas).max():.0f}")
    print(f"  EA  Max |delta|: {np.abs(ea_deltas).max():.0f}")

    print("\n--- Delta Statistics (filtered <10MB jumps) ---")
    print_stats("IP deltas", ip_deltas, ip_filtered)
    print_stats("EA deltas", ea_deltas, ea_filtered)

    plot_distributions(ip_filtered, ea_filtered)

if __name__ == '__main__':
    main()