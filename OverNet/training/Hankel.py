import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('/home/kalzemic/Projects/Overseer/OverNet/training')
from Data import MemDataset

def compute_hankel_rank(pages, max_size=100, threshold=1e-6):
    """
    Compute the numerical rank of the Hankel matrix for a page sequence.
    pages: 1D array of page integers
    max_size: rows/cols of Hankel matrix
    """
    n = len(pages)
    size = min(max_size, n // 2)
    
    # build Hankel matrix
    H = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            H[i, j] = pages[i + j]
    
    # compute singular values
    svs = np.linalg.svd(H, compute_uv=False)
    
    # numerical rank = number of singular values above threshold * max_sv
    max_sv = svs[0]
    rank = np.sum(svs > threshold * max_sv)
    
    return rank, svs

def pages_from_bits(bits):
    """Convert bit arrays to page integers"""
    powers = 2 ** np.arange(bits.shape[-1], dtype=np.float64)
    result = (bits.astype(np.float64) * powers).sum(axis=-1)
    return result.astype(np.int64).flatten()

def main():
    # load DLRM dataset
    print("Loading DLRM dataset...")
    dataset = MemDataset(
        '/home/kalzemic/Projects/Overseer/OverNet/training/hpl/hpl_data_val.bin',
        seq_len=64
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ranks = []
    sv_profiles = []
    n_samples = 500

    print(f"Computing Hankel rank on {n_samples} sequences...")
    for i, (x, y) in enumerate(tqdm(loader)):
        if i >= n_samples:
            break

        # convert bits to integers
        x_np = x.squeeze(0).cpu().numpy()  # (64, 36)
        pages = pages_from_bits(x_np)  # (64,)

        rank, svs = compute_hankel_rank(pages, max_size=32)
        ranks.append(rank)
        sv_profiles.append(svs)

    ranks = np.array(ranks)
    print(f"\n=== DLRM Hankel Analysis ===")
    print(f"Mean rank: {ranks.mean():.2f}")
    print(f"Median rank: {np.median(ranks):.2f}")
    print(f"Max rank: {ranks.max()}")
    print(f"Min rank: {ranks.min()}")
    print(f"Rank distribution:")
    unique, counts = np.unique(ranks, return_counts=True)
    for r, c in zip(unique, counts):
        print(f"  Rank {r}: {c} sequences ({100*c/len(ranks):.1f}%)")

    # average singular value decay
    avg_svs = np.mean(sv_profiles, axis=0)
    print(f"\nAverage singular value decay (first 10):")
    for i, sv in enumerate(avg_svs[:10]):
        print(f"  SV[{i}]: {sv:.4f}")
    print(f"  Ratio SV[0]/SV[-1]: {avg_svs[0]/avg_svs[-1]:.2f}")

if __name__ == '__main__':
    main()