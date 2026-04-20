import torch
import numpy as np
from Data import MemDataset
from tqdm import tqdm

DATAPATH =  './data/hpl/hpl_data_val.bin'

def entropy_check():
    dataset = MemDataset([DATAPATH], shuffle_files=False)
    
    bit_counts = np.zeros(36)
    total_pages = 0

    for x, y in tqdm(dataset):
        # y shape: (batch, 64, 36)
        pages = y.reshape(-1, 36).cpu().numpy()
        bit_counts += pages.sum(axis=0)
        total_pages += pages.shape[0]

    bit_probs = bit_counts / total_pages
    entropy = -bit_probs * np.log2(bit_probs + 1e-10) - (1 - bit_probs) * np.log2(1 - bit_probs + 1e-10)

    print(f"Total pages: {total_pages}")
    print(f"\nBit probabilities (P(bit=1)):")
    for i, p in enumerate(bit_probs):
        print(f"  Bit {i:2d}: {p:.4f}  entropy: {entropy[i]:.4f}")
    
    print(f"\nMean entropy: {entropy.mean():.4f}")
    print(f"Max entropy bit: {entropy.argmax()} ({entropy.max():.4f})")
    print(f"Min entropy bit: {entropy.argmin()} ({entropy.min():.4f})")

if __name__ == '__main__':
    entropy_check()