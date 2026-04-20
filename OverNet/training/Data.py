import torch
from torch.utils.data import IterableDataset
import numpy as np
import random

class MemDataset(IterableDataset):
    def __init__(self, bin_paths, seq_len=64, block_size=3_000_000, batch_size=64, shuffle_files=True):
        super().__init__()
        if isinstance(bin_paths, str):
            bin_paths = [bin_paths]
        self.bin_paths = bin_paths
        self.seq_len = seq_len
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def load_block(self, data, start):
        end = min(start + self.block_size, len(data))
        block = np.array(data[start:end])          # (N, 2) uint64
        return torch.from_numpy(block).to(self.device)  # (N, 2) uint64

    def __iter__(self):
        sources = []

        for path in self.bin_paths:
            data = np.memmap(path, dtype=np.uint64, mode='r').reshape(-1, 2)
            sources.append({'data': data, 'idx': 0})

        while sources:
            if self.shuffle_files:
                random.shuffle(sources)

            torch.cuda.empty_cache()
            exhausted = []

            for i, src in enumerate(sources):
                if src['idx'] >= len(src['data']) - 2 * self.seq_len:
                    exhausted.append(i)
                    continue

                block = self.load_block(src['data'], src['idx'])
                # block: (N, 2) uint64
                state_index = 0

                windows = block.unfold(0, 2 * self.seq_len, self.seq_len)
                # shape: (num_windows, 2, 2*seq_len)

                windows = windows.permute(0, 2, 1)
                # shape: (num_windows, 2*seq_len, 2)

                x_all = windows[:, :self.seq_len, :]   # (num_windows, seq_len, 2)
                y_all = windows[:, self.seq_len:, :]   # (num_windows, seq_len, 2)

                for j in range(0, len(x_all), self.batch_size):
                    yield state_index, x_all[j:j+self.batch_size], y_all[j:j+self.batch_size]
                    state_index += 1

                src['idx'] += self.block_size

            for i in reversed(exhausted):
                sources.pop(i)




class MemIPDataset(IterableDataset):
    def __init__(self, bin_paths, seq_len=64, block_size=3_000_000, batch_size=64, shuffle_files=True):
        super().__init__()
        if isinstance(bin_paths, str):
            bin_paths = [bin_paths]
        self.bin_paths = bin_paths
        self.seq_len = seq_len
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def load_block(self, data, start):
        end = min(start + self.block_size, len(data))
        block = np.array(data[start:end])             # (N, 2) uint64
        bits = ((block[:, :, None] >> np.arange(48, dtype=np.uint64)) & 1).astype(np.float32)
        # shape: (N, 2, 48)
        return torch.from_numpy(bits).to(self.device)

    def __iter__(self):
        sources = []
        
        for path in self.bin_paths:
            data = np.memmap(path, dtype=np.uint64, mode='r').reshape(-1, 2)
            sources.append({'data': data, 'idx': 0})

        while sources:
            if self.shuffle_files:
                random.shuffle(sources)

            torch.cuda.empty_cache()
            exhausted = []

            for i, src in enumerate(sources):
                if src['idx'] >= len(src['data']) - 2 * self.seq_len:
                    exhausted.append(i)
                    continue

                block = self.load_block(src['data'], src['idx'])
                # block: (N, 2, 48)
                state_index = 0

                # unfold over sequence dimension (dim=0), window size = 2*seq_len, step = seq_len
                windows = block.unfold(0, 2 * self.seq_len, self.seq_len)
                # shape: (num_windows, 2, 48, 2*seq_len)

                windows = windows.permute(0, 3, 1, 2)
                # shape: (num_windows, 2*seq_len, 2, 48)

                x_all = windows[:, :self.seq_len, :, :]   # (num_windows, seq_len, 2, 48)
                y_all = windows[:, self.seq_len:, :, :]   # (num_windows, seq_len, 2, 48)

                for j in range(0, len(x_all), self.batch_size):
                    yield state_index, x_all[j:j+self.batch_size], y_all[j:j+self.batch_size]
                    state_index += 1

                src['idx'] += self.block_size

            for i in reversed(exhausted):
                sources.pop(i)


