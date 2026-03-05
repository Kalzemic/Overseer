import torch
import numpy as np
from Data import MemDataset
from tqdm import tqdm
from collections import defaultdict

MLPERF_VAL = './mlperf/mlperf_val.bin'

K = 64 # markov order
MAX_SAMPLES = 500000  # cap to avoid memory explosion

def bits_to_int(bits):
    result = 0
    for b in bits:
        result = (result << 1) | int(b)
    return result

def markov_entropy_check():
    dataset = MemDataset([MLPERF_VAL])
    
    
    transitions = defaultdict(lambda: defaultdict(int))
    
    all_pages = []
    total = 0
    
    print("Collecting pages...")
    for x, y in tqdm(dataset):
        pages = y.reshape(-1, 36).cpu().numpy().astype(np.int8)
        for page in pages:
            all_pages.append(tuple(page))
            total += 1
            if total >= MAX_SAMPLES:
                break
        if total >= MAX_SAMPLES:
            break

    print(f"Collected {total} pages, building transition table...")
    
    for i in range(K, len(all_pages)):
        state = tuple(all_pages[i-K:i])
        next_page = all_pages[i]
        transitions[state][next_page] += 1

    print(f"Unique states: {len(transitions)}")
    
    # compute conditional entropy H(X_t | X_{t-1}, ..., X_{t-k})
    total_conditional_entropy = 0
    total_states_visited = 0

    for state, next_counts in transitions.items():
        total_from_state = sum(next_counts.values())
        state_entropy = 0
        for count in next_counts.values():
            p = count / total_from_state
            state_entropy -= p * np.log2(p + 1e-10)
        total_conditional_entropy += state_entropy * total_from_state
        total_states_visited += total_from_state

    conditional_entropy = total_conditional_entropy / total_states_visited

    # also compute average branching factor
    avg_branching = np.mean([len(v) for v in transitions.values()])

    print(f"\nMarkov Order: {K}")
    print(f"Conditional Entropy H(X_t | X_{{t-{K}}}...X_{{t-1}}): {conditional_entropy:.4f} bits")
    print(f"Average branching factor (unique next pages per state): {avg_branching:.2f}")
    print(f"Max branching factor: {max(len(v) for v in transitions.values())}")
    print(f"States with only 1 next page (fully determined): {sum(1 for v in transitions.values() if len(v) == 1)}")
    print(f"Total unique states: {len(transitions)}")

if __name__ == '__main__':
    markov_entropy_check()