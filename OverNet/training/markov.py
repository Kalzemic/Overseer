import numpy as np
from tqdm import tqdm
from collections import defaultdict

RESNET_VAL = './data/resnet/resnet_val.bin'
K = 64
MAX_SAMPLES = 500_000

def load_pages(path, max_samples):
    data = np.memmap(path, dtype=np.uint64, mode='r').reshape(-1, 2)
    pages = data[:max_samples, 1]  # column 1 = page address
    # convert to 36-bit tuples
    bits = ((pages[:, None] >> np.arange(36, dtype=np.uint64)) & 1).astype(np.int8)
    return [tuple(row) for row in bits]

def markov_entropy_check():
    print("Loading pages...")
    all_pages = load_pages(RESNET_VAL, MAX_SAMPLES)
    print(f"Collected {len(all_pages)} pages, building transition table...")

    transitions = defaultdict(lambda: defaultdict(int))

    for i in tqdm(range(K, len(all_pages))):
        state = tuple(all_pages[i-K:i])
        next_page = all_pages[i]
        transitions[state][next_page] += 1

    print(f"Unique states: {len(transitions)}")

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
    avg_branching = np.mean([len(v) for v in transitions.values()])

    print(f"\nMarkov Order: {K}")
    print(f"Conditional Entropy H(X_t | X_{{t-{K}}}...X_{{t-1}}): {conditional_entropy:.4f} bits")
    print(f"Average branching factor: {avg_branching:.2f}")
    print(f"Max branching factor: {max(len(v) for v in transitions.values())}")
    print(f"States with only 1 next page (fully determined): {sum(1 for v in transitions.values() if len(v) == 1)}")
    print(f"Total unique states: {len(transitions)}")

if __name__ == '__main__':
    markov_entropy_check()