"""
Verify Hilbert curve properties for various N.
- Check all cells visited exactly once
- Show stride distribution for small N (should be ~25% each for ±1, ±N)
- Check stride distribution converges toward 25% each as N grows
"""
from collections import Counter

def hilbert_d2xy(n, d):
    x = y = 0
    s = 1
    t = d
    while s < n:
        rx = 1 & (t >> 1)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y

for log_n in [1, 2, 3, 4, 5]:
    N = 1 << log_n
    pages = []
    for d in range(N * N):
        x, y = hilbert_d2xy(N, d)
        page_idx = y * N + x  # row-major
        pages.append(page_idx)

    # Verify all pages visited exactly once
    assert sorted(pages) == list(range(N * N)), "FAIL: not all pages visited"

    deltas = [pages[i+1] - pages[i] for i in range(len(pages)-1)]
    cnt = Counter(deltas)
    n_trans = len(deltas)

    print(f"\nN={N:4d} (log_n={log_n})  transitions={n_trans}")
    for stride, count in sorted(cnt.items()):
        print(f"  stride {stride:+5d}  {count:6d} ({count/n_trans*100:.1f}%)")
    dominant = cnt.most_common(1)[0]
    print(f"  Dominant: {dominant[0]:+d}  {dominant[1]/n_trans*100:.1f}%")
