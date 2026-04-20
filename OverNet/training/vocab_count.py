import numpy as np
from collections import Counter

RESNET_TRAIN = [f'./data/resnet/resnet_train_0{i}.bin' for i in range(1, 10)]

unique_pages = set()
unique_pcs   = set()

for path in RESNET_TRAIN:
    data = np.memmap(path, dtype=np.uint64, mode='r').reshape(-1, 2)
    # column 0 = ip, column 1 = page
    unique_pcs.update(data[:, 0].tolist())
    unique_pages.update(data[:, 1].tolist())
    print(f"{path}: {len(data):,} rows | running unique pages: {len(unique_pages):,} | unique pcs: {len(unique_pcs):,}")

print(f"\ntotal unique pages: {len(unique_pages):,}")
print(f"total unique pcs:   {len(unique_pcs):,}")

# also look at frequency distribution
page_counts = Counter()
for path in RESNET_TRAIN:
    data = np.memmap(path, dtype=np.uint64, mode='r').reshape(-1, 2)
    page_counts.update(data[:, 1].tolist())

total = sum(page_counts.values())
top_512  = sum(v for _, v in page_counts.most_common(512))
top_4k   = sum(v for _, v in page_counts.most_common(4096))
top_8k   = sum(v for _, v in page_counts.most_common(8192))
top_32k  = sum(v for _, v in page_counts.most_common(32768))

print(f"\ncoverage of top-512  pages: {100*top_512/total:.1f}%")
print(f"coverage of top-4k   pages: {100*top_4k/total:.1f}%")
print(f"coverage of top-8k   pages: {100*top_8k/total:.1f}%")
print(f"coverage of top-32k  pages: {100*top_32k/total:.1f}%")