import numpy as np
import os

RESNET_TRAIN = [f'./data/resnet{j}/resnet{j}_train_0{i}.bin' for i in range(1,9) for j in ['',2,3]]
RESNET_VAL   = [f'./data/resnet{j}/resnet{j}_val.bin' for j in ['',2,3]]

def get_uniques(paths, label):
    ips = set()
    pages = set()
    total = 0
    for p in paths:
        if not os.path.exists(p):
            print(f"  MISSING: {p}")
            continue
        data = np.memmap(p, dtype=np.uint64, mode='r').reshape(-1, 2)
        total += len(data)
        file_ips = set(data[:, 0].tolist())
        file_pages = set(data[:, 1].tolist())
        ips.update(file_ips)
        pages.update(file_pages)
        print(f"  {p}: {len(data):,} rows, {len(file_ips):,} IPs, {len(file_pages):,} pages")
    print(f"  {label} total: {total:,} rows, {len(ips):,} unique IPs, {len(pages):,} unique pages\n")
    return ips, pages

print("=== TRAIN ===")
train_ips, train_pages = get_uniques(RESNET_TRAIN, "TRAIN")

print("=== VAL ===")
val_ips, val_pages = get_uniques(RESNET_VAL, "VAL")

print("=== OVERLAP ===")
ip_overlap = train_ips & val_ips
page_overlap = train_pages & val_pages

print(f"IP overlap:   {len(ip_overlap):,} / {len(val_ips):,} val IPs seen in train ({100*len(ip_overlap)/max(len(val_ips),1):.1f}%)")
print(f"Page overlap: {len(page_overlap):,} / {len(val_pages):,} val pages seen in train ({100*len(page_overlap)/max(len(val_pages),1):.1f}%)")

print(f"\nTrain-only IPs:   {len(train_ips - val_ips):,}")
print(f"Val-only IPs:     {len(val_ips - train_ips):,}")
print(f"Train-only pages: {len(train_pages - val_pages):,}")
print(f"Val-only pages:   {len(val_pages - train_pages):,}")