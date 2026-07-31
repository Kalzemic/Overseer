import re

addrs = []
with open("trace_faults.txt") as f:
    for line in f:
        m = re.search(r'addr: (0x[0-9a-f]+)', line)
        if m:
            addrs.append(int(m.group(1), 16))

PAGE = 4096
deltas = [addrs[i+1] - addrs[i] for i in range(len(addrs)-1)]

from collections import Counter
page_deltas = [d // PAGE for d in deltas]
top = Counter(page_deltas).most_common(20)
print("Top page-delta strides:")
for stride, count in top:
    print(f"  stride={stride:+d} pages ({stride*4}KB): {count} times ({count/len(deltas)*100:.1f}%)")