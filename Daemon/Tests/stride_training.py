from transformers import BertConfig, BertForMaskedLM
import torch
import time
import os

# Disable PyTorch's caching allocator entirely — forces malloc/free directly
# This prevents MADV_FREE on tensor memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# Force glibc to not use MADV_FREE/MADV_DONTNEED on free()
# M_PERTURB causes malloc to touch pages on alloc/free keeping them resident
import ctypes
import ctypes.util

libc = ctypes.CDLL("libc.so.6")
# MALLOPT_PERTURB = -6, any nonzero value enables it
libc.mallopt(-6, 1)
# Disable madvise in glibc allocator
# M_MMAP_THRESHOLD = -3, set very high to avoid mmap for small allocs
libc.mallopt(-3, 256 * 1024 * 1024)  # 256MB threshold

# Disable kernel readahead for all current mappings
# MADV_RANDOM = 1 — tells kernel not to bother with sequential readahead
MADV_RANDOM = 1
with open('/proc/self/maps') as f:
    for line in f:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            addr_range = parts[0].split('-')
            start_addr = int(addr_range[0], 16)
            end_addr   = int(addr_range[1], 16)
            libc.madvise(
                ctypes.c_void_p(start_addr),
                ctypes.c_size_t(end_addr - start_addr),
                ctypes.c_int(MADV_RANDOM)
            )
        except Exception:
            pass

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

VOCAB_SIZE = 30522

config = BertConfig(
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=1024,
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=128
)

net = BertForMaskedLM(config)

params = sum(p.numel() for p in net.parameters())
print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")


with open('/proc/self/maps') as f:
    for line in f:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            addr_range = parts[0].split('-')
            start_addr = int(addr_range[0], 16)
            end_addr   = int(addr_range[1], 16)
            libc.madvise(
                ctypes.c_void_p(start_addr),
                ctypes.c_size_t(end_addr - start_addr),
                ctypes.c_int(MADV_RANDOM)
            )
        except Exception:
            pass

batch_size = 8
seq_len = 64
num_steps = 10

input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
labels = input_ids.clone()

# Keep gradient tensors alive rather than dropping them
# This prevents zero_grad from releasing pages back to allocator
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

# Pre-allocate and pin gradient buffers so they stay resident
for param in net.parameters():
    param.grad = torch.zeros_like(param)

start = time.perf_counter()
for i in range(num_steps):
    output = net(input_ids, labels=labels)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)

elapsed = time.perf_counter() - start
print(f"Training loop: {elapsed:.2f}s ({elapsed / num_steps * 1000:.1f}ms/step)")
