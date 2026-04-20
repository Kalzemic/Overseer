import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from Data import MemDataset
from OverNet_Ordinal import OverNet

# ── Config ────────────────────────────────────────────────────────────────────
d_model    = 256
seq_length = 32
d_state    = 256
n_heads    = 8
layers     = 1
dropout    = 0.0  # off for profiling
N_WARMUP   = 10   # batches to warm up vocab before profiling
N_PROFILE  = 20   # batches to profile

RESNET_TRAIN = [f'./data/resnet{j}/resnet{j}_train_0{i}.bin'
                for i in range(1, 9) for j in ['', 2, 3]]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ── Setup ─────────────────────────────────────────────────────────────────────
train_data = MemDataset(RESNET_TRAIN, seq_len=seq_length)

net = OverNet(
    n_layers=layers, seq_len=seq_length, n_heads=n_heads,
    d_model=d_model, dropout=dropout
).to(device)
net.eval()

state = None

# ── Warmup (populate vocab, no profiling) ────────────────────────────────────
print(f"Warming up for {N_WARMUP} batches...")
with torch.no_grad():
    for i, (index, x, y) in enumerate(train_data):
        if i >= N_WARMUP:
            break
        x = x.to(device)
        if index == 0 or state is None or x.shape[0] != state.shape[0]:
            state = torch.zeros(x.shape[0], d_state, d_model).to(device)
            net.reset_vocab()
        out, state = net(x, state)
        state = state.detach()

print(f"Vocab size after warmup: {len(net.page_map)} pages, {len(net.ip_map)} IPs")

# ── Profile ───────────────────────────────────────────────────────────────────
print(f"\nProfiling {N_PROFILE} batches...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    with torch.no_grad():
        for i, (index, x, y) in enumerate(train_data):
            if i < N_WARMUP:
                continue
            if i >= N_WARMUP + N_PROFILE:
                break

            x = x.to(device)

            with record_function("lookup"):
                mapped = net.lookup(x)

            with record_function("embedding"):
                ip_ints   = mapped[:, :, 0]
                page_ints = mapped[:, :, 1]
                ip_h   = net.ip_embedding(ip_ints)
                page_h = net.page_embedding(page_ints)

            with record_function("cross_attn"):
                fused = net.cross_attn(page_h, ip_h)
                fused = fused + net.encoder_pe

            with record_function("encoder"):
                memory = net.encoder(fused)

            with record_function("overstate"):
                new_state = net.overstate(memory, state)

            with record_function("decoder"):
                queries = net.output_queries.expand(x.shape[0], -1, -1) + net.decoder_pe
                out = net.decoder(queries, memory)

            with record_function("head"):
                out = net.head(out)

            state = new_state.detach()

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("CUDA TIME (sorted)")
print("="*80)
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

print("\n" + "="*80)
print("CPU TIME (sorted)")
print("="*80)
print(prof.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=20
))

# Save trace for Chrome viewer (optional)
prof.export_chrome_trace("overnet_trace.json")
print("\nChrome trace saved to overnet_trace.json")
print("Open at chrome://tracing to view")