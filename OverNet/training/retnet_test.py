import torch
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

VOCAB_SIZE = 32_768
D_MODEL    = 128
NHEAD      = 4
NUM_LAYERS = 4

config = RetNetConfig(
    vocab_size=VOCAB_SIZE,
    decoder_embed_dim=D_MODEL,
    decoder_retention_heads=NHEAD,
    decoder_layers=NUM_LAYERS,
    decoder_ffn_embed_dim=D_MODEL * 4,
)
retnet = RetNetDecoder(config)

print("=== architecture ===")
print(retnet)

print("\n=== named modules ===")
for name, _ in retnet.named_modules():
    print(name)

# test forward
x = torch.randint(0, VOCAB_SIZE, (2, 16))  # (batch, seq_len)
out, _ = retnet(x)
print(f"\n=== forward output shape ===")
print(f"input:  {x.shape}")
print(f"output: {out.shape}")  # should be (2, 16, d_model) or (2, 16, vocab_size)?

# test recurrent
print("\n=== recurrent forward ===")
incremental_state = {}
for t in range(4):
    xt = torch.randint(0, VOCAB_SIZE, (1, 1))
    out_t, incremental_state = retnet(xt, incremental_state=incremental_state)
    print(f"step {t}: input {xt.shape} → output {out_t.shape}")