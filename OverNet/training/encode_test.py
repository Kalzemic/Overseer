import torch

input_size = 52
bit_shifts = torch.arange(input_size)
powers = (2 ** torch.arange(input_size)).long()

def to_bits(integers):
    integers = integers >> 12
    return ((integers.unsqueeze(-1) >> bit_shifts) & 1).float()

def to_integers(bits):
    binary = (bits > 0).long()
    return (binary * powers).sum(dim=-1) << 12

# Hardcoded addresses similar to what we see in production
addrs = torch.tensor([
    0x7f6807093000,
    0x7f6807094000,
    0x7f6807095000,
    0x55f648d72000,
    0x55f648d73000,
    0x55f648e80000,
], dtype=torch.long)

# Compute deltas
deltas = torch.zeros_like(addrs)
deltas[1:] = addrs[1:] - addrs[:-1]

print("Original deltas:")
for i, d in enumerate(deltas.tolist()):
    print(f"  [{i}]: 0x{d & 0xFFFFFFFFFFFFFFFF:x}")

bits = to_bits(deltas)
reconstructed = to_integers(bits)

print("\nReconstructed deltas:")
for i, d in enumerate(reconstructed.tolist()):
    print(f"  [{i}]: 0x{d & 0xFFFFFFFFFFFFFFFF:x}")

print("\nMatch:", torch.all(reconstructed == deltas).item())