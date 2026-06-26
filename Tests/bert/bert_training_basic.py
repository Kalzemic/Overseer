from transformers import BertConfig, BertForMaskedLM
import torch, time, resource

VOCAB_SIZE = 30522
config = BertConfig(
    hidden_size=256, num_hidden_layers=6, num_attention_heads=8,
    intermediate_size=1024, vocab_size=VOCAB_SIZE, max_position_embeddings=128
)
net = BertForMaskedLM(config)

batch_size, seq_len, num_steps = 8, 64, 5
input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
labels = input_ids.clone()
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

before = resource.getrusage(resource.RUSAGE_SELF)
start = time.perf_counter()
for i in range(num_steps):
    output = net(input_ids, labels=labels)
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
elapsed = time.perf_counter() - start
after = resource.getrusage(resource.RUSAGE_SELF)

params = sum(p.numel() for p in net.parameters())
print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")
print(f"Training loop: {elapsed:.2f}s ({elapsed / num_steps * 1000:.1f}ms/step)")
print(f"Major faults: {after.ru_majflt - before.ru_majflt}")
print(f"Minor faults: {after.ru_minflt - before.ru_minflt}")