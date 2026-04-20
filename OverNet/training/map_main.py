import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from Data import MemDataset
from OverNet_Map import OverNet, OverMap

# --- config ---
VOCAB_SIZE  = 512
D_MODEL     = 64
NHEAD       = 2
NUM_LAYERS  = 1
SEQ_LEN     = 64
BATCH_SIZE  = 64
LR          = 3e-4
EPOCHS      = 3
GRAD_CLIP   = 1.0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESNET_TRAIN = [f'./data/resnet/resnet_train_0{i}.bin' for i in range(1, 10)]
RESNET_VAL   = './data/resnet/resnet_val.bin'


def interleave(batch: torch.Tensor, over_map: OverMap) -> torch.Tensor:
    # batch: (B, seq_len, 2) uint64 — columns [ip, page]
    B, S, _ = batch.shape
    ips   = batch[:, :, 0].reshape(-1).tolist()
    pages = batch[:, :, 1].reshape(-1).tolist()
    ip_ranks   = over_map.lookup_pcs(ips).reshape(B, S)
    page_ranks = over_map.lookup_pages(pages).reshape(B, S)
    interleaved = torch.stack([ip_ranks, page_ranks], dim=2)  # (B, S, 2)
    return interleaved.reshape(B, S * 2)                      # (B, S*2)


def run_epoch(model, loader, over_map, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_steps = 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for _, x, y in tqdm(loader):
            x_seq = interleave(x, over_map)  # (B, seq_len*2)
            y_seq = interleave(y, over_map)  # (B, seq_len*2)
            y_pages = y_seq[:, 1::2]         # (B, seq_len) — page targets only

            logits = model(x_seq)            # (B, seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y_pages.reshape(-1),
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss  += loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


def main():
    model    = OverNet(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS).to(DEVICE)
    over_map = OverMap(VOCAB_SIZE, DEVICE)

    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = MemDataset(RESNET_TRAIN, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    val_ds   = MemDataset(RESNET_VAL,   seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, over_map, optimizer, GRAD_CLIP)
        val_loss   = run_epoch(model, val_loader,   over_map)
        scheduler.step()

        print(f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': VOCAB_SIZE,
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'num_layers': NUM_LAYERS,
            }, 'overnet_best.pt')
            print(f"  → saved checkpoint (val {val_loss:.4f})")


if __name__ == '__main__':
    main()