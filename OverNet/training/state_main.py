import torch
import torch.nn as nn
import numpy as np
from Data import MemDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from OverNet_State import OverNet
from torch.optim.lr_scheduler import OneCycleLR

lr = 1e-4
layers = 1
epochs = 50
heads = 8
max_grad_norm = 5
l2lambda = 1e-2
d_model = 128
seq_length = 128
dropout = .3
d_state= 128
batch_size=64

RESNET_TRAIN = [f'./data/resnet_{i}.bin' for i in range(1,10)]
RESNET_VAL = ['./data/resnet_10.bin']
BERT_TRAIN = [f'./data/bert_{i}.bin' for i in range(1,10)]
BERT_VAL = ['./data/bert_10.bin']
HPL_TRAIN = [f'./data/hpl_{i}.bin' for i in range(1,10)]
HPL_VAL = ['./data/hpl_10.bin']
MLPERF_TRAIN = []
MLPERF_VAL = []
STEPS_PER_EPOCH = 7174

total_steps = epochs * STEPS_PER_EPOCH
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def compute_coverage(pred_page_ints, true_page_ints):
    true_page_ints = true_page_ints & ~0xFFF
    match = (pred_page_ints.unsqueeze(2) == true_page_ints.unsqueeze(1)).any(dim=2)
    return match.float().mean(dim=1).mean().item()
def diagnose_coverage(pred_page_ints, true_page_ints, n=3):
    """Print first n examples of predicted vs true pages"""
    pred = pred_page_ints[0, :n].tolist()
    true = true_page_ints[0, :n].tolist()
    for i in range(n):
        p = pred[i] & 0xFFFFFFFFFFFFFFFF
        t = true[i] & 0xFFFFFFFFFFFFFFFF
        match = "✓" if p == t else "✗"
        print(f"  [{i}] pred: 0x{p:x}  true: 0x{t:x}  {match}")

def main():
    train_data = MemDataset([*RESNET_TRAIN, *BERT_TRAIN, *HPL_TRAIN], seq_len=seq_length, batch_size=batch_size)
    val_data   = MemDataset([*RESNET_VAL, *BERT_VAL, *HPL_VAL],   seq_len=seq_length, batch_size=batch_size)

    overnet = OverNet(
        n_layers=layers, seq_len=seq_length, n_heads=heads,
        d_model=d_model, dropout=dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))

    # single optimizer — no VQ parameters to separate out
    optimizer = torch.optim.AdamW(overnet.parameters(), lr=lr, weight_decay=l2lambda)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    scaler = torch.amp.GradScaler()
    overnet, history = train(overnet, criterion, optimizer, scheduler,
                             train_data, val_data, scaler)
    plot_history(history)


def make_delta_bits(x, y, net):
    true_page_ints = y[:, :, 1].long() 
    last_x_page    = x[:, -1, 1].long()

    first_delta = true_page_ints[:, :1] - last_x_page.unsqueeze(1)
    rest_deltas = true_page_ints[:, 1:] - true_page_ints[:, :-1]
    true_deltas = torch.cat([first_delta, rest_deltas], dim=1)

    true_delta_bits = net.to_bits(true_deltas)
    return true_delta_bits, true_page_ints


def train(net, criterion, optimizer, scheduler, train_data, val_data, scaler):

    history = dict(train_loss=[], train_acc=[], train_coverage=[],
                   val_loss=[], val_acc=[], val_coverage=[])
    best_val_acc = 0.0
    state = None

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        net.train()

        batch_losses   = []
        batch_acc      = []
        batch_coverage = []

        for index, x, y in tqdm(train_data, desc="train"):
            x = x.to(device)
            y = y.to(device)

            if index == 0 or state is None or x.shape[0] != state.shape[0]:
               state = torch.zeros(x.shape[0],d_state, d_model).to(device)

            true_delta_bits, true_page_ints = make_delta_bits(x, y, net)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                delta_logits, pred_page_ints, new_state = net(x, state)
                loss = criterion(delta_logits, true_delta_bits)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            state = new_state.detach()

            batch_losses.append(loss.item())
            with torch.no_grad():
                acc      = ((delta_logits > 0) == true_delta_bits.bool()).float().mean().item()
                coverage = compute_coverage(pred_page_ints, true_page_ints)
                if index == 0:  # only first batch
                    print("\nCoverage diagnosis:")
                    diagnose_coverage(pred_page_ints, true_page_ints)
            batch_acc.append(acc)
            batch_coverage.append(coverage)

        epoch_loss     = float(np.mean(batch_losses))
        epoch_acc      = float(np.mean(batch_acc))
        epoch_coverage = float(np.mean(batch_coverage))
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_coverage'].append(epoch_coverage)
        print(f"Train | loss: {epoch_loss:.4f}  acc: {epoch_acc*100:.2f}%  coverage: {epoch_coverage*100:.2f}%")

        # ── Validation ──────────────────────────────────────────────────
        net.eval()
        batch_losses   = []
        batch_acc      = []
        batch_coverage = []

        with torch.no_grad():
            for index, x, y in tqdm(val_data, desc="val"):
                x = x.to(device)
                y = y.to(device)

                if index == 0 or state is None or x.shape[0] != state.shape[0]:
                    state = torch.zeros(x.shape[0], d_state, d_model).to(device)

                if not torch.isfinite(x.float()).all():
                    continue

                true_delta_bits, true_page_ints = make_delta_bits(x, y, net)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    delta_logits, pred_page_ints, new_state = net(x, state)

                    if not torch.isfinite(delta_logits).all():
                        continue

                    loss = criterion(delta_logits, true_delta_bits)

                state = new_state
                batch_losses.append(loss.item())
                acc      = ((delta_logits > 0) == true_delta_bits.bool()).float().mean().item()
                coverage = compute_coverage(pred_page_ints, true_page_ints)
                if index == 0:  # only first batch
                    print("\nCoverage diagnosis:")
                    diagnose_coverage(pred_page_ints, true_page_ints)
                batch_acc.append(acc)
                batch_coverage.append(coverage)

        epoch_loss     = float(np.mean(batch_losses))
        epoch_acc      = float(np.mean(batch_acc))
        epoch_coverage = float(np.mean(batch_coverage))
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        history['val_coverage'].append(epoch_coverage)
        print(f"Val   | loss: {epoch_loss:.4f}  acc: {epoch_acc*100:.2f}%  coverage: {epoch_coverage*100:.2f}%")

        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(net.state_dict(), f'OverNet_Bitwise_{layers}L.pt')
            print(f"  → saved checkpoint")

        torch.cuda.empty_cache()

    return net, history


def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_acc'], label='train')
    axes[1].plot(history['val_acc'],   label='val')
    axes[1].set_title('Bit Accuracy')
    axes[1].legend()
    axes[2].plot(history['train_coverage'], label='train')
    axes[2].plot(history['val_coverage'],   label='val')
    axes[2].set_title('Coverage')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('training_history_bitwise.png')
    plt.show()


if __name__ == '__main__':
    main()