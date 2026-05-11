import torch
import torch.nn as nn
import numpy as np
from Data import MemDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from OverNet_reg import OverNet
from torch.optim.lr_scheduler import OneCycleLR

lr = 5e-5
layers = 1
epochs = 50
heads = 8
max_grad_norm = 1
l2lambda = 1e-2
d_model = 128
seq_length = 128
dropout = 0.3
d_state = 128
batch_size = 64
PAGE_SIZE = 4096

RESNET_TRAIN = [f'./data/resnet_{i}.bin' for i in range(1, 10)]
RESNET_VAL = ['./data/resnet_10.bin']
BERT_TRAIN = [f'./data/bert_{i}.bin' for i in range(1, 10)]
BERT_VAL = ['./data/bert_10.bin']
HPL_TRAIN = [f'./data/hpl_{i}.bin' for i in range(1, 10)]
HPL_VAL = ['./data/hpl_10.bin']

STEPS_PER_EPOCH = 7174
total_steps = epochs * STEPS_PER_EPOCH
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def compute_coverage(pred_page_ints, true_page_ints):
    match = (pred_page_ints.unsqueeze(2) == true_page_ints.unsqueeze(1)).any(dim=2)
    return match.float().mean(dim=1).mean().item()


def diagnose_coverage(pred_page_ints, true_page_ints, n=3):
    true = true_page_ints[0, :n].tolist()
    pred = pred_page_ints[0, :n].tolist()
    for i in range(n):
        p = pred[i] & 0xFFFFFFFFFFFFFFFF
        t = true[i] & 0xFFFFFFFFFFFFFFFF
        match = "✓" if p == t else "✗"
        #print(f"  [{i}] pred: 0x{p:x}  true: 0x{t:x}  {match}")


def make_batch(x, y):
    page_x = x[:, :, 1].float()
    ip_x   = x[:, :, 0].float()

    page_deltas_x = torch.zeros_like(page_x)
    page_deltas_x[:, 1:] = page_x[:, 1:] - page_x[:, :-1]

    ip_deltas_x = torch.zeros_like(ip_x)
    ip_deltas_x[:, 1:] = ip_x[:, 1:] - ip_x[:, :-1]

    # normalize by page size
    page_deltas_x = page_deltas_x / PAGE_SIZE
    ip_deltas_x   = ip_deltas_x / PAGE_SIZE

    page_y = y[:, :, 1].float()
    last_x_page = page_x[:, -1:]

    true_first_delta = page_y[:, :1] - last_x_page
    true_rest_deltas = page_y[:, 1:] - page_y[:, :-1]
    true_page_deltas = torch.cat([true_first_delta, true_rest_deltas], dim=1)

    # normalize true deltas
    true_page_deltas = true_page_deltas / PAGE_SIZE

    true_page_ints = y[:, :, 1].long()

    return page_deltas_x, ip_deltas_x, true_page_deltas, true_page_ints


def train(net, criterion, optimizer, scheduler, train_data, val_data, scaler):
    history = dict(train_loss=[], train_coverage=[],
                   val_loss=[], val_coverage=[])
    best_val_loss = float('inf')
    state = None

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        net.train()

        batch_losses   = []
        batch_coverage = []

        for index, x, y in tqdm(train_data, desc="train"):
            x = x.to(device)
            y = y.to(device)

            if index == 0 or state is None or x.shape[0] != state.shape[0]:
                state = torch.zeros(x.shape[0], d_state, d_model).to(device)

            page_deltas_x, ip_deltas_x, true_page_deltas, true_page_ints = make_batch(x, y)

            pred_deltas, new_state = net(page_deltas_x, ip_deltas_x, state)
            loss = criterion(pred_deltas, true_page_deltas)

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
                anchor = x[:, -1, 1].long()
                # denormalize predictions
                pred_page_ints = anchor.unsqueeze(1) + (pred_deltas * PAGE_SIZE).cumsum(dim=1).round().long()
                coverage = compute_coverage(pred_page_ints, true_page_ints)

            if index % 365 == 0:
                #print("\nCoverage diagnosis:")
                diagnose_coverage(pred_page_ints, true_page_ints)

            batch_coverage.append(coverage)

        epoch_loss     = float(np.mean(batch_losses))
        epoch_coverage = float(np.mean(batch_coverage))
        history['train_loss'].append(epoch_loss)
        history['train_coverage'].append(epoch_coverage)
        print(f"Train | loss: {epoch_loss:.4f}  coverage: {epoch_coverage*100:.2f}%")

        # Validation
        net.eval()
        batch_losses   = []
        batch_coverage = []

        with torch.no_grad():
            for index, x, y in tqdm(val_data, desc="val"):
                x = x.to(device)
                y = y.to(device)

                if index == 0 or state is None or x.shape[0] != state.shape[0]:
                    state = torch.zeros(x.shape[0], d_state, d_model).to(device)

                if not torch.isfinite(x.float()).all():
                    continue

                page_deltas_x, ip_deltas_x, true_page_deltas, true_page_ints = make_batch(x, y)

                pred_deltas, new_state = net(page_deltas_x, ip_deltas_x, state)
                loss = criterion(pred_deltas, true_page_deltas)

                if not torch.isfinite(loss):
                    continue

                state = new_state

                anchor = x[:, -1, 1].long()
                # denormalize predictions
                pred_page_ints = anchor.unsqueeze(1) + (pred_deltas * PAGE_SIZE).cumsum(dim=1).round().long()
                coverage = compute_coverage(pred_page_ints, true_page_ints)

                if index == 0:
                    print("\nCoverage diagnosis:")
                    diagnose_coverage(pred_page_ints, true_page_ints)

                batch_losses.append(loss.item())
                batch_coverage.append(coverage)

        epoch_loss     = float(np.mean(batch_losses))
        epoch_coverage = float(np.mean(batch_coverage))
        history['val_loss'].append(epoch_loss)
        history['val_coverage'].append(epoch_coverage)
        print(f"Val   | loss: {epoch_loss:.4f}  coverage: {epoch_coverage*100:.2f}%")

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(net.state_dict(), f'OverNet_Regression_{layers}L.pt')
            print(f"  → saved checkpoint")

        torch.cuda.empty_cache()

    return net, history


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_coverage'], label='train')
    axes[1].plot(history['val_coverage'],   label='val')
    axes[1].set_title('Coverage')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('training_history_regression.png')
    plt.show()


def main():
    train_data = MemDataset([*RESNET_TRAIN, *BERT_TRAIN, *HPL_TRAIN],
                            seq_len=seq_length, batch_size=batch_size)
    val_data   = MemDataset([*RESNET_VAL, *BERT_VAL, *HPL_VAL],
                            seq_len=seq_length, batch_size=batch_size)

    net = OverNet(
        n_layers=layers, seq_len=seq_length, n_heads=heads,
        d_model=d_model, dropout=dropout, d_state=d_state
    ).to(device)

    criterion = nn.HuberLoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2lambda)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    scaler = torch.amp.GradScaler()
    net, history = train(net, criterion, optimizer, scheduler,
                         train_data, val_data, scaler)
    plot_history(history)


if __name__ == '__main__':
    main()