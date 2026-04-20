import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from Data import MemIPDataset
from OverNet_Sharded import OverNet, OverTable


lr            = 1e-4
n_layers      = 1
epochs        = 50
n_heads       = 8
max_grad_norm = 5
l2_lambda     = 1e-2
d_model       = 256
seq_length    = 64
vocab_size    = 3000
accum_steps   = 8

RESNET_TRAIN = [f'./data/resnet/resnet_train_0{i}.bin' for i in range(1, 10)]
RESNET_VAL   = './data/resnet2/resnet2_val.bin'

STEPS_PER_EPOCH = 250000
total_steps     = epochs * STEPS_PER_EPOCH // accum_steps
device          = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def bits_to_idx(bits, vocab_size):
    powers = (2 ** torch.arange(48, device=bits.device)).long()
    raw    = (bits.long() * powers).sum(dim=-1)
    return (raw % vocab_size).long()


def bits_to_raw(bits):
    powers = (2 ** torch.arange(48, device=bits.device)).long()
    return (bits.long() * powers).sum(dim=-1)


def prepare_batch(x, y, vocab_size):
    x_ips = bits_to_idx(x[:, :, 0, :], vocab_size)
    y_ips = bits_to_idx(y[:, :, 0, :], vocab_size)
    x_eas = bits_to_raw(x[:, :, 1, :])
    y_eas = bits_to_raw(y[:, :, 1, :])
    return x_ips, y_ips, x_eas, y_eas


def compute_coverage(predicted_ips, y_eas, table):
    """
    Fully GPU — no Python loops.
    predicted_ips: (batch, seq) on device
    y_eas:         (batch, seq) on device
    """
    predicted_eas = table.lookup(predicted_ips)          # (batch, seq)
    # check if each predicted EA appears anywhere in y_eas for that batch row
    # predicted_eas: (batch, seq, 1) vs y_eas: (batch, 1, seq)
    match = (predicted_eas.unsqueeze(2) == y_eas.unsqueeze(1)).any(dim=2)  # (batch, seq)
    return match.float().mean().item()


def main():
    train_data = MemIPDataset(RESNET_TRAIN, seq_len=seq_length)
    val_data   = MemIPDataset([RESNET_VAL],  seq_len=seq_length)

    model = OverNet(
        seq_length = seq_length,
        d_model    = d_model,
        n_heads    = n_heads,
        n_layers   = n_layers,
        dropout    = 0.2,
        vocab_size = vocab_size,
    ).to(device)

    table = OverTable(vocab_size=vocab_size, device=device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
    scheduler = OneCycleLR(
        optimizer,
        max_lr           = lr,
        total_steps      = total_steps,
        pct_start        = 0.05,
        anneal_strategy  = 'cos',
        div_factor       = 10,
        final_div_factor = 100,
    )
    scaler = torch.amp.GradScaler()

    model, history = train(model, table, criterion, optimizer, scheduler, scaler, train_data, val_data)
    plot_history(history)


def train(model, table, criterion, optimizer, scheduler, scaler, train_data, val_data):

    history       = dict(train_loss=[], train_exact=[], train_coverage=[],
                         val_loss=[], val_exact=[], val_coverage=[])
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ----------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------
        model.train()
        batch_losses   = []
        batch_exact    = []
        batch_coverage = []

        for index, x, y in tqdm(train_data, desc="train", total=STEPS_PER_EPOCH):

            x_ips, y_ips, x_eas, y_eas = prepare_batch(x, y, vocab_size)

            is_boundary = (index % accum_steps == accum_steps - 1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x_ips)
                loss   = criterion(
                    logits.reshape(-1, vocab_size),
                    y_ips.reshape(-1)
                )

            if index % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss / accum_steps).backward()

            if is_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # update OverTable — fully GPU, no loops
            table.update(x_ips.reshape(-1), x_eas.reshape(-1))

            # evaluate coverage — fully GPU
            with torch.no_grad():
                predicted_ips = logits.argmax(dim=-1)
                coverage = compute_coverage(predicted_ips, y_eas, table)
                batch_coverage.append(coverage)

            batch_losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            exact = (preds == y_ips).float().mean().item()
            batch_exact.append(exact)

            if index >= STEPS_PER_EPOCH:
                break

        epoch_loss     = float(np.mean(batch_losses))
        epoch_exact    = float(np.mean(batch_exact))
        epoch_coverage = float(np.mean(batch_coverage))
        history['train_loss'].append(epoch_loss)
        history['train_exact'].append(epoch_exact)
        history['train_coverage'].append(epoch_coverage)
        print(f"Train | loss: {epoch_loss:.4f}  ip_exact: {epoch_exact * 100:.2f}%  coverage: {epoch_coverage * 100:.2f}%")

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        model.eval()
        batch_losses   = []
        batch_exact    = []
        batch_coverage = []

        for index, x, y in tqdm(val_data, desc="val"):

            x_ips, y_ips, x_eas, y_eas = prepare_batch(x, y, vocab_size)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x_ips)
                    loss   = criterion(
                        logits.reshape(-1, vocab_size),
                        y_ips.reshape(-1)
                    )

                if not torch.isfinite(logits).all():
                    continue

                predicted_ips = logits.argmax(dim=-1)
                coverage = compute_coverage(predicted_ips, y_eas, table)
                batch_coverage.append(coverage)

                batch_losses.append(loss.item())
                preds = logits.argmax(dim=-1)
                exact = (preds == y_ips).float().mean().item()
                batch_exact.append(exact)

            # update table from val observations
            table.update(x_ips.reshape(-1), x_eas.reshape(-1))

        epoch_loss     = float(np.mean(batch_losses))
        epoch_exact    = float(np.mean(batch_exact))
        epoch_coverage = float(np.mean(batch_coverage))
        history['val_loss'].append(epoch_loss)
        history['val_exact'].append(epoch_exact)
        history['val_coverage'].append(epoch_coverage)
        print(f"Val   | loss: {epoch_loss:.4f}  ip_exact: {epoch_exact * 100:.2f}%  coverage: {epoch_coverage * 100:.2f}%")

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(model.state_dict(), f'OverNet_Sharded_{n_layers}L.pt')
            torch.save(table.table, f'OverTable_{n_layers}L.pt')
            print(f"  → saved checkpoint")

        torch.cuda.empty_cache()

    return model, history


def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_exact'], label='train')
    axes[1].plot(history['val_exact'],   label='val')
    axes[1].set_title('IP Exact Match')
    axes[1].legend()
    axes[2].plot(history['train_coverage'], label='train')
    axes[2].plot(history['val_coverage'],   label='val')
    axes[2].set_title('EA Coverage')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == '__main__':
    main()