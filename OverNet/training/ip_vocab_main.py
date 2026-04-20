import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

from Data import MemIPDataset
from OverNetIPVocab import OverNet


lr            = 1e-4
n_layers      = 1
epochs        = 50
n_heads       = 8
max_grad_norm = 5
l2_lambda     = 1e-2
d_model       = 256
seq_length    = 64
accum_steps   = 8

RESNET_TRAIN = [f'./data/resnet/resnet_train_0{i}.bin' for i in range(1, 10)]
RESNET_VAL   = './data/resnet/resnet_val.bin'

STEPS_PER_EPOCH = 250000
total_steps     = epochs * STEPS_PER_EPOCH // accum_steps
device          = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def bits_to_raw(bits):
    """
    Convert (batch, seq, 48) float bit tensor to (batch, seq) long raw addresses.
    No modulo hashing — raw value preserved.
    """
    powers = (2 ** torch.arange(48, device=bits.device)).long()
    return (bits.long() * powers).sum(dim=-1)


def prepare_batch(x, y):
    """
    Args:
        x: (batch, seq, 2, 48)
        y: (batch, seq, 2, 48)
    Returns:
        x_pages: (batch, seq) long — raw page addresses
        x_ips:   (batch, seq) long — raw ip addresses
        y_pages: (batch, seq) long — raw page addresses
    """
    x_pages = bits_to_raw(x[:, :, 1, :])
    x_ips   = bits_to_raw(x[:, :, 0, :])
    y_pages = bits_to_raw(y[:, :, 1, :])
    return x_pages, x_ips, y_pages


def chunk_targets(y_pages):
    """
    Split raw page addresses into 4 chunk targets for loss computation.
    Args:
        y_pages: (batch, seq) long
    Returns:
        4 tensors of (batch, seq) long, each in [0, 4096)
    """
    c0 = (y_pages >>  0) & 0xFFF
    c1 = (y_pages >> 12) & 0xFFF
    c2 = (y_pages >> 24) & 0xFFF
    c3 = (y_pages >> 36) & 0xFFF
    return c0, c1, c2, c3


def compute_loss(criterion, logits, y_pages):
    """
    Sum cross entropy loss over all 4 chunk heads.
    Args:
        logits:  tuple of 4 (batch, seq, 4096)
        y_pages: (batch, seq) long — raw page addresses
    """
    c0, c1, c2, c3 = chunk_targets(y_pages)
    l0 = criterion(logits[0].reshape(-1, 4096), c0.reshape(-1))
    l1 = criterion(logits[1].reshape(-1, 4096), c1.reshape(-1))
    l2 = criterion(logits[2].reshape(-1, 4096), c2.reshape(-1))
    l3 = criterion(logits[3].reshape(-1, 4096), c3.reshape(-1))
    return l0 + l1 + l2 + l3


def compute_exact(logits, y_pages):
    """
    Exact match — all 4 chunks must be correct.
    Args:
        logits:  tuple of 4 (batch, seq, 4096)
        y_pages: (batch, seq) long
    """
    c0, c1, c2, c3 = chunk_targets(y_pages)
    p0 = logits[0].argmax(dim=-1)
    p1 = logits[1].argmax(dim=-1)
    p2 = logits[2].argmax(dim=-1)
    p3 = logits[3].argmax(dim=-1)
    correct = (p0 == c0) & (p1 == c1) & (p2 == c2) & (p3 == c3)
    return correct.float().mean().item()



def main():
    train_data = MemIPDataset(RESNET_TRAIN, seq_len=seq_length)
    val_data   = MemIPDataset([RESNET_VAL],  seq_len=seq_length)

    model = OverNet(
        seq_len  = seq_length,
        d_model  = d_model,
        n_heads  = n_heads,
        n_layers = n_layers,
        dropout  = 0.2,
    ).to(device)

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

    model, history = train(model, criterion, optimizer, scheduler, scaler, train_data, val_data)
    plot_history(history)


def train(model, criterion, optimizer, scheduler, scaler, train_data, val_data):

    history       = dict(train_loss=[], train_exact=[], val_loss=[], val_exact=[])
    best_val_loss = float('inf')
    state         = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ----------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------
        model.train()
        batch_losses   = []
        batch_vq_losses = []
        batch_exact    = []

        for index, x, y in tqdm(train_data, desc="train"):

            if index == 0 or state is None or x.shape[0] != state.shape[0]:
                state = torch.zeros(x.shape[0], d_model, device=device)

            x_pages, x_ips, y_pages = prepare_batch(x, y)
           

            is_boundary = (index % accum_steps == accum_steps - 1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, new_state, continuous, phase_idx = model(x_pages, x_ips, state)
                task_loss = compute_loss(criterion, logits, y_pages)
                vq_loss   = model.overstate.loss(continuous, phase_idx)
                loss      = task_loss + 0.01 * vq_loss

            if index % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward(retain_graph=not is_boundary)

            if is_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            state = new_state.detach() if is_boundary else new_state

            batch_losses.append(task_loss.item())
            batch_vq_losses.append(vq_loss.item())
            batch_exact.append(compute_exact(logits, y_pages))

            if index >= STEPS_PER_EPOCH:
                break

        epoch_loss    = float(np.mean(batch_losses))
        epoch_vq_loss = float(np.mean(batch_vq_losses))
        epoch_exact   = float(np.mean(batch_exact))
        history['train_loss'].append(epoch_loss)
        history['train_exact'].append(epoch_exact)
        print(f"Train | loss: {epoch_loss:.4f}  vq: {epoch_vq_loss:.4f}  exact: {epoch_exact * 100:.2f}%")

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        model.eval()
        batch_losses = []
        batch_exact  = []

        with torch.no_grad():
            for index, x, y in tqdm(val_data, desc="val"):

                if index == 0 or state is None or x.shape[0] != state.shape[0]:
                    state = torch.zeros(x.shape[0], d_model, device=device)

                if not torch.isfinite(x).all():
                    continue

                x_pages, x_ips, y_pages = prepare_batch(x, y)
                

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, new_state, continuous, phase_idx = model(x_pages, x_ips, state)
                    loss = compute_loss(criterion, logits, y_pages)

                if not torch.isfinite(logits[0]).all():
                    continue

                state = new_state
                batch_losses.append(loss.item())
                batch_exact.append(compute_exact(logits, y_pages))

        epoch_loss  = float(np.mean(batch_losses))
        epoch_exact = float(np.mean(batch_exact))
        history['val_loss'].append(epoch_loss)
        history['val_exact'].append(epoch_exact)
        print(f"Val   | loss: {epoch_loss:.4f}  exact: {epoch_exact * 100:.2f}%")

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(model.state_dict(), f'OverNet_Chunked_{n_layers}L.pt')
            print(f"  → saved checkpoint")

        torch.cuda.empty_cache()

    return model, history


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_exact'], label='train')
    axes[1].plot(history['val_exact'],   label='val')
    axes[1].set_title('Exact Match')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == '__main__':
    main()