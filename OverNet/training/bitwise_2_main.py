import torch
import torch.nn as nn
import numpy as np 
from Data import MemDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from OverNet_Bitwise import OverNet
from torch.optim.lr_scheduler import OneCycleLR

lr = 1e-4
layers = 1
epochs = 50
heads = 8
max_grad_norm = 5
l2lambda = 1e-2
d_model = 256
seq_length = 64

RESNET_TRAIN = [f'./data/resnet{j}/resnet{j}_train_0{i}.bin' for i in range(1,10) for j in ['']]
RESNET_VAL = './data/resnet/resnet_val.bin'
STEPS_PER_EPOCH = 250000

total_steps = epochs * STEPS_PER_EPOCH
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

powers = (2 ** torch.arange(48)).long().to(device)


def to_bits(x):
    # x: (B, L) int64
    # returns: (B, L, 48) float32
    return ((x.unsqueeze(-1) >> torch.arange(48, device=x.device, dtype=torch.int64)) & 1).float()


def bits_to_pages(bits):
    """
    Convert bit tensor to raw page values.
    Args:
        bits: (batch, seq, 48) float or bool
    Returns:
        (batch, seq) long
    """
    return (bits.long() * powers).sum(dim=-1)


def compute_coverage(pred_bits, true_bits):
    """
    Set-based coverage — did each predicted page appear anywhere
    in the true future page sequence, regardless of position.
    Args:
        pred_bits: (batch, seq, 48) — predicted bit logits
        true_bits: (batch, seq, 48) — true page bits
    Returns:
        float — coverage score
    """
    pred_pages = bits_to_pages(pred_bits > 0)   # (batch, seq)
    true_pages = bits_to_pages(true_bits)        # (batch, seq)
    match = (pred_pages.unsqueeze(2) == true_pages.unsqueeze(1)).any(dim=2)  # (batch, seq)
    return match.float().mean().item()


def main():
    train_data = MemDataset(RESNET_TRAIN, seq_len=seq_length)
    val_data   = MemDataset([RESNET_VAL], seq_len=seq_length)
    overnet    = OverNet(n_layers=layers, seq_len=seq_length, n_heads=heads, d_model=d_model, dropout=0.2).to(device)
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.AdamW(overnet.parameters(), lr=lr, weight_decay=l2lambda)
    scheduler  = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    scaler = torch.amp.GradScaler()
    overnet, history = train(overnet, criterion, optimizer, train_data, val_data, scaler, scheduler)
    plot_history(history)


def train(net, criterion, optimizer, train_data, val_data, scaler, scheduler):

    history = dict(train_loss=[], train_acc=[], train_coverage=[],
                   val_loss=[], val_acc=[], val_coverage=[])
    best_val_loss = float('inf')
    state = None

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        net.train()

        batch_losses    = []
        batch_vq_losses = []
        batch_acc       = []
        batch_coverage  = []

        for index, x, y in tqdm(train_data, desc="train"):

            if index == 0 or state is None or x.shape[0] != state.shape[0]:
                state = torch.zeros(x.shape[0], d_model).to(device)

            y_bits = to_bits(y[:, :, 1].to(torch.int64))  # (batch, seq_len, 48)

            if index % 8 == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output, new_state, continuous, phase_idx = net(x, state)
                task_loss = criterion(output, y_bits)
                vq_loss   = net.overstate.loss(continuous.detach(), phase_idx)
                loss      = (task_loss + 0.01 * vq_loss) / 8

            scaler.scale(loss).backward(retain_graph=(index % 8 != 7))

            if index % 8 == 7:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            state = new_state.detach() if index % 8 == 7 else new_state

            batch_losses.append(task_loss.item())
            batch_vq_losses.append(vq_loss.item())
            acc      = ((output > 0) == y_bits).float().mean().item()
            coverage = compute_coverage(output.float(), y_bits)
            batch_acc.append(acc)
            batch_coverage.append(coverage)

        epoch_loss     = float(np.mean(batch_losses))
        epoch_vq_loss  = float(np.mean(batch_vq_losses))
        epoch_acc      = float(np.mean(batch_acc))
        epoch_coverage = float(np.mean(batch_coverage))
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_coverage'].append(epoch_coverage)
        print(f"Train | loss: {epoch_loss:.4f}  vq: {epoch_vq_loss:.4f}  acc: {epoch_acc*100:.2f}%  coverage: {epoch_coverage*100:.2f}%")

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        net.eval()
        batch_losses   = []
        batch_acc      = []
        batch_coverage = []

        with torch.no_grad():
            for index, x, y in tqdm(val_data, desc="val"):

                if index == 0 or state is None or x.shape[0] != state.shape[0]:
                    state = torch.zeros(x.shape[0], d_model).to(device)

                y_bits = to_bits(y[:, :, 1].to(torch.int64))  # (batch, seq_len, 48)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output, new_state, continuous, phase_idx = net(x, state)

                    if not torch.isfinite(output).all():
                        continue

                    loss = criterion(output, y_bits)

                state = new_state
                batch_losses.append(loss.item())
                acc      = ((output > 0) == y_bits).float().mean().item()
                coverage = compute_coverage(output.float(), y_bits)
                batch_acc.append(acc)
                batch_coverage.append(coverage)

        epoch_loss     = float(np.mean(batch_losses))
        epoch_acc      = float(np.mean(batch_acc))
        epoch_coverage = float(np.mean(batch_coverage))
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        history['val_coverage'].append(epoch_coverage)
        print(f"Val   | loss: {epoch_loss:.4f}  acc: {epoch_acc*100:.2f}%  coverage: {epoch_coverage*100:.2f}%")

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
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