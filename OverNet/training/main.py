import torch
import torch.nn as nn
import numpy as np
from Data import MemDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from OverNet_Ordinal import OverNet
from torch.optim.lr_scheduler import OneCycleLR
from LookupTable import LookupTable


lr = 1e-4
layers = 1
epochs = 50
heads = 8
max_grad_norm = 5
l2lambda = 1e-2
d_model = 256
seq_length = 256
dropout = .3
d_state = 128
vocab_size=2 ** 9
batch_size = 64


# RESNET_TRAIN = [f'./data/resnet{j}/resnet{j}_train_0{i}.bin' for i in range(1, 9) for j in ['', 2]] + \
#                [f'./data/resnet{j}/resnet{j}_val.bin' for j in ['', 2]]
# RESNET_VAL   = [f'./data/resnet3/resnet3_train_0{i}.bin' for i in range(1, 9)]  + \
#                [f'./data/resnet3/resnet3_val.bin']
RESNET_TRAIN = [f'./data/resnet_{i}.bin' for i in range(1,10)]
RESNET_VAL = ['./data/resnet_10.bin']
BERT_TRAIN = [f'./data/bert_{i}.bin' for i in range(1,10)]
BERT_VAL = ['./data/bert_10.bin']
HPL_TRAIN = [f'./data/hpl_{i}.bin' for i in range(1,10)]
HPL_VAL = ['./data/hpl_10.bin']
MLPERF_TRAIN = []
MLPERF_VAL = []
STEPS_PER_EPOCH = 12465 // 4

total_steps = epochs * STEPS_PER_EPOCH
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def compute_deltas(x, anchor):
    x = x.long()
    anchor = anchor.long()
    deltas = torch.empty_like(x)
    deltas[:, 0, :] = x[:, 0, :] - anchor
    deltas[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return deltas

def compute_coverage(pred_page_ranks, true_page_ranks):
    match = (pred_page_ranks.unsqueeze(2) == true_page_ranks.unsqueeze(1)).any(dim=2)
    return match.float().mean(dim=1).mean().item()


def main():
    train_data = MemDataset([*RESNET_TRAIN, *BERT_TRAIN, *HPL_TRAIN], seq_len=seq_length, batch_size=batch_size)
    val_data   = MemDataset([*RESNET_VAL, *BERT_VAL, *HPL_VAL],   seq_len=seq_length, batch_size=batch_size)

    overnet = OverNet(
        n_layers=layers, seq_len=seq_length, n_heads=heads,
        d_model=d_model, dropout=dropout,vocab_size=vocab_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(overnet.parameters(), lr=lr, weight_decay=l2lambda)
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
    overnet, history = train(overnet, criterion, optimizer, scheduler,
                             train_data, val_data, scaler)
    plot_history(history)


def train(net: OverNet, criterion, optimizer, scheduler, train_data, val_data, scaler):

    history = dict(train_loss=[], train_acc=[], train_coverage=[],
                   val_loss=[], val_acc=[], val_coverage=[])
    best_val_acc = 0.0
    state = None
    lut =LookupTable(vocab_size=vocab_size)
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        net.train()

        batch_losses   = []
        batch_acc      = []
        batch_coverage = []
        lut.reset_vocab()

        for index, x, y in tqdm(train_data, desc="train"):
            x = x.to(device)
            y = y.to(device)

            y = compute_deltas(y,x[:,-1,:])
            x = compute_deltas(x,torch.zeros_like(x[:,0,:],device=device))
            
            if index == 0 or state is None or x.shape[0] != state.shape[0]:
                state = torch.zeros(x.shape[0], d_state, d_model).to(device)
                

            # convert y pages to ordinal ranks — (batch, seq)
            x = lut.lookup(x).to(device)
            true_page_ranks = lut.lookup(y)[:, :, 1].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out, new_state = net(x, state)
                # if index == 0: 
                #     print(out.permute(0, 2, 1))
                loss = criterion(out.permute(0, 2, 1), true_page_ranks)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            state = new_state.detach()

            pred_page_ranks = out.argmax(dim=-1)  # (batch, seq)

            batch_losses.append(loss.item())
            with torch.no_grad():
                acc      = (pred_page_ranks == true_page_ranks).float().mean().item()
                coverage = compute_coverage(pred_page_ranks, true_page_ranks)
            batch_acc.append(acc)
            batch_coverage.append(coverage)

        epoch_loss     = float(np.mean(batch_losses))
        epoch_acc      = float(np.mean(batch_acc))
        epoch_coverage = float(np.mean(batch_coverage))
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_coverage'].append(epoch_coverage)
        print(f"Train | loss: {epoch_loss:.4f}  acc: {epoch_acc*100:.2f}%  coverage: {epoch_coverage*100:.2f}%")

    
        net.eval()
        batch_losses   = []
        batch_acc      = []
        batch_coverage = []
        lut.reset_vocab()


        with torch.no_grad():
            for index, x, y in tqdm(val_data, desc="val"):
                x = x.to(device)
                y = y.to(device)

                y = compute_deltas(y,x[:,-1,:])
                x = compute_deltas(x,torch.zeros_like(x[:,0,:],device=device))

                if index == 0 or state is None or x.shape[0] != state.shape[0]:
                    state = torch.zeros(x.shape[0], d_state, d_model).to(device)

                if not torch.isfinite(x.float()).all():
                    continue
                
                x = lut.lookup(x).to(device)
                true_page_ranks = lut.lookup(y)[:, :, 1].to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out, new_state = net(x, state)

                    if not torch.isfinite(out).all():
                        continue

                    loss = criterion(out.permute(0, 2, 1), true_page_ranks)

                state = new_state
                pred_page_ranks = out.argmax(dim=-1)

                batch_losses.append(loss.item())
                acc      = (pred_page_ranks == true_page_ranks).float().mean().item()
                coverage = compute_coverage(pred_page_ranks, true_page_ranks)
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
            torch.save(net.state_dict(), f'OverNet_Ordinal_{layers}L.pt')
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
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[2].plot(history['train_coverage'], label='train')
    axes[2].plot(history['val_coverage'],   label='val')
    axes[2].set_title('Coverage')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('training_history_ordinal.png')
    plt.show()


if __name__ == '__main__':
    main()