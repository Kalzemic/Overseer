import torch
import torch.nn as nn
import numpy as np 
from Data import MemIPDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from OverNetIP import OverNet
from torch.optim.lr_scheduler import OneCycleLR


lr = 1e-4
layers = 1
epochs = 50
max_grad_norm= 5
l2lambda = 1e-4
d_state= 256
seq_length = 64

MLPERF_TRAIN = [f'./mlperf_ip/mlperf_ip_train_0{i}.bin' for i in range(10)]
MLPERF_VAL = './mlperf_ip/mlperf_ip_val.bin'
BERT_TRAIN = [f'./bert_ip/bert_ip_train_0{i}.bin' for i in range(10)]
BERT_VAL = './bert_ip/bert_ip_val.bin'
MLPERF_STEPS_PER_EPOCH = 232610
HPL_STEPS_PER_EPOCH = 1
total_steps = epochs * MLPERF_STEPS_PER_EPOCH
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

def main():
    train_data = MemIPDataset(MLPERF_TRAIN, seq_len=seq_length)
    val_data = MemIPDataset([MLPERF_VAL], seq_len=seq_length)
    overnet = OverNet(n_layers=layers, d_model=d_state,dropout=0.2).to(device)
    #print(overnet.causal_mask[:4,:4])
    criterion = nn.BCEWithLogitsLoss()
    # x_test = torch.randn(2, 64, 36).cuda()
    # proj_out = overnet.projection(x_test)
    # print("Projection output norm:", proj_out.norm(dim=-1).mean())
    # print("Projection output std:", proj_out.std())
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(overnet.parameters(), lr=lr, weight_decay=l2lambda)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=total_steps,
        pct_start=0.05,       # 5% warmup
        anneal_strategy='cos',
        div_factor=10,         # start_lr = 3e-4 / 10 = 3e-5
        final_div_factor=100   # end_lr = 3e-4 / 100 = 3e-6
    )
    scaler = torch.amp.grad_scaler.GradScaler()
    overnet, train_losses, train_acc, val_losses, val_acc= train(overnet,criterion,optimizer,train_data,val_data, scaler, scheduler)

    plt.plot(range(epochs),train_losses,label='Train Losses')
    plt.legend()
    plt.show()
    plt.plot(range(epochs),train_acc,label='Train Accuracy')
    plt.legend()
    plt.show()
    plt.plot(range(epochs),val_losses,label='Validation Losses')
    plt.legend()
    plt.show()
    plt.plot(range(epochs),val_acc,label='Validation Accuracy')
    plt.legend()
    plt.show()

        


def train(net:nn.Module ,criterion: nn.Module, optimizer: torch.optim.Optimizer, train: torch.utils.data.IterableDataset, val:torch.utils.data.IterableDataset, 
          scaler, scheduler):

    train_losses = []
    train_acc =[]
    val_losses = []
    val_acc = []
    val_exact = []
    best_val_loss = 10
    best_state_dict = None
    

    for epoch in range(epochs):
        
        print(f'Epoch {epoch + 1} Running ')
        net.train()

        batch_losses = []
        batch_acc = []
        
        for index,x,y in tqdm(train):

            if index == 0 or x.shape[0] != state.shape[0]:
                state = torch.zeros(x.shape[0],d_state).to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output, new_state = net(x, state)
                loss = criterion(output,y[:,:,1,:])

            batch_losses.append(loss.item())
            acc = torch.mean(((output > 0) == y[:,:,1,:]).float()).item()
            batch_acc.append(acc)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            # if total_steps % 100 == 0: 
            #     print(f"Grad norm: {grad_norm.item():.4f} | Clipped: {grad_norm.item() > max_grad_norm}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state = new_state.detach()

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        batch_losses = []
        batch_acc = []
        print(f'Train Loss: {epoch_loss} Train Accuracy: {epoch_acc}')


        net.eval()

        batch_exact = []
        with torch.no_grad():
            
            
            for i, (index, x, y) in enumerate(tqdm(val)):

                
                if index == 0 or x.shape[0] != state.shape[0]:
                    state = torch.zeros(x.shape[0],d_state).to(device)
                if not torch.isfinite(x).all():
                    print(f"Non-finite in x at step {i}")
                    continue

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output, new_state = net(x, state)
                
                    if not torch.isfinite(output).all():
                        print(f"Non-finite in output at step {i}")
                        continue
                
                    loss = criterion(output, y[:,:,1,:])
                
                state = new_state
                batch_losses.append(loss.item())
                acc = torch.mean(((output > 0) == y[:,:,1,:]).float()).item()
                batch_acc.append(acc)
                
                exact = ((output > 0) == y[:,:,1,:]).all(dim=-1).float().mean().item()
                batch_exact.append(exact)

        
            
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        epoch_exact = np.mean(batch_exact)
            
        if epoch_loss < best_val_loss:
                
            best_val_loss = epoch_loss
            best_state_dict = {k: v.clone().detach() for k, v in net.state_dict().items()}
            torch.save(best_state_dict,f'OverNet_Bitwise_{layers}M.pt')
            
        val_losses.append(epoch_loss)
        val_acc.append(epoch_acc)
        val_exact.append(epoch_exact)
        batch_losses = []
        batch_acc = []
        print(f'Validation Loss: {epoch_loss} Validation Accuracy: {epoch_acc} Exact Match: {100 * epoch_exact}%')
        torch.cuda.empty_cache()

    return net, train_losses, train_acc, val_losses, val_acc

if __name__ == '__main__':
    main()