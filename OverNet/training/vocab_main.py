import torch
import torch.nn as nn
import numpy as np 
from Data import MemDataset

import matplotlib.pyplot as plt
from tqdm import tqdm

from OverNetVocab import OverNetVocab, prepare_input, page_to_vocab_idx, bits_to_uint64
from torch.optim.lr_scheduler import OneCycleLR
# HPL_TRAIN = './hpl/hpl_data_val.bin'
# HPL_VAL = './hpl/hpl_data_test.bin'

# BERT_TRAIN = './bert/bert_data_val.bin'
# BERT_VAL = './bert/bert_data_test.bin'

# POLYBENCH_TRAIN = './polybench_gemm/gemm_large_val.bin'
# POLYBENCH_VAL = './polybench_gemm/gemm_large_test.bin'
MLPERF_TRAIN ={f'MLPERF{j}_TRAIN_0{i}':f'./mlperf/mlperf_train_0{i}.bin' for i in range(10) for j in ['',2]}

MLPERF_VAL = './mlperf/mlperf_val.bin'
MLPERF2_VAL = './mlperf2/mlperf2_val.bin'

# GEMM2_TRAIN_08 = './gemm2/gemm2_train_08.bin'
# GEMM2_VAL = './gemm2/gemm2_val.bin'


# BERT_2_TRAIN_05 = './bert_2/bert_2_train_05.bin'
# BERT_2_VAL = './bert_2/bert_2_val.bin'


lr = 1e-4
layers = 2
epochs = 50
max_grad_norm= 5
l2lambda = 1e-5
d_state= 256
seq_length = 64
vocab_size = 2000
dropout = 0.3
STEPS_PER_EPOCH = 254890

total_steps = epochs * STEPS_PER_EPOCH
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

def main():
    train_data = MemDataset([ MLPERF_TRAIN[f'MLPERF{j}_TRAIN_0{i}'] for i in range(10) for j in ['',2]], seq_len=seq_length)
    val_data = MemDataset([ MLPERF_VAL, MLPERF2_VAL], seq_len=seq_length)
    overnet = OverNetVocab(n_layers=layers, d_model=d_state,dropout=dropout,vocab_size=vocab_size).to(device)
    #print(overnet.causal_mask[:4,:4])
    criterion = nn.CrossEntropyLoss()
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
    #val_exact = []
    best_val_loss = 10
    best_state_dict = None
    

    for epoch in range(epochs):
        
        print(f'Epoch {epoch + 1} Running ')
        net.train()

        batch_losses = []
        batch_acc = []
        
        for i,(x,y) in enumerate(tqdm(train)):

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x =prepare_input(x, vocab_size=vocab_size).to(device) 
                output = net(x)
                y = prepare_input(y, vocab_size=vocab_size).to(device) 
                loss = criterion(output.reshape(-1, vocab_size),y.reshape(-1))
            
            batch_losses.append(loss.item())
            preds = output.argmax(dim=-1)
            acc = (preds == y).float().mean()
            batch_acc.append(acc.item())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            # if i % 100 == 0: 
            #     print(f"Grad norm: {grad_norm.item():.4f} | Clipped: {grad_norm.item() > max_grad_norm}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        batch_losses = []
        batch_acc = []
        print(f'Train Loss: {epoch_loss} Train Accuracy: {epoch_acc}')


        net.eval()

        #batch_exact = []
        with torch.no_grad(),torch.autocast(device_type='cuda', dtype=torch.float16):
            
            
            for i, (x, y) in enumerate(tqdm(val)):


                if not torch.isfinite(x).all():
                    print(f"Non-finite in x at step {i}")
                    continue

                x = prepare_input(x, vocab_size=vocab_size).to(device)
                output = net(x)
                
                if not torch.isfinite(output).all():
                    print(f"Non-finite in output at step {i}")
                    continue
                y = prepare_input(y, vocab_size=vocab_size).to(device)

                loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
                batch_losses.append(loss.item())
                preds = output.argmax(dim=-1)
                acc = (preds == y).float().mean()
                batch_acc.append(acc.item())
                
                #exact = ((output > 0) == y).all(dim=-1).float().mean().item()
                #batch_exact.append(exact)

        
            
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        #epoch_exact = np.mean(batch_exact)
            
        if epoch_loss < best_val_loss:
                
            best_val_loss = epoch_loss
            best_state_dict = {k: v.clone().detach() for k, v in net.state_dict().items()}
            torch.save(best_state_dict,f'OverNet_Vocab_{layers}M.pt')
            
        val_losses.append(epoch_loss)
        val_acc.append(epoch_acc)
        #val_exact.append(epoch_exact)
        batch_losses = []
        batch_acc = []
        print(f'Validation Loss: {epoch_loss} Validation Accuracy: {epoch_acc} ')
        torch.cuda.empty_cache()

    return net, train_losses, train_acc, val_losses, val_acc

if __name__ == '__main__':
    main()