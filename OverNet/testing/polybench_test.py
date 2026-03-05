import torch 
import torch.nn as nn
import numpy as np 
import pandas as pd
from OverNet_CarryOver import OverNet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BITS = 36
seq_length = 64

def convert_hex(x):
    try:
        return int(str(x).strip(), 16)
    except (ValueError, TypeError):
        return np.int64(-1)


def main():
    
    overnet = OverNet(input_size=BITS, n_layers=2)
    overnet.load_state_dict(torch.load("OverNet.pt", map_location=device, weights_only=True))
    overnet = overnet.to(device)
    overnet.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    polybench_gemm_small = pd.read_csv(
        "../Data/gemm_small.out",
        sep=r"\s+",
        names=["ip", "op", "ea"],
        engine="python",
    )

    data = polybench_gemm_small.drop(['op','ip'], axis=1)

    data['ea'] = (
        data['ea']
        .map(convert_hex)          
        .astype(np.int64)          
    )

    ea = data['ea'].to_numpy(np.int64)

    page_id = (ea >> 12).astype(np.int64)

    mask = np.ones(len(page_id), dtype=bool)
    mask[1:] = page_id[1:] != page_id[:-1]
    page_id = page_id[mask]

    ea_bin = ((page_id[:, None] >> np.arange(BITS)) & 1).astype(np.float32)

    features = ea_bin.astype(np.float32)
    data_tensor = torch.from_numpy(features).unsqueeze(0)

    test_results = []
    
    # Allocate inference cache for carryover state
    state = overnet.allocate_inference_cache(batch_size=1, device=device)

    with torch.no_grad():
        for i in range(0, data_tensor.shape[1] - 2*seq_length, seq_length): 
            x = data_tensor[:, i:i+seq_length, :].to(device)
            y_true = data_tensor[:, i+seq_length:i+2*seq_length, :].to(device)
            
            # Process token-by-token to use .step() with state carryover
            logits_list = []
            for t in range(x.shape[1]):
                x_t = x[:, t:t+1, :]  # shape: [batch, 1, features]
                logit_t, state = overnet(x_t, state=state)
                logits_list.append(logit_t)
            
            logits = torch.cat(logits_list, dim=1)  # shape: [batch, seq_length, BITS]
            y_pred = (torch.sigmoid(logits) > 0.5).float()  
        
            bit_accuracy = (y_pred == y_true).float().mean(dim=(0,1)) 
            overall_accuracy = (y_pred == y_true).float().mean()
        
            loss = criterion(logits, y_true)
        
            test_results.append({
                'loss': loss.item(),
                'accuracy': overall_accuracy.item(),
                'exact_match': (y_pred == y_true).all(dim=2).float().mean().item()  
            })

    print("=" * 60)
    print("TEST SET EVALUATION (WITH STATE CARRYOVER)")
    print("=" * 60)
    avg_loss = np.mean([r['loss'] for r in test_results])
    avg_acc = np.mean([r['accuracy'] for r in test_results])
    avg_exact = np.mean([r['exact_match'] for r in test_results])

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Bit-level Accuracy: {avg_acc:.2%}")
    print(f"Exact Page Match Rate: {avg_exact:.2%}")
    print()

    # Per-bit analysis on first chunk (fresh state)
    state_fresh = overnet.allocate_inference_cache(batch_size=1, device=device)
    
    with torch.no_grad():
        x_sample = data_tensor[:, 0:seq_length, :].to(device)
        y_sample = data_tensor[:, seq_length:2*seq_length, :].to(device)
        
        logits_list = []
        for t in range(x_sample.shape[1]):
            x_t = x_sample[:, t:t+1, :]
            logit_t, state_fresh = overnet(x_t, state=state_fresh)
            logits_list.append(logit_t)
        
        logits = torch.cat(logits_list, dim=1)
        y_pred = (torch.sigmoid(logits) > 0.5).float()
    
        per_bit_acc = (y_pred == y_sample).float().mean(dim=(0,1))
    
        print("Per-bit accuracy:")
        for i in range(len(per_bit_acc)):
            print(f"  Bit {i:2d}: {per_bit_acc[i]:.2%}")
    
        print()
        print("Baseline comparisons:")
    
        always_zero_acc = (y_sample == 0).float().mean()
        always_one_acc = (y_sample == 1).float().mean()
    
        print(f"  Always predict 0: {always_zero_acc:.2%}")
        print(f"  Always predict 1: {always_one_acc:.2%}")
        print(f"  Random (50/50): ~50%")


if __name__ == '__main__':
    main()