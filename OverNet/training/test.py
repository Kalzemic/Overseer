import torch
import torch.nn as nn
from Data import MemDataset
from OverNet import OverNet
from tqdm import tqdm
from OverNetTransformer import OverNetTransformer
HPL_VAL = './hpl/hpl_data_test.bin'

GEMM2_VAL = './gemm2/gemm2_val.bin'


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

layers = 1
d_state = 256

def evaluate(net, dataset, name):
    net.eval()
    total_exact = 0
    total_pages = 0
    total_correct_bits = 0
    total_bits = 0

    with torch.no_grad():
        for x, y in tqdm(dataset, desc=name):
            output = net(x)
            total_exact += ((output > 0) == y).all(dim=-1).sum().item()
            total_pages += y.shape[0] * y.shape[1]
            total_correct_bits += ((output > 0) == y).sum().item()
            total_bits += y.numel()

    exact = 100 * total_exact / total_pages
    bit_acc = 100 * total_correct_bits / total_bits
    print(f'{name} -> Exact Match: {exact:.4f}% | Bit Accuracy: {bit_acc:.4f}%')

def main():
    overnet = OverNetTransformer(n_layers=layers, d_model=d_state).to(device)
    overnet.load_state_dict(torch.load(f'OverNet_{layers}M.pt', map_location=device))

    hpl_val = MemDataset([HPL_VAL])
    dlrm_val = MemDataset([GEMM2_VAL])

    evaluate(overnet, hpl_val, 'HPL')
    evaluate(overnet, dlrm_val, 'GEMM')

if __name__ == '__main__':
    main()