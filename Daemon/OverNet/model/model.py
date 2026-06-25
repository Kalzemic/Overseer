import torch
from OverNet import OverNet

#INPUT_SIZE = 2
SEQ_LEN = 64
D_MODEL = 256   
VOCAB_SIZE=2**7
net = OverNet(n_layers=1, d_model=D_MODEL, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)


net.load_state_dict(torch.load("OverNet.pt", map_location="cpu"))


net.eval()

dummy_ip = torch.randint(0, VOCAB_SIZE , (1, SEQ_LEN))
dummy_pages = torch.randint(0, VOCAB_SIZE , (1, SEQ_LEN))


print(net(dummy_ip, dummy_pages))


torch.onnx.export(
    net,
    (dummy_ip, dummy_pages),
    "OverNet.onnx",
    opset_version=18,
    input_names=["ip", "pages"],
    output_names=["pred_pages"]
)

print("Exported to OverNet.onnx")