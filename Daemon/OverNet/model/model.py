import torch
from OverNet import OverNet

#INPUT_SIZE = 2
SEQ_LEN = 64
D_MODEL = 128   
D_STATE = 128
VOCAB_SIZE=2**9
net = OverNet(n_layers=1, d_model=D_MODEL, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)


net.load_state_dict(torch.load("OverNet.pt", map_location="cpu"))


net.eval()

dummy_x = torch.randint(0, VOCAB_SIZE , (1, SEQ_LEN))
dummy_state = torch.randn((1, D_STATE, D_MODEL))


print(net(dummy_x, dummy_x, dummy_state))


torch.onnx.export(
    net,
    (dummy_x, dummy_x, dummy_state),
    "OverNet.onnx",
    opset_version=17,
    input_names=["ip", "pages", "state"],
    output_names=["pred_pages", "new_state"]
)

print("Exported to OverNet.onnx")