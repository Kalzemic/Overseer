import torch 
import torch.nn as nn
import torch.nn.functional as F
from OverNet import OverNet
INPUT_SIZE = 2
SEQ_LEN = 64
D_MODEL = 256
D_STATE = 256
VOCAB_SIZE = 2**14



net = OverNet(n_layers=2)
net.eval()

dummy_x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN, 2))  # randint gives long by default
dummy_state = torch.randn((1,D_STATE,D_MODEL))

print(net(dummy_x,dummy_state))

torch.onnx.export(
    net,
    (dummy_x, dummy_state),
    "OverNet.onnx",
    opset_version=17,
    input_names=["x", "state"],
    output_names=["y", 'new_state']
)
print("Exported to over_net.onnx")