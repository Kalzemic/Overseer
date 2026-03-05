import torch
from OverNet import OverNet

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create OverNet
net = OverNet(
    input_size=36,
    d_model=64,
    d_state=32,
    conv_size=4,
    expand=2,
    n_layers=2
).to(device)

net.eval()

# Allocate state
state = net.allocate_inference_cache(batch_size=1, device=device)

# Single timestep input: (B, L=1, input_size=36)
x_t = torch.randn(1, 1, 36, device=device)

with torch.no_grad():
    # First step
    y1, state = net(x_t, state=state)
    
    # Second step (same input, but state carries over)
    y2, state = net(x_t, state=state)

print("y1:", y1)
print("y2:", y2)
print("Outputs identical?", torch.allclose(y1, y2))
print("State carryover working:", not torch.allclose(y1, y2))