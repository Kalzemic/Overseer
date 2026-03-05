
import torch 
import torch.nn as nn 
import torch.nn.functional as F


from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm





class OverNet(nn.Module):
    def __init__(self, input_size=36, d_model=256, d_state=256, conv_size=4, expand=2, n_layers=4):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers

        self.projection = OverProjection(input_size, d_model)


        self.encoder = nn.Sequential(
            CausalConv1d(d_model, d_model, kernel_size=5, dilation=1),
            nn.GELU(),
            CausalConv1d(d_model, d_model, kernel_size=5, dilation=2),
            nn.GELU(),
            CausalConv1d(d_model, d_model, kernel_size=5, dilation=4),
            nn.GELU(),
            CausalConv1d(d_model, d_model, kernel_size=5, dilation=8),
            nn.GELU(),
        )


        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba(d_model=d_model, d_state=d_state, d_conv=conv_size, expand=expand),
                'norm1': RMSNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU()
                ),
                'norm2': RMSNorm(d_model)
            })
            for _ in range(n_layers)
        ])
        

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_size)
        )
    
    def allocate_inference_cache(self, batch_size=1, device='cuda'):
        state = []
        for layer in self.layers:
            conv_state, ssm_state = layer['mamba'].allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=1,
                device=device
            )
            state.append((conv_state, ssm_state))
        return state
    
    def forward(self, x, state=None):
        x = self.projection(x)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        x = F.dropout(x, p=0.1, training=self.training)

        if  state is not None:

            for i, layer in enumerate(self.layers):
                conv_state, ssm_state = state[i]
                
                normed = layer['norm1'](x)
                y, conv_state, ssm_state = layer['mamba'].step(normed, conv_state, ssm_state)
                
                
                state[i] = (conv_state, ssm_state)
                
                x = x + y
                x = x + layer['ffn'](layer['norm2'](x))
        else:
        
            for layer in self.layers:
                y = layer['mamba'](layer['norm1'](x))
                x = x + y
                x = x + layer['ffn'](layer['norm2'](x))

        x = F.dropout(x, p=0.1, training=self.training)
        out = self.decoder(x)

        if not self.training:
            return out, state
        else:
            return out


       






class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):

        x = self.conv(x)
        return x[:, :, :-self.padding]
    

class OverProjection(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.raw_proj = nn.Linear(input_size, d_model)
        self.delta_proj = nn.Linear(input_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        delta = F.pad(x[:, 1:, :] - x[:, :-1, :], (0, 0, 1, 0))
        h = self.raw_proj(x) + self.delta_proj(delta)
        return self.gelu(h)