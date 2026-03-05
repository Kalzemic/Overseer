import torch 
import torch.nn as nn 
import torch.nn.functional as F


from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm





class OverNetMamba(nn.Module):
    def __init__(self, input_size=36, d_model=256, d_state=256, conv_size=4, expand=2, n_layers=4,):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers

        self.projection = OverProjection(input_size, d_model)


        self.encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, dilation=1, padding='same'),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, dilation=2, padding='same'),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, dilation=4, padding='same'),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, dilation=8, padding='same'),
            nn.GELU(),
        )
        # self.encoder = nn.MultiheadAttention(
        #     embed_dim=d_model,
        #     num_heads=n_heads,
        #     dropout=0.1,
        #     batch_first=True
        # )
        # self.encoder_norm = RMSNorm(d_model)

        self.context_proj = nn.Linear(d_model, d_model)
        #self.downsample = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)

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
                'context_inject': nn.Linear(d_model, d_model),
                'norm2': RMSNorm(d_model)
            })
            for _ in range(n_layers)
        ])
        

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_size)
        )
    

    
    def forward(self, x, state=None):
        x = self.projection(x)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        #x = self.downsample(x)
        x = x.transpose(1, 2)

        x = F.dropout(x, p=0.3, training=self.training)

        context = x.mean(dim=1)  
        context = self.context_proj(context)
        
        for layer in self.layers:
            x_conditioned = x + layer['context_inject'](context).unsqueeze(1)
            y = layer['mamba'](layer['norm1'](x_conditioned))
            x = x + y
            x = x + layer['ffn'](layer['norm2'](x))

        x = F.dropout(x, p=0.3, training=self.training)
        out = self.decoder(x)


        return out


       
class OverProjection(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.raw_proj = nn.Linear(input_size, d_model)
        self.delta_proj = nn.Linear(input_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (B, L, F)
        delta = x[:, 1:, :] - x[:, :-1, :]
        delta = torch.cat([torch.zeros_like(delta[:, :1, :]), delta], dim=1)
        h = self.raw_proj(x) + self.delta_proj(delta)
        return self.gelu(h)
