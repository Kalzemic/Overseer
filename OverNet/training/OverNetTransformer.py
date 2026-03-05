import torch
import torch.nn as nn
import numpy as np



class OverNetTransformer(nn.Module):
    def __init__(self, input_size=36, seq_len=64, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # encode input pages into d_model space
        self.projection = OverProjection(input_size, d_model)

        # sinusoidal PE for the input sequence
        pe = self._make_pe(seq_len * 2, d_model)
        self.register_buffer('encoder_pe', pe[:, :seq_len, :])   
        self.register_buffer('decoder_pe', pe[:, seq_len:, :])

        # encoder: processes the 64 input pages
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)


        self.register_buffer('state',torch.zeros(d_model))
        self.state_gate = nn.Linear(d_model* 2, d_model)
        # learned output queries — one per future step
        # each query learns to represent "I'm predicting step t+k"
        self.output_queries = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # decoder: cross-attends output queries into encoder memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)

        # final head: d_model -> 36 bit logits
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_size)
        )

    def _make_pe(self, length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch = x.shape[0]

        # encode input sequence
        raw = x 
        x = self.projection(x)
        x = x + self.encoder_pe
        x = self.dropout(x)
        memory = self.encoder(x)  # (batch, 64, d_model)

        # context = memory.mean(dim=1)
        # gate = torch.sigmoid(self.state_gate(torch.cat([self.state.detach().expand(batch, -1), context], dim=-1)))
        # new_state = (gate * self.state.detach() + (1 - gate) * context).mean(dim=0)
        # self.state.copy_(new_state.detach())
        
        
        # expand learned queries for the batch
        queries = self.output_queries.expand(batch, -1, -1)  + self.decoder_pe
        #+ self.state.unsqueeze(1) 

        # decode: queries cross-attend into encoded memory
        out = self.decoder(queries, memory)  # (batch, 64, d_model)
        out = self.dropout(out)

        out = self.head(out)


        return out + raw[:, -1:, :] # (batch, 64, 36)
    


class OverProjection(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        #self.raw_proj = nn.Linear(input_size, d_model)
        self.delta_proj = nn.Linear(input_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (B, L, F)
        delta = x[:, 1:, :] - x[:, :-1, :]
        delta = torch.cat([torch.zeros_like(delta[:, :1, :]), delta], dim=1)
        h =  self.delta_proj(delta)
        return self.gelu(h)
