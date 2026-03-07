import torch
import torch.nn as nn
import numpy as np




class OverNet(nn.Module):
    def __init__(self, input_size=48, seq_len=64, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # independent delta projections for each stream
        self.page_projection = OverProjection(input_size, d_model)
        self.ip_projection   = OverProjection(input_size, d_model)

        # cross-attention: page queries into IP context
        self.ip_page_cross_attn = IPPageCrossAttention(d_model, n_heads, dropout)

        # sinusoidal PE
        pe = self._make_pe(seq_len * 2, d_model)
        self.register_buffer('encoder_pe', pe[:, :seq_len, :])
        self.register_buffer('decoder_pe', pe[:, seq_len:, :])

        # encoder: processes IP-conditioned page sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)


        self.register_buffer('internal_state', torch.zeros(1, self.d_model))
        # GRU cell for program-level state — updated once per window
        self.state_cell = nn.GRUCell(input_size=d_model, hidden_size=d_model)
        self.state_norm = nn.LayerNorm(d_model)

        # project state into query space for decoder conditioning
        self.state_proj = nn.Linear(d_model, d_model)


        # learned output queries — one per future step
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

        # output head: predicts future PAGE bits only
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_size)  # 48 page bit logits
        )


    def reset_state(self):
        self.internal_state.zero_()
    
    
    def _make_pe(self, length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, state=None):
        
        
        if state is None:
            current_state = self.internal_state
        else:
            current_state = state
        # x: (batch, seq_len, 2, 48)
        #     dim 2: 0 = IP, 1 = page
        batch = x.shape[0]
        
        pages = x[:, :, 1, :]  # (batch, seq_len, 48)
        ips   = x[:, :, 0, :]  # (batch, seq_len, 48)

        # project each stream independently via delta encoding
        page_h = self.page_projection(pages)  # (batch, seq_len, d_model)
        ip_h   = self.ip_projection(ips)      # (batch, seq_len, d_model)

        # enrich page representation with IP context via cross-attention
        fused = self.ip_page_cross_attn(page_h, ip_h)  # (batch, seq_len, d_model)

        # encode the IP-conditioned page sequence
        fused = fused + self.encoder_pe
        fused = self.dropout(fused)
        memory = self.encoder(fused)  # (batch, seq_len, d_model)

        context = memory.mean(dim=1)                          # (batch, d_model)
        new_state = self.state_cell(context, current_state)           # (batch, d_model)
        new_state = self.state_norm(new_state)

        # inject state into decoder queries — "predict given where we are in the program"
        state_bias = self.state_proj(new_state).unsqueeze(1)  # (batch, 1, d_model)
        queries = self.output_queries.expand(batch, -1, -1) + self.decoder_pe + state_bias

        out = self.decoder(queries, memory)  # (batch, seq_len, d_model)
        out = self.dropout(out)

        out = self.head(out)  # (batch, seq_len, 48)

        if state is None:
            self.internal_state = new_state.detach()
        # residual anchor: bias toward last seen page
        return out + pages[:, -1:, :], new_state  # (batch, seq_len, 48)
    



class OverProjection(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.delta_proj = nn.Linear(input_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (B, L, F)
        delta = x[:, 1:, :] - x[:, :-1, :]
        delta = torch.cat([torch.zeros_like(delta[:, :1, :]), delta], dim=1)
        return self.gelu(self.delta_proj(delta))


class IPPageCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, page_h, ip_h):
        # page_h: (B, L, d_model) — queries
        # ip_h:   (B, L, d_model) — keys and values
        # "given the IP context at each timestep, enrich the page representation"
        attn_out, _ = self.cross_attn(query=page_h, key=ip_h, value=ip_h)
        return self.norm(page_h + self.dropout(attn_out))