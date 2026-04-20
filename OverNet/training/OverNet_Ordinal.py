import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class OverNet(nn.Module):
    def __init__(self, vocab_size=2**14, d_model=256, n_heads=8, dropout=.1, seq_len=64, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.ip_map = {}
        self.page_map = {}
        self._cache_0 = None
        self._cache_1 = None

        self.ip_embedding = nn.Embedding(vocab_size, d_model)
        self.page_embedding = nn.Embedding(vocab_size, d_model)
        self.cross_attn = IPPageCrossAttention(d_model, n_heads, dropout)
        
        pe = self._make_pe(seq_len * 2, d_model)
        self.register_buffer('encoder_pe', pe[:, :seq_len, :])
        self.register_buffer('decoder_pe', pe[:, seq_len:, :])

        self.overstate = OverState(d_model, n_heads=4, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_queries = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(d_model, self.vocab_size)


    def reset_vocab(self):
        self.ip_map.clear()
        self.page_map.clear()
        self._cache_0 = None
        self._cache_1 = None

    def forward(self, x, state):
        batch = x.shape[0]

        # x = self.lookup(x)
        ip_ints   = x[:, :, 0]
        page_ints = x[:, :, 1]

        ip_h   = self.ip_embedding(ip_ints)
        page_h = self.page_embedding(page_ints)

        fused  = self.cross_attn(page_h, ip_h)
        fused  = fused + self.encoder_pe
        fused  = self.dropout(fused)

        memory    = self.encoder(fused)
        new_state = self.overstate(memory, state)

        memory = torch.cat([memory, new_state], dim=1)

        queries = self.output_queries.expand(batch, -1, -1) + self.decoder_pe
        out     = self.decoder(queries, memory)
        out     = self.dropout(out)
        out     = self.head(out)

        return out, new_state

    def _make_pe(self, length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe


class IPPageCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, page_h, ip_h):
        attn_out, _ = self.cross_attn(query=page_h, key=ip_h, value=ip_h)
        return self.norm(page_h + self.dropout(attn_out))


class OverState(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory, state):
        """
        memory: (batch, seq_len, d_model) — full encoder output
        state:  (batch, d_state, d_model) — previous state matrix

        Returns new_state: (batch, d_state, d_model)
        """
        attended, _ = self.cross_attn(query=state, key=memory, value=memory)
        return self.norm(state + attended)