import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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


class GRUHead(nn.Module):

    def __init__(self, d_model, input_size):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.act = nn.GELU()
        self.fc  = nn.Linear(d_model, input_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.act(out))


class OverNet(nn.Module):
    def __init__(self, input_size=48, seq_len=64, d_model=256, n_heads=8, n_layers=4,
                 dropout=0.1, d_state=128):
        super().__init__()

        self.d_model    = d_model
        self.seq_len    = seq_len
        self.input_size = input_size
        self.d_state    = d_state

        self.register_buffer('powers',     (2 ** torch.arange(input_size)).long())
        self.register_buffer('bit_shifts', torch.arange(input_size))

        self.page_projection    = OverFusion(input_size, d_model)
        self.ip_projection      = OverFusion(input_size, d_model)
        self.ip_page_cross_attn = IPPageCrossAttention(d_model, n_heads, dropout)

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

        self.head = GRUHead(d_model, input_size)

    def to_bits(self, integers):
        mask = (1 << self.input_size) - 1
        integers = integers & mask
        return ((integers.unsqueeze(-1) >> self.bit_shifts) & 1).float()

    def to_integers(self, bits):
        binary = (bits > 0).long()
        return (binary * self.powers).sum(dim=-1)

    def _make_pe(self, length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, state):
        """
        x:     (batch, seq_len, 2)       — raw [ip, page] integers
        state: (batch, d_state, d_model) — previous state matrix

        Returns:
            out:            (batch, seq_len, 48)       — predicted delta bit logits
            pred_page_ints: (batch, seq_len)            — reconstructed absolute pages
            new_state:      (batch, d_state, d_model)  — updated state matrix
        """
        batch = x.shape[0]

        page_ints = x[:, :, 1].long()
        ip_ints   = x[:, :, 0].long()

        page_deltas = torch.zeros_like(page_ints)
        page_deltas[:, 1:] = page_ints[:, 1:] - page_ints[:, :-1]

        ip_deltas = torch.zeros_like(ip_ints)
        ip_deltas[:, 1:] = ip_ints[:, 1:] - ip_ints[:, :-1]

        page_bits = self.to_bits(page_deltas)
        ip_bits   = self.to_bits(ip_deltas)

        page_h = self.page_projection(page_bits)
        ip_h   = self.ip_projection(ip_bits)
        fused  = self.ip_page_cross_attn(page_h, ip_h)
        fused  = fused + self.encoder_pe
        fused  = self.dropout(fused)

        memory    = self.encoder(fused)
        new_state = self.overstate(memory, state)

        # decoder sees encoder memory + state slots as context
        memory_and_state = torch.cat([memory, new_state], dim=1)

        queries = self.output_queries.expand(batch, -1, -1) + self.decoder_pe
        out     = self.decoder(queries, memory_and_state)
        out     = self.dropout(out)
        out     = self.head(out)

        pred_delta_ints = self.to_integers(out)
        last_page       = page_ints[:, -1:]
        pred_page_ints  = last_page + pred_delta_ints.cumsum(dim=1)

        return out, pred_page_ints, new_state


class OverFusion(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.GELU(),
            nn.Linear(input_size * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.proj(x)


class IPPageCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, page_h, ip_h):
        attn_out, _ = self.cross_attn(query=page_h, key=ip_h, value=ip_h)
        return self.norm(page_h + self.dropout(attn_out))