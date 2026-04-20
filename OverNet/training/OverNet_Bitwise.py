import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class OverState(nn.Module):
    def __init__(self, d_model, n_phases=16, beta=0.25):
        super().__init__()
        self.d_model = d_model
        self.n_phases = n_phases
        self.beta = beta

        self.gru = nn.GRUCell(input_size=d_model, hidden_size=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.codebook = nn.Embedding(n_phases, d_model)

        nn.init.orthogonal_(self.codebook.weight)

    def forward(self, context, state):
        continuous = self.gru(context, state)
        continuous = self.norm(continuous)

        dists = torch.cdist(
            continuous.unsqueeze(1),
            self.codebook.weight.unsqueeze(0)
        ).squeeze(1)
        phase_idx = dists.argmin(dim=-1)
        quantized = self.codebook(phase_idx)

        quantized_st = continuous + (quantized - continuous).detach()

        return quantized_st, continuous, phase_idx

    def loss(self, continuous, phase_idx):
        quantized = self.codebook(phase_idx)
        commitment_loss = F.mse_loss(continuous, quantized.detach())
        codebook_loss   = F.mse_loss(continuous.detach(), quantized)
        return codebook_loss + self.beta * commitment_loss


class BitHead(nn.Module):
    def __init__(self, d_model, n_bits=48):
        super().__init__()
        self.n_bits = n_bits
        self.d_bit = d_model // 2

        # project decoder output to per-bit feature space
        self.proj = nn.Linear(d_model, n_bits * self.d_bit)

        # 48 embedding tables of size 2 — one per bit position
        # shape: (n_bits, 2, d_bit)
        self.bit_embeddings = nn.Parameter(torch.randn(n_bits, 2, self.d_bit) * 0.02)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape

        # project to per-bit feature space
        x = self.proj(x)                                      # (batch, seq_len, n_bits * d_bit)
        x = x.view(batch, seq_len, self.n_bits, self.d_bit)   # (batch, seq_len, n_bits, d_bit)

        # similarity to "0" and "1" embedding for each bit position
        e0 = self.bit_embeddings[:, 0, :]  # (n_bits, d_bit) — "0" prototype per bit
        e1 = self.bit_embeddings[:, 1, :]  # (n_bits, d_bit) — "1" prototype per bit

        sim0 = (x * e0).sum(-1)  # (batch, seq_len, n_bits)
        sim1 = (x * e1).sum(-1)  # (batch, seq_len, n_bits)

        # logit is difference — positive means closer to "1"
        return sim1 - sim0       # (batch, seq_len, n_bits)


class OverNet(nn.Module):
    def __init__(self, input_size=48, seq_len=64, d_model=256, n_heads=8, n_layers=4,
                 dropout=0.1, n_phases=16):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # cross-attention: page queries into IP context
        self.ip_page_cross_attn = IPPageCrossAttention(d_model, n_heads, dropout)

        # sinusoidal PE for sequence positions
        pe = self._make_pe(seq_len * 2, d_model)
        self.register_buffer('encoder_pe', pe[:, :seq_len, :])
        self.register_buffer('decoder_pe', pe[:, seq_len:, :])

        # precompute sine encoding frequencies for addresses
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # OverState: VQ phase tracker with GRU memory
        self.register_buffer('internal_state', torch.zeros(1, d_model))
        self.overstate = OverState(d_model, n_phases=n_phases)

        # project quantized state into decoder query space
        self.state_proj = nn.Linear(d_model, d_model)

        # learned output queries — one per future step
        self.output_queries = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # decoder
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

        # BitHead: per-bit embedding classification
        self.head = BitHead(d_model, n_bits=input_size)

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

    def to_sine(self, x):
        # x: (B, L) int64
        # returns: (B, L, d_model) float32
        x_f = x.float().unsqueeze(-1)          # (B, L, 1)
        enc = torch.zeros(*x.shape[:2], self.d_model, device=x.device)
        enc[..., 0::2] = torch.sin(x_f * self.div_term)
        enc[..., 1::2] = torch.cos(x_f * self.div_term)
        return enc                              # (B, L, d_model)

    def forward(self, x, state=None):
        if state is None:
            current_state = self.internal_state
        else:
            current_state = state

        # x: (batch, seq_len, 2) uint64 — dim 2: 0=IP, 1=page
        batch = x.shape[0]

        pages_raw = x[:, :, 1]  # (batch, seq_len) uint64
        ips_raw   = x[:, :, 0]  # (batch, seq_len) uint64

        # page deltas in integer space — numerically meaningful
        pages_int   = pages_raw.to(torch.int64)
        page_deltas = pages_int - torch.roll(pages_int, 1, dims=1)
        page_deltas[:, 0] = 0

        # sine encode — respects numerical proximity
        page_h = self.to_sine(pages_int) + self.to_sine(page_deltas)  # (batch, seq_len, d_model)
        ip_h   = self.to_sine(ips_raw.to(torch.int64))                # (batch, seq_len, d_model)

        # enrich page representation with IP context via cross-attention
        fused = self.ip_page_cross_attn(page_h, ip_h)  # (batch, seq_len, d_model)

        # encode the IP-conditioned page sequence
        fused = fused + self.encoder_pe
        fused = self.dropout(fused)
        memory = self.encoder(fused)  # (batch, seq_len, d_model)

        # OverState: update VQ phase tracker
        context = memory.mean(dim=1)
        new_state, continuous, phase_idx = self.overstate(context, current_state)

        # inject quantized phase state into decoder queries
        state_bias = self.state_proj(new_state).unsqueeze(1)
        queries = self.output_queries.expand(batch, -1, -1) + self.decoder_pe + state_bias

        out = self.decoder(queries, memory)   # (batch, seq_len, d_model)
        out = self.dropout(out)
        out = self.head(out)                  # (batch, seq_len, 48)

        if state is None:
            self.internal_state = new_state.detach()

        return out, new_state, continuous, phase_idx


class IPPageCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, page_h, ip_h):
        # page_h: (B, L, d_model) — queries
        # ip_h:   (B, L, d_model) — keys and values
        L = page_h.shape[1]
        causal_mask = torch.triu(
            torch.ones(L, L, device=page_h.device), diagonal=1
        ).bool()

        attn_out, _ = self.cross_attn(
            query=page_h,
            key=ip_h,
            value=ip_h,
            attn_mask=causal_mask
        )
        return self.norm(page_h + self.dropout(attn_out))