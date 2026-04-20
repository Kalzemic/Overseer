import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def split_chunks(x):
    """
    Split a 48-bit address into 4 chunks of 12 bits each.
    Args:
        x: (batch, seq_len) long
    Returns:
        4 tensors of (batch, seq_len) long, each in [0, 4096)
    """
    c0 = (x >>  0) & 0xFFF
    c1 = (x >> 12) & 0xFFF
    c2 = (x >> 24) & 0xFFF
    c3 = (x >> 36) & 0xFFF
    return c0, c1, c2, c3


class ChunkedEmbedding(nn.Module):
    """
    Embeds a 48-bit address by splitting into 4 chunks of 12 bits,
    embedding each chunk separately, concatenating, and projecting to d_model.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        chunk_dim = d_model // 4
        self.embed0 = nn.Embedding(4096, chunk_dim)
        self.embed1 = nn.Embedding(4096, chunk_dim)
        self.embed2 = nn.Embedding(4096, chunk_dim)
        self.embed3 = nn.Embedding(4096, chunk_dim)
        # self.proj   = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) long — raw 48-bit address
        Returns:
            (batch, seq_len, d_model)
        """
        c0, c1, c2, c3 = split_chunks(x)
        e = torch.cat([
            self.embed0(c0),
            self.embed1(c1),
            self.embed2(c2),
            self.embed3(c3),
        ], dim=-1)  # (batch, seq_len, d_model)
        return e


class OverNet(nn.Module):
    def __init__(self, seq_len=64, d_model=256, n_heads=8, n_layers=4,
                 dropout=0.1, n_phases=16):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # chunked embeddings for page and IP
        self.page_embedding = ChunkedEmbedding(d_model, dropout)
        self.ip_embedding   = ChunkedEmbedding(d_model, dropout)

        # cross-attention: IP context into page representation
        self.ip_page_cross_attn = IPPageCrossAttention(d_model, n_heads, dropout)

        # sinusoidal PE
        pe = self._make_pe(seq_len * 2, d_model)
        self.register_buffer('encoder_pe', pe[:, :seq_len, :])
        self.register_buffer('decoder_pe', pe[:, seq_len:, :])

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

        # OverState
        self.overstate  = OverState(d_model, n_phases=n_phases)
        self.state_proj = nn.Linear(d_model, d_model)

        # learned output queries
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

        # 4 output heads — one per chunk, each classifying over 4096
        self.head0 = nn.Linear(d_model, 4096)
        self.head1 = nn.Linear(d_model, 4096)
        self.head2 = nn.Linear(d_model, 4096)
        self.head3 = nn.Linear(d_model, 4096)

    def _make_pe(self, length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, pages, ips, state):
        """
        Args:
            pages: (batch, seq_len) long — raw page addresses
            ips:   (batch, seq_len) long — raw ip addresses
            state: (batch, d_model) float — OverState hidden state

        Returns:
            logits: tuple of 4 tensors (batch, seq_len, 4096) — one per chunk
            new_state:  (batch, d_model)
            continuous: (batch, d_model)
            phase_idx:  (batch,)
        """
        batch = pages.shape[0]

        # embed via chunked embeddings
        page_h = self.page_embedding(pages)  # (batch, seq_len, d_model)
        ip_h   = self.ip_embedding(ips)      # (batch, seq_len, d_model)

        # fuse IP context into page representation
        fused = self.ip_page_cross_attn(page_h, ip_h)

        # encode
        fused  = fused + self.encoder_pe
        fused  = self.dropout(fused)
        memory = self.encoder(fused)

        # OverState
        context    = memory.mean(dim=1)
        new_state, continuous, phase_idx = self.overstate(context, state)
        state_bias = self.state_proj(new_state).unsqueeze(1)

        # decode
        queries = self.output_queries.expand(batch, -1, -1) + self.decoder_pe + state_bias
        out     = self.decoder(queries, memory)
        out     = self.dropout(out)

        # 4 separate logits — one per chunk
        logits = (
            self.head0(out),  # (batch, seq_len, 4096)
            self.head1(out),
            self.head2(out),
            self.head3(out),
        )

        return logits, new_state, continuous, phase_idx


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
    def __init__(self, d_model, n_phases=16, beta=0.25):
        super().__init__()
        self.d_model  = d_model
        self.n_phases = n_phases
        self.beta     = beta

        self.gru      = nn.GRUCell(input_size=d_model, hidden_size=d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.codebook = nn.Embedding(n_phases, d_model)

        nn.init.orthogonal_(self.codebook.weight)

    def forward(self, context, state):
        continuous = self.gru(context, state)
        continuous = self.norm(continuous)

        dists     = torch.cdist(
            continuous.unsqueeze(1),
            self.codebook.weight.unsqueeze(0)
        ).squeeze(1)
        phase_idx = dists.argmin(dim=-1)
        quantized = self.codebook(phase_idx)

        quantized_st = continuous + (quantized - continuous).detach()

        return quantized_st, continuous, phase_idx

    def loss(self, continuous, phase_idx):
        quantized        = self.codebook(phase_idx)
        commitment_loss  = F.mse_loss(continuous, quantized.detach())
        codebook_loss    = F.mse_loss(continuous.detach(), quantized)
        return codebook_loss + self.beta * commitment_loss