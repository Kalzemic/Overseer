import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CachedAttention(nn.Module):
    """State-dict compatible with nn.MultiheadAttention.
    Supports three call modes: self-attn step, cross-attn step, and cross K/V precompute."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Names MUST match nn.MultiheadAttention for state_dict loading
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x):  # (B, T, D) -> (B, H, T, Dh)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):  # (B, H, T, Dh) -> (B, T, D)
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    # --- self-attention step (causal handled by cache structure) ---
    def self_step(self, x, k_cache, v_cache):
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)

        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = self.out_proj(self._merge(out))
        return out, k, v

    # --- cross-attention step (K/V already cached from encoder memory) ---
    def cross_step(self, x, k_cache, v_cache):
        W_q = self.in_proj_weight[: self.d_model]
        b_q = self.in_proj_bias[: self.d_model]
        q = self._split(F.linear(x, W_q, b_q))

        out = F.scaled_dot_product_attention(q, k_cache, v_cache)
        out = self.out_proj(self._merge(out))
        return out

    # --- precompute cross K/V once from encoder memory ---
    def encode_kv(self, memory):
        W_kv = self.in_proj_weight[self.d_model :]
        b_kv = self.in_proj_bias[self.d_model :]
        kv = F.linear(memory, W_kv, b_kv)
        k, v = kv.chunk(2, dim=-1)
        return self._split(k), self._split(v)


class CachedDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward):
        super().__init__()
        self.self_attn = CachedAttention(d_model, n_heads)
        self.multihead_attn = CachedAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def step(self, x, cross_k, cross_v, self_k, self_v):
        # pre-norm self-attn
        sa, sk, sv = self.self_attn.self_step(self.norm1(x), self_k, self_v)
        x = x + sa
        # pre-norm cross-attn
        x = x + self.multihead_attn.cross_step(self.norm2(x), cross_k, cross_v)
        # pre-norm ffn
        x = x + self.linear2(F.relu(self.linear1(self.norm3(x))))
        return x, sk, sv


class CachedDecoder(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [CachedDecoderLayer(d_model, n_heads, dim_feedforward) for _ in range(n_layers)]
        )

    def precompute_cross_kv(self, memory):
        return [layer.multihead_attn.encode_kv(memory) for layer in self.layers]

    def step(self, x, cross_kvs, self_kvs):
        new_self_kvs = []
        for layer, (ck, cv), (sk_c, sv_c) in zip(self.layers, cross_kvs, self_kvs):
            x, sk, sv = layer.step(x, ck, cv, sk_c, sv_c)
            new_self_kvs.append((sk, sv))
        return x, new_self_kvs


class IPPageCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, page_h, ip_h):
        attn_out, _ = self.cross_attn(
            query=page_h,
            key=ip_h,
            value=ip_h,
        )

        return self.norm(page_h + self.dropout(attn_out))


class OverNet(nn.Module):
    def __init__(
        self,
        vocab_size=2**14,
        d_model=256,
        n_heads=8,
        dropout=0.0,
        seq_len=64,
        n_layers=1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads

        self.ip_embedding = nn.Embedding(vocab_size, d_model)
        self.page_embedding = nn.Embedding(vocab_size, d_model)
        self.cross_attn = IPPageCrossAttention(d_model, n_heads, dropout)

        pe = self._make_pe(seq_len * 2, d_model)
        self.register_buffer("encoder_pe", pe[:, :seq_len, :])
        self.register_buffer("decoder_pe", pe[:, seq_len:, :])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cached decoder — same state_dict layout as nn.TransformerDecoder
        self.decoder = CachedDecoder(d_model, n_heads, d_model * 4, n_layers)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab_size)

    # ---- Export-friendly halves ----

    def encode(self, ip, pages):
        """Run encoder + precompute cross-attention K/V. Call once per sample."""
        ip_h = self.ip_embedding(ip.long())
        page_h = self.page_embedding(pages.long())
        fused = self.cross_attn(page_h, ip_h) + self.encoder_pe
        memory = self.encoder(fused)
        cross_kvs = self.decoder.precompute_cross_kv(memory)
        first_token = pages[:, -1:]  # the starting token for the loop
        return memory, cross_kvs, first_token

    def decode_step(self, token, step_idx, cross_kvs, self_kvs):
        """One autoregressive step.
        token:     (B, 1)
        step_idx:  int (position for PE)
        cross_kvs: list of (k, v) per layer  — fixed after encode
        self_kvs:  list of (k_cache, v_cache) per layer — grows each step
        """
        x = self.page_embedding(token.long()) + self.decoder_pe[:, step_idx : step_idx + 1, :]
        x, new_self_kvs = self.decoder.step(x, cross_kvs, self_kvs)
        logits = self.head(x)  # (B, 1, vocab)
        next_token = torch.argmax(logits, dim=-1)
        return next_token, new_self_kvs

    # ---- Full PyTorch inference (sanity / reference) ----

    def forward(self, ip, pages):
        B = pages.shape[0]
        device = pages.device

        _, cross_kvs, current = self.encode(ip, pages)
        self_kvs = [
            (
                torch.zeros(B, self.n_heads, 0, self.head_dim, device=device),
                torch.zeros(B, self.n_heads, 0, self.head_dim, device=device),
            )
            for _ in range(self.n_layers)
        ]

        generated = []
        for step in range(self.seq_len):
            current, self_kvs = self.decode_step(current, step, cross_kvs, self_kvs)
            generated.append(current)

        return torch.cat(generated, dim=1)

    def _make_pe(self, length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe