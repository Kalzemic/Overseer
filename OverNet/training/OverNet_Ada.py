import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class OverState(nn.Module):
   
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.gru  = nn.GRUCell(input_size=d_model, hidden_size=d_model)
        self.norm = nn.LayerNorm(d_model)

        # AdaLN projection: state → (γ, β) for each encoder LayerNorm call
        # output dim = 2 * d_model: first half is γ, second half is β
        # zero-initialized so training starts from identity (AdaLN-Zero)
        self.adaLN_proj = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.adaLN_proj.weight)
        nn.init.zeros_(self.adaLN_proj.bias)

        # internal state buffer — set by forward(), read by modulate()
        self._state = None

    def forward(self, context, state):

        new_state = self.gru(context, state)
        new_state = self.norm(new_state)
        self._state = new_state
        return new_state

    def prime(self, state):

        self._state = state

    def modulate(self, x, norm):

        assert self._state is not None, "Call prime() before encoder forward."

        # normalize without affine transformation
        # we're supplying our own γ and β from the state
        x_norm = F.layer_norm(x, norm.normalized_shape, eps=norm.eps)

        # generate per-feature scale and shift from current state
        # state: (batch, d_model) → proj: (batch, 2*d_model)
        proj = self.adaLN_proj(self._state)                 # (batch, 2*d_model)
        proj = proj.unsqueeze(1)                             # (batch, 1, 2*d_model) for broadcasting
        gamma, beta = proj.chunk(2, dim=-1)                  # each (batch, 1, d_model)

        # (1 + gamma) centers the scaling around identity at zero-init
        return (1 + gamma) * x_norm + beta


class AdaLNEncoderLayer(nn.TransformerEncoderLayer):
   

    def __init__(self, d_model, nhead, dim_feedforward, dropout, overstate):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm — norms sit before attention and ffn
        )
        # reference to OverState — not owned here, shared across all layers
        self.overstate = overstate

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):

        x = src

        # pre-norm self-attention block with AdaLN on norm1
        x_normed = self.overstate.modulate(x, self.norm1)
        attn_out, _ = self.self_attn(
            x_normed, x_normed, x_normed,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # pre-norm feedforward block with AdaLN on norm2
        x_normed = self.overstate.modulate(x, self.norm2)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_normed))))
        x = x + self.dropout2(ff_out)

        return x


class OverNet(nn.Module):
    def __init__(self, input_size=48, seq_len=64, d_model=256, n_heads=8, n_layers=4,
                 dropout=0.1):
        super().__init__()

        self.d_model    = d_model
        self.seq_len    = seq_len
        self.input_size = input_size

        # bit unpacking buffers
        self.register_buffer('powers',     (2 ** torch.arange(input_size)).long())
        self.register_buffer('bit_shifts', torch.arange(input_size))

        # input projections
        self.page_projection = OverFusion(input_size, d_model)
        self.ip_projection   = OverFusion(input_size, d_model)

        # cross-attention: page queries into IP context
        self.ip_page_cross_attn = IPPageCrossAttention(d_model, n_heads, dropout)

        # sinusoidal PE
        pe = self._make_pe(seq_len * 2, d_model)
        self.register_buffer('encoder_pe', pe[:, :seq_len, :])
        self.register_buffer('decoder_pe', pe[:, seq_len:, :])

        
        self.overstate = OverState(d_model)


        self.encoder = nn.ModuleList([
            AdaLNEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                overstate=self.overstate,   # shared reference
            )
            for _ in range(n_layers)
        ])
        # final norm after encoder stack
        self.encoder_norm = nn.LayerNorm(d_model)

        # project state into decoder query space
        self.state_proj = nn.Linear(d_model, d_model)

        # learned output queries
        self.output_queries = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # decoder — standard, unchanged
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

        # output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_size),
        )

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

    def _encode(self, x):
      
        for layer in self.encoder:
            x = layer(x)
        return self.encoder_norm(x)

    def forward(self, x, state):
        """
        x:     (batch, seq_len, 2) — raw [ip, page] integers
        state: (batch, d_model)    — previous GRU hidden state

        Returns:
            out:           (batch, seq_len, 48)  — predicted delta bit logits
            pred_page_ints:(batch, seq_len)       — reconstructed absolute pages
            new_state:     (batch, d_model)       — updated GRU state
        """
        batch = x.shape[0]

        # unpack
        page_ints = x[:, :, 1].long()
        ip_ints   = x[:, :, 0].long()

        # delta encode in integer space
        page_deltas = torch.zeros_like(page_ints)
        page_deltas[:, 1:] = page_ints[:, 1:] - page_ints[:, :-1]

        ip_deltas = torch.zeros_like(ip_ints)
        ip_deltas[:, 1:] = ip_ints[:, 1:] - ip_ints[:, :-1]

        # bit arrays
        page_bits = self.to_bits(page_deltas)   # (batch, seq_len, 48)
        ip_bits   = self.to_bits(ip_deltas)     # (batch, seq_len, 48)

        # fuse ip and page streams
        page_h = self.page_projection(page_bits)
        ip_h   = self.ip_projection(ip_bits)
        fused  = self.ip_page_cross_attn(page_h, ip_h)
        fused  = fused + self.encoder_pe
        fused  = self.dropout(fused)

        self.overstate.prime(state)

        # encode — each AdaLNEncoderLayer calls overstate.modulate() internally
        memory = self._encode(fused)             # (batch, seq_len, d_model)

        # update GRU state from the encoded sequence summary
        context   = memory.mean(dim=1)           # (batch, d_model)
        new_state = self.overstate(context, state)

        # decoder queries biased by new state
        state_bias = self.state_proj(new_state).unsqueeze(1)
        queries    = self.output_queries.expand(batch, -1, -1) + self.decoder_pe + state_bias

        out = self.decoder(queries, memory)
        out = self.dropout(out)
        out = self.head(out)                     # (batch, seq_len, 48)

        # reconstruct absolute pages from predicted deltas
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