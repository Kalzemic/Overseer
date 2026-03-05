import torch
import torch.nn as nn
import numpy as np


class OverNetVocab(nn.Module):
    def __init__(self, seq_len=64, d_model=256, n_heads=8, n_layers=4, dropout=0.1, vocab_size=16384):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size


        self.page_embedding = nn.Embedding(vocab_size, d_model)


        pe = self._make_pe(seq_len, d_model)
        self.register_buffer('encoder_pe', pe)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)


        self.output_queries = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)


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

        # classification head over vocab
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
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
        """
        Args:
            x: (batch, seq_len) long — hashed page indices in [0, vocab_size)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch = x.shape[0]

        # --- delta encoding ---
        raw = x                                          # (batch, seq_len) — save for residual
        deltas = torch.zeros_like(x)
        deltas[:, 1:] = (x[:, 1:] - x[:, :-1]) % self.vocab_size
        x = deltas
        # ----------------------

        x = self.page_embedding(x)     # (batch, seq_len, d_model)
        x = x + self.encoder_pe
        x = self.dropout(x)
        memory = self.encoder(x)       # (batch, seq_len, d_model)

        # decode
        queries = self.output_queries.expand(batch, -1, -1)
        out = self.decoder(queries, memory)  # (batch, seq_len, d_model)
        out = self.dropout(out)

        logits = self.head(out)        # (batch, seq_len, vocab_size)

        # --- residual: bias logits toward raw input position ---
        logits[torch.arange(batch).unsqueeze(1),
                torch.arange(self.seq_len).unsqueeze(0),
                raw] += 1.0
        # -------------------------------------------------------

        return logits
# ===========================================================================
# Hashing utilities
# ===========================================================================

KNUTH = 11400714819323198485  # 2^64 / golden ratio

def page_to_vocab_idx(page_uint64, vocab_size=16384):
    """Hash uint64 page numbers to vocab indices."""
    if isinstance(page_uint64, torch.Tensor):
        hashed = (page_uint64.long() * (KNUTH % (2**63))) % vocab_size
        return hashed.abs()
    else:
        page_uint64 = np.asarray(page_uint64, dtype=np.uint64)
        hashed = (page_uint64 * np.uint64(KNUTH)) >> np.uint64(64 - int(np.log2(vocab_size)))
        return (hashed % vocab_size).astype(np.int64)


def bits_to_uint64(bits):
    """Convert (batch, seq_len, 36) bit tensor to (batch, seq_len) int64."""
    bits_int = bits.long()
    powers = (2 ** torch.arange(36, device=bits.device)).long()
    return (bits_int * powers).sum(dim=-1)


def prepare_input(x_bits, vocab_size=16384):
    """
    Convert bit arrays from MemDataset into hashed vocab indices.
    Call this before model.forward().

    Args:
        x_bits: (batch, seq_len, 36) from MemDataset
        vocab_size: int

    Returns:
        (batch, seq_len) long tensor of vocab indices
    """
    pages = bits_to_uint64(x_bits)
    return page_to_vocab_idx(pages, vocab_size)


def check_collisions(trace_path, vocab_size=16384, max_pages=None):
    """Check hash collision rate for a trace file."""
    data = np.memmap(trace_path, dtype=np.uint64, mode='r')
    if max_pages:
        data = data[:max_pages]

    unique_pages = np.unique(data)
    n_unique = len(unique_pages)

    indices = page_to_vocab_idx(unique_pages, vocab_size)
    unique_indices = len(np.unique(indices))

    collisions = n_unique - unique_indices
    collision_rate = collisions / n_unique if n_unique > 0 else 0

    print(f"Trace: {trace_path}")
    print(f"Unique pages:       {n_unique:,}")
    print(f"Vocab size:         {vocab_size:,}")
    print(f"Unique hash slots:  {unique_indices:,}")
    print(f"Collisions:         {collisions} ({collision_rate:.2%})")

    if collision_rate > 0.05:
        suggested = 2 ** int(np.ceil(np.log2(n_unique * 4)))
        print(f"WARNING: High collisions. Suggested vocab_size: {suggested}")

    return n_unique, collisions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_collisions', type=str)
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--max_pages', type=int, default=None)
    args = parser.parse_args()

    if args.check_collisions:
        check_collisions(args.check_collisions, args.vocab_size, args.max_pages)