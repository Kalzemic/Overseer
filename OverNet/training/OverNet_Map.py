import torch
import torch.nn as nn
from threading import Lock
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder


class OverNet(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.config = RetNetConfig(
            vocab_size=vocab_size,
            decoder_embed_dim=d_model,
            decoder_retention_heads=nhead,
            decoder_layers=num_layers,
            decoder_ffn_embed_dim=d_model * 4,
        )
        embed_tokens = nn.Embedding(vocab_size, d_model)
        self.retnet = RetNetDecoder(self.config, embed_tokens=embed_tokens)
        # output_projection is built internally and tied to embed_tokens.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.retnet(x)
        return out[:, 1::2, :]

    def forward_recurrent(self, x, incremental_state):
        out, incremental_state = self.retnet(x, incremental_state=incremental_state)
        return out[:, 1:2, :], incremental_state

class OverMap:
    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        # page address → ordinal (online, built at runtime)
        self.addr_to_rank: dict[int, int] = {}
        self._next_rank = 0

    def lookup_pages(self, addresses: list[int]) -> torch.Tensor:
        # online ordinal assignment for pages
        for addr in addresses:
            if addr not in self.addr_to_rank and self._next_rank < self.vocab_size:
                self.addr_to_rank[addr] = self._next_rank
                self._next_rank += 1
        ranks = [self.addr_to_rank.get(addr, self.vocab_size - 1) for addr in addresses]
        return torch.tensor(ranks, dtype=torch.long, device=self.device)

    def lookup_pcs(self, pcs: list[int]) -> torch.Tensor:
        # just mod — no map needed
        ranks = [pc % self.vocab_size for pc in pcs]
        return torch.tensor(ranks, dtype=torch.long, device=self.device)
    
