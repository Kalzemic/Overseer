import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


class OverNet(nn.Module):
    def __init__(self, n_layers=1,n_heads=8,vocab_size=3000,seq_length=64, d_model=256,dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)

        pe = self._make_pe(seq_length * 2, d_model)
        
        self.register_buffer('encoder_pe', pe[:, :seq_length, :])   
        self.register_buffer('decoder_pe', pe[:, seq_length:, :])
      
        
        encoderConfig = nn.TransformerEncoderLayer(d_model=d_model,
                                                        dropout=dropout,
                                                        nhead=n_heads, 
                                                        dim_feedforward=d_model * 4,
                                                        batch_first=True,norm_first=True
                                                        )
        self.encoder = nn.TransformerEncoder(encoderConfig,num_layers=n_layers)
        self.output_queries = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.02)
        decoderConfig = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(decoderConfig, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        
        batch_dim = x.shape[0]
        

        embeddings = self.dropout(self.embedding(x) + self.encoder_pe)

        memory = self.encoder(embeddings) 

        queries = self.output_queries.expand(batch_dim, -1, -1) + self.decoder_pe

        output = self.decoder(queries,memory)

        output = self.dropout(output)

        return self.head(output)
    


    def _make_pe(self, length, d_model):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    




# replace OverTable with a tensor-based version
class OverTable:
    def __init__(self, vocab_size, device):
        self.vocab_size = vocab_size
        self.device     = device
        self.table      = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        self.best_count = torch.zeros((vocab_size,), dtype=torch.long, device=device)

    def update(self, ips, eas):
        current_best = self.table[ips]          # (N,) current best EA per IP
        matches      = (current_best == eas)    # where new EA matches stored best

        # increment count where EA matches current best
        self.best_count.scatter_add_(0, ips[matches], torch.ones_like(ips[matches]))

        # decrement count where EA doesn't match
        no_match = ~matches
        self.best_count.scatter_add_(0, ips[no_match], -torch.ones_like(ips[no_match]))

        # where count dropped below 0, replace with new EA and reset count to 1
        replace = self.best_count[ips] < 0
        self.table[ips[replace]]      = eas[replace]
        self.best_count[ips[replace]] = 1

    def lookup(self, ips):
        return self.table[ips]  # (batch, seq) — direct tensor indexing