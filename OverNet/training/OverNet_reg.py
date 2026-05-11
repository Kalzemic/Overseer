import torch 
import torch.nn as nn

class OverNet(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=1, seq_len=128, dropout=0.1, d_state=128):
        super().__init__()
        
        # 2 inputs per token: page_delta + ip_delta
        self.input_proj = nn.Linear(2, d_model)
        
        self.overstate = OverState(d_model, n_heads=4, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_queries = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        
        # single output: page delta
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, page_deltas, ip_deltas, state):
        # (batch, seq_len)
        x = torch.stack([page_deltas, ip_deltas], dim=-1).float()  # (batch, seq_len, 2)
        x = self.input_proj(x)
        x = self.dropout(x)
        
        memory = self.encoder(x)
        new_state = self.overstate(memory, state)
        
        memory_and_state = torch.cat([memory, new_state], dim=1)
        queries = self.output_queries.expand(x.shape[0], -1, -1)
        out = self.decoder(queries, memory_and_state)
        out = self.dropout(out)
        
        pred_deltas = self.head(out).squeeze(-1)  # (batch, seq_len)
        return pred_deltas, new_state
    
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
