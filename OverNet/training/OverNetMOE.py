import torch
import torch.nn as nn
import torch.nn.functional as F

class OvernetLatent(nn.Module):
    def __init__(self, history_len=64, forecast_len=64, num_experts=4, d_model=256):
        super().__init__()
        self.forecast_len = forecast_len
        self.num_experts = num_experts
        
        # 1. Page-to-Embedding (Increased d_model to 256 for DLRM capacity)
        self.bit_encoder = nn.Linear(36, d_model) 
        
        # 2. Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 3. MoE Output: Each expert is a specialist
        self.prediction_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, forecast_len * 36)
            ) for _ in range(num_experts)
        ])
        
        # 4. Gating Network (Now looks at the WHOLE history, not just the last page)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, x):
        # x: [B, 64, 36]
        batch_size = x.shape[0]
        
        # Transform bits to latent space
        x = self.bit_encoder(x) 
        latent_seq = self.transformer(x) # [B, 64, d_model]
        
        # Global Pooling: Capture the "vibe" of the last 64 pages
        # This helps the gate decide: "Is this a DLRM lookup storm?"
        global_context = torch.mean(latent_seq, dim=1) # [B, d_model]
        
        # Expert opinions: [B, Exp, 64, 36]
        expert_outputs = torch.stack([
            exp(global_context).view(batch_size, self.forecast_len, 36) 
            for exp in self.prediction_experts
        ], dim=1) 
        
        # Gate weights: [B, num_experts]
        gate_scores = torch.softmax(self.gate(global_context), dim=-1) 
        
        # Final prediction (Weighted sum)
        # Reshape gate for broadcasting: [B, Exp, 1, 1]
        final_prediction = torch.sum(
            expert_outputs * gate_scores.view(batch_size, self.num_experts, 1, 1), 
            dim=1
        )
        
        return final_prediction, gate_scores
# --- Loss Function ---

class OvernetLoss(nn.Module):
    def __init__(self, gamma=2.0, gate_weight=0.1):
        super().__init__()
        self.gamma = gamma
        self.gate_weight = gate_weight

    def forward(self, pred_logits, target_bits, gate_weights):
        """
        pred_logits: (Batch, 64, 36)
        target_bits: (Batch, 64, 36)
        gate_weights: (Batch, num_experts)
        """
        # 1. Focal Loss (Binary Cross Entropy with focus on 'hard' bit flips)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_bits, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # 2. Gate Entropy (Encourages model to commit to one expert)
        gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1).mean()
        
        return focal_loss + (self.gate_weight * gate_entropy)