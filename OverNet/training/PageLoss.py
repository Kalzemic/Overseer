import torch
import torch.nn as nn
import torch.nn.functional as F


class PageLoss(nn.Module):
    def __init__(self, bce_weight=1.0, exact_weight=0.5, alpha=1.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.exact_weight = exact_weight
        self.alpha = alpha

    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target)
        
        probs = torch.sigmoid(output)
        per_bit_correct = probs * target + (1 - probs) * (1 - target)
        per_bit_correct = per_bit_correct.clamp(min=1e-7)
        
        log_exact = (per_bit_correct ** self.alpha).log().sum(dim=-1)
        exact_loss = -log_exact.mean()
        
        return self.bce_weight * bce + self.exact_weight * exact_loss




# class PageLoss(nn.Module):
#     def __init__(self, bce_weight=1.0, exact_weight=0.5, alpha=1.5):
#         super().__init__()
#         self.bce_weight = bce_weight
#         self.exact_weight = exact_weight
#         self.alpha = alpha

#     def forward(self, output, target):
#         bce = F.binary_cross_entropy_with_logits(output, target)

#         probs = torch.sigmoid(output)
#         per_bit_correct = probs * target + (1 - probs) * (1 - target)
#         per_bit_correct = per_bit_correct.clamp(min=1e-7)

#         # Bottleneck: loss is dominated by the worst bit in each prediction
#         worst_bit = (per_bit_correct ** self.alpha).min(dim=-1).values
#         exact_loss = -worst_bit.log().mean()

#         return self.bce_weight * bce + self.exact_weight * exact_loss