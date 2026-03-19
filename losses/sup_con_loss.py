import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute supervised contrastive loss.

        Args:
            features: Tensor shaped [batch, views, dim] or [batch, dim].
            labels: Tensor shaped [batch].
            mask: Pair mask shaped [batch, batch].
        """
        if len(features.shape) < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        view_count = features.shape[1]

        features = F.normalize(features, dim=2)
        features = features.view(batch_size * view_count, -1)

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(features.device)
        elif mask is None:
            mask = torch.eye(batch_size * view_count, device=features.device)
        else:
            mask = mask.float()

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * view_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(batch_size, view_count).mean()
