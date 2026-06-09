from typing import Dict

import torch
import torch.nn.functional as F


def compute_losses(
    distance_pred: torch.Tensor,
    action_pred: torch.Tensor,
    distance_target: torch.Tensor,
    action_target: torch.Tensor,
    action_mask: torch.Tensor,
    alpha: float,
) -> Dict[str, torch.Tensor]:
    distance_loss = F.mse_loss(distance_pred.squeeze(-1), distance_target)
    per_sample_action = F.mse_loss(
        action_pred, action_target, reduction="none"
    ).mean(dim=(1, 2))
    action_loss = (per_sample_action * action_mask).sum() / action_mask.sum().clamp_min(1.0)
    total_loss = alpha * 1e-2 * distance_loss + (1.0 - alpha) * action_loss
    return {
        "loss": total_loss,
        "distance_loss": distance_loss,
        "action_loss": action_loss,
    }

