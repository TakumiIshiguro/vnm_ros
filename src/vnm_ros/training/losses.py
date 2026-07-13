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
    sample_weights: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    distance_per_sample = F.mse_loss(
        distance_pred.squeeze(-1), distance_target, reduction="none"
    )
    per_sample_action = F.mse_loss(
        action_pred, action_target, reduction="none"
    ).mean(dim=(1, 2))
    if sample_weights is None:
        sample_weights = torch.ones_like(distance_per_sample)
    distance_loss = (
        distance_per_sample * sample_weights
    ).sum() / sample_weights.sum().clamp_min(1.0)
    action_weights = action_mask * sample_weights
    action_loss = (per_sample_action * action_weights).sum() / action_weights.sum().clamp_min(1.0)
    total_loss = alpha * 1e-2 * distance_loss + (1.0 - alpha) * action_loss
    return {
        "loss": total_loss,
        "distance_loss": distance_loss,
        "action_loss": action_loss,
    }
