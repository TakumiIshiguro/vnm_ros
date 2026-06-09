from typing import Dict

import torch


def batch_metrics(
    distance_pred: torch.Tensor,
    action_pred: torch.Tensor,
    distance_target: torch.Tensor,
    action_target: torch.Tensor,
    action_mask: torch.Tensor,
) -> Dict[str, float]:
    distance_mae = torch.mean(torch.abs(distance_pred.squeeze(-1) - distance_target))
    position_error = torch.linalg.vector_norm(
        action_pred[:, :, :2] - action_target[:, :, :2], dim=-1
    ).mean(dim=1)
    masked_position_error = (
        position_error * action_mask
    ).sum() / action_mask.sum().clamp_min(1.0)
    return {
        "distance_mae": float(distance_mae.item()),
        "position_error": float(masked_position_error.item()),
    }

