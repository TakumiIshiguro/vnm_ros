import os
from typing import Optional

import torch


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_validation_loss: float,
    config: dict,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_validation_loss": best_validation_loss,
            "config": config,
        },
        path,
    )


def load_training_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

