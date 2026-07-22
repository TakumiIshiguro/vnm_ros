import os

import torch
import yaml

from vnm_ros.models.model_loader import validate_checkpoint_action_scale


def save_config_yaml(path: str, config: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def checkpoint_config_path(checkpoint_path: str) -> str:
    stem, _ = os.path.splitext(checkpoint_path)
    return stem + ".yaml"


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_validation_loss: float,
    config: dict,
    inference_state_dict=None,
    training_state_dict=None,
    ema_state_dict=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "state_dict": (
            inference_state_dict
            if inference_state_dict is not None
            else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_validation_loss": best_validation_loss,
        "config": config,
    }
    if training_state_dict is not None:
        checkpoint["model_state_dict"] = training_state_dict
    if ema_state_dict is not None:
        checkpoint["ema_state_dict"] = ema_state_dict
    torch.save(checkpoint, path)
    save_config_yaml(checkpoint_config_path(path), config)


def load_training_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
    expected_model_config=None,
):
    checkpoint = torch.load(path, map_location=device)
    if expected_model_config is not None:
        validate_checkpoint_action_scale(checkpoint, expected_model_config, path)
    state_dict = checkpoint.get(
        "model_state_dict", checkpoint.get("state_dict", checkpoint)
    )
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
