import os
import re

import torch


def _compact_value(value):
    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_") or "none"


def _freeze_layers_tag(training_cfg):
    layers = training_cfg.get("freeze_layers", [])
    if not layers:
        return "none"
    names = [_compact_value(layer) for layer in layers if str(layer).strip()]
    if not names:
        return "none"
    tag = "-".join(names)
    return tag[:80]


def training_checkpoint_filename(config: dict, epoch: int) -> str:
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    training_cfg = train_cfg.get("training", {})
    model_type = _compact_value(model_cfg.get("model_type", "model"))
    parts = [
        model_type,
        f"lr{_compact_value(training_cfg.get('learning_rate', 'na'))}",
        f"bs{_compact_value(training_cfg.get('batch_size', 'na'))}",
        f"ep{_compact_value(training_cfg.get('epochs', 'na'))}",
        f"sched{_compact_value(training_cfg.get('scheduler', 'none'))}",
        f"warm{_compact_value(training_cfg.get('warmup_epochs', 0))}",
        f"alpha{_compact_value(training_cfg.get('alpha', 'na'))}",
        f"wd{_compact_value(training_cfg.get('weight_decay', 'na'))}",
        f"cmdw{int(bool(training_cfg.get('cmd_dir_loss_weighting', False)))}",
        f"cmdwp{_compact_value(training_cfg.get('cmd_dir_loss_weight_power', 'na'))}",
        f"distfreeze{int(bool(training_cfg.get('freeze_dist_pred_net', False)))}",
        f"freeze{_freeze_layers_tag(training_cfg)}",
        f"epoch{epoch:03d}",
    ]
    return "_".join(parts) + ".pth"


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
