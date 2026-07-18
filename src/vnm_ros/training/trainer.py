import json
import os
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize

from vnm_ros.training.checkpoint import save_checkpoint
from vnm_ros.training.losses import compute_losses
from vnm_ros.training.metrics import batch_metrics

CMD_DIR_NAMES = ("straight", "left", "right")


def cmd_dir_labels(cmd_dir):
    if cmd_dir is None:
        return None
    labels = torch.argmax(cmd_dir[:, :3], dim=1)
    valid = torch.sum(cmd_dir[:, :3], dim=1) > 0.0
    return torch.where(valid, labels, torch.zeros_like(labels))


def direction_metric_values(
    labels,
    distance_pred,
    action_pred,
    distance_target,
    action_target,
    action_mask,
    alpha,
):
    if labels is None:
        return {}
    distance_per_sample = torch.nn.functional.mse_loss(
        distance_pred.squeeze(-1), distance_target, reduction="none"
    )
    action_per_sample = torch.nn.functional.mse_loss(
        action_pred, action_target, reduction="none"
    ).mean(dim=(1, 2))
    total_per_sample = alpha * 1e-2 * distance_per_sample + (
        1.0 - alpha
    ) * action_per_sample
    position_error = torch.linalg.vector_norm(
        action_pred[:, :, :2] - action_target[:, :, :2], dim=-1
    ).mean(dim=1)

    values = {}
    for index, name in enumerate(CMD_DIR_NAMES):
        selection = labels == index
        count = int(selection.sum().item())
        if count <= 0:
            continue
        values[f"direction/{name}/loss"] = (
            float(total_per_sample[selection].mean().item()),
            count,
        )
        values[f"direction/{name}/distance_loss"] = (
            float(distance_per_sample[selection].mean().item()),
            count,
        )
        action_selection = selection & (action_mask > 0.0)
        action_count = int(action_selection.sum().item())
        if action_count > 0:
            values[f"direction/{name}/action_loss"] = (
                float(action_per_sample[action_selection].mean().item()),
                action_count,
            )
            values[f"direction/{name}/position_error"] = (
                float(position_error[action_selection].mean().item()),
                action_count,
            )
    return values


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        run_dir: str,
        weights_dir: str,
        config: dict,
        alpha: float,
        gradient_clip: float = 0.0,
        enable_tensorboard: bool = True,
        frozen_modules=None,
        cmd_dir_loss_weights=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir
        self.weights_dir = weights_dir
        self.config = config
        self.alpha = alpha
        self.gradient_clip = gradient_clip
        self.frozen_modules = list(frozen_modules or [])
        self.cmd_dir_loss_weights = (
            torch.as_tensor(cmd_dir_loss_weights, dtype=torch.float32, device=device)
            if cmd_dir_loss_weights is not None
            else None
        )
        self.writer = (
            SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
            if enable_tensorboard
            else None
        )
        self.best_validation_loss = float("inf")
        training_cfg = config.get("train", {}).get("training", {})
        self.final_checkpoint_name = training_cfg.get("final_checkpoint_name", "")
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)

    def _sample_loss_weights(self, cmd_dir):
        if self.cmd_dir_loss_weights is None or cmd_dir is None:
            return None
        labels = cmd_dir_labels(cmd_dir)
        return self.cmd_dir_loss_weights[labels]

    def _prepare(self, batch):
        observation = batch["observation"]
        chunks = torch.split(observation, 3, dim=1)
        observation = torch.cat([self.normalize(chunk) for chunk in chunks], dim=1)
        prepared = {
            "observation": observation.to(self.device),
            "distance": batch["distance"].to(self.device),
            "actions": batch["actions"].to(self.device),
            "action_mask": batch["action_mask"].to(self.device),
        }
        if "goal" in batch:
            prepared["goal"] = self.normalize(batch["goal"]).to(self.device)
        else:
            prepared["goal"] = None
        if "cmd_dir" in batch:
            prepared["cmd_dir"] = batch["cmd_dir"].to(self.device)
        else:
            prepared["cmd_dir"] = None
        return prepared

    def run_epoch(self, loader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        for module in self.frozen_modules:
            module.eval()
        totals = {}
        counts = {}
        sample_count = 0

        for batch in loader:
            data = self._prepare(batch)
            with torch.set_grad_enabled(training):
                distance_pred, action_pred = self.model(
                    data["observation"],
                    data["goal"],
                    cmd_dir=data["cmd_dir"],
                )
                losses = compute_losses(
                    distance_pred,
                    action_pred,
                    data["distance"],
                    data["actions"],
                    data["action_mask"],
                    self.alpha,
                    sample_weights=self._sample_loss_weights(data["cmd_dir"]),
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()

            metrics = batch_metrics(
                distance_pred,
                action_pred,
                data["distance"],
                data["actions"],
                data["action_mask"],
            )
            values = {**{k: float(v.item()) for k, v in losses.items()}, **metrics}
            batch_size = data["distance"].shape[0]
            sample_count += batch_size
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
                counts[key] = counts.get(key, 0) + batch_size
            direction_values = direction_metric_values(
                cmd_dir_labels(data["cmd_dir"]),
                distance_pred,
                action_pred,
                data["distance"],
                data["actions"],
                data["action_mask"],
                self.alpha,
            )
            for key, (value, count) in direction_values.items():
                totals[key] = totals.get(key, 0.0) + value * count
                counts[key] = counts.get(key, 0) + count

        return {key: value / max(counts.get(key, sample_count), 1) for key, value in totals.items()}

    def fit(self, train_loader, validation_loader, start_epoch: int, epochs: int):
        history_path = os.path.join(self.run_dir, "metrics.jsonl")
        try:
            for epoch in range(start_epoch, epochs):
                learning_rate = self.optimizer.param_groups[0]["lr"]
                train_metrics = self.run_epoch(train_loader, training=True)
                validation_metrics = (
                    self.run_epoch(validation_loader, training=False)
                    if validation_loader is not None
                    else None
                )
                if self.scheduler is not None:
                    self.scheduler.step()

                record = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "learning_rate": learning_rate,
                }
                if validation_metrics is None:
                    record.pop("validation")
                print(json.dumps(record, sort_keys=True))
                with open(history_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                if self.writer is not None:
                    for name, value in train_metrics.items():
                        self.writer.add_scalar(f"train/{name}", value, epoch)
                    if validation_metrics is not None:
                        for name, value in validation_metrics.items():
                            self.writer.add_scalar(f"test/{name}", value, epoch)
                    self.writer.add_scalar("training/learning_rate", learning_rate, epoch)
                    self.writer.flush()

                monitored_loss = (
                    validation_metrics["loss"]
                    if validation_metrics is not None
                    else train_metrics["loss"]
                )
                is_best = monitored_loss < self.best_validation_loss
                self.best_validation_loss = min(self.best_validation_loss, monitored_loss)
                common = dict(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_validation_loss=self.best_validation_loss,
                    config=self.config,
                )
                save_checkpoint(os.path.join(self.weights_dir, "latest.pth"), **common)
                if is_best:
                    save_checkpoint(os.path.join(self.weights_dir, "best.pth"), **common)
                if epoch + 1 == epochs and self.final_checkpoint_name:
                    save_checkpoint(
                        os.path.join(self.weights_dir, self.final_checkpoint_name),
                        **common,
                    )
        finally:
            if self.writer is not None:
                self.writer.close()
