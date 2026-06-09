import json
import os
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize

from vnm_ros.training.checkpoint import save_checkpoint
from vnm_ros.training.losses import compute_losses
from vnm_ros.training.metrics import batch_metrics


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
        self.writer = (
            SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
            if enable_tensorboard
            else None
        )
        self.best_validation_loss = float("inf")
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)

    def _prepare(self, batch):
        observation = batch["observation"]
        chunks = torch.split(observation, 3, dim=1)
        observation = torch.cat([self.normalize(chunk) for chunk in chunks], dim=1)
        goal = self.normalize(batch["goal"])
        return {
            "observation": observation.to(self.device),
            "goal": goal.to(self.device),
            "distance": batch["distance"].to(self.device),
            "actions": batch["actions"].to(self.device),
            "action_mask": batch["action_mask"].to(self.device),
        }

    def run_epoch(self, loader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        totals = {}
        sample_count = 0

        for batch in loader:
            data = self._prepare(batch)
            with torch.set_grad_enabled(training):
                distance_pred, action_pred = self.model(
                    data["observation"], data["goal"]
                )
                losses = compute_losses(
                    distance_pred,
                    action_pred,
                    data["distance"],
                    data["actions"],
                    data["action_mask"],
                    self.alpha,
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

        return {key: value / max(sample_count, 1) for key, value in totals.items()}

    def fit(self, train_loader, validation_loader, start_epoch: int, epochs: int):
        history_path = os.path.join(self.run_dir, "metrics.jsonl")
        for filename in os.listdir(self.weights_dir):
            if filename.startswith("epoch_") and filename.endswith(".pth"):
                os.remove(os.path.join(self.weights_dir, filename))
        try:
            for epoch in range(start_epoch, epochs):
                train_metrics = self.run_epoch(train_loader, training=True)
                validation_metrics = (
                    self.run_epoch(validation_loader, training=False)
                    if validation_loader is not None
                    else None
                )
                if self.scheduler is not None:
                    self.scheduler.step()

                learning_rate = self.optimizer.param_groups[0]["lr"]
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
        finally:
            if self.writer is not None:
                self.writer.close()
