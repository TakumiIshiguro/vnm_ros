import json
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize

from vnm_ros.training.checkpoint import save_checkpoint


class NoMaDTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        noise_scheduler,
        device,
        run_dir: str,
        weights_dir: str,
        config: dict,
        gradient_clip: float = 0.0,
        enable_tensorboard: bool = True,
        frozen_modules=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.run_dir = run_dir
        self.weights_dir = weights_dir
        self.config = config
        self.gradient_clip = gradient_clip
        self.frozen_modules = list(frozen_modules or [])
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
        return {
            "observation": observation.to(self.device),
            "actions": batch["actions"].to(self.device),
            "cmd_dir": batch["cmd_dir"].to(self.device),
        }

    def _obs_cond(self, data):
        batch_size = data["observation"].shape[0]
        image_size = data["observation"].shape[-2:]
        fake_goal = torch.randn(
            (batch_size, 3, image_size[0], image_size[1]),
            dtype=data["observation"].dtype,
            device=self.device,
        )
        goal_mask = torch.ones(batch_size, dtype=torch.long, device=self.device)
        obs_cond = self.model(
            "vision_encoder",
            obs_img=data["observation"],
            goal_img=fake_goal,
            input_goal_mask=goal_mask,
            cmd_dir=data["cmd_dir"],
        )
        return obs_cond

    def run_epoch(self, loader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        for module in self.frozen_modules:
            module.eval()

        total_loss = 0.0
        sample_count = 0
        for batch in loader:
            data = self._prepare(batch)
            batch_size = data["actions"].shape[0]
            with torch.set_grad_enabled(training):
                obs_cond = self._obs_cond(data)
                noise = torch.randn_like(data["actions"])
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=self.device,
                ).long()
                noisy_actions = self.noise_scheduler.add_noise(
                    data["actions"], noise, timesteps
                )
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=noisy_actions,
                    timestep=timesteps,
                    global_cond=obs_cond,
                )
                loss = F.mse_loss(noise_pred, noise)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
        return {"loss": total_loss / max(sample_count, 1)}

    def fit(self, train_loader, validation_loader, start_epoch: int, epochs: int):
        history_path = os.path.join(self.run_dir, "metrics.jsonl")
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
        if self.writer is not None:
            self.writer.close()
