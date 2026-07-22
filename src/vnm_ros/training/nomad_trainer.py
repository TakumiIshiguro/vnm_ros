import json
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize

from vnm_ros.training.checkpoint import save_checkpoint
from vnm_ros.training.optimizer import optimizer_learning_rates
from vnm_ros.training.trainer import CMD_DIR_NAMES, cmd_dir_labels


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
        cmd_dir_loss_weights=None,
        ema_model=None,
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
        self.cmd_dir_loss_weights = (
            torch.as_tensor(cmd_dir_loss_weights, dtype=torch.float32, device=device)
            if cmd_dir_loss_weights is not None
            else None
        )
        if ema_model is None:
            raise ValueError("NoMaDTrainer requires an EMA model")
        self.ema_model = ema_model
        self.writer = (
            SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
            if enable_tensorboard
            else None
        )
        self.best_validation_loss = float("inf")
        training_cfg = config.get("train", {}).get("training", {})
        self.best_checkpoint_name = training_cfg.get(
            "best_checkpoint_name", "best.pth"
        )
        self.final_checkpoint_name = training_cfg.get("final_checkpoint_name", "")
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)

    def _sample_loss_weights(self, cmd_dir):
        if self.cmd_dir_loss_weights is None:
            return None
        labels = cmd_dir_labels(cmd_dir)
        return self.cmd_dir_loss_weights[labels]

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

        totals = {}
        counts = {}
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
                per_sample_loss = F.mse_loss(
                    noise_pred, noise, reduction="none"
                ).mean(dim=tuple(range(1, noise_pred.ndim)))
                sample_weights = self._sample_loss_weights(data["cmd_dir"])
                if sample_weights is None:
                    loss = per_sample_loss.mean()
                else:
                    loss = (
                        per_sample_loss * sample_weights
                    ).sum() / sample_weights.sum().clamp_min(1.0)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()
                    self.ema_model.step(self.model.parameters())
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item()) * batch_size
            counts["loss"] = counts.get("loss", 0) + batch_size
            labels = cmd_dir_labels(data["cmd_dir"])
            for index, name in enumerate(CMD_DIR_NAMES):
                selection = labels == index
                count = int(selection.sum().item())
                if count <= 0:
                    continue
                key = f"direction/{name}/loss"
                value = float(per_sample_loss[selection].mean().item())
                totals[key] = totals.get(key, 0.0) + value * count
                counts[key] = counts.get(key, 0) + count
        return {key: value / max(counts[key], 1) for key, value in totals.items()}

    def fit(self, train_loader, validation_loader, start_epoch: int, epochs: int):
        history_path = os.path.join(self.run_dir, "metrics.jsonl")
        for epoch in range(start_epoch, epochs):
            learning_rates = optimizer_learning_rates(self.optimizer)
            learning_rate = next(iter(learning_rates.values()))
            train_metrics = self.run_epoch(train_loader, training=True)
            validation_metrics = None
            if validation_loader is not None:
                self.ema_model.store(self.model.parameters())
                self.ema_model.copy_to(self.model.parameters())
                try:
                    validation_metrics = self.run_epoch(
                        validation_loader, training=False
                    )
                finally:
                    self.ema_model.restore(self.model.parameters())
            if self.scheduler is not None:
                self.scheduler.step()

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rate": learning_rate,
                "learning_rates": learning_rates,
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
                for group_name, group_rate in learning_rates.items():
                    self.writer.add_scalar(
                        f"training/learning_rate/{group_name}", group_rate, epoch
                    )
                self.writer.flush()

            monitored_loss = (
                validation_metrics["loss"]
                if validation_metrics is not None
                else train_metrics["loss"]
            )
            is_best = monitored_loss < self.best_validation_loss
            self.best_validation_loss = min(self.best_validation_loss, monitored_loss)
            training_state_dict = self.model.state_dict()
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())
            try:
                inference_state_dict = {
                    key: value.detach().clone()
                    for key, value in self.model.state_dict().items()
                }
            finally:
                self.ema_model.restore(self.model.parameters())
            common = dict(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_validation_loss=self.best_validation_loss,
                config=self.config,
                inference_state_dict=inference_state_dict,
                training_state_dict=training_state_dict,
                ema_state_dict=self.ema_model.state_dict(),
            )
            if is_best:
                save_checkpoint(
                    os.path.join(self.weights_dir, self.best_checkpoint_name),
                    **common,
                )
            if epoch + 1 == epochs and self.final_checkpoint_name:
                save_checkpoint(
                    os.path.join(self.weights_dir, self.final_checkpoint_name),
                    **common,
                )
        if self.writer is not None:
            self.writer.close()
