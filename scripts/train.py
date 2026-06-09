#!/usr/bin/env python3
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from vnm_ros.datasets import ViNTDataset
from vnm_ros.models.model_loader import _build_model
from vnm_ros.training.checkpoint import load_training_checkpoint
from vnm_ros.training.trainer import Trainer
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path


def make_dataset(config, dataset_type):
    dataset = config["dataset"]
    data_dir_key = (
        "train_data_dir" if dataset_type == "train" else "test_data_dir"
    )
    return ViNTDataset(
        data_dir=resolve_path(dataset[data_dir_key], package_root()),
        image_size=dataset["image_size"],
        context_size=int(dataset["context_size"]),
        len_traj_pred=int(dataset["len_traj_pred"]),
        waypoint_spacing=int(dataset["waypoint_spacing"]),
        min_goal_distance=int(dataset["min_goal_distance"]),
        max_goal_distance=int(dataset["max_goal_distance"]),
        min_action_distance=int(dataset["min_action_distance"]),
        max_action_distance=int(dataset["max_action_distance"]),
        metric_waypoint_spacing=float(dataset["metric_waypoint_spacing"]),
        normalize=bool(dataset["normalize"]),
        learn_angle=bool(dataset["learn_angle"]),
        negative_mining=bool(dataset["negative_mining"]) if dataset_type == "train" else False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--use-test", choices=["true", "false"], default=None)
    args = parser.parse_args()
    cfg = load_runtime_config(args.config_dir)
    train_cfg = cfg["train"]
    model_cfg = dict(cfg["model"])
    model_cfg.update(
        {
            "context_size": train_cfg["dataset"]["context_size"],
            "len_traj_pred": train_cfg["dataset"]["len_traj_pred"],
            "learn_angle": train_cfg["dataset"]["learn_angle"],
            "image_size": train_cfg["dataset"]["image_size"],
            "normalize": train_cfg["dataset"]["normalize"],
        }
    )

    seed = int(train_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = train_cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = _build_model(model_cfg).to(device)

    training = train_cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = None
    if training.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training["epochs"])
        )

    start_epoch = 0
    resume = training.get("resume", "")
    if resume:
        checkpoint = load_training_checkpoint(
            resolve_path(resume, package_root()), model, optimizer, scheduler, device
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    use_test = bool(training.get("use_test", True))
    if args.use_test is not None:
        use_test = args.use_test == "true"
    train_dataset = make_dataset(train_cfg, "train")
    validation_dataset = make_dataset(train_cfg, "test") if use_test else None
    loader_args = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_args)
        if validation_dataset is not None
        else None
    )
    run_dir = resolve_path(os.path.join("runs", train_cfg["run_name"]), package_root())
    weights_dir = resolve_path("weights", package_root())
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        weights_dir=weights_dir,
        config={"model": model_cfg, "train": train_cfg},
        alpha=float(training["alpha"]),
        gradient_clip=float(training.get("gradient_clip", 0.0)),
        enable_tensorboard=bool(training.get("tensorboard", True)),
    )
    test_samples = len(validation_dataset) if validation_dataset is not None else 0
    print(
        f"device={device} train_samples={len(train_dataset)} "
        f"use_test={use_test} test_samples={test_samples}"
    )
    trainer.fit(train_loader, validation_loader, start_epoch, int(training["epochs"]))


if __name__ == "__main__":
    main()
