#!/usr/bin/env python3
import argparse
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from vnm_ros.datasets import DirectionViNTDataset, ViNTDataset
from vnm_ros.models.model_loader import (
    build_model,
    freeze_image_encoders,
    load_observation_encoder_weights,
    load_model_weights,
)
from vnm_ros.training.checkpoint import load_training_checkpoint
from vnm_ros.training.trainer import Trainer
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path


def make_dataset(config, dataset_type, model_type="vint"):
    dataset = config["dataset"]
    data_dir_key = (
        "train_data_dir" if dataset_type == "train" else "test_data_dir"
    )
    data_dir = resolve_path(dataset[data_dir_key], package_root())
    common_args = dict(
        data_dir=data_dir,
        image_size=dataset["image_size"],
        context_size=int(dataset["context_size"]),
        len_traj_pred=int(dataset["len_traj_pred"]),
        waypoint_spacing=int(dataset["waypoint_spacing"]),
        metric_waypoint_spacing=float(dataset["metric_waypoint_spacing"]),
        normalize=bool(dataset["normalize"]),
        learn_angle=bool(dataset["learn_angle"]),
    )
    if model_type == "direction_vint":
        return DirectionViNTDataset(
            **common_args,
            min_action_distance=int(dataset["min_action_distance"]),
            max_action_distance=int(dataset["max_action_distance"]),
        )
    return ViNTDataset(
        **common_args,
        min_goal_distance=int(dataset["min_goal_distance"]),
        max_goal_distance=int(dataset["max_goal_distance"]),
        min_action_distance=int(dataset["min_action_distance"]),
        max_action_distance=int(dataset["max_action_distance"]),
        negative_mining=bool(dataset["negative_mining"]) if dataset_type == "train" else False,
    )


def dataset_name(data_dir):
    normalized = os.path.normpath(data_dir)
    directory_name = os.path.basename(normalized)
    if directory_name in ("train", "test"):
        return os.path.basename(os.path.dirname(normalized))
    return directory_name


def print_dataset_summary(dataset, dataset_type: str):
    trajectory_names = list(getattr(dataset, "trajectory_names", []))
    displayed_names = ",".join(trajectory_names[:5])
    if len(trajectory_names) > 5:
        displayed_names += ",..."
    print(
        f"loaded_dataset type={dataset_type} "
        f"class={type(dataset).__name__} "
        f"data_dir={dataset.data_dir} "
        f"trajectories={len(trajectory_names)}[{displayed_names}] "
        f"samples={len(dataset)}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--use-test", choices=["true", "false"], default=None)
    args, _ = parser.parse_known_args()
    cfg = load_runtime_config(args.config_dir)
    train_cfg = cfg["train"]
    model_cfg = dict(cfg["model"])
    if model_cfg["model_type"] == "nomad":
        raise NotImplementedError(
            "NoMaD diffusion training is not supported by scripts/train.py. "
            "Use this package's NoMaD support for inference with a pretrained "
            "checkpoint, or add a NoMaD-specific diffusion trainer."
        )
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
    model = build_model(model_cfg).to(device)

    training = train_cfg["training"]
    resume = training.get("resume", "")
    pretrained_checkpoint = training.get("pretrained_checkpoint", "")
    if pretrained_checkpoint and not resume:
        pretrained_path = resolve_path(pretrained_checkpoint, package_root())
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(pretrained_path)
        pretrained_load = training.get("pretrained_load", "full")
        if pretrained_load == "full":
            load_model_weights(model, pretrained_path, device, strict=True)
            print(f"loaded pretrained model from {pretrained_path}")
        elif pretrained_load == "obs_encoder":
            loaded_keys = load_observation_encoder_weights(
                model, pretrained_path, device
            )
            print(
                f"loaded {len(loaded_keys)} observation encoder tensors "
                f"from {pretrained_path}"
            )
        else:
            raise ValueError(f"Unsupported pretrained_load: {pretrained_load}")

    frozen_modules = []
    if bool(training.get("freeze_encoder", False)):
        frozen_modules = freeze_image_encoders(model)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable model parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = None
    if training.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training["epochs"])
        )

    start_epoch = 0
    if resume:
        resume_path = resolve_path(resume, package_root())
        checkpoint = load_training_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"resumed training from {resume_path} at epoch {start_epoch}")

    use_test = bool(training.get("use_test", True))
    if args.use_test is not None:
        use_test = args.use_test == "true"
    train_dataset = make_dataset(train_cfg, "train", model_cfg["model_type"])
    validation_dataset = (
        make_dataset(train_cfg, "test", model_cfg["model_type"]) if use_test else None
    )
    print_dataset_summary(train_dataset, "train")
    if validation_dataset is not None:
        print_dataset_summary(validation_dataset, "test")
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
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = resolve_path(os.path.join("runs", run_name), package_root())
    weights_dir = resolve_path("weights", package_root())
    checkpoint_train_cfg = dict(train_cfg)
    checkpoint_train_cfg["run_name"] = run_name
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        weights_dir=weights_dir,
        config={"model": model_cfg, "train": checkpoint_train_cfg},
        alpha=float(training["alpha"]),
        gradient_clip=float(training.get("gradient_clip", 0.0)),
        enable_tensorboard=bool(training.get("tensorboard", True)),
        frozen_modules=frozen_modules,
    )
    test_samples = len(validation_dataset) if validation_dataset is not None else 0
    train_data_dir = resolve_path(
        train_cfg["dataset"]["train_data_dir"], package_root()
    )
    print(
        f"run_name={run_name} run_dir={run_dir} "
        f"dataset_name={dataset_name(train_data_dir)} "
        f"train_data_dir={train_data_dir} "
        f"device={device} train_samples={len(train_dataset)} "
        f"use_test={use_test} test_samples={test_samples} "
        f"freeze_encoder={bool(frozen_modules)} "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters)}"
    )
    if validation_dataset is not None:
        test_data_dir = resolve_path(
            train_cfg["dataset"]["test_data_dir"], package_root()
        )
        print(
            f"test_dataset_name={dataset_name(test_data_dir)} "
            f"test_data_dir={test_data_dir}"
        )
    trainer.fit(train_loader, validation_loader, start_epoch, int(training["epochs"]))


if __name__ == "__main__":
    main()
