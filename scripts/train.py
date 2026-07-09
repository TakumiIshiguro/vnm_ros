#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from vnm_ros.datasets import NoMaDDirectionDataset, ViNTDataset, ViNTDirectionDataset
from vnm_ros.models.model_loader import (
    build_model,
    freeze_image_encoders,
    load_model_weights,
)
from vnm_ros.training.checkpoint import load_training_checkpoint
from vnm_ros.training.nomad_trainer import NoMaDTrainer
from vnm_ros.training.trainer import Trainer
from vnm_ros.utils.config import load_runtime_config, model_dataset_dir, package_root, resolve_path


def make_dataset(config, model_cfg, dataset_type):
    dataset = config["dataset"]
    data_dir = model_dataset_dir(dataset, dataset_type, model_cfg["model_type"])
    if bool(model_cfg.get("direction_conditioning", False)):
        return ViNTDirectionDataset(
            data_dir=data_dir,
            image_size=dataset["image_size"],
            context_size=int(dataset["context_size"]),
            len_traj_pred=int(dataset["len_traj_pred"]),
            waypoint_spacing=int(dataset["waypoint_spacing"]),
            metric_waypoint_spacing=float(dataset["metric_waypoint_spacing"]),
            normalize=bool(dataset["normalize"]),
            learn_angle=bool(dataset["learn_angle"]),
            center_crop=bool(model_cfg.get("image_center_crop", False)),
        )
    return ViNTDataset(
        data_dir=data_dir,
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
        center_crop=bool(model_cfg.get("image_center_crop", False)),
    )


def make_nomad_dataset(config, model_cfg, dataset_type):
    dataset = config["dataset"]
    data_dir = model_dataset_dir(dataset, dataset_type, model_cfg["model_type"])
    return NoMaDDirectionDataset(
        data_dir=data_dir,
        image_size=model_cfg["image_size"],
        context_size=int(model_cfg["context_size"]),
        len_traj_pred=int(model_cfg["len_traj_pred"]),
        waypoint_spacing=int(dataset["waypoint_spacing"]),
        action_stats=model_cfg["action_stats"],
        center_crop=bool(model_cfg.get("image_center_crop", False)),
    )


def make_cmd_dir_sampler(dataset, training):
    if not bool(training.get("balance_cmd_dir_sampling", False)):
        return None, None
    if not hasattr(dataset, "sample_cmd_dir_labels"):
        raise ValueError(
            "balance_cmd_dir_sampling requires a direction-conditioned dataset"
        )
    labels = dataset.sample_cmd_dir_labels()
    class_counts = np.bincount(labels, minlength=3).astype(np.float64)
    present = class_counts > 0.0
    if np.count_nonzero(present) <= 1:
        raise ValueError(
            "balance_cmd_dir_sampling requires at least two cmd_dir classes "
            f"in the training dataset, got counts={class_counts.astype(int).tolist()}"
        )
    power = float(training.get("cmd_dir_sampling_power", 0.5))
    if power < 0.0:
        raise ValueError("cmd_dir_sampling_power must be >= 0")
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    class_weights[present] = np.power(1.0 / class_counts[present], power)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    info = {
        "counts": class_counts.astype(int).tolist(),
        "weights": class_weights.tolist(),
        "power": power,
    }
    return sampler, info


def dataset_name(data_dir):
    normalized = os.path.normpath(data_dir)
    directory_name = os.path.basename(normalized)
    if directory_name in ("train", "test"):
        return os.path.basename(os.path.dirname(normalized))
    return directory_name


def build_scheduler(optimizer, training):
    scheduler_name = str(training.get("scheduler", "")).lower()
    if scheduler_name in ("", "none", "null"):
        return None

    epochs = int(training["epochs"])
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if scheduler_name not in ("warmup_cosine", "cosine_warmup"):
        raise ValueError(f"Unsupported scheduler: {training.get('scheduler')}")

    warmup_epochs = int(training.get("warmup_epochs", 1))
    warmup_start_factor = float(training.get("warmup_start_factor", 0.1))
    min_lr_factor = float(training.get("min_lr_factor", 0.0))
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if not 0.0 <= min_lr_factor <= 1.0:
        raise ValueError("min_lr_factor must be in [0, 1]")

    def lr_lambda(epoch):
        if epochs <= 0:
            return 1.0
        if warmup_epochs > 0 and epoch < warmup_epochs:
            if warmup_epochs == 1:
                return warmup_start_factor
            progress = epoch / float(warmup_epochs - 1)
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress

        cosine_epochs = max(1, epochs - warmup_epochs)
        progress = min(max((epoch - warmup_epochs) / float(cosine_epochs), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--use-test", choices=["true", "false"], default=None)
    args, unknown_args = parser.parse_known_args()
    unexpected_args = [
        arg for arg in unknown_args if not arg.startswith("__") and ":=" not in arg
    ]
    if unexpected_args:
        parser.error(f"unrecognized arguments: {' '.join(unexpected_args)}")
    cfg = load_runtime_config(args.config_dir)
    train_cfg = cfg["train"]
    model_cfg = dict(cfg["model"])
    if model_cfg["model_type"] == "nomad":
        model_cfg["direction_conditioning"] = True
        train_nomad_direction(args, train_cfg, model_cfg)
        return
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
    pretrained_weights_path = training.get("pretrained_weights_path", "")
    if pretrained_weights_path and not resume:
        pretrained_weights_path = resolve_path(pretrained_weights_path, package_root())
        if not os.path.isfile(pretrained_weights_path):
            raise FileNotFoundError(pretrained_weights_path)
        load_model_weights(
            model,
            pretrained_weights_path,
            device,
            strict=not bool(model_cfg.get("direction_conditioning", False)),
        )
        print(f"loaded pretrained model from {pretrained_weights_path}")

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
    scheduler = build_scheduler(optimizer, training)

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
    train_dataset = make_dataset(train_cfg, model_cfg, "train")
    validation_dataset = make_dataset(train_cfg, model_cfg, "test") if use_test else None
    loader_args = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    cmd_dir_sampler, cmd_dir_sampler_info = make_cmd_dir_sampler(
        train_dataset, training
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=cmd_dir_sampler is None,
        sampler=cmd_dir_sampler,
        **loader_args,
    )
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
    train_data_dir = model_dataset_dir(
        train_cfg["dataset"], "train", model_cfg["model_type"]
    )
    print(
        f"run_name={run_name} run_dir={run_dir} "
        f"dataset_name={dataset_name(train_data_dir)} "
        f"train_data_dir={train_data_dir} "
        f"device={device} train_samples={len(train_dataset)} "
        f"use_test={use_test} test_samples={test_samples} "
        f"direction_conditioning={bool(model_cfg.get('direction_conditioning', False))} "
        f"skipped_mixed_cmd_dir_samples="
        f"{getattr(train_dataset, 'skipped_mixed_cmd_dir_samples', 0)} "
        f"balance_cmd_dir_sampling={cmd_dir_sampler is not None} "
        f"cmd_dir_sampler_info={cmd_dir_sampler_info} "
        f"freeze_encoder={bool(frozen_modules)} "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters)}"
    )
    if validation_dataset is not None:
        test_data_dir = model_dataset_dir(
            train_cfg["dataset"], "test", model_cfg["model_type"]
        )
        print(
            f"test_dataset_name={dataset_name(test_data_dir)} "
            f"test_data_dir={test_data_dir}"
        )
    trainer.fit(train_loader, validation_loader, start_epoch, int(training["epochs"]))


def train_nomad_direction(args, train_cfg, model_cfg):
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
    pretrained_weights_path = training.get("pretrained_weights_path", "")
    if pretrained_weights_path and not resume:
        pretrained_weights_path = resolve_path(pretrained_weights_path, package_root())
        if not os.path.isfile(pretrained_weights_path):
            raise FileNotFoundError(pretrained_weights_path)
        load_model_weights(model, pretrained_weights_path, device, strict=False)
        print(f"loaded pretrained NoMaD model from {pretrained_weights_path}")

    frozen_modules = []
    if bool(training.get("freeze_encoder", True)):
        model.vision_encoder.requires_grad_(False)
        model.vision_encoder.eval()
        frozen_modules.append(model.vision_encoder)
    if bool(training.get("freeze_dist_pred_net", True)):
        model.dist_pred_net.requires_grad_(False)
        model.dist_pred_net.eval()
        frozen_modules.append(model.dist_pred_net)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable model parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, training)

    try:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError as exc:
        raise ImportError("NoMaD direction fine-tuning requires diffusers.") from exc
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(model_cfg.get("num_diffusion_iters", 10)),
        beta_schedule=model_cfg.get("beta_schedule", "squaredcos_cap_v2"),
        clip_sample=bool(model_cfg.get("clip_sample", True)),
        prediction_type=model_cfg.get("prediction_type", "epsilon"),
    )

    start_epoch = 0
    if resume:
        resume_path = resolve_path(resume, package_root())
        checkpoint = load_training_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"resumed NoMaD training from {resume_path} at epoch {start_epoch}")

    use_test = bool(training.get("use_test", True))
    if args.use_test is not None:
        use_test = args.use_test == "true"
    train_dataset = make_nomad_dataset(train_cfg, model_cfg, "train")
    validation_dataset = make_nomad_dataset(train_cfg, model_cfg, "test") if use_test else None
    loader_args = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    cmd_dir_sampler, cmd_dir_sampler_info = make_cmd_dir_sampler(
        train_dataset, training
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=cmd_dir_sampler is None,
        sampler=cmd_dir_sampler,
        **loader_args,
    )
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
    trainer = NoMaDTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        device=device,
        run_dir=run_dir,
        weights_dir=weights_dir,
        config={"model": model_cfg, "train": checkpoint_train_cfg},
        gradient_clip=float(training.get("gradient_clip", 0.0)),
        enable_tensorboard=bool(training.get("tensorboard", True)),
        frozen_modules=frozen_modules,
    )
    test_samples = len(validation_dataset) if validation_dataset is not None else 0
    train_data_dir = model_dataset_dir(
        train_cfg["dataset"], "train", model_cfg["model_type"]
    )
    print(
        f"run_name={run_name} run_dir={run_dir} "
        f"dataset_name={dataset_name(train_data_dir)} "
        f"train_data_dir={train_data_dir} "
        f"device={device} train_samples={len(train_dataset)} "
        f"skipped_mixed_cmd_dir_samples="
        f"{getattr(train_dataset, 'skipped_mixed_cmd_dir_samples', 0)} "
        f"use_test={use_test} test_samples={test_samples} "
        f"test_skipped_mixed_cmd_dir_samples="
        f"{getattr(validation_dataset, 'skipped_mixed_cmd_dir_samples', 0) if validation_dataset is not None else 0} "
        f"balance_cmd_dir_sampling={cmd_dir_sampler is not None} "
        f"cmd_dir_sampler_info={cmd_dir_sampler_info} "
        f"freeze_encoder={bool(training.get('freeze_encoder', True))} "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters)}"
    )
    trainer.fit(train_loader, validation_loader, start_epoch, int(training["epochs"]))


if __name__ == "__main__":
    main()
