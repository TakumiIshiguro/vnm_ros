#!/usr/bin/env python3
import argparse
import copy
import math
import os
import random
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from vnm_ros.datasets import NoMaDDirectionDataset, ViNTDataset, ViNTDirectionDataset
from vnm_ros.models.model_loader import (
    build_model,
    load_model_weights,
    validate_checkpoint_action_scale,
)
from vnm_ros.training.checkpoint import load_training_checkpoint, save_config_yaml
from vnm_ros.training.nomad_trainer import NoMaDTrainer
from vnm_ros.training.optimizer import build_parameter_groups
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
        metric_waypoint_spacing=float(dataset["metric_waypoint_spacing"]),
        action_stats=model_cfg["action_stats"],
        normalize=bool(model_cfg.get("normalize", True)),
        center_crop=bool(model_cfg.get("image_center_crop", False)),
    )


def make_cmd_dir_loss_weights(dataset, training):
    if not bool(training.get("cmd_dir_loss_weighting", False)):
        return None, None
    if not hasattr(dataset, "sample_cmd_dir_labels"):
        raise ValueError(
            "cmd_dir_loss_weighting requires a direction-conditioned dataset"
        )
    labels = dataset.sample_cmd_dir_labels()
    class_counts = np.bincount(labels, minlength=3).astype(np.float64)
    present = class_counts > 0.0
    if np.count_nonzero(present) <= 1:
        raise ValueError(
            "cmd_dir_loss_weighting requires at least two cmd_dir classes "
            f"in the training dataset, got counts={class_counts.astype(int).tolist()}"
        )
    power = float(training.get("cmd_dir_loss_weight_power", 1.0))
    if power < 0.0:
        raise ValueError("cmd_dir_loss_weight_power must be >= 0")
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    class_weights[present] = np.power(1.0 / class_counts[present], power)
    class_weights *= len(labels) / np.sum(class_weights[labels])
    info = {
        "counts": class_counts.astype(int).tolist(),
        "weights": class_weights.tolist(),
        "power": power,
    }
    return class_weights.astype(np.float32), info


def dataset_name(data_dir, model_type=None):
    normalized = os.path.normpath(data_dir)
    directory_name = os.path.basename(normalized)
    if model_type is not None and directory_name == str(model_type):
        normalized = os.path.dirname(normalized)
        directory_name = os.path.basename(normalized)
    if directory_name in ("train", "test"):
        return os.path.basename(os.path.dirname(normalized))
    return directory_name


def architecture_name(model_cfg):
    return str(model_cfg["model_type"])


def configured_model_name(training):
    model_name = str(training.get("model_name", "")).strip()
    if model_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_name):
        raise ValueError(
            "training.model_name must start with an ASCII letter or digit and "
            "contain only letters, digits, '_', '-', and '.'"
        )
    return model_name


def named_checkpoint(model_name, checkpoint_name):
    return f"{model_name}_{checkpoint_name}" if model_name else checkpoint_name


def make_run_name(training):
    model_name = configured_model_name(training)
    model_part = f"_{model_name}" if model_name else ""
    return (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        f"{model_part}_ep{int(training['epoch']):03d}"
    )


def training_artifact_dirs(model_cfg, dataset, run_name):
    architecture = architecture_name(model_cfg)
    run_dir = resolve_path(
        os.path.join("runs", architecture, dataset, run_name), package_root()
    )
    weights_dir = resolve_path(
        os.path.join("weights", architecture, dataset), package_root()
    )
    return run_dir, weights_dir


def training_run_config(
    model_cfg,
    train_cfg,
    run_name,
    dataset,
    train_data_dir,
    run_dir,
    weights_dir,
    device,
    train_samples,
    test_samples,
    use_test,
    cmd_dir_loss_weight_info,
):
    effective_train_cfg = copy.deepcopy(train_cfg)
    effective_training_cfg = effective_train_cfg["training"]
    effective_training_cfg["use_test"] = bool(use_test)
    effective_training_cfg["cmd_dir_loss_weight_info"] = cmd_dir_loss_weight_info
    return {
        "model": copy.deepcopy(model_cfg),
        "train": effective_train_cfg,
        "artifacts": {
            "run_name": run_name,
            "model_name": configured_model_name(effective_training_cfg),
            "dataset_name": dataset,
            "train_data_dir": train_data_dir,
            "run_dir": run_dir,
            "weights_dir": weights_dir,
            "device": str(device),
            "train_samples": int(train_samples),
            "test_samples": int(test_samples),
        },
    }


def configured_epochs(training):
    epoch_values = training.get("epoch")
    if not isinstance(epoch_values, (list, tuple)) or not epoch_values:
        raise ValueError("training.epoch must be a non-empty list of epoch counts")

    epochs = []
    for epoch_count in epoch_values:
        if (
            isinstance(epoch_count, bool)
            or not isinstance(epoch_count, int)
            or epoch_count <= 0
        ):
            raise ValueError("training.epoch must contain positive integer epoch counts")
        epochs.append(epoch_count)
    return epochs


def train_cfg_for_epoch_count(train_cfg, epoch_count):
    run_train_cfg = copy.deepcopy(train_cfg)
    training = run_train_cfg["training"]
    training["epoch"] = int(epoch_count)
    model_name = configured_model_name(training)
    training["best_checkpoint_name"] = named_checkpoint(model_name, "best.pth")
    training["final_checkpoint_name"] = named_checkpoint(
        model_name, f"epoch{int(epoch_count):03d}.pth"
    )
    return run_train_cfg


def run_training_jobs(train_fn, args, train_cfg, model_cfg):
    if args.list_freeze_layers:
        train_fn(args, train_cfg, dict(model_cfg))
        return

    training = train_cfg["training"]
    epoch_counts = configured_epochs(training)
    if training.get("resume", "") and len(epoch_counts) > 1:
        raise ValueError("resume can only be used with one training.epoch value")

    for epoch_count in epoch_counts:
        run_train_cfg = train_cfg_for_epoch_count(train_cfg, epoch_count)
        print(f"starting training run epoch={epoch_count}")
        train_fn(args, run_train_cfg, dict(model_cfg))


def freeze_named_layers(model, layer_names):
    if not layer_names:
        return [], []
    modules = dict(model.named_modules())
    frozen_modules = []
    frozen_names = []
    seen_modules = set()
    for layer_name in layer_names:
        layer_name = str(layer_name).strip()
        if not layer_name:
            continue
        module = modules.get(layer_name)
        if module is None:
            matches = [
                name for name, _ in model.named_parameters()
                if name == layer_name or name.startswith(layer_name + ".")
            ]
            if not matches:
                available = ", ".join(sorted(name for name in modules if name)[:30])
                raise ValueError(
                    f"freeze_layers entry not found: {layer_name}. "
                    f"Use a module or parameter prefix from model.named_modules(). "
                    f"Examples: {available}"
                )
            for param_name, parameter in model.named_parameters():
                if param_name == layer_name or param_name.startswith(layer_name + "."):
                    parameter.requires_grad_(False)
            frozen_names.append(layer_name)
            continue
        module.requires_grad_(False)
        module.eval()
        if id(module) not in seen_modules:
            frozen_modules.append(module)
            seen_modules.add(id(module))
        frozen_names.append(layer_name)
    return frozen_modules, frozen_names


def frozen_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def print_freeze_layer_names(model):
    try:
        for name, module in model.named_modules():
            if name:
                print(f"{name}\t{type(module).__name__}")
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except OSError:
            pass


def build_scheduler(optimizer, training):
    scheduler_name = str(training.get("scheduler", "")).lower()
    if scheduler_name in ("", "none", "null"):
        return None

    epochs = int(training["epoch"])
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
    parser.add_argument("--list-freeze-layers", action="store_true")
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
        run_training_jobs(train_nomad_direction, args, train_cfg, model_cfg)
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
    run_training_jobs(train_vint_direction, args, train_cfg, model_cfg)
    return


def train_vint_direction(args, train_cfg, model_cfg):
    seed = int(train_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = train_cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = build_model(model_cfg).to(device)
    if args.list_freeze_layers:
        print_freeze_layer_names(model)
        return

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
    frozen_layer_names = []
    layer_modules, layer_names = freeze_named_layers(
        model, training.get("freeze_layers", [])
    )
    frozen_modules.extend(layer_modules)
    frozen_layer_names.extend(layer_names)

    parameter_groups, learning_rate_groups = build_parameter_groups(model, training)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameter_groups,
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
    cmd_dir_loss_weights, cmd_dir_loss_weight_info = make_cmd_dir_loss_weights(
        train_dataset, training
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_args)
        if validation_dataset is not None
        else None
    )
    run_name = make_run_name(training)
    train_data_dir = model_dataset_dir(
        train_cfg["dataset"], "train", model_cfg["model_type"]
    )
    dataset = dataset_name(train_data_dir, model_cfg["model_type"])
    run_dir, weights_dir = training_artifact_dirs(model_cfg, dataset, run_name)
    test_samples = len(validation_dataset) if validation_dataset is not None else 0
    checkpoint_config = training_run_config(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        run_name=run_name,
        dataset=dataset,
        train_data_dir=train_data_dir,
        run_dir=run_dir,
        weights_dir=weights_dir,
        device=device,
        train_samples=len(train_dataset),
        test_samples=test_samples,
        use_test=use_test,
        cmd_dir_loss_weight_info=cmd_dir_loss_weight_info,
    )
    save_config_yaml(os.path.join(run_dir, "training_config.yaml"), checkpoint_config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        weights_dir=weights_dir,
        config=checkpoint_config,
        alpha=float(training["alpha"]),
        gradient_clip=float(training.get("gradient_clip", 0.0)),
        enable_tensorboard=bool(training.get("tensorboard", True)),
        frozen_modules=frozen_modules,
        cmd_dir_loss_weights=cmd_dir_loss_weights,
    )
    print(
        f"run_name={run_name} architecture={architecture_name(model_cfg)} "
        f"model_name={configured_model_name(training) or '(default)'} "
        f"run_dir={run_dir} weights_dir={weights_dir} "
        f"dataset_name={dataset} "
        f"train_data_dir={train_data_dir} "
        f"device={device} train_samples={len(train_dataset)} "
        f"use_test={use_test} test_samples={test_samples} "
        f"direction_conditioning={bool(model_cfg.get('direction_conditioning', False))} "
        f"relabelled_mixed_cmd_dir_samples="
        f"{getattr(train_dataset, 'relabelled_mixed_cmd_dir_samples', 0)} "
        f"skipped_tied_cmd_dir_samples="
        f"{getattr(train_dataset, 'skipped_tied_cmd_dir_samples', 0)} "
        f"cmd_dir_loss_weighting={cmd_dir_loss_weights is not None} "
        f"cmd_dir_loss_weight_info={cmd_dir_loss_weight_info} "
        f"freeze_layers={frozen_layer_names} "
        f"learning_rate_groups={learning_rate_groups} "
        f"frozen_parameters={frozen_parameter_count(model)} "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters)}"
    )
    if validation_dataset is not None:
        test_data_dir = model_dataset_dir(
            train_cfg["dataset"], "test", model_cfg["model_type"]
        )
        print(
            f"test_dataset_name={dataset_name(test_data_dir, model_cfg['model_type'])} "
            f"test_data_dir={test_data_dir}"
        )
    trainer.fit(train_loader, validation_loader, start_epoch, int(training["epoch"]))


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
    if args.list_freeze_layers:
        print_freeze_layer_names(model)
        return

    training = train_cfg["training"]
    resume = training.get("resume", "")
    pretrained_weights_path = training.get("pretrained_weights_path", "")
    if pretrained_weights_path and not resume:
        pretrained_weights_path = resolve_path(pretrained_weights_path, package_root())
        if not os.path.isfile(pretrained_weights_path):
            raise FileNotFoundError(pretrained_weights_path)
        checkpoint = load_model_weights(
            model, pretrained_weights_path, device, strict=False
        )
        validate_checkpoint_action_scale(
            checkpoint,
            model_cfg,
            pretrained_weights_path,
        )
        print(f"loaded pretrained NoMaD model from {pretrained_weights_path}")

    frozen_modules = []
    frozen_layer_names = []
    if bool(training.get("freeze_dist_pred_net", True)):
        model.dist_pred_net.requires_grad_(False)
        model.dist_pred_net.eval()
        frozen_modules.append(model.dist_pred_net)
        frozen_layer_names.append("dist_pred_net")
    layer_modules, layer_names = freeze_named_layers(
        model, training.get("freeze_layers", [])
    )
    frozen_modules.extend(layer_modules)
    frozen_layer_names.extend(layer_names)

    parameter_groups, learning_rate_groups = build_parameter_groups(model, training)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameter_groups,
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
            resume_path,
            model,
            optimizer,
            scheduler,
            device,
            expected_model_config=model_cfg,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"resumed NoMaD training from {resume_path} at epoch {start_epoch}")

    try:
        from diffusers.training_utils import EMAModel
    except ImportError as exc:
        raise ImportError("NoMaD direction fine-tuning requires diffusers.") from exc
    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,
        use_ema_warmup=True,
        power=0.75,
    )
    if resume and checkpoint.get("ema_state_dict"):
        ema_model.load_state_dict(checkpoint["ema_state_dict"])

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
    cmd_dir_loss_weights, cmd_dir_loss_weight_info = make_cmd_dir_loss_weights(
        train_dataset, training
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_args)
        if validation_dataset is not None
        else None
    )
    run_name = make_run_name(training)
    train_data_dir = model_dataset_dir(
        train_cfg["dataset"], "train", model_cfg["model_type"]
    )
    dataset = dataset_name(train_data_dir, model_cfg["model_type"])
    run_dir, weights_dir = training_artifact_dirs(model_cfg, dataset, run_name)
    test_samples = len(validation_dataset) if validation_dataset is not None else 0
    checkpoint_config = training_run_config(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        run_name=run_name,
        dataset=dataset,
        train_data_dir=train_data_dir,
        run_dir=run_dir,
        weights_dir=weights_dir,
        device=device,
        train_samples=len(train_dataset),
        test_samples=test_samples,
        use_test=use_test,
        cmd_dir_loss_weight_info=cmd_dir_loss_weight_info,
    )
    save_config_yaml(os.path.join(run_dir, "training_config.yaml"), checkpoint_config)
    trainer = NoMaDTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        device=device,
        run_dir=run_dir,
        weights_dir=weights_dir,
        config=checkpoint_config,
        gradient_clip=float(training.get("gradient_clip", 0.0)),
        enable_tensorboard=bool(training.get("tensorboard", True)),
        frozen_modules=frozen_modules,
        cmd_dir_loss_weights=cmd_dir_loss_weights,
        ema_model=ema_model,
    )
    print(
        f"run_name={run_name} architecture={architecture_name(model_cfg)} "
        f"model_name={configured_model_name(training) or '(default)'} "
        f"run_dir={run_dir} weights_dir={weights_dir} "
        f"dataset_name={dataset} "
        f"train_data_dir={train_data_dir} "
        f"device={device} train_samples={len(train_dataset)} "
        f"action_normalize={train_dataset.normalize} "
        f"metric_waypoint_spacing={train_dataset.metric_waypoint_spacing} "
        f"relabelled_mixed_cmd_dir_samples="
        f"{getattr(train_dataset, 'relabelled_mixed_cmd_dir_samples', 0)} "
        f"skipped_tied_cmd_dir_samples="
        f"{getattr(train_dataset, 'skipped_tied_cmd_dir_samples', 0)} "
        f"use_test={use_test} test_samples={test_samples} "
        f"test_relabelled_mixed_cmd_dir_samples="
        f"{getattr(validation_dataset, 'relabelled_mixed_cmd_dir_samples', 0) if validation_dataset is not None else 0} "
        f"test_skipped_tied_cmd_dir_samples="
        f"{getattr(validation_dataset, 'skipped_tied_cmd_dir_samples', 0) if validation_dataset is not None else 0} "
        f"cmd_dir_loss_weighting={cmd_dir_loss_weights is not None} "
        f"cmd_dir_loss_weight_info={cmd_dir_loss_weight_info} "
        f"freeze_layers={frozen_layer_names} "
        f"learning_rate_groups={learning_rate_groups} "
        f"frozen_parameters={frozen_parameter_count(model)} "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters)}"
    )
    trainer.fit(train_loader, validation_loader, start_epoch, int(training["epoch"]))


if __name__ == "__main__":
    main()
