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

from vnm_ros.datasets import NoMaDDirectionDataset, ViNTDataset
from vnm_ros.datasets.dataset_utils import to_local_coords
from vnm_ros.models.model_loader import (
    build_model,
    freeze_image_encoders,
    load_model_weights,
)
from vnm_ros.training.checkpoint import load_training_checkpoint
from vnm_ros.training.nomad_trainer import NoMaDTrainer
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


def make_nomad_dataset(config, model_cfg, dataset_type):
    dataset = config["dataset"]
    data_dir_key = (
        "train_data_dir" if dataset_type == "train" else "test_data_dir"
    )
    return NoMaDDirectionDataset(
        data_dir=resolve_path(dataset[data_dir_key], package_root()),
        image_size=model_cfg["image_size"],
        context_size=int(model_cfg["context_size"]),
        len_traj_pred=int(model_cfg["len_traj_pred"]),
        waypoint_spacing=int(dataset["waypoint_spacing"]),
        action_stats=model_cfg["action_stats"],
    )


def dataset_name(data_dir):
    normalized = os.path.normpath(data_dir)
    directory_name = os.path.basename(normalized)
    if directory_name in ("train", "test"):
        return os.path.basename(os.path.dirname(normalized))
    return directory_name


def cmd_dir_name(cmd_dir):
    names = ("straight", "left", "right")
    cmd_dir = np.asarray(cmd_dir, dtype=np.float32)
    if cmd_dir.size < 3 or float(np.sum(cmd_dir[:3])) <= 0.0:
        return "none"
    return names[int(np.argmax(cmd_dir[:3]))]


def preview_indices(length, count):
    count = min(int(count), int(length))
    if count <= 0:
        return []
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def preview_indices_per_direction(dataset, count_per_direction):
    by_direction = {"straight": [], "left": [], "right": []}
    for sample_index, (name, current) in enumerate(dataset.samples):
        trajectory = dataset.trajectory(name)
        direction = cmd_dir_name(trajectory["cmd_dir"][current])
        if direction in by_direction:
            by_direction[direction].append(sample_index)

    selected = []
    for direction in ("straight", "left", "right"):
        indices = by_direction[direction]
        local_indices = preview_indices(len(indices), count_per_direction)
        selected.extend((direction, indices[index]) for index in local_indices)
    return selected


def draw_local_trajectory(draw, box, points, color):
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    margin = 28
    points = np.asarray(points, dtype=np.float32)
    max_abs_y = max(float(np.max(np.abs(points[:, 1]))), 0.1)
    max_x = max(float(np.max(points[:, 0])), 0.1)
    scale = min((width - 2 * margin) / (2.0 * max_abs_y), (height - 2 * margin) / max_x)
    origin = (x0 + width // 2, y1 - margin)

    def project(point):
        px = origin[0] - float(point[1]) * scale
        py = origin[1] - float(point[0]) * scale
        return (px, py)

    draw.rectangle(box, outline=(190, 190, 190), width=1)
    draw.line((origin[0], y0 + margin, origin[0], y1 - margin), fill=(220, 220, 220))
    draw.line((x0 + margin, origin[1], x1 - margin, origin[1]), fill=(220, 220, 220))
    projected = [project(point) for point in points]
    if len(projected) >= 2:
        draw.line(projected, fill=color, width=4, joint="curve")
    for index, point in enumerate(projected):
        radius = 5 if index == 0 else 4
        fill = (30, 140, 70) if index == 0 else color
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=fill,
        )


def save_nomad_training_previews(dataset, output_dir, count_per_direction=10):
    from PIL import Image, ImageDraw

    os.makedirs(output_dir, exist_ok=True)
    colors = {
        "straight": (40, 150, 80),
        "left": (60, 110, 220),
        "right": (220, 80, 70),
        "none": (120, 120, 120),
    }
    written = []
    selected_samples = preview_indices_per_direction(dataset, count_per_direction)
    direction_counts = {"straight": 0, "left": 0, "right": 0}
    for preview_id, (_, sample_index) in enumerate(selected_samples):
        name, current = dataset.samples[sample_index]
        trajectory = dataset.trajectory(name)
        action_indices = current + np.arange(dataset.len_traj_pred + 1) * dataset.waypoint_spacing
        positions = trajectory["position"][action_indices]
        yaw = float(trajectory["yaw"][current])
        local_positions = to_local_coords(positions, positions[0], yaw)
        cmd_name = cmd_dir_name(trajectory["cmd_dir"][current])

        context_indices = current + np.arange(-dataset.context_size, 1) * dataset.waypoint_spacing
        context_images = []
        for context_index in context_indices:
            with Image.open(dataset.image_path(name, int(context_index))) as source:
                context_images.append(source.convert("RGB").resize((128, 128)))

        canvas_width = 64 + len(context_images) * 144 + 328
        canvas = Image.new("RGB", (canvas_width, 320), (250, 250, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), f"{preview_id:02d} {name} idx={current} cmd={cmd_name}", fill=(20, 20, 20))
        for image_id, image in enumerate(context_images):
            x = 24 + image_id * 144
            canvas.paste(image, (x, 64))
            frame_color = (30, 140, 70) if image_id == len(context_images) - 1 else (170, 170, 170)
            draw.rectangle((x, 64, x + 128, 192), outline=frame_color, width=3)
            label = "current" if image_id == len(context_images) - 1 else f"t-{len(context_images) - 1 - image_id}"
            draw.text((x, 42), f"{label} idx={int(context_indices[image_id])}", fill=(20, 20, 20))

        traj_x0 = 48 + len(context_images) * 144
        draw.text((traj_x0, 18), "teacher trajectory in robot frame", fill=(20, 20, 20))
        draw_local_trajectory(
            draw,
            (traj_x0, 48, traj_x0 + 304, 304),
            local_positions,
            colors.get(cmd_name, colors["none"]),
        )
        direction_id = direction_counts.get(cmd_name, 0)
        direction_counts[cmd_name] = direction_id + 1
        output_path = os.path.join(
            output_dir,
            f"{cmd_name}_{direction_id:02d}_sample_{preview_id:02d}.png",
        )
        canvas.save(output_path)
        written.append(output_path)
    return written


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
        load_model_weights(model, pretrained_weights_path, device, strict=True)
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
    scheduler = None
    if training.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training["epochs"])
        )

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
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_args)
        if validation_dataset is not None
        else None
    )
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = resolve_path(os.path.join("runs", run_name), package_root())
    weights_dir = resolve_path("weights", package_root())
    preview_count = int(
        training.get(
            "plot_training_samples_per_direction",
            training.get("plot_training_samples", 10),
        )
    )
    preview_paths = save_nomad_training_previews(
        train_dataset,
        os.path.join(run_dir, "training_samples"),
        count_per_direction=preview_count,
    )
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
    train_data_dir = resolve_path(
        train_cfg["dataset"]["train_data_dir"], package_root()
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
        f"freeze_encoder={bool(training.get('freeze_encoder', True))} "
        f"trainable_parameters={sum(p.numel() for p in trainable_parameters)} "
        f"training_sample_plots={len(preview_paths)}"
    )
    trainer.fit(train_loader, validation_loader, start_epoch, int(training["epochs"]))


if __name__ == "__main__":
    main()
