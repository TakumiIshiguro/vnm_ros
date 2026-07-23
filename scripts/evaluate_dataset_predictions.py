#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from PIL import Image, ImageDraw

from vnm_ros.control.cmd_dir_action_selector import CmdDirActionSelector
from vnm_ros.datasets import NoMaDDirectionDataset, ViNTDirectionDataset
from vnm_ros.datasets.dataset_utils import to_local_coords
from vnm_ros.datasets.trajectory_dataset import discover_trajectory_names
from vnm_ros.models.vnm_model import VNMModel
from vnm_ros.utils.config import (
    load_runtime_config,
    model_dataset_dir,
    package_root,
    resolve_path,
)


DIRECTION_NAMES = ("straight", "left", "right")
COMMAND_COLORS = {
    "straight": (215, 45, 45),
    "left": (40, 105, 210),
    "right": (225, 135, 20),
}


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value}")


def safe_name(value):
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return result or "unknown"


def checkpoint_name(path):
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return safe_name(f"{parent}_{stem}" if parent else stem)


def make_dataset(cfg, data_dir, trajectory_names):
    model_cfg = cfg["model"]
    dataset_cfg = cfg["train"]["dataset"]
    if model_cfg["model_type"] == "nomad":
        return NoMaDDirectionDataset(
            data_dir=data_dir,
            image_size=model_cfg["image_size"],
            context_size=int(model_cfg["context_size"]),
            len_traj_pred=int(model_cfg["len_traj_pred"]),
            waypoint_spacing=int(dataset_cfg["waypoint_spacing"]),
            metric_waypoint_spacing=float(dataset_cfg["metric_waypoint_spacing"]),
            action_stats=model_cfg["action_stats"],
            normalize=bool(model_cfg.get("normalize", True)),
            center_crop=bool(model_cfg.get("image_center_crop", False)),
            trajectory_names=trajectory_names,
        )
    if not bool(model_cfg.get("direction_conditioning", False)):
        raise ValueError(
            f"{model_cfg['model_type']} dataset evaluation without a goal image "
            "requires direction_conditioning=true"
        )
    return ViNTDirectionDataset(
        data_dir=data_dir,
        image_size=dataset_cfg["image_size"],
        context_size=int(dataset_cfg["context_size"]),
        len_traj_pred=int(dataset_cfg["len_traj_pred"]),
        waypoint_spacing=int(dataset_cfg["waypoint_spacing"]),
        metric_waypoint_spacing=float(dataset_cfg["metric_waypoint_spacing"]),
        normalize=bool(dataset_cfg["normalize"]),
        learn_angle=bool(dataset_cfg["learn_angle"]),
        center_crop=bool(model_cfg.get("image_center_crop", False)),
        trajectory_names=trajectory_names,
    )


def command_name(cmd_dir):
    cmd_dir = np.asarray(cmd_dir[:3])
    if cmd_dir.size < 3 or np.sum(cmd_dir) <= 0.0:
        return "straight"
    return DIRECTION_NAMES[int(np.argmax(cmd_dir))]


def command_vector(direction):
    command = np.zeros(3, dtype=np.float32)
    command[DIRECTION_NAMES.index(direction)] = 1.0
    return command


def selected_sample_indices(
    dataset, count_per_direction, explicit_indices, sample_seed=0
):
    if explicit_indices:
        result = []
        for index in explicit_indices:
            if index < 0 or index >= len(dataset):
                raise IndexError(f"sample index {index} is outside [0, {len(dataset)})")
            result.append(index)
        return result

    labels = dataset.sample_cmd_dir_labels()
    rng = np.random.default_rng(int(sample_seed))
    result = []
    for label in range(3):
        indices = np.flatnonzero(labels == label)
        if count_per_direction < 0:
            result.extend(indices.tolist())
            continue
        count = min(int(count_per_direction), len(indices))
        if count > 0:
            selected = rng.choice(indices, size=count, replace=False)
            result.extend(np.sort(selected).tolist())
    return result


def teacher_trajectory(dataset, sample_index):
    name, current = dataset.samples[sample_index]
    trajectory = dataset.trajectory(name)
    indices = current + np.arange(dataset.len_traj_pred + 1) * dataset.waypoint_spacing
    positions = trajectory["position"][indices]
    local = to_local_coords(positions, positions[0], float(trajectory["yaw"][current]))
    cmd_dir = dataset.sample_cmd_dir(sample_index)
    return name, int(current), cmd_dir, local[:, :2]


def context_images(dataset, name, current):
    indices = current + np.arange(-dataset.context_size, 1) * dataset.waypoint_spacing
    images = []
    for index in indices:
        with Image.open(dataset.image_path(name, int(index))) as image:
            images.append(image.convert("RGB").copy())
    return indices, images


def predictions_in_dataset_units(model, model_cfg, dataset_cfg, images, cmd_dir):
    predictions = np.asarray(model.predict_explore(images, cmd_dir=cmd_dir))
    if predictions.ndim == 2:
        predictions = predictions[np.newaxis, ...]
    if predictions.ndim != 3 or predictions.shape[-1] < 2:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    predictions = predictions[..., :2].astype(np.float32)
    if bool(model_cfg.get("normalize", True)):
        scale = float(dataset_cfg["metric_waypoint_spacing"]) * int(
            dataset_cfg["waypoint_spacing"]
        )
        predictions *= scale
    return predictions


def choose_prediction(predictions, cmd_dir, strategy, threshold_deg, waypoint_index):
    if strategy == "mean":
        return predictions.mean(axis=0), -1, None, None
    if strategy == "first":
        return predictions[0], 0, None, None
    if strategy != "cmd_dir":
        raise ValueError(f"Unsupported prediction strategy: {strategy}")
    selector = CmdDirActionSelector(theta_threshold_deg=threshold_deg)
    selector.update(cmd_dir)
    selector.select(predictions, waypoint_index)
    return (
        selector.selected_action,
        selector.selected_sample,
        selector.selected_theta,
        selector.selected_score,
    )


def endpoint_angle(points):
    endpoint = points[-1]
    return math.degrees(math.atan2(float(endpoint[1]), float(endpoint[0])))


def angle_difference_deg(a, b):
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def evaluation_metrics(teacher, prediction):
    target = teacher[1 : 1 + len(prediction)]
    if len(target) != len(prediction):
        raise ValueError(
            f"Teacher horizon {len(target)} does not match prediction {len(prediction)}"
        )
    errors = np.linalg.norm(prediction - target, axis=1)
    teacher_angle = endpoint_angle(target)
    prediction_angle = endpoint_angle(prediction)
    return {
        "ade_m": float(np.mean(errors)),
        "fde_m": float(errors[-1]),
        "teacher_endpoint_angle_deg": teacher_angle,
        "prediction_endpoint_angle_deg": prediction_angle,
        "endpoint_angle_error_deg": angle_difference_deg(prediction_angle, teacher_angle),
    }


def resized(image, height):
    height = int(height)
    width = max(1, int(round(image.width * height / image.height)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def prepare_image_panel(images, context_indices, context_height):
    if not images or len(images) != len(context_indices):
        raise ValueError("images and context_indices must have the same non-zero length")

    history = list(zip(images[:-1], context_indices[:-1]))[-3:]
    context_previews = [
        (resized(image, context_height), int(index))
        for image, index in history
    ]
    observation_height = max(int(context_height) + 1, int(context_height) * 2)
    observation = resized(images[-1], observation_height)
    gap = 12
    context_width = max(
        (image.width for image, _ in context_previews),
        default=0,
    )
    context_stack_height = (
        sum(image.height for image, _ in context_previews)
        + gap * max(0, len(context_previews) - 1)
    )
    panel_height = max(context_stack_height, observation.height)
    observation_x = context_width + (gap if context_previews else 0)
    return {
        "contexts": context_previews,
        "observation": observation,
        "observation_index": int(context_indices[-1]),
        "gap": gap,
        "context_width": context_width,
        "observation_x": observation_x,
        "width": observation_x + observation.width,
        "height": panel_height,
    }


def draw_image_panel(canvas, draw, panel, origin):
    origin_x, origin_y = origin
    contexts = panel["contexts"]
    y = origin_y
    for offset, (image, index) in enumerate(contexts):
        x = origin_x + (panel["context_width"] - image.width) // 2
        canvas.paste(image, (x, y))
        draw.rectangle(
            (x, y, x + image.width - 1, y + image.height - 1),
            outline=(170, 170, 170),
            width=3,
        )
        label = f"t-{len(contexts) - offset} idx={index}"
        draw.rectangle((x + 5, y + 5, x + 105, y + 24), fill=(0, 0, 0))
        draw.text((x + 9, y + 7), label, fill=(255, 255, 255))
        y += image.height + panel["gap"]

    observation = panel["observation"]
    observation_x = origin_x + panel["observation_x"]
    observation_y = origin_y + (panel["height"] - observation.height) // 2
    canvas.paste(observation, (observation_x, observation_y))
    draw.rectangle(
        (
            observation_x,
            observation_y,
            observation_x + observation.width - 1,
            observation_y + observation.height - 1,
        ),
        outline=(30, 140, 70),
        width=4,
    )
    label = f"current idx={panel['observation_index']}"
    draw.rectangle(
        (observation_x + 7, observation_y + 7, observation_x + 127, observation_y + 28),
        fill=(0, 0, 0),
    )
    draw.text(
        (observation_x + 11, observation_y + 9),
        label,
        fill=(255, 255, 255),
    )


def draw_path_plot(draw, box, teacher, predictions, selected):
    x0, y0, x1, y1 = box
    margin = 34
    paths = [teacher] + [np.vstack((np.zeros((1, 2)), path)) for path in predictions]
    selected_path = np.vstack((np.zeros((1, 2)), selected))
    paths.append(selected_path)
    all_points = np.concatenate(paths, axis=0)
    min_x = min(-0.05, float(np.min(all_points[:, 0])))
    max_x = max(0.1, float(np.max(all_points[:, 0])))
    max_y = max(0.1, float(np.max(np.abs(all_points[:, 1]))))
    scale = min(
        (x1 - x0 - 2 * margin) / (2.0 * max_y),
        (y1 - y0 - 2 * margin) / (max_x - min_x),
    )
    origin_x = (x0 + x1) / 2.0
    bottom = y1 - margin

    def point(value):
        return (
            origin_x - float(value[1]) * scale,
            bottom - (float(value[0]) - min_x) * scale,
        )

    draw.rectangle(box, outline=(170, 170, 170), width=1)
    grid_spacing = 0.1
    for forward in np.arange(
        math.floor(min_x / grid_spacing) * grid_spacing,
        max_x + grid_spacing,
        grid_spacing,
    ):
        left = point((forward, -max_y))
        right = point((forward, max_y))
        draw.line((left, right), fill=(228, 228, 228), width=1)
    for lateral in np.arange(-max_y, max_y + grid_spacing, grid_spacing):
        near = point((min_x, lateral))
        far = point((max_x, lateral))
        draw.line((near, far), fill=(228, 228, 228), width=1)

    for prediction in predictions:
        path = np.vstack((np.zeros((1, 2)), prediction))
        draw.line([point(value) for value in path], fill=(130, 190, 235), width=2)
    draw.line([point(value) for value in teacher], fill=(20, 20, 20), width=6)
    draw.line([point(value) for value in teacher], fill=(40, 185, 90), width=3)
    draw.line([point(value) for value in selected_path], fill=(210, 45, 45), width=4)
    for value in selected_path[1:]:
        px, py = point(value)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(210, 45, 45))
    px, py = point((0.0, 0.0))
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(20, 20, 20))


def draw_command_comparison_plot(draw, box, teacher, selected_by_command):
    x0, y0, x1, y1 = box
    margin = 38
    paths = [teacher] + [
        np.vstack((np.zeros((1, 2)), selected))
        for selected in selected_by_command.values()
    ]
    all_points = np.concatenate(paths, axis=0)
    min_x = min(-0.05, float(np.min(all_points[:, 0])))
    max_x = max(0.1, float(np.max(all_points[:, 0])))
    max_y = max(0.1, float(np.max(np.abs(all_points[:, 1]))))
    scale = min(
        (x1 - x0 - 2 * margin) / (2.0 * max_y),
        (y1 - y0 - 2 * margin) / (max_x - min_x),
    )
    origin_x = (x0 + x1) / 2.0
    bottom = y1 - margin

    def point(value):
        return (
            origin_x - float(value[1]) * scale,
            bottom - (float(value[0]) - min_x) * scale,
        )

    draw.rectangle(box, outline=(170, 170, 170), width=1)
    grid_spacing = 0.1
    for forward in np.arange(
        math.floor(min_x / grid_spacing) * grid_spacing,
        max_x + grid_spacing,
        grid_spacing,
    ):
        draw.line(
            (point((forward, -max_y)), point((forward, max_y))),
            fill=(228, 228, 228),
            width=1,
        )
    for lateral in np.arange(-max_y, max_y + grid_spacing, grid_spacing):
        draw.line(
            (point((min_x, lateral)), point((max_x, lateral))),
            fill=(228, 228, 228),
            width=1,
        )

    draw.line([point(value) for value in teacher], fill=(20, 20, 20), width=6)
    draw.line([point(value) for value in teacher], fill=(40, 185, 90), width=3)
    for direction in DIRECTION_NAMES:
        selected = selected_by_command[direction]
        path = np.vstack((np.zeros((1, 2)), selected))
        color = COMMAND_COLORS[direction]
        draw.line([point(value) for value in path], fill=color, width=4)
        for value in selected:
            px, py = point(value)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    px, py = point((0.0, 0.0))
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(20, 20, 20))


def render_preview(
    output_path,
    images,
    context_indices,
    teacher,
    predictions,
    selected,
    record,
    image_height,
):
    image_panel = prepare_image_panel(images, context_indices, image_height)
    plot_width = 440
    plot_height = 400
    canvas_width = 24 + image_panel["width"] + 24 + plot_width + 24
    canvas_height = max(
        500,
        92 + image_panel["height"] + 24,
        68 + plot_height + 24,
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"sample={record['sample_index']} {record['trajectory']} "
        f"idx={record['current_index']} cmd={record['command']}"
    )
    draw.text((24, 16), title, fill=(20, 20, 20))
    draw.text(
        (24, 40),
        f"ADE={record['ade_m']:.3f}m  FDE={record['fde_m']:.3f}m  "
        f"angle_error={record['endpoint_angle_error_deg']:+.1f}deg",
        fill=(20, 20, 20),
    )
    draw_image_panel(canvas, draw, image_panel, (24, 92))

    plot_x = 24 + image_panel["width"] + 24
    draw.text((plot_x, 16), "robot frame: x=forward, y=left", fill=(20, 20, 20))
    draw.text(
        (plot_x, 40),
        "teacher=green/black  selected=red  candidates=blue",
        fill=(20, 20, 20),
    )
    draw_path_plot(
        draw,
        (plot_x, 68, plot_x + plot_width, 68 + plot_height),
        teacher,
        predictions,
        selected,
    )
    canvas.save(output_path)


def render_command_comparison(
    output_path,
    images,
    context_indices,
    teacher,
    selected_by_command,
    sample_index,
    trajectory,
    current_index,
    teacher_command,
    image_height,
):
    image_panel = prepare_image_panel(images, context_indices, image_height)
    plot_width = 500
    plot_height = 430
    canvas_width = 24 + image_panel["width"] + 24 + plot_width + 24
    canvas_height = max(
        520,
        92 + image_panel["height"] + 24,
        76 + plot_height + 24,
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 16),
        f"sample={sample_index} {trajectory} idx={current_index} "
        f"teacher_cmd={teacher_command}",
        fill=(20, 20, 20),
    )

    draw_image_panel(canvas, draw, image_panel, (24, 92))

    plot_x = 24 + image_panel["width"] + 24
    draw.text((plot_x, 16), "same context and diffusion noise", fill=(20, 20, 20))
    legend_x = plot_x
    draw.text((legend_x, 40), "teacher", fill=(40, 160, 75))
    legend_x += 62
    for direction in DIRECTION_NAMES:
        angle = endpoint_angle(selected_by_command[direction])
        draw.text(
            (legend_x, 40),
            f"{direction} {angle:+.1f}deg",
            fill=COMMAND_COLORS[direction],
        )
        legend_x += 112
    draw_command_comparison_plot(
        draw,
        (plot_x, 68, plot_x + plot_width, 68 + plot_height),
        teacher,
        selected_by_command,
    )
    canvas.save(output_path)


def write_summary(records, output_path):
    fields = [
        "direction",
        "count",
        "ade_mean_m",
        "fde_mean_m",
        "abs_endpoint_angle_error_mean_deg",
    ]
    rows = []
    for direction in DIRECTION_NAMES + ("all",):
        selected = records if direction == "all" else [
            record for record in records if record["command"] == direction
        ]
        if not selected:
            continue
        rows.append(
            {
                "direction": direction,
                "count": len(selected),
                "ade_mean_m": float(np.mean([row["ade_m"] for row in selected])),
                "fde_mean_m": float(np.mean([row["fde_m"] for row in selected])),
                "abs_endpoint_angle_error_mean_deg": float(
                    np.mean([abs(row["endpoint_angle_error_deg"]) for row in selected])
                ),
            }
        )
    with open(output_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model predictions with dataset teacher trajectories."
    )
    parser.add_argument("--config-dir", default=os.path.join(package_root(), "config"))
    parser.add_argument("--dataset-type", choices=("train", "test"), default=None)
    parser.add_argument("--trajectory-name", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples-per-direction", type=int, default=None)
    parser.add_argument("--sample-index", type=int, action="append", default=[])
    parser.add_argument("--include-augmented", type=parse_bool, default=None)
    parser.add_argument(
        "--strategy",
        choices=("configured", "mean", "first", "cmd_dir"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--preview-image-height", type=int, default=None)
    args, unknown_args = parser.parse_known_args()
    unexpected_args = [
        arg for arg in unknown_args if not arg.startswith("__") and ":=" not in arg
    ]
    if unexpected_args:
        parser.error(f"unrecognized arguments: {' '.join(unexpected_args)}")
    return args


def main():
    import torch

    args = parse_args()
    cfg = load_runtime_config(resolve_path(args.config_dir, package_root()))
    model_cfg = dict(cfg["model"])
    dataset_cfg = cfg["train"]["dataset"]
    evaluation_cfg = cfg["visualization"].get("evaluation", {})
    dataset_type = args.dataset_type or str(
        evaluation_cfg.get("dataset_type", "train")
    )
    trajectory_name = (
        args.trajectory_name
        if args.trajectory_name is not None
        else str(evaluation_cfg.get("trajectory_name", ""))
    )
    samples_per_direction = (
        args.samples_per_direction
        if args.samples_per_direction is not None
        else int(evaluation_cfg.get("samples_per_direction", 5))
    )
    include_augmented = (
        args.include_augmented
        if args.include_augmented is not None
        else bool(evaluation_cfg.get("include_augmented", False))
    )
    compare_all_commands = bool(
        evaluation_cfg.get("compare_all_commands", False)
    )
    configured_sample_indices = evaluation_cfg.get("sample_indices", []) or []
    if not isinstance(configured_sample_indices, list):
        raise ValueError("visualization.evaluation.sample_indices must be a list")
    explicit_sample_indices = args.sample_index or [
        int(index) for index in configured_sample_indices
    ]
    seed = args.seed if args.seed is not None else int(evaluation_cfg.get("seed", 0))
    sample_seed = (
        args.sample_seed
        if args.sample_seed is not None
        else int(evaluation_cfg.get("sample_seed", 0))
    )
    preview_image_height = (
        args.preview_image_height
        if args.preview_image_height is not None
        else int(evaluation_cfg.get("preview_image_height", 160))
    )
    data_dir = model_dataset_dir(dataset_cfg, dataset_type, model_cfg["model_type"])
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir)

    if trajectory_name:
        trajectory_names = [trajectory_name]
    else:
        trajectory_names = discover_trajectory_names(data_dir)
        if not include_augmented:
            trajectory_names = [name for name in trajectory_names if os.sep not in name]
    dataset = make_dataset(cfg, data_dir, trajectory_names)

    checkpoint = args.checkpoint_path or model_cfg["checkpoint_path"]
    checkpoint = resolve_path(checkpoint, package_root())
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    model_cfg["checkpoint_path"] = checkpoint
    model = VNMModel(model_cfg, checkpoint)

    strategy = args.strategy or str(evaluation_cfg.get("strategy", "configured"))
    if strategy == "configured":
        strategy = str(model_cfg.get("action_sample_strategy", "first"))
    sample_indices = selected_sample_indices(
        dataset, samples_per_direction, explicit_sample_indices, sample_seed
    )
    if not sample_indices:
        raise ValueError("No evaluation samples selected")

    dataset_name = os.path.basename(os.path.dirname(data_dir))
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else str(evaluation_cfg.get("output_dir", "auto"))
    )
    if output_dir in ("", "auto"):
        output_dir = os.path.join(
            package_root(),
            "plots",
            "evaluation",
            safe_name(dataset_name),
            safe_name(model_cfg["model_type"]),
            checkpoint_name(checkpoint),
            dataset_type,
        )
    else:
        output_dir = resolve_path(output_dir, package_root())
    os.makedirs(output_dir, exist_ok=True)
    for output_name in os.listdir(output_dir):
        if output_name.lower().endswith(".png") or output_name in (
            "metrics.csv",
            "summary.csv",
        ):
            os.remove(os.path.join(output_dir, output_name))

    records = []
    for preview_index, sample_index in enumerate(sample_indices):
        name, current, cmd_dir, teacher = teacher_trajectory(dataset, sample_index)
        indices, images = context_images(dataset, name, current)
        teacher_command = command_name(cmd_dir)
        input_commands = (
            [command_vector(direction) for direction in DIRECTION_NAMES]
            if compare_all_commands
            else [cmd_dir]
        )
        selected_by_command = {}
        for input_cmd_dir in input_commands:
            input_command = command_name(input_cmd_dir)
            torch.manual_seed(seed + int(sample_index))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + int(sample_index))
            predictions = predictions_in_dataset_units(
                model, model_cfg, dataset_cfg, images, input_cmd_dir
            )
            selected, selected_index, selected_theta, selected_score = choose_prediction(
                predictions,
                input_cmd_dir,
                strategy,
                float(model_cfg.get("cmd_dir_theta_threshold_deg", 15.0)),
                int(model_cfg.get("waypoint_index", 0)),
            )
            metrics = evaluation_metrics(teacher, selected)
            record = {
                "sample_index": int(sample_index),
                "trajectory": name,
                "current_index": current,
                "teacher_command": teacher_command,
                "command": input_command,
                "teacher_command_match": input_command == teacher_command,
                "strategy": strategy,
                "selected_sample": selected_index,
                "selected_theta_deg": ""
                if selected_theta is None
                else math.degrees(float(selected_theta)),
                "selected_score": ""
                if selected_score is None
                else float(selected_score),
                **metrics,
            }
            records.append(record)
            selected_by_command[input_command] = selected
            if not compare_all_commands:
                output_name = (
                    f"{record['command']}_{preview_index:04d}_"
                    f"sample_{sample_index:06d}_{safe_name(name)}.png"
                )
                render_preview(
                    os.path.join(output_dir, output_name),
                    images,
                    indices,
                    teacher,
                    predictions,
                    selected,
                    record,
                    preview_image_height,
                )
        if compare_all_commands:
            output_name = (
                f"commands_{preview_index:04d}_sample_{sample_index:06d}_"
                f"{safe_name(name)}.png"
            )
            render_command_comparison(
                os.path.join(output_dir, output_name),
                images,
                indices,
                teacher,
                selected_by_command,
                int(sample_index),
                name,
                current,
                teacher_command,
                preview_image_height,
            )
        print(
            f"[{preview_index + 1}/{len(sample_indices)}] "
            f"sample={sample_index} teacher_cmd={teacher_command} "
            f"inputs={','.join(selected_by_command)}"
        )

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary = write_summary(records, summary_path)
    print(
        f"evaluated samples={len(records)} sample_seed={sample_seed} "
        f"strategy={strategy} checkpoint={checkpoint} "
        f"output_dir={output_dir}"
    )
    for row in summary:
        print(
            f"{row['direction']}: count={row['count']} "
            f"ADE={row['ade_mean_m']:.3f}m FDE={row['fde_mean_m']:.3f}m "
            f"angle_abs_error={row['abs_endpoint_angle_error_mean_deg']:.1f}deg"
        )


if __name__ == "__main__":
    main()
