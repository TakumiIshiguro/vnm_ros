#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import yaml
from PIL import Image
from PIL import ImageDraw

from vnm_ros.datasets.dataset_utils import to_local_coords
from vnm_ros.datasets.nomad_direction_dataset import NoMaDDirectionDataset
from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset
from vnm_ros.datasets.vint_direction_dataset import ViNTDirectionDataset
from vnm_ros.utils.config import load_runtime_config, model_dataset_dir, package_root, resolve_path
from vnm_ros.utils.logger import info


CMD_DIR_COLORS = {
    -1: (140, 140, 140),
    0: (40, 220, 40),
    1: (40, 100, 255),
    2: (255, 60, 40),
}
CMD_DIR_NAMES = {
    -1: "none",
    0: "straight",
    1: "left",
    2: "right",
}


def workspace_root():
    return os.path.abspath(os.path.join(package_root(), ".."))


def default_map_yaml():
    path = os.path.join(
        workspace_root(),
        "orne_navigation",
        "orne_navigation_executor",
        "config",
        "maps",
        "cit_3f_map.yaml",
    )
    return path if os.path.isfile(path) else None


def resolve_optional_path(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.join(package_root(), path),
        os.path.join(workspace_root(), path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)


def default_output_dir(data_dir, dataset_type):
    normalized = os.path.normpath(data_dir)
    try:
        relative = os.path.relpath(normalized, package_root())
    except ValueError:
        relative = os.path.basename(normalized)
    if relative.startswith(".."):
        relative = os.path.basename(normalized)
    parts = [part for part in relative.split(os.sep) if part and part != "."]
    if parts and parts[0] == "dataset":
        parts = parts[1:]
    if not parts:
        parts = [os.path.basename(normalized)]
    return os.path.join(package_root(), "plots", "dataset", *parts, dataset_type)


def select_trajectories(data_dir, requested_name):
    if requested_name:
        return [requested_name]
    names = sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, name, "traj_data.pkl"))
    )
    if not names:
        raise ValueError(f"No trajectory directories found in {data_dir}")
    return names


def cmd_dir_index(cmd_dir):
    if cmd_dir is None or len(cmd_dir) < 3:
        return -1
    cmd_dir = np.asarray(cmd_dir[:3], dtype=np.float32)
    if np.sum(cmd_dir) <= 0.0:
        return -1
    return int(np.argmax(cmd_dir))


def color_for_trajectory(index):
    colors = [
        (40, 220, 40),
        (40, 120, 255),
        (255, 140, 40),
        (255, 220, 40),
        (190, 80, 255),
        (40, 220, 210),
    ]
    return colors[index % len(colors)]


class MapCanvas:
    def __init__(self, map_yaml=None, bounds=None, image_size=1200, padding_m=1.0):
        self.map_yaml = map_yaml
        self.bounds = bounds
        self.image_size = int(image_size)
        self.padding_m = float(padding_m)
        self.map_image = None
        self.resolution = None
        self.origin = None
        self.crop_box = None
        self.display_scale = 1.0
        if map_yaml:
            self._load_map(map_yaml)

    def _load_map(self, map_yaml):
        with open(map_yaml, "r") as f:
            metadata = yaml.safe_load(f)
        image_path = metadata["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml), image_path)
        self.map_image = Image.open(image_path).convert("RGB")
        self.resolution = float(metadata["resolution"])
        self.origin = np.asarray(metadata["origin"][:2], dtype=np.float64)

    def world_to_map_pixel(self, position):
        xy = (np.asarray(position, dtype=np.float64) - self.origin) / self.resolution
        x = xy[..., 0]
        y = self.map_image.height - 1 - xy[..., 1]
        return np.stack([x, y], axis=-1)

    def world_to_blank_pixel(self, position):
        x_min, y_min, x_max, y_max = self.bounds
        points = np.asarray(position, dtype=np.float64)
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)
        usable = self.image_size - 2 * self.padding_px
        x = self.padding_px + (points[..., 0] - x_min) / span_x * usable
        y = self.image_size - self.padding_px - (points[..., 1] - y_min) / span_y * usable
        return np.stack([x, y], axis=-1)

    def make(self, positions):
        self.display_scale = 1.0
        if self.map_image is None:
            return self._make_blank()
        return self._make_map(positions)

    def _make_blank(self):
        self.padding_px = 80
        return Image.new("RGB", (self.image_size, self.image_size), (245, 245, 245))

    def _make_map(self, positions):
        if positions is None or len(positions) == 0:
            return self.map_image.copy()
        pixels = self.world_to_map_pixel(positions)
        pad_px = max(20, int(self.padding_m / self.resolution))
        x0 = int(max(np.floor(np.min(pixels[:, 0])) - pad_px, 0))
        y0 = int(max(np.floor(np.min(pixels[:, 1])) - pad_px, 0))
        x1 = int(min(np.ceil(np.max(pixels[:, 0])) + pad_px, self.map_image.width - 1))
        y1 = int(min(np.ceil(np.max(pixels[:, 1])) + pad_px, self.map_image.height - 1))
        if x1 <= x0 or y1 <= y0:
            self.crop_box = None
            return self.map_image.copy()
        self.crop_box = (x0, y0, x1, y1)
        image = self.map_image.crop(self.crop_box)
        width, height = image.size
        self.display_scale = min(
            self.image_size / max(width, 1),
            self.image_size / max(height, 1),
        )
        output_size = (
            max(1, int(round(width * self.display_scale))),
            max(1, int(round(height * self.display_scale))),
        )
        return image.resize(output_size, Image.Resampling.NEAREST)

    def world_to_pixel(self, position):
        if self.map_image is None:
            return self.world_to_blank_pixel(position)
        pixels = self.world_to_map_pixel(position)
        if self.crop_box is not None:
            pixels = pixels - np.asarray(self.crop_box[:2], dtype=np.float64)
        pixels = pixels * self.display_scale
        return pixels

    def visible_world_bounds(self):
        if self.map_image is None:
            return self.bounds
        if self.crop_box is None:
            x0, y0 = 0, 0
            x1, y1 = self.map_image.width - 1, self.map_image.height - 1
        else:
            x0, y0, x1, y1 = self.crop_box
        corners = np.asarray(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float64,
        )
        world_x = corners[:, 0] * self.resolution + self.origin[0]
        world_y = (self.map_image.height - 1 - corners[:, 1]) * self.resolution + self.origin[1]
        return (
            float(np.min(world_x)),
            float(np.min(world_y)),
            float(np.max(world_x)),
            float(np.max(world_y)),
        )


def bounds_for_positions(position_sets, padding=1.0):
    all_positions = np.concatenate(position_sets, axis=0)
    x_min, y_min = np.min(all_positions, axis=0) - padding
    x_max, y_max = np.max(all_positions, axis=0) + padding
    return float(x_min), float(y_min), float(x_max), float(y_max)


def draw_polyline(draw, pixels, color, width):
    if len(pixels) < 2:
        return
    points = [tuple(map(float, point)) for point in pixels]
    draw.line(points, fill=color, width=width, joint="curve")


def draw_direction_segments(draw, pixels, cmd_dirs, width):
    if len(pixels) < 2:
        return
    for index in range(len(pixels) - 1):
        direction = cmd_dir_index(cmd_dirs[index]) if cmd_dirs is not None else -1
        color = CMD_DIR_COLORS.get(direction, CMD_DIR_COLORS[-1])
        draw.line(
            [tuple(map(float, pixels[index])), tuple(map(float, pixels[index + 1]))],
            fill=color,
            width=width,
        )


def draw_world_grid(draw, canvas, image, spacing_m):
    spacing_m = float(spacing_m)
    if spacing_m <= 0.0:
        return
    x_min, y_min, x_max, y_max = canvas.visible_world_bounds()
    x_start = np.floor(x_min / spacing_m) * spacing_m
    x_end = np.ceil(x_max / spacing_m) * spacing_m
    y_start = np.floor(y_min / spacing_m) * spacing_m
    y_end = np.ceil(y_max / spacing_m) * spacing_m
    grid_color = (205, 205, 205) if canvas.map_image is None else (165, 165, 165)
    axis_color = (145, 145, 145) if canvas.map_image is None else (110, 110, 110)
    width = 1

    x_values = np.arange(x_start, x_end + spacing_m * 0.5, spacing_m)
    for x in x_values:
        pixels = canvas.world_to_pixel([[x, y_min], [x, y_max]])
        color = axis_color if abs(x) < spacing_m * 0.5 else grid_color
        draw.line(
            [tuple(map(float, pixels[0])), tuple(map(float, pixels[1]))],
            fill=color,
            width=width,
        )

    y_values = np.arange(y_start, y_end + spacing_m * 0.5, spacing_m)
    for y in y_values:
        pixels = canvas.world_to_pixel([[x_min, y], [x_max, y]])
        color = axis_color if abs(y) < spacing_m * 0.5 else grid_color
        draw.line(
            [tuple(map(float, pixels[0])), tuple(map(float, pixels[1]))],
            fill=color,
            width=width,
        )

    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(120, 120, 120), width=1)


def draw_start_end(draw, pixels):
    if len(pixels) == 0:
        return
    radius = 5
    start = tuple(map(float, pixels[0]))
    end = tuple(map(float, pixels[-1]))
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
        fill=(0, 180, 0),
        outline=(0, 0, 0),
    )
    draw.rectangle(
        (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
        fill=(220, 0, 0),
        outline=(0, 0, 0),
    )


def draw_text_box(draw, lines, xy=(10, 10)):
    pad = 5
    line_boxes = [draw.textbbox((0, 0), line) for line in lines]
    width = max(box[2] - box[0] for box in line_boxes) if line_boxes else 0
    height = sum(box[3] - box[1] + 3 for box in line_boxes)
    x, y = xy
    draw.rectangle((x - pad, y - pad, x + width + pad, y + height + pad), fill=(255, 255, 255))
    cursor_y = y
    for line, box in zip(lines, line_boxes):
        draw.text((x, cursor_y), line, fill=(0, 0, 0))
        cursor_y += box[3] - box[1] + 3


def draw_legend(draw, image):
    lines = [
        "start: green circle",
        "end: red square",
        "straight: green",
        "left: blue",
        "right: red",
        "none: gray",
    ]
    draw_text_box(draw, lines, xy=(10, max(10, image.height - 115)))


def direction_counts(cmd_dirs):
    counts = {name: 0 for name in CMD_DIR_NAMES.values()}
    if cmd_dirs is None:
        return counts
    for cmd_dir in cmd_dirs:
        counts[CMD_DIR_NAMES.get(cmd_dir_index(cmd_dir), "none")] += 1
    return counts


def preview_indices(length, count):
    if count < 0:
        return list(range(length))
    count = min(int(count), int(length))
    if count <= 0:
        return []
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def preview_indices_per_direction(dataset, count_per_direction):
    by_direction = {"straight": [], "left": [], "right": []}
    for sample_index, (name, current) in enumerate(dataset.samples):
        trajectory = dataset.trajectory(name)
        direction = CMD_DIR_NAMES.get(cmd_dir_index(trajectory["cmd_dir"][current]), "none")
        if direction in by_direction:
            by_direction[direction].append(sample_index)

    selected = []
    for direction in ("straight", "left", "right"):
        indices = by_direction[direction]
        local_indices = preview_indices(len(indices), count_per_direction)
        selected.extend((direction, indices[index]) for index in local_indices)
    return selected


def draw_local_trajectory(
    draw,
    box,
    points,
    color,
    grid_spacing_m=0.1,
    expected_forward_m=None,
):
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    margin = 28
    points = np.asarray(points, dtype=np.float32)
    expected_forward_m = (
        None if expected_forward_m is None else max(float(expected_forward_m), 0.0)
    )
    max_abs_y = max(float(np.max(np.abs(points[:, 1]))), 0.1)
    max_x = max(float(np.max(points[:, 0])), expected_forward_m or 0.0, 0.1)
    scale = min((width - 2 * margin) / (2.0 * max_abs_y), (height - 2 * margin) / max_x)
    origin = (x0 + width // 2, y1 - margin)

    def project(point):
        px = origin[0] - float(point[1]) * scale
        py = origin[1] - float(point[0]) * scale
        return (px, py)

    draw.rectangle(box, outline=(190, 190, 190), width=1)
    grid_spacing_m = float(grid_spacing_m)
    if grid_spacing_m > 0.0:
        grid_color = (230, 230, 230)
        for forward_x in np.arange(0.0, max_x + grid_spacing_m * 0.01, grid_spacing_m):
            left = project((forward_x, -max_abs_y))
            right = project((forward_x, max_abs_y))
            draw.line((left[0], left[1], right[0], right[1]), fill=grid_color)
        y_grid_min = np.floor(-max_abs_y / grid_spacing_m) * grid_spacing_m
        y_grid_max = np.ceil(max_abs_y / grid_spacing_m) * grid_spacing_m
        for lateral_y in np.arange(y_grid_min, y_grid_max + grid_spacing_m * 0.5, grid_spacing_m):
            near = project((0.0, lateral_y))
            far = project((max_x, lateral_y))
            draw.line((near[0], near[1], far[0], far[1]), fill=grid_color)
    draw.line((origin[0], y0 + margin, origin[0], y1 - margin), fill=(200, 200, 200))
    draw.line((x0 + margin, origin[1], x1 - margin, origin[1]), fill=(200, 200, 200))
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


def make_nomad_dataset(cfg, data_dir, trajectory_names):
    model_cfg = cfg["model"]
    dataset_cfg = cfg["train"]["dataset"]
    return NoMaDDirectionDataset(
        data_dir=data_dir,
        image_size=model_cfg["image_size"],
        context_size=int(model_cfg["context_size"]),
        len_traj_pred=int(model_cfg["len_traj_pred"]),
        waypoint_spacing=int(dataset_cfg["waypoint_spacing"]),
        action_stats=model_cfg["action_stats"],
        center_crop=bool(model_cfg.get("image_center_crop", False)),
        trajectory_names=trajectory_names,
    )


def make_vint_dataset(cfg, data_dir, trajectory_names):
    model_cfg = cfg["model"]
    dataset_cfg = cfg["train"]["dataset"]
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


def make_direction_dataset(cfg, data_dir, trajectory_names):
    model_type = cfg["model"]["model_type"]
    if model_type == "nomad":
        return make_nomad_dataset(cfg, data_dir, trajectory_names)
    if model_type == "vint" and bool(cfg["model"].get("direction_conditioning", False)):
        return make_vint_dataset(cfg, data_dir, trajectory_names)
    raise ValueError(
        "training sample previews require NoMaD or ViNT with direction_conditioning=true"
    )


def resize_to_height(image, target_height):
    target_height = int(target_height)
    if target_height <= 0 or image.height == target_height:
        return image.copy()
    scale = target_height / max(image.height, 1)
    target_size = (max(1, int(round(image.width * scale))), target_height)
    return image.resize(target_size, Image.Resampling.BILINEAR)


def render_training_samples(dataset, output_dir, count_per_direction, preview_image_height):
    if count_per_direction == 0:
        return []
    os.makedirs(output_dir, exist_ok=True)
    remove_pngs(output_dir)
    colors = {
        "straight": CMD_DIR_COLORS[0],
        "left": CMD_DIR_COLORS[1],
        "right": CMD_DIR_COLORS[2],
        "none": CMD_DIR_COLORS[-1],
    }
    written = []
    selected_samples = preview_indices_per_direction(dataset, count_per_direction)
    direction_counts = {"straight": 0, "left": 0, "right": 0}
    expected_forward_m = None
    if hasattr(dataset, "metric_waypoint_spacing"):
        expected_forward_m = (
            float(dataset.metric_waypoint_spacing)
            * int(dataset.waypoint_spacing)
            * int(dataset.len_traj_pred)
        )
    for preview_id, (_, sample_index) in enumerate(selected_samples):
        name, current = dataset.samples[sample_index]
        trajectory = dataset.trajectory(name)
        cmd_name = CMD_DIR_NAMES.get(cmd_dir_index(trajectory["cmd_dir"][current]), "none")
        action_indices = current + np.arange(dataset.len_traj_pred + 1) * dataset.waypoint_spacing
        positions = trajectory["position"][action_indices]
        local_positions = to_local_coords(
            positions,
            positions[0],
            float(trajectory["yaw"][current]),
        )
        context_indices = current + np.arange(-dataset.context_size, 1) * dataset.waypoint_spacing
        context_images = []
        for context_index in context_indices:
            with Image.open(dataset.image_path(name, int(context_index))) as source:
                context_images.append(resize_to_height(source, preview_image_height))

        image_gap = 16
        image_top = 64
        image_widths = [image.width for image in context_images]
        image_heights = [image.height for image in context_images]
        max_image_height = max(image_heights) if image_heights else 0
        images_width = sum(image_widths) + image_gap * max(len(context_images) - 1, 0)
        traj_width = 304
        traj_height = 256
        traj_gap = 24
        canvas_width = 24 + images_width + traj_gap + traj_width + 24
        canvas_height = max(320, image_top + max_image_height + 32, 48 + traj_height + 16)
        canvas = Image.new("RGB", (canvas_width, canvas_height), (250, 250, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), f"{preview_id:04d} {name} idx={current} cmd={cmd_name}", fill=(20, 20, 20))
        x = 24
        for image_id, image in enumerate(context_images):
            canvas.paste(image, (x, image_top))
            frame_color = (30, 140, 70) if image_id == len(context_images) - 1 else (170, 170, 170)
            draw.rectangle(
                (x, image_top, x + image.width, image_top + image.height),
                outline=frame_color,
                width=3,
            )
            label = "current" if image_id == len(context_images) - 1 else f"t-{len(context_images) - 1 - image_id}"
            draw.text((x, 42), f"{label} idx={int(context_indices[image_id])}", fill=(20, 20, 20))
            x += image.width + image_gap

        traj_x0 = 24 + images_width + traj_gap
        draw.text((traj_x0, 18), "teacher trajectory in robot frame", fill=(20, 20, 20))
        draw_local_trajectory(
            draw,
            (traj_x0, 48, traj_x0 + traj_width, 48 + traj_height),
            local_positions,
            colors.get(cmd_name, colors["none"]),
            expected_forward_m=expected_forward_m,
        )
        direction_id = direction_counts.get(cmd_name, 0)
        direction_counts[cmd_name] = direction_id + 1
        output_path = os.path.join(
            output_dir,
            f"{cmd_name}_{direction_id:04d}_sample_{preview_id:04d}.png",
        )
        canvas.save(output_path)
        written.append(output_path)
    return written


def remove_pngs(directory):
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.lower().endswith(".png"):
            os.remove(os.path.join(directory, name))


def render_trajectory(name, trajectory, canvas, output_path, grid_spacing_m):
    positions = trajectory["position"]
    cmd_dirs = trajectory.get("cmd_dir")
    image = canvas.make(positions)
    draw = ImageDraw.Draw(image)
    draw_world_grid(draw, canvas, image, grid_spacing_m)
    pixels = canvas.world_to_pixel(positions)
    draw_direction_segments(draw, pixels, cmd_dirs, width=4)
    draw_start_end(draw, pixels)
    counts = direction_counts(cmd_dirs)
    lines = [
        name,
        f"samples: {len(positions)}",
        "cmd_dir: "
        + ", ".join(f"{key}={value}" for key, value in counts.items() if value),
    ]
    draw_text_box(draw, lines)
    draw_legend(draw, image)
    image.save(output_path)


def render_overview(dataset, trajectory_names, canvas, output_path, grid_spacing_m):
    position_sets = [dataset.trajectory(name)["position"] for name in trajectory_names]
    image = canvas.make(np.concatenate(position_sets, axis=0))
    draw = ImageDraw.Draw(image)
    draw_world_grid(draw, canvas, image, grid_spacing_m)
    for index, name in enumerate(trajectory_names):
        positions = dataset.trajectory(name)["position"]
        pixels = canvas.world_to_pixel(positions)
        draw_polyline(draw, pixels, color_for_trajectory(index), width=2)
        draw_start_end(draw, pixels)
    lines = [
        "dataset overview",
        f"trajectories: {len(trajectory_names)}",
        f"samples: {sum(len(dataset.trajectory(name)['position']) for name in trajectory_names)}",
    ]
    draw_text_box(draw, lines)
    draw_legend(draw, image)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--dataset-type", choices=["train", "test"], default=None)
    parser.add_argument("--trajectory-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--map-yaml", default=None)
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--image-size", type=int, default=1200)
    parser.add_argument("--padding-m", type=float, default=1.0)
    parser.add_argument("--grid-spacing-m", type=float, default=1.0)
    parser.add_argument(
        "--training-samples-per-direction",
        type=int,
        default=10,
        help="Direction training sample previews per direction. Use -1 for all, 0 to disable.",
    )
    parser.add_argument(
        "--preview-image-height",
        type=int,
        default=160,
        help="Preview image height in pixels. Aspect ratio is preserved.",
    )
    args, unknown_args = parser.parse_known_args()
    unexpected_args = [
        arg for arg in unknown_args if not arg.startswith("__") and ":=" not in arg
    ]
    if unexpected_args:
        parser.error(f"unrecognized arguments: {' '.join(unexpected_args)}")

    cfg = load_runtime_config(args.config_dir)
    model_cfg = cfg["model"]
    dataset_cfg = cfg["train"]["dataset"]
    visualization_cfg = cfg["visualization"]["dataset"]
    dataset_type = args.dataset_type or visualization_cfg["dataset_type"]
    data_dir = model_dataset_dir(dataset_cfg, dataset_type, model_cfg["model_type"])
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"{data_dir}. Create a {model_cfg['model_type']} dataset first."
        )
    trajectory_name = (
        args.trajectory_name
        if args.trajectory_name is not None
        else visualization_cfg["trajectory_name"]
    )
    trajectory_names = select_trajectories(data_dir, trajectory_name)
    dataset = TrajectoryDataset(data_dir, trajectory_names)

    map_yaml = None
    if not args.no_map:
        map_yaml = resolve_optional_path(args.map_yaml) if args.map_yaml else default_map_yaml()
        if map_yaml and not os.path.isfile(map_yaml):
            raise FileNotFoundError(map_yaml)

    output_dir = (
        default_output_dir(data_dir, dataset_type)
        if args.output_dir in (None, "", "auto")
        else args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    trajectories_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)
    remove_pngs(output_dir)
    remove_pngs(trajectories_dir)

    all_positions = [dataset.trajectory(name)["position"] for name in trajectory_names]
    blank_bounds = bounds_for_positions(all_positions, padding=args.padding_m)

    overview_canvas = MapCanvas(
        map_yaml=map_yaml,
        bounds=blank_bounds,
        image_size=args.image_size,
        padding_m=args.padding_m,
    )
    overview_path = os.path.join(output_dir, "overview.png")
    render_overview(
        dataset,
        trajectory_names,
        overview_canvas,
        overview_path,
        args.grid_spacing_m,
    )

    for name in trajectory_names:
        trajectory = dataset.trajectory(name)
        bounds = bounds_for_positions([trajectory["position"]], padding=args.padding_m)
        canvas = MapCanvas(
            map_yaml=map_yaml,
            bounds=bounds,
            image_size=args.image_size,
            padding_m=args.padding_m,
        )
        render_trajectory(
            name,
            trajectory,
            canvas,
            os.path.join(trajectories_dir, f"{name}.png"),
            args.grid_spacing_m,
        )

    training_sample_count = 0
    if args.training_samples_per_direction != 0:
        try:
            sample_dataset = make_direction_dataset(cfg, data_dir, trajectory_names)
        except ValueError as exc:
            info(f"skipped training sample previews: {exc}")
        else:
            training_sample_paths = render_training_samples(
                sample_dataset,
                os.path.join(output_dir, "training_samples"),
                args.training_samples_per_direction,
                args.preview_image_height,
            )
            training_sample_count = len(training_sample_paths)

    map_text = map_yaml if map_yaml else "none"
    info(
        f"plotted dataset={dataset_type} trajectories={len(trajectory_names)} "
        f"training_samples={training_sample_count} "
        f"map={map_text} output_dir={output_dir}"
    )
    print(overview_path)


if __name__ == "__main__":
    main()
