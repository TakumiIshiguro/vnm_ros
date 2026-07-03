#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import yaml
from PIL import Image
from PIL import ImageDraw

from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
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


def remove_pngs(directory):
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.lower().endswith(".png"):
            os.remove(os.path.join(directory, name))


def render_trajectory(name, trajectory, canvas, output_path):
    positions = trajectory["position"]
    cmd_dirs = trajectory.get("cmd_dir")
    image = canvas.make(positions)
    draw = ImageDraw.Draw(image)
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


def render_overview(dataset, trajectory_names, canvas, output_path):
    position_sets = [dataset.trajectory(name)["position"] for name in trajectory_names]
    image = canvas.make(np.concatenate(position_sets, axis=0))
    draw = ImageDraw.Draw(image)
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
    args, unknown_args = parser.parse_known_args()
    unexpected_args = [
        arg for arg in unknown_args if not arg.startswith("__") and ":=" not in arg
    ]
    if unexpected_args:
        parser.error(f"unrecognized arguments: {' '.join(unexpected_args)}")

    cfg = load_runtime_config(args.config_dir)
    dataset_cfg = cfg["train"]["dataset"]
    visualization_cfg = cfg["visualization"]["dataset"]
    dataset_type = args.dataset_type or visualization_cfg["dataset_type"]
    data_dir_key = "train_data_dir" if dataset_type == "train" else "test_data_dir"
    data_dir = resolve_path(dataset_cfg[data_dir_key], package_root())
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

    output_dir = args.output_dir or os.path.join(
        package_root(), "plots", "dataset", dataset_type
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
    render_overview(dataset, trajectory_names, overview_canvas, overview_path)

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
        )

    map_text = map_yaml if map_yaml else "none"
    info(
        f"plotted dataset={dataset_type} trajectories={len(trajectory_names)} "
        f"map={map_text} output_dir={output_dir}"
    )
    print(overview_path)


if __name__ == "__main__":
    main()
