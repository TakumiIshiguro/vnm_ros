import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


TARGET_DIRECTION_NAMES = {
    0: "straight",
    1: "left",
    2: "right",
}


def decode_action_candidates(data: Sequence[float]) -> Optional[Dict]:
    """Decode the compact Float32MultiArray representation used by vnm_node."""
    if len(data) < 5:
        return None
    selected = int(data[0])
    waypoint_index = int(data[1])
    sample_count = int(data[2])
    horizon = int(data[3])
    dimensions = int(data[4])
    if sample_count <= 0 or horizon <= 0 or dimensions < 2:
        return None
    expected = 5 + sample_count * horizon * dimensions
    if len(data) < expected:
        return None

    values = np.asarray(data[5:expected], dtype=np.float32)
    if not np.isfinite(values).all():
        return None
    actions = values.reshape(sample_count, horizon, dimensions)
    target_direction = int(data[expected + 1]) if len(data) > expected + 1 else -1
    return {
        "selected": max(0, min(selected, sample_count - 1)),
        "waypoint_index": max(0, min(waypoint_index, horizon - 1)),
        "actions": actions,
        "avoidance_active": bool(data[expected]) if len(data) > expected else False,
        "target_direction": TARGET_DIRECTION_NAMES.get(
            target_direction,
            "none",
        ),
    }


TARGET_ARROW_COLOR = (45, 120, 230)
TARGET_ARROW_OUTLINE = (20, 45, 95)
MINIMUM_ARROW_PIXELS = 48.0


def _ticks(minimum: float, maximum: float, spacing: float):
    """Tick values anchored at zero.

    Stepping from `minimum` instead skipped the origin whenever the range was
    not an even multiple of the spacing, which put two rounded "0" labels
    either side of the centre line and none on it.
    """
    first = int(math.ceil(minimum / spacing - 1e-6))
    last = int(math.floor(maximum / spacing + 1e-6))
    return [index * spacing for index in range(first, last + 1)]


def _format_tick(value: float, signed: bool = False) -> str:
    if abs(value) < 1e-6:
        return "0"
    magnitude = abs(value)
    text = (
        f"{magnitude:.0f}"
        if abs(magnitude - round(magnitude)) < 1e-6
        else f"{magnitude:.1f}"
    )
    if value < 0.0:
        return "-" + text
    return ("+" + text) if signed else text


def _draw_target_arrow(draw: ImageDraw.ImageDraw, box, direction: str) -> None:
    """Draw the commanded direction as one large arrow above the plot."""
    if direction not in ("straight", "left", "right"):
        return
    left, top, right, bottom = box
    size = min((bottom - top) * 0.8, (right - left) * 0.45)
    if size < MINIMUM_ARROW_PIXELS:
        return
    half = size / 2.0
    head = size * 0.42
    head_half = size * 0.30
    shaft_half = size * 0.13
    points = [
        (0.0, -half),
        (-head_half, -half + head),
        (-shaft_half, -half + head),
        (-shaft_half, half),
        (shaft_half, half),
        (shaft_half, -half + head),
        (head_half, -half + head),
    ]
    if direction == "left":
        points = [(dy, -dx) for dx, dy in points]
    elif direction == "right":
        points = [(-dy, dx) for dx, dy in points]
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    draw.polygon(
        [(center_x + dx, center_y + dy) for dx, dy in points],
        fill=TARGET_ARROW_COLOR,
        outline=TARGET_ARROW_OUTLINE,
    )


def _validate_points(points, columns: int, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, columns), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < columns:
        raise ValueError(f"{name} must have shape (N, >= {columns})")
    points = points[:, :columns]
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def render_care_bev(
    obstacle_points,
    candidates: Optional[Dict],
    map_width_m: float = 1.0,
    map_height_m: float = 1.0,
    image_size: Tuple[int, int] = (640, 640),
    grid_spacing_m: float = 0.5,
    obstacle_radius_pixels: int = 2,
    all_obstacle_points=None,
) -> Image.Image:
    """Render obstacles and NoMaD trajectories in the robot coordinate frame.

    Robot coordinates are x-forward and y-left in metres. Positive y is shown
    on the left side of the image so the BEV matches the robot's viewpoint.
    """
    map_width_m = float(map_width_m)
    map_height_m = float(map_height_m)
    width, height = map(int, image_size)
    if map_width_m <= 0.0 or map_height_m <= 0.0:
        raise ValueError("map dimensions must be positive")
    if width < 256 or height < 256:
        raise ValueError("image dimensions must be at least 256 pixels")
    if grid_spacing_m <= 0.0:
        raise ValueError("grid_spacing_m must be positive")
    if obstacle_radius_pixels < 0:
        raise ValueError("obstacle_radius_pixels must be non-negative")

    obstacles = _validate_points(obstacle_points, 2, "obstacle_points")
    all_obstacles = _validate_points(
        [] if all_obstacle_points is None else all_obstacle_points,
        2,
        "all_obstacle_points",
    )
    image = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(image)
    # Keep a band above the plot for the target-direction arrow. The plot is
    # bottom-anchored, so a wide map leaves the arrow more room than this
    # minimum; reserving it stops the arrow from disappearing when the map
    # aspect ratio fills the image.
    arrow_band_top = 35
    arrow_band = int(round(height * 0.18))
    available_plot = (70, arrow_band_top + arrow_band, width - 25, height - 70)
    available_width = available_plot[2] - available_plot[0]
    available_height = available_plot[3] - available_plot[1]
    pixels_per_metre = min(
        available_width / map_width_m,
        available_height / map_height_m,
    )
    plot_width = int(round(map_width_m * pixels_per_metre))
    plot_height = int(round(map_height_m * pixels_per_metre))
    plot_left = available_plot[0] + (available_width - plot_width) // 2
    plot = (
        plot_left,
        available_plot[3] - plot_height,
        plot_left + plot_width,
        available_plot[3],
    )
    pixels_per_metre_x = pixels_per_metre
    pixels_per_metre_y = pixels_per_metre

    def point(robot_xy):
        forward = float(robot_xy[0])
        left = float(robot_xy[1])
        return (
            int(round(plot[0] + plot_width / 2.0 - left * pixels_per_metre_x)),
            int(round(plot[3] - forward * pixels_per_metre_y)),
        )

    lateral_values = np.arange(
        -map_width_m / 2.0,
        map_width_m / 2.0 + grid_spacing_m * 0.5,
        grid_spacing_m,
    )
    forward_values = np.arange(
        0.0,
        map_height_m + grid_spacing_m * 0.5,
        grid_spacing_m,
    )
    for lateral in lateral_values:
        x, _ = point((0.0, lateral))
        draw.line((x, plot[1], x, plot[3]), fill=(220, 224, 228), width=1)
    for forward in forward_values:
        _, y = point((forward, 0.0))
        draw.line((plot[0], y, plot[2], y), fill=(220, 224, 228), width=1)

    center_x, _ = point((0.0, 0.0))
    draw.line((center_x, plot[1], center_x, plot[3]), fill=(145, 151, 158), width=2)
    draw.rectangle(plot, outline=(90, 96, 102), width=2)

    half_width = map_width_m / 2.0
    for lateral in _ticks(-half_width, half_width, grid_spacing_m):
        x, _ = point((0.0, lateral))
        label = _format_tick(lateral, signed=True)
        draw.text(
            (x - draw.textlength(label) / 2.0, plot[3] + 8),
            label,
            fill=(45, 50, 55),
        )
    for forward in _ticks(0.0, map_height_m, grid_spacing_m):
        _, y = point((forward, 0.0))
        label = _format_tick(forward)
        draw.text(
            (plot[0] - 8 - draw.textlength(label), y - 6),
            label,
            fill=(45, 50, 55),
        )
    draw.text((plot[0] - 34, plot[1] - 18), "x [m]", fill=(45, 50, 55))
    draw.text((center_x - 14, plot[3] + 26), "y [m]", fill=(45, 50, 55))

    all_obstacle_pixels = set()
    for obstacle in all_obstacles:
        forward, left = map(float, obstacle)
        if not (0.0 <= forward <= map_height_m):
            continue
        if abs(left) > half_width:
            continue
        all_obstacle_pixels.add(point(obstacle))
    all_radius = max(1, obstacle_radius_pixels - 1)
    for x, y in all_obstacle_pixels:
        draw.ellipse(
            (x - all_radius, y - all_radius, x + all_radius, y + all_radius),
            fill=(125, 132, 142),
        )

    for obstacle in obstacles:
        forward, left = map(float, obstacle)
        if not (0.0 <= forward <= map_height_m):
            continue
        if abs(left) > half_width:
            continue
        x, y = point(obstacle)
        radius = obstacle_radius_pixels
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(225, 55, 55),
        )

    candidate_count = 0
    selected_index = -1
    avoidance_active = False
    target_direction = "none"
    if candidates is not None:
        actions = np.asarray(candidates.get("actions", []), dtype=np.float32)
        if (
            actions.ndim != 3
            or actions.shape[0] == 0
            or actions.shape[1] == 0
            or actions.shape[2] < 2
        ):
            raise ValueError("candidates.actions must have shape (S, T, >= 2)")
        if not np.isfinite(actions).all():
            raise ValueError("candidates.actions must contain only finite values")
        candidate_count = actions.shape[0]
        selected_index = max(
            0,
            min(int(candidates.get("selected", 0)), candidate_count - 1),
        )
        waypoint_index = max(
            0,
            min(int(candidates.get("waypoint_index", 0)), actions.shape[1] - 1),
        )
        avoidance_active = bool(candidates.get("avoidance_active", False))
        target_direction = str(candidates.get("target_direction", "none"))
        origin = point((0.0, 0.0))
        for index, action in enumerate(actions):
            pixels = [origin] + [point(waypoint[:2]) for waypoint in action]
            if index != selected_index:
                draw.line(pixels, fill=(64, 156, 255), width=2)
        selected_pixels = [origin] + [
            point(waypoint[:2]) for waypoint in actions[selected_index]
        ]
        draw.line(selected_pixels, fill=(25, 25, 25), width=7)
        selected_color = (
            (225, 70, 235)
            if avoidance_active
            else (255, 211, 40)
        )
        draw.line(selected_pixels, fill=selected_color, width=4)
        waypoint_pixel = selected_pixels[waypoint_index + 1]
        draw.ellipse(
            (
                waypoint_pixel[0] - 6,
                waypoint_pixel[1] - 6,
                waypoint_pixel[0] + 6,
                waypoint_pixel[1] + 6,
            ),
            fill=(255, 125, 30),
            outline=(25, 25, 25),
            width=2,
        )

    origin = point((0.0, 0.0))
    draw.polygon(
        (
            (origin[0], origin[1] - 14),
            (origin[0] - 9, origin[1]),
            (origin[0] + 9, origin[1]),
        ),
        fill=(40, 190, 95),
        outline=(20, 80, 45),
    )
    _draw_target_arrow(
        draw,
        (plot[0], arrow_band_top, plot[2], plot[1]),
        target_direction,
    )
    return image
