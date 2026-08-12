from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


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
    return {
        "selected": max(0, min(selected, sample_count - 1)),
        "waypoint_index": max(0, min(waypoint_index, horizon - 1)),
        "actions": actions,
        "avoidance_active": bool(data[expected]) if len(data) > expected else False,
    }


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
    available_plot = (70, 35, width - 25, height - 70)
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
    for lateral in np.arange(-half_width, half_width + 0.1, 1.0):
        x, _ = point((0.0, lateral))
        draw.text((x - 10, plot[3] + 8), f"{lateral:+.0f}", fill=(45, 50, 55))
    for forward in np.arange(0.0, map_height_m + 0.1, 1.0):
        _, y = point((forward, 0.0))
        draw.text((plot[0] - 28, y - 5), f"{forward:.0f}", fill=(45, 50, 55))
    draw.text((plot[0] + 4, 8), "CARE: obstacles + NoMaD candidates", fill=(20, 25, 30))
    draw.text((plot[0] - 50, plot[1] - 2), "x [m]", fill=(45, 50, 55))
    draw.text((center_x - 24, height - 33), "y [m]", fill=(45, 50, 55))
    draw.text((plot[0], height - 18), "+left", fill=(45, 50, 55))
    draw.text((plot[2] - 30, height - 18), "right-", fill=(45, 50, 55))

    visible_all_obstacles = 0
    all_obstacle_pixels = set()
    for obstacle in all_obstacles:
        forward, left = map(float, obstacle)
        if not (0.0 <= forward <= map_height_m):
            continue
        if abs(left) > half_width:
            continue
        all_obstacle_pixels.add(point(obstacle))
        visible_all_obstacles += 1
    all_radius = max(1, obstacle_radius_pixels - 1)
    for x, y in all_obstacle_pixels:
        draw.ellipse(
            (x - all_radius, y - all_radius, x + all_radius, y + all_radius),
            fill=(125, 132, 142),
        )

    visible_obstacles = 0
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
        visible_obstacles += 1

    candidate_count = 0
    selected_index = -1
    avoidance_active = False
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
    draw.text(
        (plot[0] + 8, plot[1] + 8),
        f"all={visible_all_obstacles}  CARE bins={visible_obstacles}  "
        f"candidates={candidate_count}  selected={selected_index}  "
        f"avoidance={'on' if avoidance_active else 'off'}",
        fill=(20, 25, 30),
    )
    legend_x = plot[2] - 185
    draw.rectangle(
        (legend_x, plot[1] + 8, legend_x + 11, plot[1] + 19),
        fill=(125, 132, 142),
    )
    draw.text((legend_x + 17, plot[1] + 7), "all points", fill=(20, 25, 30))
    draw.rectangle(
        (legend_x, plot[1] + 25, legend_x + 11, plot[1] + 36),
        fill=(225, 55, 55),
    )
    draw.text((legend_x + 17, plot[1] + 24), "CARE bins", fill=(20, 25, 30))
    draw.line(
        (legend_x, plot[1] + 48, legend_x + 11, plot[1] + 48),
        fill=(64, 156, 255),
        width=3,
    )
    draw.text((legend_x + 17, plot[1] + 42), "candidate", fill=(20, 25, 30))
    draw.line(
        (legend_x, plot[1] + 64, legend_x + 11, plot[1] + 64),
        fill=(
            (225, 70, 235)
            if avoidance_active
            else (255, 211, 40)
        ),
        width=4,
    )
    selected_label = (
        "CARE avoidance"
        if avoidance_active
        else "selected"
    )
    draw.text((legend_x + 17, plot[1] + 58), selected_label, fill=(20, 25, 30))
    return image
