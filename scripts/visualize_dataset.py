#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from PIL import Image as PILImage
from PIL import ImageDraw
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from vnm_ros.datasets.cmd_dir_utils import hold_cmd_dir_changes
from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
from vnm_ros.utils.image_utils import pil_to_msg
from vnm_ros.utils.logger import info


def yaw_to_quaternion(yaw):
    half_yaw = 0.5 * yaw
    return math.sin(half_yaw), math.cos(half_yaw)


def pose_stamped(position, yaw, frame_id, stamp):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = float(position[0])
    pose.pose.position.y = float(position[1])
    pose.pose.orientation.z, pose.pose.orientation.w = yaw_to_quaternion(float(yaw))
    return pose


def cmd_dir_name(cmd_dir):
    if sum(float(value) for value in cmd_dir[:3]) <= 0.0:
        return "none"
    index = int(max(range(len(cmd_dir)), key=lambda i: cmd_dir[i]))
    return ["straight", "left", "right"][index] if index < 3 else "unknown"


def cmd_dir_index(cmd_dir):
    if sum(float(value) for value in cmd_dir[:3]) <= 0.0:
        return -1
    return int(max(range(len(cmd_dir)), key=lambda i: cmd_dir[i]))


def cmd_dir_color(index):
    colors = {
        -1: (0.55, 0.55, 0.55, 1.0),
        0: (0.1, 0.9, 0.1, 1.0),
        1: (0.1, 0.35, 1.0, 1.0),
        2: (1.0, 0.15, 0.1, 1.0),
    }
    return colors.get(index, (1.0, 1.0, 1.0, 1.0))


def direction_marker(marker_id, name, color, frame_id, stamp):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = f"dataset_cmd_dir_{name}"
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.16
    marker.scale.y = 0.16
    r, g, b, a = color
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a
    return marker


def direction_marker_array(positions, cmd_dirs, frame_id, stamp):
    markers = MarkerArray()
    if cmd_dirs is None:
        return markers
    markers.markers = [
        direction_marker(0, "none", cmd_dir_color(-1), frame_id, stamp),
        direction_marker(1, "straight", cmd_dir_color(0), frame_id, stamp),
        direction_marker(2, "left", cmd_dir_color(1), frame_id, stamp),
        direction_marker(3, "right", cmd_dir_color(2), frame_id, stamp),
    ]
    for position, cmd_dir in zip(positions, cmd_dirs):
        point = Point()
        point.x = float(position[0])
        point.y = float(position[1])
        point.z = 0.05
        index = cmd_dir_index(cmd_dir)
        marker_index = index + 1
        if 0 <= marker_index < len(markers.markers):
            markers.markers[marker_index].points.append(point)
    return markers


def draw_overlay(image, trajectory_name, index, sample_count, cmd_dir):
    draw = ImageDraw.Draw(image)
    lines = [f"{trajectory_name} {index + 1}/{sample_count}"]
    if cmd_dir is not None:
        values = ", ".join(f"{float(value):.0f}" for value in cmd_dir)
        lines.append(f"cmd_dir: {cmd_dir_name(cmd_dir)} [{values}]")
    margin = 8
    line_height = 0
    line_width = 0
    for line in lines:
        box = draw.textbbox((margin, margin), line)
        line_width = max(line_width, box[2] - box[0])
        line_height = max(line_height, box[3] - box[1])
    spacing = 2
    pad = 4
    rect = (
        margin - pad,
        margin - pad,
        margin + line_width + pad,
        margin + len(lines) * line_height + (len(lines) - 1) * spacing + pad,
    )
    draw.rectangle(rect, fill=(0, 0, 0))
    for line_index, line in enumerate(lines):
        y = margin + line_index * (line_height + spacing)
        draw.text((margin, y), line, fill=(255, 255, 255))
    return image


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


def trajectory_path(trajectory, frame_id, stamp):
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    path.poses = [
        pose_stamped(position, yaw, frame_id, stamp)
        for position, yaw in zip(trajectory["position"], trajectory["yaw"])
    ]
    return path


def trajectory_color(index):
    colors = [
        (0.1, 0.9, 0.1, 1.0),
        (0.1, 0.45, 1.0, 1.0),
        (1.0, 0.35, 0.1, 1.0),
        (1.0, 0.9, 0.1, 1.0),
        (0.8, 0.2, 1.0, 1.0),
        (0.1, 0.9, 0.85, 1.0),
    ]
    return colors[index % len(colors)]


def trajectory_marker_array(dataset, trajectory_names, frame_id, stamp):
    markers = MarkerArray()
    for marker_id, name in enumerate(trajectory_names):
        trajectory = dataset.trajectory(name)
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "dataset_trajectories"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.04
        r, g, b, a = trajectory_color(marker_id)
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        for position in trajectory["position"]:
            point = Point()
            point.x = float(position[0])
            point.y = float(position[1])
            point.z = 0.02
            marker.points.append(point)
        markers.markers.append(marker)
    return markers


def trajectory_cmd_dirs(trajectory, hold_samples_after_change):
    cmd_dirs = trajectory.get("cmd_dir")
    if cmd_dirs is None:
        return None
    return hold_cmd_dir_changes(cmd_dirs, hold_samples_after_change)


def all_direction_marker_array(dataset, trajectory_names, cmd_dir_cache, frame_id, stamp):
    markers = MarkerArray()
    markers.markers = [
        direction_marker(0, "none", cmd_dir_color(-1), frame_id, stamp),
        direction_marker(1, "straight", cmd_dir_color(0), frame_id, stamp),
        direction_marker(2, "left", cmd_dir_color(1), frame_id, stamp),
        direction_marker(3, "right", cmd_dir_color(2), frame_id, stamp),
    ]
    for name in trajectory_names:
        trajectory = dataset.trajectory(name)
        cmd_dirs = cmd_dir_cache.get(name)
        if cmd_dirs is None:
            continue
        for position, cmd_dir in zip(trajectory["position"], cmd_dirs):
            point = Point()
            point.x = float(position[0])
            point.y = float(position[1])
            point.z = 0.05
            index = cmd_dir_index(cmd_dir)
            marker_index = index + 1
            if 0 <= marker_index < len(markers.markers):
                markers.markers[marker_index].points.append(point)
    if all(not marker.points for marker in markers.markers):
        return MarkerArray()
    return markers


def main():
    rospy.init_node("vnm_visualize_dataset")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    model_cfg = cfg["model"]
    dataset_cfg = cfg["train"]["dataset"]
    visualization_cfg = cfg["visualization"]["dataset"]

    dataset_type = visualization_cfg["dataset_type"]
    if dataset_type not in ("train", "test"):
        raise ValueError(f"dataset_type must be train or test: {dataset_type}")

    data_dir_key = "train_data_dir" if dataset_type == "train" else "test_data_dir"
    data_dir = resolve_path(dataset_cfg[data_dir_key], package_root())
    requested_name = visualization_cfg["trajectory_name"]
    trajectory_names = select_trajectories(data_dir, requested_name)
    dataset = TrajectoryDataset(data_dir, trajectory_names)
    cmd_dir_hold_samples_after_change = int(
        dataset_cfg.get("cmd_dir_hold_samples_after_change", 0)
    )

    frame_id = visualization_cfg["frame_id"]
    rate = float(visualization_cfg["rate"])
    loop = bool(visualization_cfg["loop"])
    if rate <= 0.0:
        raise ValueError(f"rate must be positive: {rate}")

    image_pub = rospy.Publisher("~image", Image, queue_size=1)
    path_pub = rospy.Publisher("~path", Path, queue_size=1, latch=True)
    pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=1)
    cmd_dir_pub = rospy.Publisher("~cmd_dir", Float32MultiArray, queue_size=1)
    direction_markers_pub = rospy.Publisher(
        "~direction_markers", MarkerArray, queue_size=1, latch=True
    )
    trajectory_markers_pub = rospy.Publisher(
        "~trajectory_markers", MarkerArray, queue_size=1, latch=True
    )

    stamp = rospy.Time.now()
    first_trajectory = dataset.trajectory(trajectory_names[0])
    cmd_dir_cache = {
        name: trajectory_cmd_dirs(
            dataset.trajectory(name),
            cmd_dir_hold_samples_after_change,
        )
        for name in trajectory_names
    }
    path_pub.publish(trajectory_path(first_trajectory, frame_id, stamp))
    trajectory_markers_pub.publish(
        trajectory_marker_array(dataset, trajectory_names, frame_id, stamp)
    )
    direction_markers_pub.publish(
        all_direction_marker_array(dataset, trajectory_names, cmd_dir_cache, frame_id, stamp)
    )
    sample_counts = {
        name: len(dataset.trajectory(name)["position"]) for name in trajectory_names
    }
    waypoint_spacing = int(dataset_cfg.get("waypoint_spacing", 1))
    required_samples = (
        int(model_cfg.get("context_size", dataset_cfg.get("context_size", 0)))
        + int(model_cfg.get("len_traj_pred", dataset_cfg.get("len_traj_pred", 0)))
    ) * waypoint_spacing + 1
    too_short = [
        f"{name}({sample_counts[name]})"
        for name in trajectory_names
        if sample_counts[name] < required_samples
    ]
    total_samples = sum(sample_counts.values())
    has_cmd_dir = any(
        cmd_dir_cache.get(name) is not None for name in trajectory_names
    )
    print(
        f"loaded_dataset type={dataset_type} "
        f"class={type(dataset).__name__} "
        f"data_dir={data_dir} "
        f"trajectories={len(trajectory_names)} "
        f"samples={total_samples} "
        f"required_trainable_samples={required_samples} "
        f"has_cmd_dir={has_cmd_dir} "
        f"cmd_dir_hold_samples_after_change={cmd_dir_hold_samples_after_change}",
        flush=True,
    )
    if too_short:
        info(
            "visualization includes trajectories shorter than one trainable sample: "
            + ", ".join(too_short[:20])
            + (" ..." if len(too_short) > 20 else "")
        )
    info(
        f"visualizing {dataset_type} trajectories={len(trajectory_names)} "
        f"samples={total_samples} at {rate:.2f} Hz"
    )

    trajectory_index = 0
    sample_index = 0
    current_trajectory_name = trajectory_names[trajectory_index]
    playback_rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        trajectory_name = trajectory_names[trajectory_index]
        trajectory = dataset.trajectory(trajectory_name)
        sample_count = sample_counts[trajectory_name]
        cmd_dirs = cmd_dir_cache.get(trajectory_name)
        if trajectory_name != current_trajectory_name:
            path_pub.publish(trajectory_path(trajectory, frame_id, stamp))
            current_trajectory_name = trajectory_name
            info(f"visualizing trajectory {trajectory_name}: {sample_count} samples")
        with PILImage.open(dataset.image_path(trajectory_name, sample_index)) as source:
            image = source.convert("RGB")
        cmd_dir = cmd_dirs[sample_index] if cmd_dirs is not None else None
        image = draw_overlay(image, trajectory_name, sample_index, sample_count, cmd_dir)
        image_msg = pil_to_msg(image)
        image_msg.header.frame_id = frame_id
        image_msg.header.stamp = stamp
        image_pub.publish(image_msg)
        if cmd_dir is not None:
            cmd_dir_pub.publish(Float32MultiArray(data=list(map(float, cmd_dir))))
        pose_pub.publish(
            pose_stamped(
                trajectory["position"][sample_index],
                trajectory["yaw"][sample_index],
                frame_id,
                stamp,
            )
        )

        sample_index += 1
        if sample_index >= sample_count:
            sample_index = 0
            trajectory_index += 1
            if trajectory_index >= len(trajectory_names):
                if loop:
                    trajectory_index = 0
                else:
                    info("dataset playback finished")
                    rospy.spin()
                    return
        playback_rate.sleep()


if __name__ == "__main__":
    main()
