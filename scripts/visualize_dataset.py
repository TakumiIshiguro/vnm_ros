#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from PIL import Image as PILImage
from PIL import ImageDraw
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

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
    index = int(max(range(len(cmd_dir)), key=lambda i: cmd_dir[i]))
    return ["straight", "left", "right"][index] if index < 3 else "unknown"


def cmd_dir_index(cmd_dir):
    return int(max(range(len(cmd_dir)), key=lambda i: cmd_dir[i]))


def cmd_dir_color(index):
    colors = {
        0: (0.1, 0.9, 0.1, 1.0),
        1: (0.1, 0.35, 1.0, 1.0),
        2: (1.0, 0.15, 0.1, 1.0),
    }
    return colors.get(index, (1.0, 1.0, 1.0, 1.0))


def direction_marker(marker_id, position, cmd_dir, frame_id, stamp):
    index = cmd_dir_index(cmd_dir)
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "dataset_cmd_dir"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = 0.05
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.18
    marker.scale.y = 0.18
    marker.scale.z = 0.18
    r, g, b, a = cmd_dir_color(index)
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
        direction_marker(i, position, cmd_dir, frame_id, stamp)
        for i, (position, cmd_dir) in enumerate(zip(positions, cmd_dirs))
    ]
    return markers


def draw_cmd_dir(image, cmd_dir):
    if cmd_dir is None:
        return image
    draw = ImageDraw.Draw(image)
    values = ", ".join(f"{float(value):.0f}" for value in cmd_dir)
    text = f"cmd_dir: {cmd_dir_name(cmd_dir)} [{values}]"
    margin = 8
    box = draw.textbbox((margin, margin), text)
    pad = 4
    rect = (
        box[0] - pad,
        box[1] - pad,
        box[2] + pad,
        box[3] + pad,
    )
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((margin, margin), text, fill=(255, 255, 255))
    return image


def select_trajectory(data_dir, requested_name):
    if requested_name:
        return requested_name
    names = sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    )
    if not names:
        raise ValueError(f"No trajectory directories found in {data_dir}")
    return names[-1]


def main():
    rospy.init_node("vnm_visualize_dataset")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    dataset_cfg = cfg["train"]["dataset"]
    visualization_cfg = cfg["visualization"]["dataset"]

    dataset_type = visualization_cfg["dataset_type"]
    if dataset_type not in ("train", "test"):
        raise ValueError(f"dataset_type must be train or test: {dataset_type}")

    data_dir_key = "train_data_dir" if dataset_type == "train" else "test_data_dir"
    data_dir = resolve_path(dataset_cfg[data_dir_key], package_root())
    requested_name = visualization_cfg["trajectory_name"]
    trajectory_name = select_trajectory(data_dir, requested_name)
    dataset = TrajectoryDataset(data_dir, [trajectory_name])
    trajectory = dataset.trajectory(trajectory_name)

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

    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = rospy.Time.now()
    path.poses = [
        pose_stamped(position, yaw, frame_id, path.header.stamp)
        for position, yaw in zip(trajectory["position"], trajectory["yaw"])
    ]
    path_pub.publish(path)

    sample_count = len(path.poses)
    cmd_dirs = trajectory.get("cmd_dir")
    direction_markers_pub.publish(
        direction_marker_array(
            trajectory["position"], cmd_dirs, frame_id, path.header.stamp
        )
    )
    print(
        f"loaded_dataset type={dataset_type} "
        f"class={type(dataset).__name__} "
        f"data_dir={data_dir} "
        f"trajectory={trajectory_name} "
        f"samples={sample_count} "
        f"has_cmd_dir={cmd_dirs is not None}",
        flush=True,
    )
    info(
        f"visualizing {dataset_type} trajectory {trajectory_name}: "
        f"{sample_count} samples at {rate:.2f} Hz"
    )

    index = 0
    playback_rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        with PILImage.open(dataset.image_path(trajectory_name, index)) as source:
            image = source.convert("RGB")
        cmd_dir = cmd_dirs[index] if cmd_dirs is not None else None
        image = draw_cmd_dir(image, cmd_dir)
        image_msg = pil_to_msg(image)
        image_msg.header.frame_id = frame_id
        image_msg.header.stamp = stamp
        image_pub.publish(image_msg)
        if cmd_dir is not None:
            cmd_dir_pub.publish(Float32MultiArray(data=list(map(float, cmd_dir))))
        pose_pub.publish(
            pose_stamped(
                trajectory["position"][index],
                trajectory["yaw"][index],
                frame_id,
                stamp,
            )
        )

        index += 1
        if index >= sample_count:
            if loop:
                index = 0
            else:
                info("dataset playback finished")
                rospy.spin()
                return
        playback_rate.sleep()


if __name__ == "__main__":
    main()
