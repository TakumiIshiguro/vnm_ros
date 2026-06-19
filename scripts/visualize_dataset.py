#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from PIL import Image as PILImage
from sensor_msgs.msg import Image

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

    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = rospy.Time.now()
    path.poses = [
        pose_stamped(position, yaw, frame_id, path.header.stamp)
        for position, yaw in zip(trajectory["position"], trajectory["yaw"])
    ]
    path_pub.publish(path)

    sample_count = len(path.poses)
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
        image_msg = pil_to_msg(image)
        image_msg.header.frame_id = frame_id
        image_msg.header.stamp = stamp
        image_pub.publish(image_msg)
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
