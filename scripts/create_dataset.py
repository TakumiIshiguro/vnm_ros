#!/usr/bin/env python3
import math
import os
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import rospy

from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
from vnm_ros.utils.image_utils import center_crop_resize, msg_to_pil
from vnm_ros.utils.logger import info, warn


def required_bag_path(path):
    if not path:
        raise ValueError("collection.bag_path must be set")
    resolved = resolve_path(path, package_root())
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


def stamp_to_sec(msg, bag_time):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is not None:
        stamp_sec = stamp.to_sec()
        if stamp_sec > 0.0:
            return stamp_sec
    return bag_time.to_sec()


def pose_message_to_xy_yaw(msg):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    robot_position = np.array([position.x, position.y], dtype=np.float32)
    robot_yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )
    return robot_position, robot_yaw


def pose_input(collection_cfg, topics):
    pose_source = collection_cfg.get("pose_source", "odometry")
    if pose_source == "odometry":
        return pose_source, topics.get("odometry_topic", "/odom")
    if pose_source == "amcl":
        return pose_source, topics.get("amcl_pose_topic", "/amcl_pose")
    raise ValueError(f"pose_source must be odometry or amcl: {pose_source}")


def main():
    rospy.init_node("vnm_create_dataset")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    dataset_cfg = train_cfg["dataset"]
    collection_cfg = train_cfg["collection"]
    topics = cfg["topics"]

    dataset_type = collection_cfg["dataset_type"]
    if dataset_type not in ("train", "test"):
        raise ValueError(f"dataset_type must be train or test: {dataset_type}")
    trajectory_name = collection_cfg["trajectory_name"]
    name = trajectory_name or datetime.now().strftime("traj_%Y%m%d_%H%M%S")
    bag_path = required_bag_path(collection_cfg.get("bag_path", ""))
    data_dir_key = "train_data_dir" if dataset_type == "train" else "test_data_dir"
    data_dir = resolve_path(dataset_cfg[data_dir_key], package_root())
    trajectory_dir = os.path.join(data_dir, name)
    if os.path.exists(trajectory_dir):
        raise FileExistsError(trajectory_dir)
    os.makedirs(trajectory_dir)

    positions = []
    yaws = []
    cmd_dirs = []
    sample_dt = float(collection_cfg["sample_dt"])
    last_saved = float("-inf")
    image_topic = topics["image_topic"]
    cmd_dir_topic = topics.get("cmd_dir_topic")
    pose_source, pose_topic = pose_input(collection_cfg, topics)

    def finalize():
        if not positions:
            warn(f"no samples collected in {trajectory_dir}")
            return
        with open(os.path.join(trajectory_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(
                {
                    "position": np.asarray(positions, dtype=np.float32),
                    "yaw": np.asarray(yaws, dtype=np.float32),
                    "cmd_dir": np.asarray(cmd_dirs, dtype=np.float32),
                },
                f,
            )
        info(
            f"saved {dataset_type} trajectory {name}: "
            f"{len(positions)} samples"
        )

    import rosbag

    current_position = None
    current_yaw = None
    current_cmd_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    extension = collection_cfg.get("image_format", "jpg")
    image_size = tuple(model_cfg["image_size"])
    info(
        f"creating {dataset_type} trajectory {name} every "
        f"{sample_dt:.3f}s from bag {bag_path} using "
        f"{pose_source} pose {pose_topic} saved_image_size={image_size}"
    )
    with rosbag.Bag(bag_path, "r") as bag:
        bag_topics = [image_topic, pose_topic]
        if cmd_dir_topic:
            bag_topics.append(cmd_dir_topic)
        for topic, msg, bag_time in bag.read_messages(topics=bag_topics):
            msg_time = stamp_to_sec(msg, bag_time)
            if topic == pose_topic:
                current_position, current_yaw = pose_message_to_xy_yaw(msg)
                continue
            if topic == cmd_dir_topic:
                current_cmd_dir = np.asarray(msg.cmd_dir[:3], dtype=np.float32)
                continue
            if current_position is None or current_yaw is None:
                continue
            if msg_time - last_saved < sample_dt:
                continue
            index = len(positions)
            image = center_crop_resize(msg_to_pil(msg), image_size)
            image.save(os.path.join(trajectory_dir, f"{index}.{extension}"))
            positions.append(current_position.copy())
            yaws.append(float(current_yaw))
            cmd_dirs.append(current_cmd_dir.copy())
            last_saved = msg_time
            info(f"saved sample {index}")
    finalize()


if __name__ == "__main__":
    main()
