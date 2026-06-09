#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
from vnm_ros.utils.image_utils import msg_to_pil
from vnm_ros.utils.logger import info, warn

latest_image = None
latest_position = None
latest_yaw = None


def optional_path(path):
    if not path:
        return None
    return resolve_path(path, package_root())


def stamp_to_sec(msg, bag_time):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is not None:
        stamp_sec = stamp.to_sec()
        if stamp_sec > 0.0:
            return stamp_sec
    return bag_time.to_sec()


def image_callback(msg: Image):
    global latest_image
    latest_image = msg_to_pil(msg).convert("RGB")


def odometry_to_pose(msg: Odometry):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    robot_position = np.array([position.x, position.y], dtype=np.float32)
    robot_yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )
    return robot_position, robot_yaw


def odometry_callback(msg: Odometry):
    global latest_position, latest_yaw
    latest_position, latest_yaw = odometry_to_pose(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--trajectory-name", default=None)
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--bag-path", default=None)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("vnm_create_dataset")
    cfg = load_runtime_config(args.config_dir)
    train_cfg = cfg["train"]
    dataset_cfg = train_cfg["dataset"]
    collection_cfg = train_cfg["collection"]
    topics = cfg["topics"]

    name = args.trajectory_name or datetime.now().strftime("traj_%Y%m%d_%H%M%S")
    data_dir_key = "train_data_dir" if args.dataset == "train" else "test_data_dir"
    data_dir = resolve_path(dataset_cfg[data_dir_key], package_root())
    trajectory_dir = os.path.join(data_dir, name)
    if os.path.exists(trajectory_dir):
        raise FileExistsError(trajectory_dir)
    os.makedirs(trajectory_dir)

    positions = []
    yaws = []
    sample_dt = float(collection_cfg["sample_dt"])
    last_saved = float("-inf")
    image_topic = topics["image_topic"]
    odometry_topic = topics.get("odometry_topic", "/odom")
    bag_path = (
        optional_path(args.bag_path)
        or optional_path(collection_cfg.get(f"{args.dataset}_bag_path", ""))
        or optional_path(collection_cfg.get("bag_path", ""))
    )

    def finalize():
        if not positions:
            warn(f"no samples collected in {trajectory_dir}")
            return
        with open(os.path.join(trajectory_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(
                {
                    "position": np.asarray(positions, dtype=np.float32),
                    "yaw": np.asarray(yaws, dtype=np.float32),
                },
                f,
            )
        info(
            f"saved {args.dataset} trajectory {name}: "
            f"{len(positions)} samples"
        )

    if bag_path:
        import rosbag

        current_position = None
        current_yaw = None
        extension = collection_cfg.get("image_format", "jpg")
        info(
            f"creating {args.dataset} trajectory {name} every "
            f"{sample_dt:.3f}s from bag {bag_path}"
        )
        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, bag_time in bag.read_messages(topics=[image_topic, odometry_topic]):
                msg_time = stamp_to_sec(msg, bag_time)
                if topic == odometry_topic:
                    current_position, current_yaw = odometry_to_pose(msg)
                    continue
                if current_position is None or current_yaw is None:
                    continue
                if msg_time - last_saved < sample_dt:
                    continue
                index = len(positions)
                image = msg_to_pil(msg).convert("RGB")
                image.save(os.path.join(trajectory_dir, f"{index}.{extension}"))
                positions.append(current_position.copy())
                yaws.append(float(current_yaw))
                last_saved = msg_time
                info(f"saved sample {index}")
        finalize()
        return

    rospy.Subscriber(image_topic, Image, image_callback, queue_size=1)
    rospy.Subscriber(odometry_topic, Odometry, odometry_callback, queue_size=1)

    rospy.on_shutdown(finalize)
    rate = rospy.Rate(max(10.0 / sample_dt, 10.0))
    info(
        f"recording {args.dataset} trajectory {name} every "
        f"{sample_dt:.3f}s from {image_topic}"
    )

    while not rospy.is_shutdown():
        now = time.monotonic()
        if (
            latest_image is not None
            and latest_position is not None
            and latest_yaw is not None
            and now - last_saved >= sample_dt
        ):
            index = len(positions)
            extension = collection_cfg.get("image_format", "jpg")
            latest_image.save(os.path.join(trajectory_dir, f"{index}.{extension}"))
            positions.append(latest_position.copy())
            yaws.append(float(latest_yaw))
            last_saved = now
            info(f"saved sample {index}")
        rate.sleep()


if __name__ == "__main__":
    main()
