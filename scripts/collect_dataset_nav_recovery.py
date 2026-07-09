#!/usr/bin/env python3
import math
import os
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from scenario_navigation_msgs.msg import cmd_dir_intersection
from sensor_msgs.msg import Image

from vnm_ros.datasets.cmd_dir_utils import CmdDirHoldFilter
from vnm_ros.utils.config import load_runtime_config, model_dataset_dir, package_root
from vnm_ros.utils.image_utils import msg_to_pil, preprocess_image
from vnm_ros.utils.logger import info, warn


def stamp_to_sec(msg):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is not None:
        stamp_sec = stamp.to_sec()
        if stamp_sec > 0.0:
            return stamp_sec
    return rospy.get_time()


def pose_message_to_xy_yaw(msg):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    robot_position = np.array([position.x, position.y], dtype=np.float32)
    robot_yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )
    return robot_position, robot_yaw


def pose_topic_and_type(collection_cfg, topics):
    pose_source = collection_cfg.get("pose_source", "amcl")
    if pose_source == "amcl":
        return pose_source, topics.get("amcl_pose_topic", "/mcl_pose"), PoseWithCovarianceStamped
    if pose_source == "odometry":
        return pose_source, topics.get("odometry_topic", "/odom"), Odometry
    raise ValueError(f"pose_source must be amcl or odometry: {pose_source}")


def required_trajectory_samples(model_cfg, dataset_cfg):
    context_size = int(model_cfg["context_size"])
    len_traj_pred = int(model_cfg["len_traj_pred"])
    waypoint_spacing = int(dataset_cfg.get("waypoint_spacing", 1))
    return (context_size + len_traj_pred) * waypoint_spacing + 1


def required_cmd_dir_hold_samples(model_cfg, dataset_cfg):
    len_traj_pred = int(model_cfg["len_traj_pred"])
    waypoint_spacing = int(dataset_cfg.get("waypoint_spacing", 1))
    return len_traj_pred * waypoint_spacing


def resolve_cmd_dir_hold_samples(model_cfg, dataset_cfg):
    required = required_cmd_dir_hold_samples(model_cfg, dataset_cfg)
    configured = dataset_cfg.get("cmd_dir_hold_samples_after_change", "auto")
    if configured is None or configured == "" or str(configured).lower() == "auto":
        return required, required, "auto"

    configured = int(configured)
    if configured < required:
        warn(
            f"cmd_dir_hold_samples_after_change={configured} is too small for "
            f"len_traj_pred={model_cfg['len_traj_pred']} "
            f"waypoint_spacing={dataset_cfg.get('waypoint_spacing', 1)}; "
            f"using {required}"
        )
        return required, configured, "raised"
    return configured, required, "manual"


def resolve_min_trajectory_samples(collection_cfg, model_cfg, dataset_cfg):
    required = required_trajectory_samples(model_cfg, dataset_cfg)
    configured = collection_cfg.get("min_trajectory_samples", "auto")
    if configured is None or configured == "" or str(configured).lower() == "auto":
        return required, required, "auto"

    configured = int(configured)
    if configured < required:
        warn(
            f"min_trajectory_samples={configured} is too small for "
            f"context_size={model_cfg['context_size']} "
            f"len_traj_pred={model_cfg['len_traj_pred']} "
            f"waypoint_spacing={dataset_cfg.get('waypoint_spacing', 1)}; "
            f"using {required}"
        )
        return required, configured, "raised"
    return configured, required, "manual"


class NavRecoveryDatasetCollector:
    def __init__(self):
        rospy.init_node("vnm_collect_dataset_nav_recovery")
        config_dir = rospy.get_param("~config_dir", None)
        cfg = load_runtime_config(config_dir)
        self.topics = cfg["topics"]
        self.train_cfg = cfg["train"]
        self.model_cfg = cfg["model"]
        self.dataset_cfg = self.train_cfg["dataset"]
        self.collection_cfg = self.train_cfg["collection"]

        dataset_type = self.collection_cfg["dataset_type"]
        if dataset_type not in ("train", "test"):
            raise ValueError(f"dataset_type must be train or test: {dataset_type}")
        self.trajectory_name_prefix = self.collection_cfg["trajectory_name"]
        self.data_dir = model_dataset_dir(
            self.dataset_cfg, dataset_type, self.model_cfg["model_type"]
        )
        os.makedirs(self.data_dir, exist_ok=True)

        self.sample_dt = float(self.collection_cfg["sample_dt"])
        self.extension = self.collection_cfg.get("image_format", "jpg")
        self.image_size = tuple(self.model_cfg["image_size"])
        self.center_crop = bool(self.model_cfg.get("image_center_crop", False))
        (
            self.cmd_dir_hold_samples_after_change,
            self.required_cmd_dir_hold_samples,
            self.cmd_dir_hold_samples_source,
        ) = resolve_cmd_dir_hold_samples(self.model_cfg, self.dataset_cfg)
        self.recovery_start_distance = float(
            self.collection_cfg.get("recovery_start_distance", 0.145)
        )
        self.vint_resume_distance = float(
            self.collection_cfg.get("vint_resume_distance", 0.1)
        )
        if self.vint_resume_distance > self.recovery_start_distance:
            raise ValueError("vint_resume_distance must be <= recovery_start_distance")
        self.angular_recovery_enabled = bool(
            self.collection_cfg.get("angular_recovery_enabled", True)
        )
        self.angular_recovery_start_error = float(
            self.collection_cfg.get("angular_recovery_start_error", 0.4)
        )
        self.angular_recovery_resume_error = float(
            self.collection_cfg.get("angular_recovery_resume_error", 0.1)
        )
        if self.angular_recovery_resume_error > self.angular_recovery_start_error:
            raise ValueError(
                "angular_recovery_resume_error must be <= angular_recovery_start_error"
            )

        self.nav_path_topic = self.collection_cfg.get(
            "nav_path_topic", "/move_base/NavfnROS/plan"
        )
        self.nav_cmd_vel_topic = self.collection_cfg.get("nav_cmd_vel_topic", "/nav_vel")
        self.vint_cmd_vel_topic = self.collection_cfg.get(
            "vint_cmd_vel_topic", self.topics.get("cmd_vel_debug_topic", "/vnm/cmd_vel_debug")
        )
        self.publish_zero_when_idle = bool(
            self.collection_cfg.get("publish_zero_when_idle", True)
        )
        self.split_trajectory_on_mode_switch = bool(
            self.collection_cfg.get("split_trajectory_on_mode_switch", False)
        )
        self.split_trajectory_on_save_gap = bool(
            self.collection_cfg.get("split_trajectory_on_save_gap", True)
        )
        self.trajectory_save_gap_factor = float(
            self.collection_cfg.get("trajectory_save_gap_factor", 2.5)
        )
        if self.trajectory_save_gap_factor <= 1.0:
            raise ValueError("trajectory_save_gap_factor must be > 1.0")
        (
            self.min_trajectory_samples,
            self.required_trajectory_samples,
            self.min_trajectory_samples_source,
        ) = resolve_min_trajectory_samples(
            self.collection_cfg, self.model_cfg, self.dataset_cfg
        )
        self.output_cmd_vel_topic = self.topics.get("cmd_vel_topic", "/cmd_vel")
        self.image_topic = self.topics["image_topic"]
        self.cmd_dir_topic = self.topics.get("cmd_dir_topic", "/cmd_dir_intersection")
        self.pose_source, self.pose_topic, self.pose_type = pose_topic_and_type(
            self.collection_cfg, self.topics
        )

        self.latest_image = None
        self.latest_image_time = None
        self.current_position = None
        self.current_yaw = None
        self.current_cmd_dir = None
        self.current_path = None
        self.path_distance = None
        self.vint_cmd_vel = None
        self.nav_cmd_vel = None
        self.mode = "vint"
        self.inputs_ready = False
        self.segment_index = 0
        self.name = None
        self.trajectory_dir = None
        self.last_saved = float("-inf")
        self.trajectory_switch_pending = False
        self.cmd_dir_filter = CmdDirHoldFilter(self.cmd_dir_hold_samples_after_change)

        self.positions = []
        self.yaws = []
        self.cmd_dirs = []

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.Subscriber(self.pose_topic, self.pose_type, self.pose_callback, queue_size=1)
        rospy.Subscriber(self.cmd_dir_topic, cmd_dir_intersection, self.cmd_dir_callback, queue_size=1)
        rospy.Subscriber(self.nav_path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(self.vint_cmd_vel_topic, Twist, self.vint_cmd_vel_callback, queue_size=1)
        rospy.Subscriber(self.nav_cmd_vel_topic, Twist, self.nav_cmd_vel_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher(self.output_cmd_vel_topic, Twist, queue_size=1)
        rospy.on_shutdown(self.finalize)

        info(
            f"online collection data_dir={self.data_dir} "
            f"pose={self.pose_source}:{self.pose_topic} image={self.image_topic} "
            f"cmd_dir={self.cmd_dir_topic} vint_cmd={self.vint_cmd_vel_topic} "
            f"nav_cmd={self.nav_cmd_vel_topic} output={self.output_cmd_vel_topic} "
            f"recovery_start_distance={self.recovery_start_distance:.3f} "
            f"vint_resume_distance={self.vint_resume_distance:.3f} "
            f"angular_recovery_enabled={self.angular_recovery_enabled} "
            f"angular_recovery_start_error={self.angular_recovery_start_error:.3f} "
            f"angular_recovery_resume_error={self.angular_recovery_resume_error:.3f} "
            f"min_trajectory_samples={self.min_trajectory_samples} "
            f"required_trajectory_samples={self.required_trajectory_samples} "
            f"min_trajectory_samples_source={self.min_trajectory_samples_source} "
            f"cmd_dir_hold_samples_after_change={self.cmd_dir_hold_samples_after_change} "
            f"required_cmd_dir_hold_samples={self.required_cmd_dir_hold_samples} "
            f"cmd_dir_hold_samples_source={self.cmd_dir_hold_samples_source} "
            f"split_trajectory_on_mode_switch={self.split_trajectory_on_mode_switch} "
            f"split_trajectory_on_save_gap={self.split_trajectory_on_save_gap} "
            f"trajectory_save_gap_factor={self.trajectory_save_gap_factor:.3f} "
            f"publish_zero_when_idle={self.publish_zero_when_idle}"
            f" saved_image_size={self.image_size} center_crop={self.center_crop}"
        )

    def image_callback(self, msg):
        self.latest_image = msg
        self.latest_image_time = stamp_to_sec(msg)

    def pose_callback(self, msg):
        self.current_position, self.current_yaw = pose_message_to_xy_yaw(msg)
        self.update_path_distance()

    def cmd_dir_callback(self, msg):
        self.current_cmd_dir = np.asarray(msg.cmd_dir[:3], dtype=np.float32)

    def path_callback(self, msg):
        self.current_path = msg
        self.update_path_distance()

    def vint_cmd_vel_callback(self, msg):
        self.vint_cmd_vel = msg

    def nav_cmd_vel_callback(self, msg):
        self.nav_cmd_vel = msg

    def update_path_distance(self):
        if self.current_position is None or self.current_path is None:
            self.path_distance = None
            return
        poses = self.current_path.poses
        if not poses:
            self.path_distance = None
            return
        path_xy = np.array(
            [[pose.pose.position.x, pose.pose.position.y] for pose in poses],
            dtype=np.float32,
        )
        distances = np.linalg.norm(path_xy - self.current_position.reshape(1, 2), axis=1)
        self.path_distance = float(np.min(distances))

    def angular_error(self):
        if self.nav_cmd_vel is None or self.vint_cmd_vel is None:
            return None
        return abs(
            float(self.nav_cmd_vel.angular.z) - float(self.vint_cmd_vel.angular.z)
        )

    def update_mode(self):
        angular_error = self.angular_error()
        if self.path_distance is None and angular_error is None:
            return

        distance_requires_nav = (
            self.path_distance is not None
            and self.path_distance >= self.recovery_start_distance
        )
        angular_requires_nav = (
            self.angular_recovery_enabled
            and angular_error is not None
            and angular_error >= self.angular_recovery_start_error
        )
        distance_allows_vint = (
            self.path_distance is not None
            and self.path_distance <= self.vint_resume_distance
        )
        angular_allows_vint = (
            not self.angular_recovery_enabled
            or angular_error is None
            or angular_error <= self.angular_recovery_resume_error
        )

        if self.mode == "vint" and (distance_requires_nav or angular_requires_nav):
            self.mode = "nav"
            if self.trajectory_dir is None:
                self.start_trajectory()
            reasons = []
            if distance_requires_nav:
                reasons.append(f"path_distance={self.path_distance:.3f}")
            if angular_requires_nav:
                reasons.append(f"angular_error={angular_error:.3f}")
            info(f"mode=nav {' '.join(reasons)}")
        elif self.mode == "nav" and distance_allows_vint and angular_allows_vint:
            self.mode = "vint"
            if self.split_trajectory_on_mode_switch:
                self.handle_mode_switch_to_vint()
            angular_text = (
                "none" if angular_error is None else f"{angular_error:.3f}"
            )
            info(
                f"mode=vint path_distance={self.path_distance:.3f} "
                f"angular_error={angular_text}"
            )

    def selected_cmd_vel(self):
        self.update_mode()
        if self.mode == "nav":
            return self.nav_cmd_vel
        return self.vint_cmd_vel

    def missing_inputs(self):
        missing = []
        if self.vint_cmd_vel is None:
            missing.append(self.vint_cmd_vel_topic)
        if self.nav_cmd_vel is None:
            missing.append(self.nav_cmd_vel_topic)
        if self.current_position is None:
            missing.append(self.pose_topic)
        if self.current_path is None:
            missing.append(self.nav_path_topic)
        if self.latest_image is None:
            missing.append(self.image_topic)
        if self.current_cmd_dir is None:
            missing.append(self.cmd_dir_topic)
        return missing

    def ready_to_save(self):
        return (
            self.latest_image is not None
            and self.latest_image_time is not None
            and self.current_position is not None
            and self.current_yaw is not None
            and self.current_cmd_dir is not None
        )

    def should_save_sample(self):
        if self.mode == "nav":
            return True
        if self.trajectory_dir is None:
            return False
        if self.path_distance is None:
            return False
        return self.path_distance <= self.recovery_start_distance

    def save_sample(self):
        if not self.should_save_sample():
            return
        if not self.ready_to_save():
            return
        if self.trajectory_dir is None:
            self.start_trajectory()
        elif self.should_split_on_save_gap():
            gap = self.latest_image_time - self.last_saved
            info(
                f"trajectory save gap detected {self.name}: "
                f"gap={gap:.3f}s threshold="
                f"{self.sample_dt * self.trajectory_save_gap_factor:.3f}s"
            )
            self.finalize()
            self.start_trajectory()
        if self.latest_image_time - self.last_saved < self.sample_dt:
            return
        index = len(self.positions)
        image = preprocess_image(
            msg_to_pil(self.latest_image),
            self.image_size,
            center_crop=self.center_crop,
        )
        image.save(os.path.join(self.trajectory_dir, f"{index}.{self.extension}"))
        self.positions.append(self.current_position.copy())
        self.yaws.append(float(self.current_yaw))
        self.cmd_dirs.append(self.cmd_dir_filter.update(self.current_cmd_dir))
        self.last_saved = self.latest_image_time
        path_distance = "none" if self.path_distance is None else f"{self.path_distance:.3f}"
        angular_error = self.angular_error()
        angular_error = "none" if angular_error is None else f"{angular_error:.3f}"
        info(
            f"saved sample {index} mode={self.mode} "
            f"path_distance={path_distance} angular_error={angular_error}"
        )
        self.finalize_if_switch_ready()

    def should_split_on_save_gap(self):
        if not self.split_trajectory_on_save_gap:
            return False
        if not self.positions:
            return False
        if self.latest_image_time is None or self.last_saved == float("-inf"):
            return False
        return (
            self.latest_image_time - self.last_saved
            > self.sample_dt * self.trajectory_save_gap_factor
        )

    def publish_cmd_vel(self, msg):
        if msg is not None:
            self.cmd_vel_pub.publish(msg)
            return
        if self.publish_zero_when_idle:
            self.cmd_vel_pub.publish(Twist())

    def spin(self):
        rate = rospy.Rate(float(self.collection_cfg.get("control_rate", 10.0)))
        while not rospy.is_shutdown():
            missing_inputs = self.missing_inputs()
            if missing_inputs:
                self.inputs_ready = False
                missing = ", ".join(missing_inputs)
                rospy.logwarn_throttle(
                    2.0,
                    f"waiting for dataset collection inputs; publishing zero. missing=[{missing}]",
                )
                self.publish_cmd_vel(None)
                rate.sleep()
                continue

            if not self.inputs_ready:
                self.inputs_ready = True
                info("all dataset collection inputs are ready; starting control")

            selected = self.selected_cmd_vel()
            if selected is None:
                rospy.logwarn_throttle(
                    2.0,
                    "selected cmd_vel source is not available; publishing zero",
                )
            self.publish_cmd_vel(selected)
            self.save_sample()
            rate.sleep()

    def finalize(self):
        if not self.positions:
            if self.trajectory_dir is not None:
                try:
                    os.rmdir(self.trajectory_dir)
                    info(f"removed empty trajectory {self.name}")
                except OSError:
                    pass
                self.reset_trajectory()
            return
        traj_data = {
            "position": np.asarray(self.positions, dtype=np.float32),
            "yaw": np.asarray(self.yaws, dtype=np.float32),
            "cmd_dir": np.asarray(self.cmd_dirs, dtype=np.float32),
            "metadata": {
                "model_type": self.model_cfg["model_type"],
                "image_size": list(self.image_size),
                "image_center_crop": self.center_crop,
                "sample_dt": self.sample_dt,
                "source": "online",
                "image_topic": self.image_topic,
                "pose_source": self.pose_source,
                "pose_topic": self.pose_topic,
            },
        }
        with open(os.path.join(self.trajectory_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)
        info(f"saved online trajectory {self.name}: {len(self.positions)} samples")
        self.reset_trajectory()

    def request_trajectory_switch(self):
        if self.trajectory_dir is None:
            return
        self.trajectory_switch_pending = True
        info(
            f"trajectory switch pending {self.name}: "
            f"{len(self.positions)}/{self.min_trajectory_samples} samples"
        )

    def handle_mode_switch_to_vint(self):
        if self.trajectory_dir is None:
            return
        if len(self.positions) >= self.min_trajectory_samples:
            info(
                f"trajectory switch ready {self.name}: "
                f"{len(self.positions)}/{self.min_trajectory_samples} samples"
            )
            self.finalize()
            return
        self.request_trajectory_switch()

    def finalize_if_switch_ready(self):
        if not self.trajectory_switch_pending:
            return
        if len(self.positions) < self.min_trajectory_samples:
            return
        was_nav = self.mode == "nav"
        self.finalize()
        if was_nav:
            self.start_trajectory()

    def start_trajectory(self):
        if self.trajectory_dir is not None:
            return
        if self.trajectory_name_prefix:
            self.name = f"{self.trajectory_name_prefix}_{self.segment_index:04d}"
        else:
            stamp = datetime.now().strftime("traj_%Y%m%d_%H%M%S")
            self.name = f"{stamp}_{self.segment_index:04d}"
        self.segment_index += 1
        self.trajectory_dir = os.path.join(self.data_dir, self.name)
        if os.path.exists(self.trajectory_dir):
            raise FileExistsError(self.trajectory_dir)
        os.makedirs(self.trajectory_dir)
        self.last_saved = float("-inf")
        self.trajectory_switch_pending = False
        info(f"started trajectory {self.name}: {self.trajectory_dir}")

    def reset_trajectory(self):
        self.name = None
        self.trajectory_dir = None
        self.positions = []
        self.yaws = []
        self.cmd_dirs = []
        self.last_saved = float("-inf")
        self.trajectory_switch_pending = False
        self.cmd_dir_filter.reset()


if __name__ == "__main__":
    NavRecoveryDatasetCollector().spin()
