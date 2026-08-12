#!/usr/bin/env python3
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray

from vnm_ros.care import decode_action_candidates, render_care_bev
from vnm_ros.utils.config import load_care_config
from vnm_ros.utils.image_utils import pil_to_msg


class CareOverlayNode:
    def __init__(self, config):
        self.runtime = config["runtime"]
        self.map_config = config["map"]
        self.topics = config["topics"]
        self.lock = threading.Lock()
        self.obstacle_points = np.empty((0, 2), dtype=np.float32)
        self.all_obstacle_points = np.empty((0, 2), dtype=np.float32)
        self.candidates = None
        self.obstacle_stamp = None
        self.all_obstacle_stamp = None
        self.candidates_stamp = None

        self.publisher = rospy.Publisher(
            self.topics["care_bev_topic"],
            Image,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["obstacle_points_topic"],
            PointCloud2,
            self._obstacle_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["all_obstacle_points_topic"],
            PointCloud2,
            self._all_obstacle_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["action_candidates_topic"],
            Float32MultiArray,
            self._candidates_callback,
            queue_size=1,
        )

    def _decode_obstacle_cloud(self, message, label):
        expected_frame = self.runtime["robot_frame"]
        if message.header.frame_id != expected_frame:
            rospy.logwarn_throttle(
                5.0,
                "CARE rejected %s cloud in frame '%s'; expected '%s'",
                label,
                message.header.frame_id,
                expected_frame,
            )
            return None
        points = np.asarray(
            list(
                point_cloud2.read_points(
                    message,
                    field_names=("x", "y"),
                    skip_nans=True,
                )
            ),
            dtype=np.float32,
        )
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float32)
        else:
            points = points.reshape(-1, 2)
        return points

    def _obstacle_callback(self, message):
        points = self._decode_obstacle_cloud(message, "bin-selected obstacle")
        if points is None:
            return
        with self.lock:
            self.obstacle_points = points
            self.obstacle_stamp = rospy.Time.now()

    def _all_obstacle_callback(self, message):
        points = self._decode_obstacle_cloud(message, "all-obstacle")
        if points is None:
            return
        with self.lock:
            self.all_obstacle_points = points
            self.all_obstacle_stamp = rospy.Time.now()

    def _candidates_callback(self, message):
        candidates = decode_action_candidates(message.data)
        if candidates is None:
            rospy.logwarn_throttle(
                5.0,
                "CARE received an invalid action_candidates message",
            )
            return
        with self.lock:
            self.candidates = candidates
            self.candidates_stamp = rospy.Time.now()

    def _snapshot(self):
        now = rospy.Time.now()
        timeout = rospy.Duration(self.runtime["stale_timeout_seconds"])
        with self.lock:
            obstacles = self.obstacle_points.copy()
            all_obstacles = self.all_obstacle_points.copy()
            candidates = self.candidates
            obstacle_stamp = self.obstacle_stamp
            all_obstacle_stamp = self.all_obstacle_stamp
            candidates_stamp = self.candidates_stamp
        if obstacle_stamp is None or now - obstacle_stamp > timeout:
            obstacles = np.empty((0, 2), dtype=np.float32)
        if (
            all_obstacle_stamp is None
            or now - all_obstacle_stamp > timeout
        ):
            all_obstacles = np.empty((0, 2), dtype=np.float32)
        if candidates_stamp is None or now - candidates_stamp > timeout:
            candidates = None
        return obstacles, all_obstacles, candidates, now

    def publish_once(self):
        obstacles, all_obstacles, candidates, stamp = self._snapshot()
        if (
            candidates is None
            and len(obstacles) == 0
            and len(all_obstacles) == 0
        ):
            return
        image = render_care_bev(
            obstacles,
            candidates,
            map_width_m=self.map_config["width_m"],
            map_height_m=self.map_config["height_m"],
            image_size=tuple(self.map_config["image_size"]),
            grid_spacing_m=self.map_config["grid_spacing_m"],
            obstacle_radius_pixels=self.map_config[
                "obstacle_radius_pixels"
            ],
            all_obstacle_points=all_obstacles,
        )
        message = pil_to_msg(image)
        message.header.stamp = stamp
        message.header.frame_id = self.runtime["robot_frame"]
        self.publisher.publish(message)


def main():
    rospy.init_node("care_overlay")
    config = load_care_config(rospy.get_param("~config_dir", None))
    output_topic = str(
        rospy.get_param("~care_bev_topic_override", "")
    ).strip()
    if output_topic:
        config["topics"]["care_bev_topic"] = output_topic
    node = CareOverlayNode(config)
    rospy.loginfo(
        "CARE overlay: actions=%s CARE_bins=%s all_points=%s "
        "output=%s frame=%s",
        config["topics"]["action_candidates_topic"],
        config["topics"]["obstacle_points_topic"],
        config["topics"]["all_obstacle_points_topic"],
        config["topics"]["care_bev_topic"],
        config["runtime"]["robot_frame"],
    )
    rate = rospy.Rate(config["runtime"]["rate"])
    while not rospy.is_shutdown():
        node.publish_once()
        rate.sleep()


if __name__ == "__main__":
    main()
