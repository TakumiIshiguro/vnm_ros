#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from scenario_navigation_msgs.msg import cmd_dir_intersection
from sensor_msgs.msg import Joy

from vnm_ros.control.joy_direction import DIRECTION_COMMANDS, direction_from_joy
from vnm_ros.utils.config import load_runtime_config


def make_msg(direction):
    msg = cmd_dir_intersection()
    msg.cmd_dir = DIRECTION_COMMANDS[direction]
    msg.intersection_label = [0] * 8
    msg.intersection_name = direction
    return msg


class ControllerCmdDirPublisher:
    def __init__(self):
        config_dir = rospy.get_param("~config_dir", None)
        cfg = load_runtime_config(config_dir)
        self.output_topic = cfg["topics"]["cmd_dir_topic"]
        self.joy_topic = cfg["topics"].get("joy_topic", "/joy")
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.stale_timeout = float(rospy.get_param("~stale_timeout", 0.5))
        self.mapping = {
            "horizontal_axis": int(rospy.get_param("~horizontal_axis", 6)),
            "fallback_horizontal_axis": int(
                rospy.get_param("~fallback_horizontal_axis", 0)
            ),
            "axis_threshold": float(rospy.get_param("~axis_threshold", 0.5)),
            "invert_axis": bool(rospy.get_param("~invert_axis", False)),
            "straight_button": int(rospy.get_param("~straight_button", -1)),
            "left_button": int(rospy.get_param("~left_button", -1)),
            "right_button": int(rospy.get_param("~right_button", -1)),
        }
        self.direction = "straight"
        self.last_joy_time = None
        self.publisher = rospy.Publisher(
            self.output_topic,
            cmd_dir_intersection,
            queue_size=1,
        )
        self.subscriber = rospy.Subscriber(
            self.joy_topic,
            Joy,
            self.joy_callback,
            queue_size=1,
        )

    def joy_callback(self, msg):
        direction = direction_from_joy(msg.axes, msg.buttons, **self.mapping)
        if direction != self.direction:
            rospy.loginfo(f"controller cmd_dir: {self.direction} -> {direction}")
        self.direction = direction
        self.last_joy_time = time.monotonic()

    def run(self):
        rospy.loginfo(
            f"controller cmd_dir: {self.joy_topic} -> {self.output_topic} "
            f"at {self.rate_hz:.1f} Hz"
        )
        rospy.loginfo(
            f"horizontal_axis={self.mapping['horizontal_axis']} "
            f"fallback_horizontal_axis={self.mapping['fallback_horizontal_axis']} "
            f"axis_threshold={self.mapping['axis_threshold']:.2f} "
            f"invert_axis={self.mapping['invert_axis']}"
        )
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            direction = self.direction
            if (
                self.last_joy_time is None
                or time.monotonic() - self.last_joy_time > self.stale_timeout
            ):
                direction = "straight"
            self.publisher.publish(make_msg(direction))
            rate.sleep()


def main():
    rospy.init_node("vnm_controller_cmd_dir")
    ControllerCmdDirPublisher().run()


if __name__ == "__main__":
    main()
