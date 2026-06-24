#!/usr/bin/env python3
import os
import select
import sys
import termios
import tty

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from scenario_navigation_msgs.msg import cmd_dir_intersection

from vnm_ros.utils.config import load_runtime_config


DIRECTIONS = {
    "w": ([1, 0, 0], "straight"),
    "s": ([1, 0, 0], "straight"),
    "a": ([0, 1, 0], "left"),
    "d": ([0, 0, 1], "right"),
}


def make_msg(cmd_dir, name):
    msg = cmd_dir_intersection()
    msg.cmd_dir = cmd_dir
    msg.intersection_label = [0] * 8
    msg.intersection_name = name
    return msg


def read_key(timeout):
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if readable:
        return sys.stdin.read(1)
    return None


def main():
    rospy.init_node("vnm_keyboard_cmd_dir")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    topic = cfg["topics"]["cmd_dir_topic"]
    rate_hz = float(rospy.get_param("~rate", 10.0))

    pub = rospy.Publisher(topic, cmd_dir_intersection, queue_size=1, latch=True)
    rate = rospy.Rate(rate_hz)
    current_cmd, current_name = DIRECTIONS["w"]

    print("keyboard cmd_dir publisher")
    print("  w/s: straight [1, 0, 0]")
    print("  a:   left     [0, 1, 0]")
    print("  d:   right    [0, 0, 1]")
    print("  q:   quit")
    print(f"publishing to {topic} at {rate_hz:.1f} Hz", flush=True)

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            key = read_key(0.0)
            if key == "q":
                break
            if key in DIRECTIONS:
                current_cmd, current_name = DIRECTIONS[key]
                print(f"cmd_dir={current_name} {current_cmd}", flush=True)

            pub.publish(make_msg(current_cmd, current_name))
            rate.sleep()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
