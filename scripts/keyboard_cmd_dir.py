#!/usr/bin/env python3
import os
import select
import sys
import termios
import time
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
DEFAULT_KEY = "w"


def make_msg(cmd_dir, name):
    msg = cmd_dir_intersection()
    msg.cmd_dir = cmd_dir
    msg.intersection_label = [0] * 8
    msg.intersection_name = name
    return msg


def read_latest_key(timeout):
    latest = None
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    while readable:
        latest = sys.stdin.read(1)
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    return latest


def main():
    rospy.init_node("vnm_keyboard_cmd_dir")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    topic = cfg["topics"]["cmd_dir_topic"]
    rate_hz = float(rospy.get_param("~rate", 10.0))
    hold_timeout = float(rospy.get_param("~hold_timeout", 0.35))

    pub = rospy.Publisher(topic, cmd_dir_intersection, queue_size=1)
    rate = rospy.Rate(rate_hz)
    active_key = DEFAULT_KEY
    last_key_time = 0.0

    print("keyboard cmd_dir publisher")
    print("  no key: straight [1, 0, 0]")
    print("  w/s:    straight [1, 0, 0]")
    print("  a:      left     [0, 1, 0]")
    print("  d:      right    [0, 0, 1]")
    print("  q:      quit")
    print(f"publishing to {topic} at {rate_hz:.1f} Hz", flush=True)
    print(f"hold_timeout={hold_timeout:.2f}s", flush=True)

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            key = read_latest_key(0.0)
            if key == "q":
                break
            if key in DIRECTIONS:
                active_key = key
                last_key_time = time.monotonic()

            if time.monotonic() - last_key_time > hold_timeout:
                active_key = DEFAULT_KEY

            cmd, name = DIRECTIONS[active_key]
            pub.publish(make_msg(cmd, name))
            rate.sleep()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
