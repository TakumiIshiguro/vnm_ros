#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from geometry_msgs.msg import Twist
from PIL import ImageDraw
from sensor_msgs.msg import Image

from vnm_ros.utils.config import load_runtime_config
from vnm_ros.utils.image_utils import msg_to_pil, pil_to_msg

camera_image = None
cmd_vel = None
subgoal_image = None


def camera_callback(msg: Image):
    global camera_image
    camera_image = msg_to_pil(msg).convert("RGB")


def cmd_vel_callback(msg: Twist):
    global cmd_vel
    cmd_vel = msg


def subgoal_callback(msg: Image):
    global subgoal_image
    subgoal_image = msg_to_pil(msg).convert("RGB")


def draw_cmd_vel(draw: ImageDraw.ImageDraw, image_size, cmd, max_v, max_w):
    width, height = image_size
    origin = (width // 2, int(height * 0.90))
    max_len = int(height * 0.34)
    max_side = int(width * 0.28)

    if cmd is None:
        draw.text((12, height - 34), "waiting for cmd_vel", fill=(255, 230, 80))
        return

    v = float(cmd.linear.x)
    w = float(cmd.angular.z)
    v_ratio = 0.0 if max_v <= 0 else max(-1.0, min(1.0, v / max_v))
    w_ratio = 0.0 if max_w <= 0 else max(-1.0, min(1.0, w / max_w))

    arrow_len = max(18, int(abs(v_ratio) * max_len))
    if abs(v) < 1e-4:
        arrow_len = 34

    # Robot-view convention: positive angular.z turns left, so draw it to image-left.
    lateral = int(-w_ratio * max_side)
    direction = -1 if v >= 0 else 1
    end = (
        max(0, min(width - 1, origin[0] + lateral)),
        max(0, min(height - 1, origin[1] + direction * arrow_len)),
    )

    draw.line([origin, end], fill=(0, 255, 90), width=6)

    angle = math.atan2(end[1] - origin[1], end[0] - origin[0])
    head_len = 18
    head_angle = 0.55
    left = (
        int(end[0] - head_len * math.cos(angle - head_angle)),
        int(end[1] - head_len * math.sin(angle - head_angle)),
    )
    right = (
        int(end[0] - head_len * math.cos(angle + head_angle)),
        int(end[1] - head_len * math.sin(angle + head_angle)),
    )
    draw.polygon([end, left, right], fill=(0, 255, 90))
    draw.ellipse((origin[0] - 7, origin[1] - 7, origin[0] + 7, origin[1] + 7), fill=(80, 180, 255))

    draw.text((12, height - 34), f"cmd_vel  v={v:.3f} m/s  w={w:.3f} rad/s", fill=(0, 0, 0))


def paste_subgoal(base, subgoal):
    if subgoal is None:
        return
    width, height = base.size
    inset_w = max(120, width // 4)
    inset_h = int(inset_w * subgoal.height / max(1, subgoal.width))
    inset = subgoal.resize((inset_w, inset_h))
    x0 = width - inset_w - 12
    y0 = 12
    base.paste(inset, (x0, y0))
    draw = ImageDraw.Draw(base)
    draw.rectangle((x0, y0, x0 + inset_w, y0 + inset_h), outline=(255, 255, 0), width=3)
    draw.rectangle((x0, y0 + inset_h, x0 + inset_w, y0 + inset_h + 22), fill=(0, 0, 0))
    draw.text((x0 + 6, y0 + inset_h + 4), "subgoal", fill=(255, 255, 0))


def main():
    rospy.init_node("vnm_overlay_view")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    topics = cfg["topics"]
    robot = cfg["robot"]

    rospy.Subscriber(topics["image_topic"], Image, camera_callback, queue_size=1)
    rospy.Subscriber(topics["cmd_vel_debug_topic"], Twist, cmd_vel_callback, queue_size=1)
    rospy.Subscriber(topics["topomap_image_topic"], Image, subgoal_callback, queue_size=1)
    pub = rospy.Publisher(topics["annotated_image_topic"], Image, queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if camera_image is not None:
            annotated = camera_image.copy()
            draw = ImageDraw.Draw(annotated)
            draw_cmd_vel(
                draw,
                annotated.size,
                cmd_vel,
                max_v=float(robot["max_v"]),
                max_w=float(robot["max_w"]),
            )
            paste_subgoal(annotated, subgoal_image)
            pub.publish(pil_to_msg(annotated))
        rate.sleep()


if __name__ == "__main__":
    main()
