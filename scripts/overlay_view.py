#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from PIL import ImageDraw
from geometry_msgs.msg import Twist
from scenario_navigation_msgs.msg import cmd_dir_intersection
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from vnm_ros.utils.config import load_runtime_config
from vnm_ros.utils.image_utils import msg_to_pil, pil_to_msg

camera_image = None
subgoal_image = None
action_candidates = None
current_waypoint = None
current_target_dir = "none"
current_cmd_vel = None
vnm_cmd_vel = None
nav_cmd_vel = None
camera_image_size = None


def camera_callback(msg: Image):
    global camera_image
    image = msg_to_pil(msg).convert("RGB")
    if camera_image_size is not None:
        image = image.resize(camera_image_size)
    camera_image = image


def subgoal_callback(msg: Image):
    global subgoal_image
    subgoal_image = msg_to_pil(msg).convert("RGB")


def action_candidates_callback(msg: Float32MultiArray):
    global action_candidates
    action_candidates = decode_action_candidates(msg.data)


def waypoint_callback(msg: Float32MultiArray):
    global current_waypoint
    if len(msg.data) >= 2:
        current_waypoint = [float(msg.data[0]), float(msg.data[1])]
    else:
        current_waypoint = None


def cmd_dir_name(cmd_dir):
    if len(cmd_dir) < 3:
        return "none"
    active = [index for index, value in enumerate(cmd_dir[:3]) if value > 0]
    if len(active) != 1:
        return "none"
    return ["straight", "left", "right"][active[0]]


def cmd_dir_callback(msg: cmd_dir_intersection):
    global current_target_dir
    current_target_dir = cmd_dir_name(msg.cmd_dir)


def twist_pair(msg: Twist):
    return float(msg.linear.x), float(msg.angular.z)


def cmd_vel_callback(msg: Twist):
    global current_cmd_vel
    current_cmd_vel = twist_pair(msg)


def vnm_cmd_vel_callback(msg: Twist):
    global vnm_cmd_vel
    vnm_cmd_vel = twist_pair(msg)


def nav_cmd_vel_callback(msg: Twist):
    global nav_cmd_vel
    nav_cmd_vel = twist_pair(msg)


def command_distance(first, second):
    if first is None or second is None:
        return None
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def command_source():
    vnm_distance = command_distance(current_cmd_vel, vnm_cmd_vel)
    nav_distance = command_distance(current_cmd_vel, nav_cmd_vel)
    if vnm_distance is None and nav_distance is None:
        return "unknown"
    if nav_distance is None or (
        vnm_distance is not None and vnm_distance <= nav_distance
    ):
        return "vnm"
    return "nav"


def decode_action_candidates(data):
    if len(data) < 5:
        return None
    selected = int(data[0])
    waypoint_index = int(data[1])
    sample_count = int(data[2])
    horizon = int(data[3])
    dims = int(data[4])
    expected = 5 + sample_count * horizon * dims
    if sample_count <= 0 or horizon <= 0 or dims < 2 or len(data) < expected:
        return None

    values = data[5:expected]
    actions = []
    offset = 0
    for _ in range(sample_count):
        sample = []
        for _ in range(horizon):
            sample.append(values[offset : offset + dims])
            offset += dims
        actions.append(sample)
    return {
        "selected": max(0, min(selected, sample_count - 1)),
        "waypoint_index": max(0, min(waypoint_index, horizon - 1)),
        "actions": actions,
    }


def image_point(waypoint, image_size, scale, origin):
    width, height = image_size
    x = float(waypoint[0])
    y = float(waypoint[1])
    return (
        int(max(0, min(width - 1, origin[0] - y * scale))),
        int(max(0, min(height - 1, origin[1] - x * scale))),
    )


def draw_action_candidates(draw: ImageDraw.ImageDraw, image_size, candidates, waypoint):
    if candidates is None:
        return

    width, height = image_size
    origin = (width // 2, int(height * 0.92))
    actions = candidates["actions"]
    selected = candidates["selected"]
    waypoint_index = candidates["waypoint_index"]

    max_extent = 0.0
    for sample in actions:
        for waypoint in sample:
            max_extent = max(max_extent, abs(float(waypoint[0])), abs(float(waypoint[1])))
    if max_extent < 1e-6:
        return

    scale = min(width * 0.34, height * 0.38) / max_extent

    def point(waypoint):
        return image_point(waypoint, image_size, scale, origin)

    draw.ellipse(
        (origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5),
        fill=(255, 255, 255),
        outline=(0, 0, 0),
    )

    for index, sample in enumerate(actions):
        points = [point(waypoint) for waypoint in sample]
        if len(points) < 2:
            continue
        if index == selected:
            continue
        draw.line(points, fill=(80, 170, 255), width=2)
        draw.ellipse(
            (
                points[-1][0] - 3,
                points[-1][1] - 3,
                points[-1][0] + 3,
                points[-1][1] + 3,
            ),
            fill=(80, 170, 255),
        )

    selected_points = [point(waypoint) for waypoint in actions[selected]]
    if len(selected_points) >= 2:
        draw.line(selected_points, fill=(0, 0, 0), width=8)
        draw.line(selected_points, fill=(255, 220, 40), width=5)
    selected_waypoint = selected_points[waypoint_index]
    draw.ellipse(
        (
            selected_waypoint[0] - 8,
            selected_waypoint[1] - 8,
            selected_waypoint[0] + 8,
            selected_waypoint[1] + 8,
        ),
        fill=(255, 80, 40),
        outline=(0, 0, 0),
        width=2,
    )
    if waypoint is not None:
        published_waypoint = point(waypoint)
        draw.ellipse(
            (
                published_waypoint[0] - 5,
                published_waypoint[1] - 5,
                published_waypoint[0] + 5,
                published_waypoint[1] + 5,
            ),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )
    draw.text(
        (12, 12),
        f"actions  selected={selected}  candidates={len(actions)}",
        fill=(0, 0, 0),
    )


def draw_status(draw: ImageDraw.ImageDraw, target_dir: str):
    lines = [f"target_dir: {target_dir}", f"cmd_source: {command_source()}"]
    if current_cmd_vel is not None:
        lines.append(f"cmd_vel: v={current_cmd_vel[0]:+.2f} w={current_cmd_vel[1]:+.2f}")
    if vnm_cmd_vel is not None:
        lines.append(f"vnm: v={vnm_cmd_vel[0]:+.2f} w={vnm_cmd_vel[1]:+.2f}")
    if nav_cmd_vel is not None:
        lines.append(f"nav: v={nav_cmd_vel[0]:+.2f} w={nav_cmd_vel[1]:+.2f}")

    x = 12
    y = 34
    line_boxes = [draw.textbbox((x, y), line) for line in lines]
    width = max(box[2] - box[0] for box in line_boxes)
    line_height = max(box[3] - box[1] for box in line_boxes)
    height = len(lines) * line_height + (len(lines) - 1) * 2
    draw.rectangle(
        (x - 4, y - 3, x + width + 4, y + height + 3),
        fill=(255, 255, 255),
        outline=(0, 0, 0),
    )
    for index, line in enumerate(lines):
        draw.text((x, y + index * (line_height + 2)), line, fill=(0, 0, 0))


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
    overlay_cfg = cfg["visualization"]["overlay"]
    model_cfg = cfg["model"]

    global camera_image_size
    camera_image_size = tuple(overlay_cfg.get("image_size", model_cfg["image_size"]))

    rospy.Subscriber(topics["image_topic"], Image, camera_callback, queue_size=1)
    rospy.Subscriber(topics["topomap_image_topic"], Image, subgoal_callback, queue_size=1)
    rospy.Subscriber(
        topics.get("action_candidates_topic", "/vnm/action_candidates"),
        Float32MultiArray,
        action_candidates_callback,
        queue_size=1,
    )
    rospy.Subscriber(topics["waypoint_topic"], Float32MultiArray, waypoint_callback, queue_size=1)
    rospy.Subscriber(topics["cmd_dir_topic"], cmd_dir_intersection, cmd_dir_callback, queue_size=1)
    rospy.Subscriber(topics["cmd_vel_topic"], Twist, cmd_vel_callback, queue_size=1)
    rospy.Subscriber(topics["cmd_vel_debug_topic"], Twist, vnm_cmd_vel_callback, queue_size=1)
    rospy.Subscriber(
        cfg["train"]["collection"].get("nav_cmd_vel_topic", "/nav_vel"),
        Twist,
        nav_cmd_vel_callback,
        queue_size=1,
    )
    pub = rospy.Publisher(topics["annotated_image_topic"], Image, queue_size=1)

    rate = rospy.Rate(float(overlay_cfg["rate"]))
    while not rospy.is_shutdown():
        if camera_image is not None:
            annotated = camera_image.copy()
            draw = ImageDraw.Draw(annotated)
            draw_action_candidates(draw, annotated.size, action_candidates, current_waypoint)
            draw_status(draw, current_target_dir)
            paste_subgoal(annotated, subgoal_image)
            pub.publish(pil_to_msg(annotated))
        rate.sleep()


if __name__ == "__main__":
    main()
