#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from PIL import ImageDraw
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from vnm_ros.utils.config import load_runtime_config
from vnm_ros.utils.image_utils import msg_to_pil, pil_to_msg

camera_image = None
subgoal_image = None
action_candidates = None
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


def draw_action_candidates(draw: ImageDraw.ImageDraw, image_size, candidates):
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
        x = float(waypoint[0])
        y = float(waypoint[1])
        return (
            int(max(0, min(width - 1, origin[0] - y * scale))),
            int(max(0, min(height - 1, origin[1] - x * scale))),
        )

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
    draw.text(
        (12, 12),
        f"actions  selected={selected}  candidates={len(actions)}",
        fill=(0, 0, 0),
    )


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
    pub = rospy.Publisher(topics["annotated_image_topic"], Image, queue_size=1)

    rate = rospy.Rate(float(overlay_cfg["rate"]))
    while not rospy.is_shutdown():
        if camera_image is not None:
            annotated = camera_image.copy()
            draw = ImageDraw.Draw(annotated)
            draw_action_candidates(draw, annotated.size, action_candidates)
            paste_subgoal(annotated, subgoal_image)
            pub.publish(pil_to_msg(annotated))
        rate.sleep()


if __name__ == "__main__":
    main()
