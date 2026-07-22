#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from scenario_navigation_msgs.msg import cmd_dir_intersection
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

from vnm_ros.control.cmd_dir_action_selector import CmdDirActionSelector
from vnm_ros.control.waypoint_controller import WaypointController
from vnm_ros.models.vnm_model import VNMModel
from vnm_ros.ros.cmd_vel_publisher import CmdVelPublisher
from vnm_ros.ros.image_subscriber import ImageContextSubscriber
from vnm_ros.ros.ros_utils import action_candidates_msg, waypoint_marker, waypoint_msg
from vnm_ros.topomap.subgoal_selector import SubgoalSelector
from vnm_ros.topomap.topomap import Topomap
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
from vnm_ros.utils.image_utils import pil_to_msg
from vnm_ros.utils.logger import info


def optional_bool(value, name):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("", "auto", "default"):
        return None
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name} must be true, false, or auto: {value}")


def cmd_dir_name(cmd_dir):
    if cmd_dir is None or len(cmd_dir) < 3:
        return "none"
    cmd_dir = np.asarray(cmd_dir[:3])
    if np.count_nonzero(cmd_dir > 0) != 1:
        return "none"
    return ["straight", "left", "right"][int(np.argmax(cmd_dir))]


def delete_waypoint_marker(frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "vnm_waypoint"
    marker.id = 0
    marker.action = Marker.DELETE
    return marker


def main():
    rospy.init_node("vnm_node")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    topics = cfg["topics"]
    robot = dict(cfg["robot"])
    model_cfg = dict(cfg["model"])
    topomap_cfg = cfg["topomap"]

    use_pretrained_weights = optional_bool(
        rospy.get_param("~use_pretrained_weights", False),
        "use_pretrained_weights",
    )
    use_pretrained_weights = bool(use_pretrained_weights)
    checkpoint_override = str(
        rospy.get_param("~checkpoint_path_override", "")
    ).strip()
    if checkpoint_override:
        checkpoint = checkpoint_override
    elif use_pretrained_weights:
        checkpoint = str(
            cfg["train"].get("training", {}).get("pretrained_weights_path", "")
        ).strip()
        if not checkpoint:
            raise ValueError(
                "use_pretrained_weights=true requires "
                "training.pretrained_weights_path"
            )
    else:
        checkpoint = model_cfg["checkpoint_path"]
    model_cfg["checkpoint_path"] = checkpoint

    publish_cmd_vel_override = optional_bool(
        rospy.get_param("~publish_cmd_vel_override", "auto"),
        "publish_cmd_vel_override",
    )
    if publish_cmd_vel_override is not None:
        robot["publish_cmd_vel"] = publish_cmd_vel_override

    direction_conditioning_override = optional_bool(
        rospy.get_param("~direction_conditioning_override", "auto"),
        "direction_conditioning_override",
    )
    if direction_conditioning_override is not None:
        model_cfg["direction_conditioning"] = direction_conditioning_override

    navigation_mode_override = str(
        rospy.get_param("~navigation_mode_override", "")
    ).strip()
    if navigation_mode_override:
        robot["navigation_mode"] = navigation_mode_override

    action_sample_strategy_override = str(
        rospy.get_param("~action_sample_strategy_override", "")
    ).strip()
    if action_sample_strategy_override:
        model_cfg["action_sample_strategy"] = action_sample_strategy_override

    checkpoint = resolve_path(checkpoint, package_root())
    model = VNMModel(model_cfg, checkpoint)
    navigation_mode = robot.get("navigation_mode", "topomap")
    action_sample_strategy = model_cfg.get("action_sample_strategy", "first")
    select_by_cmd_dir = action_sample_strategy == "cmd_dir"
    use_cmd_dir_input = bool(model_cfg.get("direction_conditioning", False)) or select_by_cmd_dir
    if navigation_mode == "explore" and not (
        model_cfg["model_type"] == "nomad"
        or bool(model_cfg.get("direction_conditioning", False))
    ):
        raise ValueError(
            "navigation_mode=explore requires NoMaD or direction_conditioning=true"
        )

    topo = None
    subgoal_selector = None
    goal_node = None
    topomap_dir = None
    if navigation_mode != "explore":
        topomap_dir = topomap_cfg["topomap_dir"]
        if not topomap_dir.startswith("topomaps/"):
            topomap_dir = os.path.join("topomaps", topomap_dir)
        topo = Topomap(resolve_path(topomap_dir, package_root()))

        goal_node = int(topomap_cfg["goal_node"])
        if goal_node < 0:
            goal_node = len(topo) - 1
        subgoal_selector = SubgoalSelector(
            goal_node=goal_node,
            search_radius=int(topomap_cfg["search_radius"]),
            close_threshold=float(topomap_cfg["close_threshold"]),
        )

    image_sub = ImageContextSubscriber(topics["image_topic"], model.context_size)
    action_selector = CmdDirActionSelector(
        theta_threshold_deg=float(model_cfg.get("cmd_dir_theta_threshold_deg", 15.0))
    )
    if navigation_mode == "explore" and use_cmd_dir_input:
        rospy.Subscriber(
            topics["cmd_dir_topic"],
            cmd_dir_intersection,
            lambda msg: action_selector.update(msg.cmd_dir),
            queue_size=1,
        )
    waypoint_pub = rospy.Publisher(
        topics["waypoint_topic"], waypoint_msg([]).__class__, queue_size=1
    )
    action_candidates_pub = rospy.Publisher(
        topics.get("action_candidates_topic", "/vnm/action_candidates"),
        waypoint_msg([]).__class__,
        queue_size=1,
    )
    marker_pub = rospy.Publisher(topics["marker_topic"], Marker, queue_size=1)
    subgoal_pub = rospy.Publisher(topics["topomap_image_topic"], Image, queue_size=1)
    reached_pub = rospy.Publisher(topics["reached_goal_topic"], Bool, queue_size=1)
    cmd_debug_pub = rospy.Publisher(topics["cmd_vel_debug_topic"], Twist, queue_size=1)
    cmd_pub = CmdVelPublisher(topics["cmd_vel_topic"])
    controller = WaypointController(
        dt=1.0 / float(robot["model_rate"]),
        max_v=float(robot["max_v"]),
        max_w=float(robot["max_w"]),
    )

    def reset_context_callback(msg):
        if not msg.data:
            return
        image_sub.reset()
        waypoint_pub.publish(waypoint_msg([]))
        action_candidates_pub.publish(action_candidates_msg([], selected_sample=-1, waypoint_index=0))
        marker_pub.publish(delete_waypoint_marker(topics["frame_id"]))
        cmd_debug_pub.publish(Twist())
        reached_pub.publish(Bool(data=False))
        if robot.get("publish_cmd_vel", True):
            cmd_pub.stop()
        info("reset VNM context and local path outputs")

    rospy.Subscriber(
        topics.get("reset_context_topic", "/vnm/reset_context"),
        Bool,
        reset_context_callback,
        queue_size=1,
    )

    rate = rospy.Rate(float(robot["model_rate"]))
    waypoint_index = int(model_cfg["waypoint_index"])
    info(f"loaded {model_cfg['model_type']} from {checkpoint}")
    info(f"navigation_mode={navigation_mode}")
    info(
        f"runtime publish_cmd_vel={bool(robot.get('publish_cmd_vel', True))} "
        f"use_pretrained_weights={use_pretrained_weights}"
    )
    if navigation_mode == "explore":
        info(f"action_sample_strategy={action_sample_strategy}")
        if use_cmd_dir_input:
            info(f"direction_conditioning={model_cfg.get('direction_conditioning', False)}")
            info(
                f"cmd_dir_theta_threshold_deg="
                f"{model_cfg.get('cmd_dir_theta_threshold_deg', 15.0)}"
            )
    if topo is not None:
        info(f"loaded topomap {topomap_dir} with {len(topo)} nodes")
        info(f"goal_node={goal_node}")
    reached_logged = False
    waiting_for_cmd_dir_logged = False

    while not rospy.is_shutdown():
        image_sub.sample()
        reached = False
        if subgoal_selector is not None:
            reached = subgoal_selector.reached_goal()

        if navigation_mode == "explore" and use_cmd_dir_input and action_selector.cmd_dir is None:
            if not waiting_for_cmd_dir_logged:
                info(
                    "waiting for target direction cmd_dir; "
                    "VNM will not publish motion commands until it is received"
                )
                waiting_for_cmd_dir_logged = True
            cmd_debug_pub.publish(Twist())
            if robot.get("publish_cmd_vel", True):
                cmd_pub.stop()
            reached_pub.publish(Bool(data=False))
            rate.sleep()
            continue
        waiting_for_cmd_dir_logged = False

        if image_sub.ready() and not reached:
            if navigation_mode == "explore":
                cmd_dir = action_selector.cmd_dir if use_cmd_dir_input else None
                target_name = cmd_dir_name(cmd_dir)
                actions = model.predict_explore(image_sub.context(), cmd_dir=cmd_dir)
                if select_by_cmd_dir:
                    waypoint = action_selector.select(actions, waypoint_index)
                    selected_action = action_selector.selected_action
                    selected_sample = action_selector.selected_sample
                elif action_sample_strategy == "mean":
                    selected_action = actions.mean(axis=0)
                    selected_sample = -1
                    selected_waypoint_index = min(waypoint_index, actions.shape[1] - 1)
                    waypoint = selected_action[selected_waypoint_index]
                    info(f"target_dir={target_name} mode=explore selected_sample=mean")
                else:
                    waypoint = actions[0, min(waypoint_index, actions.shape[1] - 1)]
                    selected_action = actions[0]
                    selected_sample = 0
                    info(f"target_dir={target_name} mode=explore selected_sample=0 target_angle=None")
                actions = model.scale_waypoint(
                    actions,
                    max_v=float(robot["max_v"]),
                    model_rate=float(robot["model_rate"]),
                )
                selected_action = model.scale_waypoint(
                    selected_action,
                    max_v=float(robot["max_v"]),
                    model_rate=float(robot["model_rate"]),
                )
                selected_waypoint_index = min(waypoint_index, actions.shape[1] - 1)
                waypoint = selected_action[selected_waypoint_index]
                display_actions = actions
                display_selected_sample = selected_sample
                if selected_sample < 0:
                    display_actions = np.concatenate(
                        [actions, selected_action[np.newaxis, :, :]], axis=0
                    )
                    display_selected_sample = display_actions.shape[0] - 1
                if select_by_cmd_dir:
                    info(
                        f"target_dir={action_selector.target_name} "
                        f"mode=explore selected_sample={selected_sample} "
                        f"theta={action_selector.selected_theta} "
                        f"score={action_selector.selected_score} "
                        f"waypoint=({float(waypoint[0]):.3f},{float(waypoint[1]):.3f})"
                    )
                action_candidates_pub.publish(
                    action_candidates_msg(
                        display_actions,
                        selected_sample=display_selected_sample,
                        waypoint_index=selected_waypoint_index,
                    )
                )
            else:
                start, end = subgoal_selector.window_bounds()
                distances, waypoints = model.predict(
                    image_sub.context(), topo.window(start, end)
                )
                waypoint = subgoal_selector.select(distances, waypoints, waypoint_index)
                info(
                    f"current_node={subgoal_selector.localized_node} "
                    f"last_node={goal_node}"
                )

            if navigation_mode != "explore":
                waypoint = model.scale_waypoint(
                    waypoint,
                    max_v=float(robot["max_v"]),
                    model_rate=float(robot["model_rate"]),
                )

            if robot.get("publish_waypoint", True):
                waypoint_pub.publish(waypoint_msg(waypoint))
            marker_pub.publish(waypoint_marker(waypoint, topics["frame_id"]))
            if subgoal_selector is not None:
                subgoal_pub.publish(
                    pil_to_msg(topo.images[subgoal_selector.selected_node])
                )

            v, w = controller.command(waypoint, waypoint_index)
            cmd_debug = Twist()
            cmd_debug.linear.x = v
            cmd_debug.angular.z = w
            cmd_debug_pub.publish(cmd_debug)

            if robot.get("publish_cmd_vel", True):
                cmd_pub.publish(v, w)

        reached_pub.publish(Bool(data=reached))
        if reached:
            cmd_debug_pub.publish(Twist())
        if reached and robot.get("publish_cmd_vel", True):
            cmd_pub.stop()
        if reached and not reached_logged:
            info(
                f"reached_goal=true current_node={subgoal_selector.localized_node} "
                f"last_node={goal_node}"
            )
            reached_logged = True
        rate.sleep()


if __name__ == "__main__":
    main()
