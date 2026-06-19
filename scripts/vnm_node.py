#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

from vnm_ros.control.waypoint_controller import WaypointController
from vnm_ros.models.vint import VNMModel
from vnm_ros.ros.cmd_vel_publisher import CmdVelPublisher
from vnm_ros.ros.image_subscriber import ImageContextSubscriber
from vnm_ros.ros.ros_utils import waypoint_marker, waypoint_msg
from vnm_ros.topomap.subgoal_selector import SubgoalSelector
from vnm_ros.topomap.topomap import Topomap
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
from vnm_ros.utils.image_utils import pil_to_msg
from vnm_ros.utils.logger import info


def main():
    rospy.init_node("vnm_node")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    topics = cfg["topics"]
    robot = cfg["robot"]
    model_cfg = cfg["model"]
    topomap_cfg = cfg["topomap"]

    checkpoint = model_cfg["checkpoint_path"]
    checkpoint = resolve_path(checkpoint, package_root())
    model = VNMModel(model_cfg, checkpoint)

    topomap_dir = topomap_cfg["topomap_dir"]
    if not topomap_dir.startswith("topomaps/"):
        topomap_dir = os.path.join("topomaps", topomap_dir)
    topo = Topomap(resolve_path(topomap_dir, package_root()))

    goal_node = int(topomap_cfg["goal_node"])
    if goal_node < 0:
        goal_node = len(topo) - 1
    selector = SubgoalSelector(
        goal_node=goal_node,
        search_radius=int(topomap_cfg["search_radius"]),
        close_threshold=float(topomap_cfg["close_threshold"]),
    )

    image_sub = ImageContextSubscriber(topics["image_topic"], model.context_size)
    waypoint_pub = rospy.Publisher(topics["waypoint_topic"], waypoint_msg([]).__class__, queue_size=1)
    marker_pub = rospy.Publisher(topics["marker_topic"], Marker, queue_size=1)
    subgoal_pub = rospy.Publisher(topics["topomap_image_topic"], pil_to_msg(topo.images[0]).__class__, queue_size=1)
    reached_pub = rospy.Publisher(topics["reached_goal_topic"], Bool, queue_size=1)
    cmd_debug_pub = rospy.Publisher(topics["cmd_vel_debug_topic"], Twist, queue_size=1)
    cmd_pub = CmdVelPublisher(topics["cmd_vel_topic"])
    controller = WaypointController(
        dt=1.0 / float(robot["model_rate"]),
        max_v=float(robot["max_v"]),
        max_w=float(robot["max_w"]),
    )

    rate = rospy.Rate(float(robot["model_rate"]))
    waypoint_index = int(model_cfg["waypoint_index"])
    info(f"loaded {model_cfg['model_name']} from {checkpoint}")
    info(f"loaded topomap {topomap_dir} with {len(topo)} nodes")
    info(f"goal_node={goal_node}")
    reached_logged = False

    while not rospy.is_shutdown():
        if image_sub.ready() and not selector.reached_goal():
            start, end = selector.window_bounds()
            distances, waypoints = model.predict(image_sub.context(), topo.window(start, end))
            waypoint = selector.select(distances, waypoints, waypoint_index)
            waypoint = model.scale_waypoint(
                waypoint,
                max_v=float(robot["max_v"]),
                model_rate=float(robot["model_rate"]),
            )
            info(
                f"current_node={selector.localized_node} "
                f"last_node={goal_node}"
            )

            if robot.get("publish_waypoint", True):
                waypoint_pub.publish(waypoint_msg(waypoint))
            marker_pub.publish(waypoint_marker(waypoint, topics["frame_id"]))
            subgoal_pub.publish(pil_to_msg(topo.images[selector.selected_node]))

            v, w = controller.command(waypoint)
            cmd_debug = Twist()
            cmd_debug.linear.x = v
            cmd_debug.angular.z = w
            cmd_debug_pub.publish(cmd_debug)

            if robot.get("publish_cmd_vel", True):
                cmd_pub.publish(v, w)

        reached = selector.reached_goal()
        reached_pub.publish(Bool(data=reached))
        if reached:
            cmd_debug_pub.publish(Twist())
        if reached and robot.get("publish_cmd_vel", True):
            cmd_pub.stop()
        if reached and not reached_logged:
            info(f"reached_goal=true current_node={selector.localized_node} last_node={goal_node}")
            reached_logged = True
        rate.sleep()


if __name__ == "__main__":
    main()
