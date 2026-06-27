from typing import Sequence

from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
import rospy


def waypoint_msg(waypoint: Sequence[float]) -> Float32MultiArray:
    msg = Float32MultiArray()
    msg.data = list(map(float, waypoint))
    return msg


def action_candidates_msg(actions, selected_sample: int, waypoint_index: int) -> Float32MultiArray:
    msg = Float32MultiArray()
    sample_count = len(actions)
    horizon = len(actions[0]) if sample_count > 0 else 0
    dims = len(actions[0][0]) if horizon > 0 else 0
    header = [
        float(selected_sample),
        float(waypoint_index),
        float(sample_count),
        float(horizon),
        float(dims),
    ]
    values = []
    for sample in actions:
        for waypoint in sample:
            values.extend(map(float, waypoint))
    msg.data = header + values
    return msg


def waypoint_marker(waypoint: Sequence[float], frame_id: str) -> Marker:
    dx = float(waypoint[0]) if len(waypoint) > 0 else 0.0
    dy = float(waypoint[1]) if len(waypoint) > 1 else 0.0

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "vnm_waypoint"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.04
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.2
    marker.color.a = 1.0

    p0 = Point()
    p0.z = 0.05
    p1 = Point()
    p1.x = dx
    p1.y = dy
    p1.z = 0.05
    marker.points = [p0, p1]
    return marker
