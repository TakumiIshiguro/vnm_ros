import threading

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


class ObstaclePointSubscriber:
    """Keep the latest robot-frame 2D obstacle points."""

    def __init__(self, topic: str, robot_frame: str):
        self.robot_frame = str(robot_frame)
        self._lock = threading.Lock()
        self._points = None
        self._receipt_time = None
        self._subscriber = rospy.Subscriber(
            topic,
            PointCloud2,
            self._callback,
            queue_size=1,
        )

    def _callback(self, message: PointCloud2) -> None:
        if message.header.frame_id != self.robot_frame:
            rospy.logwarn_throttle(
                5.0,
                "CARE rejected obstacle cloud in frame '%s'; expected '%s'",
                message.header.frame_id,
                self.robot_frame,
            )
            return
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
        with self._lock:
            self._points = points
            self._receipt_time = rospy.Time.now()

    def latest(self, timeout_seconds: float):
        with self._lock:
            points = None if self._points is None else self._points.copy()
            receipt_time = self._receipt_time
        if points is None or receipt_time is None:
            return None
        if rospy.Time.now() - receipt_time > rospy.Duration(timeout_seconds):
            return None
        return points
