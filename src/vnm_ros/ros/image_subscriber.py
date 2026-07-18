from collections import deque
from threading import Lock
from typing import List

import rospy
from sensor_msgs.msg import Image

from vnm_ros.utils.image_utils import msg_to_pil


class ImageContextSubscriber:
    def __init__(self, topic: str, context_size: int, queue_size: int = 1):
        self._context_size = context_size
        self._images = deque(maxlen=context_size + 1)
        self._lock = Lock()
        self._latest_msg = None
        self._received_count = 0
        self._sampled_count = 0
        self._sub = rospy.Subscriber(topic, Image, self._callback, queue_size=queue_size)

    def _callback(self, msg: Image):
        with self._lock:
            self._latest_msg = msg
            self._received_count += 1

    def sample(self) -> bool:
        with self._lock:
            if (
                self._latest_msg is None
                or self._received_count == self._sampled_count
            ):
                return False
            msg = self._latest_msg
            sampled_count = self._received_count

        image = msg_to_pil(msg)
        with self._lock:
            self._images.append(image)
            self._sampled_count = sampled_count
        return True

    def ready(self) -> bool:
        with self._lock:
            return len(self._images) > self._context_size

    def context(self) -> List:
        with self._lock:
            return list(self._images)

    def reset(self) -> None:
        with self._lock:
            self._images.clear()
            self._latest_msg = None
            self._sampled_count = self._received_count
