from collections import deque
from typing import List

import rospy
from sensor_msgs.msg import Image

from vnm_ros.utils.image_utils import msg_to_pil


class ImageContextSubscriber:
    def __init__(self, topic: str, context_size: int, queue_size: int = 1):
        self._context_size = context_size
        self._images = deque(maxlen=context_size + 1)
        self._sub = rospy.Subscriber(topic, Image, self._callback, queue_size=queue_size)

    def _callback(self, msg: Image):
        self._images.append(msg_to_pil(msg))

    def ready(self) -> bool:
        return len(self._images) > self._context_size

    def context(self) -> List:
        return list(self._images)

