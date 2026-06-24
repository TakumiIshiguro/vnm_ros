import rospy
from scenario_navigation_msgs.msg import cmd_dir_intersection


class CmdDirSubscriber:
    def __init__(self, topic: str, queue_size: int = 1):
        self._cmd_dir = None
        self._sub = rospy.Subscriber(
            topic, cmd_dir_intersection, self._callback, queue_size=queue_size
        )

    def _callback(self, msg: cmd_dir_intersection):
        self._cmd_dir = list(map(float, msg.cmd_dir))

    def ready(self) -> bool:
        return self._cmd_dir is not None

    def cmd_dir(self):
        return list(self._cmd_dir)
