import rospy
from geometry_msgs.msg import Twist


class CmdVelPublisher:
    def __init__(self, topic: str):
        self._pub = rospy.Publisher(topic, Twist, queue_size=1)

    def publish(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._pub.publish(msg)

    def stop(self):
        self.publish(0.0, 0.0)

