import rospy


def info(message: str):
    rospy.loginfo(f"[vnm_ros] {message}")


def warn(message: str):
    rospy.logwarn(f"[vnm_ros] {message}")


def error(message: str):
    rospy.logerr(f"[vnm_ros] {message}")

