#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
from sensor_msgs.msg import Image

from vnm_ros.topomap.topomap_builder import TopomapBuilder
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
from vnm_ros.utils.image_utils import msg_to_pil
from vnm_ros.utils.logger import info, warn

latest_image = None
latest_time = float("-inf")


def optional_path(path):
    if not path:
        return None
    return resolve_path(path, package_root())


def stamp_to_sec(msg, bag_time):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is not None:
        stamp_sec = stamp.to_sec()
        if stamp_sec > 0.0:
            return stamp_sec
    return bag_time.to_sec()


def image_callback(msg: Image):
    global latest_image, latest_time
    latest_image = msg_to_pil(msg)
    latest_time = time.time()


def main():
    rospy.init_node("vnm_create_topomap")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    topics = cfg["topics"]
    topomap_cfg = cfg["topomap"]

    topomap_dir = rospy.get_param("~topomap_dir", topomap_cfg["topomap_dir"])
    if not topomap_dir.startswith("topomaps/"):
        topomap_dir = os.path.join("topomaps", topomap_dir)
    output_dir = resolve_path(topomap_dir, package_root())
    dt = float(rospy.get_param("~sample_dt", topomap_cfg["sample_dt"]))
    bag_path = optional_path(
        rospy.get_param("~bag_path", topomap_cfg.get("bag_path", ""))
    )

    builder = TopomapBuilder(
        output_dir=output_dir,
        dt=dt,
        overwrite=bool(topomap_cfg.get("overwrite", True)),
    )
    builder.prepare()

    if bag_path:
        import rosbag

        image_topic = topics["image_topic"]
        info(f"saving topomap to {output_dir} from bag {bag_path}")
        with rosbag.Bag(bag_path, "r") as bag:
            for _, msg, bag_time in bag.read_messages(topics=[image_topic]):
                image = msg_to_pil(msg)
                if builder.maybe_save(image, stamp_to_sec(msg, bag_time)):
                    info(f"saved topomap image {builder.index - 1}")
        info(f"saved {builder.index} topomap images to {output_dir}")
        return

    rospy.Subscriber(topics["image_topic"], Image, image_callback, queue_size=1)
    # Run faster than the requested sampling interval so timing jitter around
    # the dt boundary does not skip a whole extra second.
    rate = rospy.Rate(max(10.0 / dt, 10.0))
    info(f"saving topomap to {output_dir} from {topics['image_topic']} every {dt:.2f}s")

    while not rospy.is_shutdown():
        if latest_image is not None and builder.maybe_save(latest_image):
            info(f"saved topomap image {builder.index - 1}")
        if (
            topomap_cfg.get("shutdown_on_stale_image", True)
            and latest_time > 0
            and time.time() - latest_time > 2.0 * dt
        ):
            warn("image topic is stale; shutting down topomap builder")
            rospy.signal_shutdown("stale image topic")
        rate.sleep()


if __name__ == "__main__":
    main()
