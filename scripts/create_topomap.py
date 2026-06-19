#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy

from vnm_ros.topomap.topomap_builder import TopomapBuilder
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path
from vnm_ros.utils.image_utils import msg_to_pil
from vnm_ros.utils.logger import info


def required_bag_path(path):
    if not path:
        raise ValueError("topomap.bag_path must be set")
    resolved = resolve_path(path, package_root())
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


def stamp_to_sec(msg, bag_time):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is not None:
        stamp_sec = stamp.to_sec()
        if stamp_sec > 0.0:
            return stamp_sec
    return bag_time.to_sec()


def main():
    rospy.init_node("vnm_create_topomap")
    config_dir = rospy.get_param("~config_dir", None)
    cfg = load_runtime_config(config_dir)
    topics = cfg["topics"]
    topomap_cfg = cfg["topomap"]

    topomap_dir = topomap_cfg["topomap_dir"]
    if not topomap_dir.startswith("topomaps/"):
        topomap_dir = os.path.join("topomaps", topomap_dir)
    output_dir = resolve_path(topomap_dir, package_root())
    dt = float(topomap_cfg["sample_dt"])
    bag_path = required_bag_path(topomap_cfg["bag_path"])

    builder = TopomapBuilder(
        output_dir=output_dir,
        dt=dt,
        overwrite=bool(topomap_cfg.get("overwrite", True)),
    )
    builder.prepare()

    import rosbag

    image_topic = topics["image_topic"]
    info(f"saving topomap to {output_dir} from bag {bag_path}")
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, bag_time in bag.read_messages(topics=[image_topic]):
            image = msg_to_pil(msg)
            if builder.maybe_save(image, stamp_to_sec(msg, bag_time)):
                info(f"saved topomap image {builder.index - 1}")
    info(f"saved {builder.index} topomap images to {output_dir}")


if __name__ == "__main__":
    main()
