#!/usr/bin/env python3
import argparse
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from PIL import Image

from vnm_ros.datasets.dataset_utils import numeric_image_files
from vnm_ros.datasets.horizontal_flip import horizontal_flip_trajectory
from vnm_ros.utils.config import (
    load_runtime_config,
    model_dataset_dir,
    package_root,
    resolve_path,
)


def source_trajectory_names(data_dir: str, requested_names):
    if requested_names:
        names = list(requested_names)
    else:
        names = sorted(
            name
            for name in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, name, "traj_data.pkl"))
        )
    if not names:
        raise ValueError(f"No source trajectories found in {data_dir}")
    return names


def flip_image(source_path: str, output_path: str):
    with Image.open(source_path) as image:
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped.save(output_path, format="PNG")


def augment_trajectory(source_dir: str, output_dir: str, name: str):
    source_trajectory_dir = os.path.join(source_dir, name)
    data_path = os.path.join(source_trajectory_dir, "traj_data.pkl")
    with open(data_path, "rb") as stream:
        trajectory = pickle.load(stream)

    image_files = numeric_image_files(source_trajectory_dir)
    if len(image_files) != len(trajectory["position"]):
        raise ValueError(
            f"{name}: images={len(image_files)} positions={len(trajectory['position'])}"
        )

    temporary_dir = output_dir + ".tmp"
    if os.path.isdir(temporary_dir):
        shutil.rmtree(temporary_dir)
    os.makedirs(temporary_dir)
    try:
        for index, image_file in enumerate(image_files):
            flip_image(
                os.path.join(source_trajectory_dir, image_file),
                os.path.join(temporary_dir, f"{index}.png"),
            )
        flipped = horizontal_flip_trajectory(trajectory, source_trajectory=name)
        with open(os.path.join(temporary_dir, "traj_data.pkl"), "wb") as stream:
            pickle.dump(flipped, stream)
        os.rename(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create horizontal-flip trajectory augmentation."
    )
    parser.add_argument(
        "--config-dir", default=os.path.join(package_root(), "config")
    )
    parser.add_argument("--dataset-type", choices=("train", "test"), default="train")
    parser.add_argument("--output-subdir", default="aug")
    parser.add_argument("--trajectory-name", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = resolve_path(args.config_dir, package_root())
    cfg = load_runtime_config(config_dir)
    model_cfg = cfg["model"]
    dataset_cfg = cfg["train"]["dataset"]
    data_dir = model_dataset_dir(dataset_cfg, args.dataset_type, model_cfg["model_type"])
    output_root = os.path.join(data_dir, args.output_subdir)
    if os.path.exists(output_root):
        raise FileExistsError(
            f"Output already exists: {output_root}. Remove it before regenerating."
        )
    temporary_root = output_root + ".tmp"
    if os.path.exists(temporary_root):
        raise FileExistsError(
            f"Temporary output already exists: {temporary_root}. Remove it before retrying."
        )
    os.makedirs(temporary_root)

    names = source_trajectory_names(data_dir, args.trajectory_name)
    try:
        for index, name in enumerate(names, start=1):
            augment_trajectory(data_dir, os.path.join(temporary_root, name), name)
            print(f"[{index}/{len(names)}] augmented {name}")
        os.rename(temporary_root, output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(f"created {len(names)} augmented trajectories in {output_root}")


if __name__ == "__main__":
    main()
