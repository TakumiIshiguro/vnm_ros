import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from vnm_ros.datasets.dataset_utils import numeric_image_files


def discover_trajectory_names(data_dir: str) -> List[str]:
    names = []
    for root, directories, files in os.walk(data_dir):
        directories.sort()
        if "traj_data.pkl" not in files:
            continue
        names.append(os.path.relpath(root, data_dir))
        directories[:] = []
    return sorted(names)


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, trajectory_names: Optional[Sequence[str]] = None):
        self.data_dir = data_dir
        if trajectory_names is None:
            trajectory_names = discover_trajectory_names(data_dir)
        self.trajectory_names = list(trajectory_names)
        if not self.trajectory_names:
            raise ValueError(f"No trajectory directories found in {data_dir}")
        self._trajectories: Dict[str, dict] = {}
        self._image_files: Dict[str, List[str]] = {}
        self._validate()

    def _validate(self):
        for name in self.trajectory_names:
            trajectory_dir = os.path.join(self.data_dir, name)
            data_path = os.path.join(trajectory_dir, "traj_data.pkl")
            if not os.path.isfile(data_path):
                raise FileNotFoundError(data_path)
            files = numeric_image_files(trajectory_dir)
            if not files:
                raise ValueError(f"No images found in {trajectory_dir}")
            trajectory = self.trajectory(name)
            length = len(trajectory["position"])
            if len(files) != length or len(trajectory["yaw"]) != length:
                raise ValueError(
                    f"{name}: images={len(files)}, positions={length}, yaw={len(trajectory['yaw'])}"
                )
            self._validate_image_size(name, trajectory_dir, files)
            validate = getattr(self, "_validate_trajectory", None)
            if validate is not None:
                validate(name, trajectory, length)
            self._image_files[name] = files

    def _validate_image_size(self, name: str, trajectory_dir: str, files: List[str]):
        expected = getattr(self, "image_size", None)
        if expected is None:
            return
        expected = tuple(int(v) for v in expected)
        metadata = self.trajectory(name).get("metadata", {})
        saved_size = metadata.get("image_size")
        if saved_size is not None and tuple(saved_size) != expected:
            raise ValueError(
                f"{name}: dataset image_size={tuple(saved_size)} does not match "
                f"training image_size={expected}. Recreate this dataset from the "
                "source bag with the selected model config."
            )
        saved_center_crop = metadata.get("image_center_crop")
        expected_center_crop = getattr(self, "center_crop", None)
        if saved_center_crop is not None and expected_center_crop is not None:
            if bool(saved_center_crop) != bool(expected_center_crop):
                raise ValueError(
                    f"{name}: dataset image_center_crop={bool(saved_center_crop)} "
                    f"does not match training image_center_crop="
                    f"{bool(expected_center_crop)}. Recreate this dataset from "
                    "the source bag with the selected model config."
                )
        first_image = os.path.join(trajectory_dir, files[0])
        with Image.open(first_image) as image:
            actual = image.size
        if actual != expected:
            raise ValueError(
                f"{name}: image file size={actual} does not match training "
                f"image_size={expected}. Recreate this dataset from the source "
                "bag instead of reusing a dataset made for another model."
            )

    def trajectory(self, name: str) -> dict:
        if name not in self._trajectories:
            path = os.path.join(self.data_dir, name, "traj_data.pkl")
            with open(path, "rb") as f:
                data = pickle.load(f)
            data["position"] = np.asarray(data["position"], dtype=np.float32)
            data["yaw"] = np.asarray(data["yaw"], dtype=np.float32).reshape(-1)
            if "cmd_dir" in data:
                data["cmd_dir"] = np.asarray(data["cmd_dir"], dtype=np.float32)
            else:
                straight = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                data["cmd_dir"] = np.tile(straight, (len(data["position"]), 1))
            self._trajectories[name] = data
        return self._trajectories[name]

    def image_path(self, name: str, index: int) -> str:
        return os.path.join(self.data_dir, name, self._image_files[name][index])
