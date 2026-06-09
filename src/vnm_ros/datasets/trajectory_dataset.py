import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from vnm_ros.datasets.dataset_utils import numeric_image_files


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, trajectory_names: Optional[Sequence[str]] = None):
        self.data_dir = data_dir
        if trajectory_names is None:
            trajectory_names = sorted(
                name
                for name in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, name))
            )
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
            self._image_files[name] = files

    def trajectory(self, name: str) -> dict:
        if name not in self._trajectories:
            path = os.path.join(self.data_dir, name, "traj_data.pkl")
            with open(path, "rb") as f:
                data = pickle.load(f)
            data["position"] = np.asarray(data["position"], dtype=np.float32)
            data["yaw"] = np.asarray(data["yaw"], dtype=np.float32).reshape(-1)
            self._trajectories[name] = data
        return self._trajectories[name]

    def image_path(self, name: str, index: int) -> str:
        return os.path.join(self.data_dir, name, self._image_files[name][index])
