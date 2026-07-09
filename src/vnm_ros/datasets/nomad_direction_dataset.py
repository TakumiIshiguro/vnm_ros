from typing import Optional, Sequence

import numpy as np
import torch

from vnm_ros.datasets.dataset_utils import load_image, to_local_coords
from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset


class NoMaDDirectionDataset(TrajectoryDataset):
    def __init__(
        self,
        data_dir: str,
        image_size: Sequence[int],
        context_size: int,
        len_traj_pred: int,
        waypoint_spacing: int,
        action_stats: dict,
        center_crop: bool = False,
        trajectory_names: Optional[Sequence[str]] = None,
    ):
        self.image_size = tuple(image_size)
        self.context_size = int(context_size)
        self.len_traj_pred = int(len_traj_pred)
        self.waypoint_spacing = int(waypoint_spacing)
        self.action_min = np.asarray(action_stats["min"], dtype=np.float32)
        self.action_max = np.asarray(action_stats["max"], dtype=np.float32)
        self.center_crop = bool(center_crop)
        self.skipped_mixed_cmd_dir_samples = 0
        super().__init__(data_dir, trajectory_names=trajectory_names)
        self.samples = self._build_index()

    def _validate_trajectory(self, name: str, trajectory: dict, length: int):
        cmd_dir = np.asarray(trajectory["cmd_dir"], dtype=np.float32)
        if cmd_dir.shape != (length, 3):
            raise ValueError(f"{name}: cmd_dir shape must be ({length}, 3), got {cmd_dir.shape}")
        trajectory["cmd_dir"] = cmd_dir

    def _build_index(self):
        samples = []
        context_offset = self.context_size * self.waypoint_spacing
        action_offset = self.len_traj_pred * self.waypoint_spacing
        for name in self.trajectory_names:
            trajectory = self.trajectory(name)
            length = len(trajectory["position"])
            for current in range(context_offset, length - action_offset):
                if not self._cmd_dir_consistent(trajectory, current):
                    self.skipped_mixed_cmd_dir_samples += 1
                    continue
                samples.append((name, current))
        if not samples:
            raise ValueError("No trainable NoMaD samples; trajectories may be too short")
        return samples

    def sample_cmd_dir_labels(self) -> np.ndarray:
        labels = []
        for name, current in self.samples:
            cmd_dir = self.trajectory(name)["cmd_dir"][current][:3]
            if np.sum(cmd_dir) <= 0.0:
                labels.append(0)
            else:
                labels.append(int(np.argmax(cmd_dir)))
        return np.asarray(labels, dtype=np.int64)

    def _cmd_dir_consistent(self, trajectory: dict, current: int) -> bool:
        indices = current + np.arange(self.len_traj_pred + 1) * self.waypoint_spacing
        cmd_dirs = trajectory["cmd_dir"][indices, :3]
        labels = np.argmax(cmd_dirs, axis=1)
        valid = np.sum(cmd_dirs, axis=1) > 0.0
        labels = np.where(valid, labels, 0)
        return bool(np.all(labels == labels[0]))

    def __len__(self):
        return len(self.samples)

    def _normalized_action_delta(self, name: str, current: int):
        trajectory = self.trajectory(name)
        indices = current + np.arange(self.len_traj_pred + 1) * self.waypoint_spacing
        positions = trajectory["position"][indices]
        yaw = float(trajectory["yaw"][current])
        local_positions = to_local_coords(positions, positions[0], yaw)
        deltas = np.diff(local_positions, axis=0).astype(np.float32)
        normalized = 2.0 * (deltas - self.action_min) / (self.action_max - self.action_min)
        return np.clip(normalized - 1.0, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, index):
        name, current = self.samples[index]
        trajectory = self.trajectory(name)
        context_indices = current + np.arange(-self.context_size, 1) * self.waypoint_spacing
        observations = torch.cat(
            [
                load_image(
                    self.image_path(name, int(i)),
                    self.image_size,
                    center_crop=self.center_crop,
                )
                for i in context_indices
            ],
            dim=0,
        )
        cmd_dir = trajectory["cmd_dir"][current][:3].astype(np.float32)
        if cmd_dir.sum() <= 0:
            cmd_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return {
            "observation": observations,
            "actions": torch.from_numpy(self._normalized_action_delta(name, current)),
            "cmd_dir": torch.from_numpy(cmd_dir),
        }
