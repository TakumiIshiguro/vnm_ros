from typing import Optional, Sequence

import numpy as np
import torch

from vnm_ros.datasets.dataset_utils import (
    cmd_dir_one_hot,
    load_image,
    majority_cmd_dir_label,
    to_local_coords,
)
from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset


class NoMaDDirectionDataset(TrajectoryDataset):
    def __init__(
        self,
        data_dir: str,
        image_size: Sequence[int],
        context_size: int,
        len_traj_pred: int,
        waypoint_spacing: int,
        metric_waypoint_spacing: float,
        action_stats: dict,
        normalize: bool = True,
        center_crop: bool = False,
        trajectory_names: Optional[Sequence[str]] = None,
    ):
        self.image_size = tuple(image_size)
        self.context_size = int(context_size)
        self.len_traj_pred = int(len_traj_pred)
        self.waypoint_spacing = int(waypoint_spacing)
        self.metric_waypoint_spacing = float(metric_waypoint_spacing)
        self.normalize = bool(normalize)
        if self.normalize and self.metric_waypoint_spacing <= 0.0:
            raise ValueError("metric_waypoint_spacing must be > 0 when normalize=true")
        self.action_min = np.asarray(action_stats["min"], dtype=np.float32)
        self.action_max = np.asarray(action_stats["max"], dtype=np.float32)
        self.center_crop = bool(center_crop)
        self.skipped_mixed_cmd_dir_samples = 0
        self.skipped_tied_cmd_dir_samples = 0
        self.relabelled_mixed_cmd_dir_samples = 0
        super().__init__(data_dir, trajectory_names=trajectory_names)
        self.samples = self._build_index()

    def _validate_trajectory(self, name: str, trajectory: dict, length: int):
        cmd_dir = np.asarray(trajectory["cmd_dir"], dtype=np.float32)
        if cmd_dir.shape != (length, 3):
            raise ValueError(f"{name}: cmd_dir shape must be ({length}, 3), got {cmd_dir.shape}")
        trajectory["cmd_dir"] = cmd_dir

    def _build_index(self):
        samples = []
        sample_labels = []
        context_offset = self.context_size * self.waypoint_spacing
        action_offset = self.len_traj_pred * self.waypoint_spacing
        for name in self.trajectory_names:
            trajectory = self.trajectory(name)
            length = len(trajectory["position"])
            for current in range(context_offset, length - action_offset):
                cmd_dirs = self._sample_cmd_dirs(trajectory, current)
                label = majority_cmd_dir_label(cmd_dirs)
                if label is None:
                    self.skipped_mixed_cmd_dir_samples += 1
                    self.skipped_tied_cmd_dir_samples += 1
                    continue
                window_labels = np.argmax(cmd_dirs[:, :3], axis=1)
                valid = np.sum(cmd_dirs[:, :3], axis=1) > 0.0
                window_labels = np.where(valid, window_labels, 0)
                if np.any(window_labels != window_labels[0]):
                    self.relabelled_mixed_cmd_dir_samples += 1
                samples.append((name, current))
                sample_labels.append(label)
        if not samples:
            raise ValueError("No trainable NoMaD samples; trajectories may be too short")
        self._sample_cmd_dir_labels = np.asarray(sample_labels, dtype=np.int64)
        return samples

    def sample_cmd_dir_labels(self) -> np.ndarray:
        return self._sample_cmd_dir_labels.copy()

    def sample_cmd_dir_label(self, index: int) -> int:
        return int(self._sample_cmd_dir_labels[index])

    def sample_cmd_dir(self, index: int) -> np.ndarray:
        return cmd_dir_one_hot(self.sample_cmd_dir_label(index))

    def _sample_cmd_dirs(self, trajectory: dict, current: int) -> np.ndarray:
        indices = current + np.arange(self.len_traj_pred + 1) * self.waypoint_spacing
        return trajectory["cmd_dir"][indices, :3]

    def __len__(self):
        return len(self.samples)

    def _normalized_action_delta(self, name: str, current: int):
        trajectory = self.trajectory(name)
        indices = current + np.arange(self.len_traj_pred + 1) * self.waypoint_spacing
        positions = trajectory["position"][indices]
        yaw = float(trajectory["yaw"][current])
        local_positions = to_local_coords(positions, positions[0], yaw)
        deltas = np.diff(local_positions, axis=0).astype(np.float32)
        if self.normalize:
            deltas /= self.metric_waypoint_spacing * self.waypoint_spacing
        normalized = 2.0 * (deltas - self.action_min) / (self.action_max - self.action_min)
        return (normalized - 1.0).astype(np.float32)

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
        cmd_dir = self.sample_cmd_dir(index)
        return {
            "observation": observations,
            "actions": torch.from_numpy(self._normalized_action_delta(name, current)),
            "cmd_dir": torch.from_numpy(cmd_dir),
        }
