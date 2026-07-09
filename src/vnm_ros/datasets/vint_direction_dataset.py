from typing import Optional, Sequence

import numpy as np
import torch

from vnm_ros.datasets.dataset_utils import angle_to_sin_cos, load_image, to_local_coords
from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset


class ViNTDirectionDataset(TrajectoryDataset):
    def __init__(
        self,
        data_dir: str,
        image_size: Sequence[int],
        context_size: int,
        len_traj_pred: int,
        waypoint_spacing: int,
        metric_waypoint_spacing: float,
        normalize: bool = True,
        learn_angle: bool = True,
        center_crop: bool = False,
        trajectory_names: Optional[Sequence[str]] = None,
    ):
        self.image_size = tuple(image_size)
        self.context_size = int(context_size)
        self.len_traj_pred = int(len_traj_pred)
        self.waypoint_spacing = int(waypoint_spacing)
        self.metric_waypoint_spacing = float(metric_waypoint_spacing)
        self.normalize = bool(normalize)
        self.learn_angle = bool(learn_angle)
        self.center_crop = bool(center_crop)
        self.skipped_mixed_cmd_dir_samples = 0
        super().__init__(data_dir, trajectory_names=trajectory_names)
        self.samples = self._build_index()

    def _validate_trajectory(self, name: str, trajectory: dict, length: int):
        cmd_dir = np.asarray(trajectory["cmd_dir"], dtype=np.float32)
        if cmd_dir.shape != (length, 3):
            raise ValueError(
                f"{name}: cmd_dir shape must be ({length}, 3), got {cmd_dir.shape}"
            )
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
            raise ValueError("No trainable ViNT direction samples; trajectories may be too short")
        return samples

    def _cmd_dir_consistent(self, trajectory: dict, current: int) -> bool:
        indices = current + np.arange(self.len_traj_pred + 1) * self.waypoint_spacing
        cmd_dirs = trajectory["cmd_dir"][indices, :3]
        labels = np.argmax(cmd_dirs, axis=1)
        valid = np.sum(cmd_dirs, axis=1) > 0.0
        labels = np.where(valid, labels, 0)
        return bool(np.all(labels == labels[0]))

    def sample_cmd_dir_labels(self) -> np.ndarray:
        labels = []
        for name, current in self.samples:
            cmd_dir = self.trajectory(name)["cmd_dir"][current][:3]
            if np.sum(cmd_dir) <= 0.0:
                labels.append(0)
            else:
                labels.append(int(np.argmax(cmd_dir)))
        return np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def _actions(self, name: str, current: int):
        trajectory = self.trajectory(name)
        indices = current + np.arange(self.len_traj_pred + 1) * self.waypoint_spacing
        positions = trajectory["position"][indices]
        yaws = trajectory["yaw"][indices]
        local_positions = to_local_coords(positions, positions[0], float(yaws[0]))
        relative_yaws = yaws[1:] - yaws[0]
        actions = np.concatenate([local_positions[1:], relative_yaws[:, None]], axis=1)
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing
        if self.learn_angle:
            return angle_to_sin_cos(actions)
        return actions[:, :2].astype(np.float32)

    def _distance(self) -> float:
        return float(self.len_traj_pred)

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
            "cmd_dir": torch.from_numpy(cmd_dir),
            "distance": torch.tensor(self._distance(), dtype=torch.float32),
            "actions": torch.from_numpy(self._actions(name, current)),
            "action_mask": torch.tensor(1.0, dtype=torch.float32),
        }
