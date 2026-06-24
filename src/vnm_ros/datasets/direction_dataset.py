from typing import Sequence

import numpy as np
import torch

from vnm_ros.datasets.dataset_utils import angle_to_sin_cos, load_image, to_local_coords
from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset


class DirectionViNTDataset(TrajectoryDataset):
    def __init__(
        self,
        data_dir: str,
        image_size: Sequence[int],
        context_size: int,
        len_traj_pred: int,
        waypoint_spacing: int,
        min_action_distance: int,
        max_action_distance: int,
        metric_waypoint_spacing: float,
        normalize: bool = True,
        learn_angle: bool = True,
    ):
        self.image_size = tuple(image_size)
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.waypoint_spacing = waypoint_spacing
        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.normalize = normalize
        self.learn_angle = learn_angle
        super().__init__(data_dir)
        self.samples = self._build_index()

    def _validate_trajectory(self, name: str, trajectory: dict, length: int):
        if "cmd_dir" not in trajectory:
            raise ValueError(f"{name}: traj_data.pkl does not contain cmd_dir")
        cmd_dir = np.asarray(trajectory["cmd_dir"], dtype=np.float32)
        if cmd_dir.shape != (length, 3):
            raise ValueError(f"{name}: cmd_dir shape must be ({length}, 3), got {cmd_dir.shape}")
        trajectory["cmd_dir"] = cmd_dir

    def _build_index(self):
        samples = []
        context_offset = self.context_size * self.waypoint_spacing
        action_offset = self.len_traj_pred * self.waypoint_spacing
        for name in self.trajectory_names:
            length = len(self.trajectory(name)["position"])
            for current in range(context_offset, length - action_offset):
                distance = self.len_traj_pred
                if self.min_action_distance < distance < self.max_action_distance:
                    samples.append((name, current, float(distance)))
        if not samples:
            raise ValueError("No trainable samples; trajectories may be too short")
        return samples

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

    def __getitem__(self, index):
        name, current, distance = self.samples[index]
        context_indices = current + np.arange(-self.context_size, 1) * self.waypoint_spacing
        observations = torch.cat(
            [load_image(self.image_path(name, int(i)), self.image_size) for i in context_indices],
            dim=0,
        )
        trajectory = self.trajectory(name)
        return {
            "observation": observations,
            "cmd_dir": torch.from_numpy(trajectory["cmd_dir"][current].astype(np.float32)),
            "distance": torch.tensor(distance, dtype=torch.float32),
            "actions": torch.from_numpy(self._actions(name, current)),
            "action_mask": torch.tensor(1.0, dtype=torch.float32),
        }
