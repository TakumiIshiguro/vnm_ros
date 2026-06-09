import random
from typing import Sequence

import numpy as np
import torch

from vnm_ros.datasets.dataset_utils import angle_to_sin_cos, load_image, to_local_coords
from vnm_ros.datasets.trajectory_dataset import TrajectoryDataset


class ViNTDataset(TrajectoryDataset):
    def __init__(
        self,
        data_dir: str,
        image_size: Sequence[int],
        context_size: int,
        len_traj_pred: int,
        waypoint_spacing: int,
        min_goal_distance: int,
        max_goal_distance: int,
        min_action_distance: int,
        max_action_distance: int,
        metric_waypoint_spacing: float,
        normalize: bool = True,
        learn_angle: bool = True,
        negative_mining: bool = True,
    ):
        self.image_size = tuple(image_size)
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.waypoint_spacing = waypoint_spacing
        self.min_goal_distance = min_goal_distance
        self.max_goal_distance = max_goal_distance
        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.normalize = normalize
        self.learn_angle = learn_angle
        self.negative_mining = negative_mining
        super().__init__(data_dir)
        self.samples = self._build_index()
        self.goal_candidates = [
            (name, index)
            for name in self.trajectory_names
            for index in range(len(self.trajectory(name)["position"]))
        ]

    def _build_index(self):
        samples = []
        context_offset = self.context_size * self.waypoint_spacing
        action_offset = self.len_traj_pred * self.waypoint_spacing
        for name in self.trajectory_names:
            length = len(self.trajectory(name)["position"])
            for current in range(context_offset, length - action_offset):
                available = (length - current - 1) // self.waypoint_spacing
                max_goal = min(self.max_goal_distance, available)
                if max_goal >= self.min_goal_distance:
                    samples.append((name, current, max_goal))
        if not samples:
            raise ValueError("No trainable samples; trajectories may be too short")
        return samples

    def __len__(self):
        return len(self.samples)

    def _sample_goal(self, name: str, current: int, max_goal: int):
        use_negative = self.negative_mining and random.random() < 0.1
        if use_negative:
            goal_name, goal_index = random.choice(self.goal_candidates)
            return goal_name, goal_index, True
        distance = random.randint(self.min_goal_distance, max_goal)
        return name, current + distance * self.waypoint_spacing, False

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
        name, current, max_goal = self.samples[index]
        goal_name, goal_index, negative = self._sample_goal(name, current, max_goal)

        context_indices = current + np.arange(-self.context_size, 1) * self.waypoint_spacing
        observations = torch.cat(
            [load_image(self.image_path(name, int(i)), self.image_size) for i in context_indices],
            dim=0,
        )
        goal = load_image(self.image_path(goal_name, goal_index), self.image_size)

        if negative:
            distance = float(self.max_goal_distance)
        else:
            distance = float((goal_index - current) // self.waypoint_spacing)

        action_mask = (
            not negative
            and self.min_action_distance < distance < self.max_action_distance
        )
        return {
            "observation": observations,
            "goal": goal,
            "distance": torch.tensor(distance, dtype=torch.float32),
            "actions": torch.from_numpy(self._actions(name, current)),
            "action_mask": torch.tensor(float(action_mask), dtype=torch.float32),
        }
