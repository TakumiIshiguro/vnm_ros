from typing import Optional, Sequence

import numpy as np


EPS = 1e-8
TARGET_NAMES = {
    0: "straight",
    1: "left",
    2: "right",
}


def clip_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


class CmdDirActionSelector:
    def __init__(self, theta_threshold_deg: float = 15.0):
        self.theta_threshold = np.deg2rad(float(theta_threshold_deg))
        self.cmd_dir: Optional[np.ndarray] = None
        self.selected_sample = 0
        self.target_angle: Optional[float] = None
        self.selected_theta: Optional[float] = None
        self.target_name = "none"
        self.selected_score: Optional[float] = None
        self.selected_action: Optional[np.ndarray] = None

    def update(self, cmd_dir: Sequence[int]) -> None:
        self.cmd_dir = np.asarray(cmd_dir, dtype=np.int8).reshape(-1)

    def select(self, actions, waypoint_index: int):
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[-1] < 2:
            raise ValueError(
                f"Expected actions shape [samples, horizon, dims], got {actions.shape}"
            )
        if actions.shape[0] == 0:
            raise ValueError("No action samples available")

        waypoint_index = min(int(waypoint_index), actions.shape[1] - 1)
        target_angle = self._target_angle()
        if target_angle is None:
            self.selected_sample = 0
            self.target_angle = None
            self.selected_theta = None
            self.target_name = "none"
            self.selected_score = None
            self.selected_action = actions[0]
            return actions[0, waypoint_index]

        cluster_indices, thetas = self._cluster_indices(actions)
        if cluster_indices.size == 0:
            self.selected_sample = 0
            self.target_angle = target_angle
            self.selected_theta = None
            self.selected_score = None
            self.selected_action = actions[0]
            return actions[0, waypoint_index]

        self.selected_sample, self.selected_score = self._medoid(actions, cluster_indices)
        self.target_angle = target_angle
        self.selected_theta = float(thetas[self.selected_sample])
        self.selected_action = actions[self.selected_sample]
        return actions[self.selected_sample, waypoint_index]

    def _cluster_indices(self, actions: np.ndarray):
        thetas = self._trajectory_thetas(actions)
        if self.target_name == "left":
            mask = thetas > self.theta_threshold
        elif self.target_name == "right":
            mask = thetas < -self.theta_threshold
        else:
            mask = np.abs(thetas) <= self.theta_threshold
        return np.flatnonzero(mask), thetas

    def _trajectory_thetas(self, actions: np.ndarray) -> np.ndarray:
        angles = np.arctan2(actions[:, :, 1], actions[:, :, 0])
        sin_mean = np.mean(np.sin(angles), axis=1)
        cos_mean = np.mean(np.cos(angles), axis=1)
        return np.arctan2(sin_mean, cos_mean)

    def _medoid(self, actions: np.ndarray, indices: np.ndarray):
        cluster = actions[indices, :, :2]
        deltas = cluster[:, np.newaxis, :, :] - cluster[np.newaxis, :, :, :]
        distances = np.linalg.norm(deltas, axis=-1).mean(axis=-1)
        costs = distances.sum(axis=1)
        local_index = int(np.argmin(costs))
        return int(indices[local_index]), float(costs[local_index])

    def _target_angle(self):
        if self.cmd_dir is None or self.cmd_dir.size < 3:
            self.target_name = "none"
            return None
        active = np.flatnonzero(self.cmd_dir[:3] > 0)
        if active.size != 1:
            self.target_name = "none"
            return None
        index = int(active[0])
        self.target_name = TARGET_NAMES.get(index, "none")
        return 0.0
