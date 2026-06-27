from typing import Optional, Sequence

import numpy as np


EPS = 1e-8
DEFAULT_TARGET_ANGLES_DEG = {
    "straight": 0.0,
    "left": 45.0,
    "right": -45.0,
}
TARGET_NAMES = {
    0: "straight",
    1: "left",
    2: "right",
}


def clip_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


class CmdDirActionSelector:
    def __init__(self, target_angles_deg=None):
        self.target_angles = self._target_angles(target_angles_deg or {})
        self.cmd_dir: Optional[np.ndarray] = None
        self.selected_sample = 0
        self.target_angle: Optional[float] = None
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
            self.target_name = "none"
            self.selected_score = None
            self.selected_action = actions[0]
            return actions[0, waypoint_index]

        waypoints = actions[:, waypoint_index, :2]
        norms = np.linalg.norm(waypoints, axis=1)
        valid = norms > EPS
        if not np.any(valid):
            self.selected_sample = 0
            self.target_angle = target_angle
            self.selected_score = None
            self.selected_action = actions[0]
            return actions[0, waypoint_index]

        self.selected_sample = self._select_angle_sample(waypoints, valid, target_angle)
        self.target_angle = target_angle
        self.selected_score = float(
            abs(
                clip_angle(
                    np.arctan2(
                        waypoints[self.selected_sample, 1],
                        waypoints[self.selected_sample, 0],
                    )
                    - target_angle
                )
            )
        )
        self.selected_action = actions[self.selected_sample]
        return actions[self.selected_sample, waypoint_index]

    def _select_angle_sample(
        self, waypoints: np.ndarray, valid: np.ndarray, target_angle: float
    ) -> int:
        angles = np.arctan2(waypoints[:, 1], waypoints[:, 0])
        scores = np.abs(clip_angle(angles - target_angle))
        scores = scores.astype(np.float64)
        scores[~valid] = np.inf
        return int(np.argmin(scores))

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
        return self.target_angles.get(self.target_name)

    def _target_angles(self, target_angles_deg):
        angles = DEFAULT_TARGET_ANGLES_DEG.copy()
        angles.update(target_angles_deg)
        return {
            name: np.deg2rad(float(angle_deg))
            for name, angle_deg in angles.items()
        }
