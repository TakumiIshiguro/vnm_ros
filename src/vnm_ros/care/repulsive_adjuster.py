from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CareAdjustment:
    action: np.ndarray
    rotation_angle: float
    desired_heading: float
    strongest_waypoint_index: Optional[int]
    rotate_in_place: bool
    stopped: bool
    reason: str
    force_balance_ratio: float = 0.0


class CareRepulsiveAdjuster:
    """CARE trajectory adjustment using artificial-potential-field repulsion."""

    def __init__(
        self,
        maximum_forward_range_m: float,
        path_influence_radius_m: float,
        depth_offset_m: float,
        minimum_force_distance_m: float,
        force_balance_ratio_threshold: float,
        theta_clip_degrees: float,
        safe_fov_threshold_degrees: float,
    ):
        self.maximum_forward_range_m = float(maximum_forward_range_m)
        self.path_influence_radius_m = float(path_influence_radius_m)
        self.depth_offset_m = float(depth_offset_m)
        self.minimum_force_distance_m = float(minimum_force_distance_m)
        self.force_balance_ratio_threshold = float(
            force_balance_ratio_threshold
        )
        self.theta_clip = np.deg2rad(float(theta_clip_degrees))
        self.safe_fov_threshold = np.deg2rad(
            float(safe_fov_threshold_degrees)
        )
        if self.maximum_forward_range_m <= 0.0:
            raise ValueError("maximum_forward_range_m must be positive")
        if self.path_influence_radius_m <= 0.0:
            raise ValueError("path_influence_radius_m must be positive")
        if self.depth_offset_m < 0.0:
            raise ValueError("depth_offset_m must be non-negative")
        if self.minimum_force_distance_m <= 0.0:
            raise ValueError("minimum_force_distance_m must be positive")
        if not 0.0 <= self.force_balance_ratio_threshold < 1.0:
            raise ValueError(
                "force_balance_ratio_threshold must be in [0, 1)"
            )
        if not 0.0 < self.theta_clip <= np.pi:
            raise ValueError("theta_clip_degrees must be in (0, 180]")
        if not 0.0 < self.safe_fov_threshold <= self.theta_clip:
            raise ValueError(
                "safe_fov_threshold_degrees must be positive and no greater "
                "than theta_clip_degrees"
            )

    def stop(self, action, reason: str) -> CareAdjustment:
        action = self._validate_action(action)
        return CareAdjustment(
            action=np.zeros_like(action),
            rotation_angle=0.0,
            desired_heading=0.0,
            strongest_waypoint_index=None,
            rotate_in_place=False,
            stopped=True,
            reason=str(reason),
            force_balance_ratio=0.0,
        )

    def adjust(self, action, obstacles) -> CareAdjustment:
        action = self._validate_action(action)
        obstacles = self._prepare_obstacles(action, obstacles)
        if len(obstacles) == 0:
            return CareAdjustment(
                action=action.copy(),
                rotation_angle=0.0,
                desired_heading=0.0,
                strongest_waypoint_index=None,
                rotate_in_place=False,
                stopped=False,
                reason="no_nearby_obstacles",
                force_balance_ratio=0.0,
            )

        forces, component_magnitude_sums = (
            self._repulsive_force_details(action, obstacles)
        )
        magnitudes = np.linalg.norm(forces, axis=1)
        strongest = int(np.argmax(magnitudes))
        strongest_force = forces[strongest]
        strongest_magnitude = float(magnitudes[strongest])
        component_magnitude_sum = float(
            component_magnitude_sums[strongest]
        )
        balance_ratio = (
            strongest_magnitude / component_magnitude_sum
            if component_magnitude_sum > np.finfo(np.float32).eps
            else 0.0
        )
        if (
            strongest_magnitude <= np.finfo(np.float32).eps
            or balance_ratio < self.force_balance_ratio_threshold
        ):
            return CareAdjustment(
                action=action.copy(),
                rotation_angle=0.0,
                desired_heading=0.0,
                strongest_waypoint_index=strongest,
                rotate_in_place=False,
                stopped=False,
                reason="balanced_repulsive_force",
                force_balance_ratio=balance_ratio,
            )

        repulsive_angle = float(
            np.arctan2(strongest_force[1], strongest_force[0])
        )
        rotation_angle = float(
            np.clip(repulsive_angle, -self.theta_clip, self.theta_clip)
        )
        cosine = np.cos(rotation_angle)
        sine = np.sin(rotation_angle)
        rotation = np.asarray(
            ((cosine, -sine), (sine, cosine)),
            dtype=np.float32,
        )
        adjusted = action.copy()
        adjusted[:, :2] = action[:, :2] @ rotation.T
        strongest_waypoint = adjusted[strongest, :2]
        desired_heading = float(
            np.arctan2(strongest_waypoint[1], strongest_waypoint[0])
        )
        return CareAdjustment(
            action=adjusted,
            rotation_angle=rotation_angle,
            desired_heading=desired_heading,
            strongest_waypoint_index=strongest,
            rotate_in_place=bool(
                abs(desired_heading) > self.safe_fov_threshold
            ),
            stopped=False,
            reason="repulsive_rotation",
            force_balance_ratio=balance_ratio,
        )

    def repulsive_forces(self, action, obstacles) -> np.ndarray:
        forces, _ = self._repulsive_force_details(action, obstacles)
        return forces

    def _repulsive_force_details(self, action, obstacles):
        action = self._validate_action(action)
        obstacles = np.asarray(obstacles, dtype=np.float32)
        if obstacles.ndim != 2 or obstacles.shape[1] != 2:
            raise ValueError("obstacles must have shape (N, 2)")
        if len(obstacles) == 0:
            return (
                np.zeros((len(action), 2), dtype=np.float32),
                np.zeros(len(action), dtype=np.float32),
            )
        difference = action[:, np.newaxis, :2] - obstacles[np.newaxis, :, :]
        distances = np.linalg.norm(difference, axis=2)
        distances = np.maximum(distances, self.minimum_force_distance_m)
        # A vector from each obstacle toward the waypoint, weighted by the
        # inverse fourth power implied by CARE Equation (1).
        components = difference / distances[:, :, np.newaxis] ** 4
        forces = np.asarray(np.sum(components, axis=1), dtype=np.float32)
        component_magnitude_sums = np.asarray(
            np.sum(np.linalg.norm(components, axis=2), axis=1),
            dtype=np.float32,
        )
        return forces, component_magnitude_sums

    def _prepare_obstacles(self, action, obstacles) -> np.ndarray:
        obstacles = np.asarray(obstacles, dtype=np.float32)
        if obstacles.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if obstacles.ndim != 2 or obstacles.shape[1] < 2:
            raise ValueError("obstacles must have shape (N, >= 2)")
        obstacles = obstacles[:, :2]
        if not np.isfinite(obstacles).all():
            raise ValueError("obstacles must contain only finite values")
        ranges = np.linalg.norm(obstacles, axis=1)
        valid = (
            (obstacles[:, 0] > 0.0)
            & (obstacles[:, 0] <= self.maximum_forward_range_m)
            & (ranges > np.finfo(np.float32).eps)
        )
        obstacles = obstacles[valid]
        ranges = ranges[valid]
        if len(obstacles) == 0:
            return obstacles
        if self.depth_offset_m > 0.0:
            adjusted_ranges = np.maximum(
                ranges - self.depth_offset_m,
                self.minimum_force_distance_m,
            )
            obstacles = obstacles * (
                adjusted_ranges / ranges
            )[:, np.newaxis]
        return self._filter_near_extended_path(action, obstacles)

    def _filter_near_extended_path(self, action, obstacles) -> np.ndarray:
        """Keep points near the predicted path extended to the CARE range."""
        path = np.concatenate(
            (
                np.zeros((1, 2), dtype=np.float32),
                np.asarray(action[:, :2], dtype=np.float32),
            ),
            axis=0,
        )
        segment_vectors = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        valid_segments = segment_lengths > np.finfo(np.float32).eps
        travelled = float(np.sum(segment_lengths[valid_segments]))
        if np.any(valid_segments):
            final_segment = segment_vectors[np.flatnonzero(valid_segments)[-1]]
            direction = final_segment / np.linalg.norm(final_segment)
        else:
            direction = np.asarray((1.0, 0.0), dtype=np.float32)
        extension_length = max(
            self.maximum_forward_range_m - travelled,
            0.0,
        )
        if extension_length > np.finfo(np.float32).eps:
            path = np.concatenate(
                (
                    path,
                    (path[-1] + direction * extension_length)[
                        np.newaxis, :
                    ],
                ),
                axis=0,
            )

        starts = path[:-1]
        vectors = np.diff(path, axis=0)
        squared_lengths = np.sum(vectors * vectors, axis=1)
        usable = squared_lengths > np.finfo(np.float32).eps
        starts = starts[usable]
        vectors = vectors[usable]
        squared_lengths = squared_lengths[usable]
        if len(starts) == 0:
            return np.empty((0, 2), dtype=np.float32)

        relative = obstacles[:, np.newaxis, :] - starts[np.newaxis, :, :]
        interpolation = np.sum(
            relative * vectors[np.newaxis, :, :],
            axis=2,
        ) / squared_lengths[np.newaxis, :]
        interpolation = np.clip(interpolation, 0.0, 1.0)
        closest = (
            starts[np.newaxis, :, :]
            + interpolation[:, :, np.newaxis]
            * vectors[np.newaxis, :, :]
        )
        distances = np.linalg.norm(
            obstacles[:, np.newaxis, :] - closest,
            axis=2,
        )
        return obstacles[
            np.min(distances, axis=1) <= self.path_influence_radius_m
        ]

    @staticmethod
    def _validate_action(action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] < 2:
            raise ValueError("action must have shape (horizon, >= 2)")
        if not np.isfinite(action).all():
            raise ValueError("action must contain only finite values")
        return action
