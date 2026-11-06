import os
from typing import Optional, Sequence

import numpy as np
import torch

from vnm_ros.datasets.dataset_utils import (
    cmd_dir_one_hot,
    load_image,
    majority_cmd_dir_label,
    sample_photometric_augmentation,
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
        augmentation: Optional[dict] = None,
        exclude_straight_passage_classes: Optional[Sequence[str]] = None,
        cmd_dir_label_mode: str = "majority",
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
        # Photometric augmentation, applied only when the caller passes a
        # config (i.e. the training split). Drawn per frame rather than once
        # per sample, matching corridor_classifier, whose production recipe
        # leaves sequence_consistent off.
        self.augmentation = dict(augmentation or {})
        self.sequence_consistent_augmentation = bool(
            self.augmentation.get("sequence_consistent", False)
        )
        self.skipped_mixed_cmd_dir_samples = 0
        self.skipped_tied_cmd_dir_samples = 0
        # Drop straight-commanded samples whose passage class is in this set.
        # Straight is three quarters plain corridor, so at a junction the
        # straight examples are averaged in with thousands of featureless
        # frames while the turn examples are not, and the command that wins
        # there is the turn. Removing the plain-corridor ones leaves the
        # straight command represented by the junctions it has to be
        # distinguished at; plain corridor is then left to the pretrained
        # weights, which already follow one.
        self.exclude_straight_passage_classes = (
            set(exclude_straight_passage_classes or ())
        )
        self.excluded_straight_samples = 0
        self.relabelled_mixed_cmd_dir_samples = 0
        # "majority" labels a sample by the most common command in its action
        # window. "current" labels it by the command at the current frame,
        # which is all the model is told at run time, and drops windows that
        # go on to a different turn command, since nothing in the input says
        # that turn is coming. Long horizons need "current": the commands
        # change 1.3-1.8 m before a turn, so a 2 m window often spans a change.
        self.cmd_dir_label_mode = str(cmd_dir_label_mode)
        if self.cmd_dir_label_mode not in ("majority", "current"):
            raise ValueError(
                "cmd_dir_label_mode must be 'majority' or 'current', "
                f"got {cmd_dir_label_mode!r}"
            )
        self.excluded_future_turn_samples = 0
        super().__init__(data_dir, trajectory_names=trajectory_names)
        self._passage_classes = {}
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
                window_labels = np.argmax(cmd_dirs[:, :3], axis=1)
                valid = np.sum(cmd_dirs[:, :3], axis=1) > 0.0
                window_labels = np.where(valid, window_labels, 0)
                if self.cmd_dir_label_mode == "current":
                    label = int(window_labels[0])
                    if np.any(
                        (window_labels != label) & (window_labels != 0)
                    ):
                        self.excluded_future_turn_samples += 1
                        continue
                else:
                    label = majority_cmd_dir_label(cmd_dirs)
                    if label is None:
                        self.skipped_mixed_cmd_dir_samples += 1
                        self.skipped_tied_cmd_dir_samples += 1
                        continue
                if np.any(window_labels != window_labels[0]):
                    self.relabelled_mixed_cmd_dir_samples += 1
                if label == 0 and self.exclude_straight_passage_classes:
                    passage = self._passage_class(name, length)
                    if (
                        passage is not None
                        and passage[current] in self.exclude_straight_passage_classes
                    ):
                        self.excluded_straight_samples += 1
                        continue
                samples.append((name, current))
                sample_labels.append(label)
        if not samples:
            raise ValueError("No trainable NoMaD samples; trajectories may be too short")
        self._sample_cmd_dir_labels = np.asarray(sample_labels, dtype=np.int64)
        return samples

    def _passage_class(self, name: str, length: int):
        """Per-frame passage class, from passage_class.npy beside traj_data.pkl.

        Written by scripts/write_passage_classes.py. Absent means the
        trajectory cannot be filtered, which is reported rather than
        silently ignored.
        """
        if name in self._passage_classes:
            return self._passage_classes[name]
        path = os.path.join(self.data_dir, name, "passage_class.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path}: exclude_straight_passage_classes needs per-frame "
                "passage labels; run scripts/write_passage_classes.py"
            )
        classes = np.load(path, allow_pickle=True)
        if len(classes) != length:
            raise ValueError(
                f"{path}: {len(classes)} labels for {length} frames"
            )
        classes = np.asarray([str(c) for c in classes], dtype=object)
        self._passage_classes[name] = classes
        return classes

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

    def turn_angle_degrees(self, index: int) -> float:
        """Bearing of the teacher trajectory's endpoint, in the robot frame.

        The scalar the self-information weighting bins on. It stands in for
        the angular velocity the method was written around: over a fixed
        horizon the endpoint's bearing is how hard this sample turns. It is
        taken from the local positions rather than from ``actions``, whose
        per-component affine normalisation would distort the angle.
        """
        name, current = self.samples[index]
        trajectory = self.trajectory(name)
        indices = current + np.arange(self.len_traj_pred + 1) * self.waypoint_spacing
        positions = trajectory["position"][indices]
        local = to_local_coords(
            positions, positions[0], float(trajectory["yaw"][current])
        )
        endpoint = local[-1]
        return float(np.degrees(np.arctan2(endpoint[1], endpoint[0])))

    def sample_turn_angles(self) -> np.ndarray:
        return np.asarray(
            [self.turn_angle_degrees(i) for i in range(len(self.samples))],
            dtype=np.float32,
        )

    def __getitem__(self, index):
        name, current = self.samples[index]
        trajectory = self.trajectory(name)
        context_indices = current + np.arange(-self.context_size, 1) * self.waypoint_spacing
        sequence_photometric = (
            sample_photometric_augmentation(self.augmentation)
            if self.sequence_consistent_augmentation
            else None
        )
        observations = torch.cat(
            [
                load_image(
                    self.image_path(name, int(i)),
                    self.image_size,
                    center_crop=self.center_crop,
                    photometric=(
                        sequence_photometric
                        if self.sequence_consistent_augmentation
                        else sample_photometric_augmentation(self.augmentation)
                    ),
                )
                for i in context_indices
            ],
            dim=0,
        )
        cmd_dir = self.sample_cmd_dir(index)
        return {
            "observation": observations,
            "actions": torch.from_numpy(self._normalized_action_delta(name, current)),
            "turn_angle": torch.tensor(
                self.turn_angle_degrees(index), dtype=torch.float32
            ),
            "cmd_dir": torch.from_numpy(cmd_dir),
        }
