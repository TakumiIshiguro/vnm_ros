import os
import pickle
import tempfile
import unittest

import numpy as np
from PIL import Image

from vnm_ros.datasets.dataset_utils import majority_cmd_dir_label
from vnm_ros.datasets.nomad_direction_dataset import NoMaDDirectionDataset
from vnm_ros.datasets.vint_direction_dataset import ViNTDirectionDataset


STRAIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float32)
LEFT = np.array([0.0, 1.0, 0.0], dtype=np.float32)
RIGHT = np.array([0.0, 0.0, 1.0], dtype=np.float32)


class CmdDirMajorityTest(unittest.TestCase):
    def _write_trajectory(self, data_dir, cmd_dirs):
        trajectory_dir = os.path.join(data_dir, "traj_000")
        os.makedirs(trajectory_dir)
        length = len(cmd_dirs)
        trajectory = {
            "position": np.stack(
                [np.arange(length, dtype=np.float32), np.zeros(length)], axis=1
            ),
            "yaw": np.zeros(length, dtype=np.float32),
            "cmd_dir": np.asarray(cmd_dirs, dtype=np.float32),
        }
        with open(os.path.join(trajectory_dir, "traj_data.pkl"), "wb") as stream:
            pickle.dump(trajectory, stream)
        for index in range(length):
            Image.new("RGB", (8, 8)).save(
                os.path.join(trajectory_dir, f"{index}.png")
            )

    def _datasets(self, data_dir, horizon):
        common = {
            "data_dir": data_dir,
            "image_size": (8, 8),
            "context_size": 1,
            "len_traj_pred": horizon,
            "waypoint_spacing": 1,
            "metric_waypoint_spacing": 1.0,
            "normalize": False,
        }
        return [
            NoMaDDirectionDataset(
                **common,
                action_stats={"min": [-2.5, -4.0], "max": [5.0, 4.0]},
            ),
            ViNTDirectionDataset(**common, learn_angle=False),
        ]

    def test_selects_unique_most_frequent_direction(self):
        self.assertEqual(
            majority_cmd_dir_label([RIGHT, LEFT, LEFT, LEFT, RIGHT]),
            1,
        )

    def test_returns_none_when_most_frequent_directions_are_tied(self):
        self.assertIsNone(majority_cmd_dir_label([LEFT, RIGHT, LEFT, RIGHT]))

    def test_dataset_relabels_mixed_window_to_majority(self):
        cmd_dirs = [STRAIGHT, RIGHT, LEFT, LEFT, LEFT, RIGHT] + [STRAIGHT] * 4
        with tempfile.TemporaryDirectory() as data_dir:
            self._write_trajectory(data_dir, cmd_dirs)
            for dataset in self._datasets(data_dir, horizon=4):
                sample_index = dataset.samples.index(("traj_000", 1))
                np.testing.assert_array_equal(
                    dataset[sample_index]["cmd_dir"].numpy(), LEFT
                )
                self.assertGreater(dataset.relabelled_mixed_cmd_dir_samples, 0)

    def test_dataset_skips_tied_window(self):
        cmd_dirs = [STRAIGHT, LEFT, RIGHT, LEFT, RIGHT] + [STRAIGHT] * 5
        with tempfile.TemporaryDirectory() as data_dir:
            self._write_trajectory(data_dir, cmd_dirs)
            for dataset in self._datasets(data_dir, horizon=3):
                self.assertNotIn(("traj_000", 1), dataset.samples)
                self.assertGreater(dataset.skipped_tied_cmd_dir_samples, 0)


if __name__ == "__main__":
    unittest.main()
