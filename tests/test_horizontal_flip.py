import os
import pickle
import tempfile
import unittest

import numpy as np
from PIL import Image

from vnm_ros.datasets.dataset_utils import to_local_coords
from vnm_ros.datasets.horizontal_flip import horizontal_flip_trajectory
from vnm_ros.datasets.trajectory_dataset import (
    TrajectoryDataset,
    discover_trajectory_names,
)


class HorizontalFlipTest(unittest.TestCase):
    def test_flips_local_trajectory_and_commands(self):
        trajectory = {
            "position": np.array(
                [[2.0, 3.0], [2.8, 3.4], [3.5, 4.2]], dtype=np.float32
            ),
            "yaw": np.array([0.4, 0.7, 1.0], dtype=np.float32),
            "cmd_dir": np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            "metadata": {"model_type": "nomad"},
        }

        flipped = horizontal_flip_trajectory(trajectory, "traj_000")
        local = to_local_coords(
            trajectory["position"], trajectory["position"][0], trajectory["yaw"][0]
        )
        flipped_local = to_local_coords(
            flipped["position"], flipped["position"][0], flipped["yaw"][0]
        )

        np.testing.assert_allclose(flipped_local[:, 0], local[:, 0], atol=1e-6)
        np.testing.assert_allclose(flipped_local[:, 1], -local[:, 1], atol=1e-6)
        np.testing.assert_array_equal(flipped["cmd_dir"][:, 0], trajectory["cmd_dir"][:, 0])
        np.testing.assert_array_equal(flipped["cmd_dir"][:, 1], trajectory["cmd_dir"][:, 2])
        np.testing.assert_array_equal(flipped["cmd_dir"][:, 2], trajectory["cmd_dir"][:, 1])
        self.assertEqual(flipped["metadata"]["augmentation"], "horizontal_flip")
        self.assertEqual(flipped["metadata"]["source_trajectory"], "traj_000")

    def test_discovers_nested_trajectories(self):
        with tempfile.TemporaryDirectory() as data_dir:
            for name in ("traj_000", os.path.join("aug", "traj_000")):
                trajectory_dir = os.path.join(data_dir, name)
                os.makedirs(trajectory_dir)
                with open(os.path.join(trajectory_dir, "traj_data.pkl"), "wb") as stream:
                    pickle.dump(
                        {
                            "position": np.zeros((1, 2), dtype=np.float32),
                            "yaw": np.zeros(1, dtype=np.float32),
                        },
                        stream,
                    )
                Image.new("RGB", (8, 8)).save(os.path.join(trajectory_dir, "0.png"))

            self.assertEqual(
                discover_trajectory_names(data_dir),
                [os.path.join("aug", "traj_000"), "traj_000"],
            )
            dataset = TrajectoryDataset(data_dir)
            self.assertEqual(len(dataset.trajectory_names), 2)


if __name__ == "__main__":
    unittest.main()
