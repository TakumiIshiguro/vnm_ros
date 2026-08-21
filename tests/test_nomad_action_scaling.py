import os
import pickle
import tempfile
import unittest

import numpy as np
from PIL import Image

from vnm_ros.datasets.nomad_direction_dataset import NoMaDDirectionDataset
from vnm_ros.models.model_loader import validate_checkpoint_action_scale
from vnm_ros.models.nomad_inference import NoMaDInference


class NoMaDActionScalingTest(unittest.TestCase):
    def _dataset(self, data_dir, metric_waypoint_spacing=0.065):
        trajectory_dir = os.path.join(data_dir, "traj_000")
        os.makedirs(trajectory_dir)
        length = 8
        positions = np.stack(
            [np.arange(length, dtype=np.float32) * 0.065, np.zeros(length)],
            axis=1,
        )
        trajectory = {
            "position": positions,
            "yaw": np.zeros(length, dtype=np.float32),
            "cmd_dir": np.tile(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                (length, 1),
            ),
        }
        with open(os.path.join(trajectory_dir, "traj_data.pkl"), "wb") as stream:
            pickle.dump(trajectory, stream)
        for index in range(length):
            Image.new("RGB", (8, 8)).save(
                os.path.join(trajectory_dir, f"{index}.png")
            )
        return NoMaDDirectionDataset(
            data_dir=data_dir,
            image_size=(8, 8),
            context_size=1,
            len_traj_pred=2,
            waypoint_spacing=1,
            metric_waypoint_spacing=metric_waypoint_spacing,
            action_stats={"min": [-2.5, -4.0], "max": [5.0, 4.0]},
            normalize=True,
        )

    def test_normalizes_metric_delta_to_dataset_steps(self):
        with tempfile.TemporaryDirectory() as data_dir:
            dataset = self._dataset(data_dir)
            actions = dataset._normalized_action_delta("traj_000", current=1)

        expected_x = 2.0 * (1.0 - (-2.5)) / (5.0 - (-2.5)) - 1.0
        np.testing.assert_allclose(actions[:, 0], expected_x, atol=1e-6)
        np.testing.assert_allclose(actions[:, 1], 0.0, atol=1e-6)

    def test_rejects_nonpositive_metric_spacing(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with self.assertRaisesRegex(ValueError, "metric_waypoint_spacing"):
                self._dataset(data_dir, metric_waypoint_spacing=0.0)

    def test_scales_normalized_inference_waypoints_to_meters(self):
        inference = NoMaDInference(model=None, config={"normalize": True}, device=None)
        waypoint = np.array([[1.0, -0.5]], dtype=np.float32)

        scaled = inference.scale_waypoint(waypoint, max_v=0.25, model_rate=4.0)

        np.testing.assert_allclose(scaled, [[0.0625, -0.03125]], atol=1e-7)
        np.testing.assert_allclose(waypoint, [[1.0, -0.5]], atol=1e-7)

    def test_rejects_checkpoint_with_different_action_scale(self):
        checkpoint = {"config": {"model": {"normalize": False}}}
        config = {"model_type": "nomad", "normalize": True}

        with self.assertRaisesRegex(ValueError, "action scale mismatch"):
            validate_checkpoint_action_scale(checkpoint, config, "old.pth")

    def test_accepts_official_state_dict_without_embedded_config(self):
        validate_checkpoint_action_scale(
            {"noise_pred_net.weight": np.zeros(1)},
            {"model_type": "nomad", "normalize": True},
            "nomad.pth",
        )

    def test_rejects_token_direction_checkpoint_for_residual_model(self):
        checkpoint = {
            "config": {
                "model": {
                    "normalize": True,
                    "direction_conditioning": True,
                    "direction_conditioning_mode": "token",
                }
            }
        }
        config = {
            "model_type": "nomad",
            "normalize": True,
            "direction_conditioning": True,
            "direction_conditioning_mode": "residual",
        }

        with self.assertRaisesRegex(ValueError, "direction conditioning mismatch"):
            validate_checkpoint_action_scale(checkpoint, config, "token.pth")

    def test_rejects_direction_normalization_mismatch(self):
        checkpoint = {
            "config": {
                "model": {
                    "model_type": "nomad",
                    "normalize": True,
                    "direction_conditioning": True,
                    "direction_conditioning_mode": "residual",
                    "direction_normalize": False,
                }
            }
        }
        config = {
            "model_type": "nomad",
            "normalize": True,
            "direction_conditioning": True,
            "direction_conditioning_mode": "residual",
            "direction_normalize": True,
        }

        with self.assertRaisesRegex(ValueError, "normalization mismatch"):
            validate_checkpoint_action_scale(
                checkpoint,
                config,
                "residual.pth",
            )

    def test_rejects_direction_scale_mismatch(self):
        checkpoint = {
            "config": {
                "model": {
                    "model_type": "nomad",
                    "normalize": True,
                    "direction_conditioning": True,
                    "direction_conditioning_mode": "residual",
                    "direction_normalize": True,
                    "direction_scale": 36.0,
                }
            }
        }
        config = {
            "model_type": "nomad",
            "normalize": True,
            "direction_conditioning": True,
            "direction_conditioning_mode": "residual",
            "direction_normalize": True,
            "direction_scale": 24.0,
        }

        with self.assertRaisesRegex(ValueError, "direction scale mismatch"):
            validate_checkpoint_action_scale(
                checkpoint,
                config,
                "fixed-residual.pth",
            )


    def test_accepts_learnable_scale_checkpoint_without_exact_scale_match(self):
        checkpoint = {
            "config": {
                "model": {
                    "model_type": "nomad",
                    "normalize": True,
                    "direction_conditioning": True,
                    "direction_conditioning_mode": "residual",
                    "direction_normalize": True,
                    "direction_scale": 4.0,
                    "direction_scale_learnable": True,
                }
            }
        }
        config = {
            "model_type": "nomad",
            "normalize": True,
            "direction_conditioning": True,
            "direction_conditioning_mode": "residual",
            "direction_normalize": True,
            "direction_scale": 4.0,
            "direction_scale_learnable": True,
        }

        validate_checkpoint_action_scale(checkpoint, config, "learnable.pth")

    def test_rejects_learnable_scale_flag_mismatch(self):
        checkpoint = {
            "config": {
                "model": {
                    "model_type": "nomad",
                    "normalize": True,
                    "direction_conditioning": True,
                    "direction_conditioning_mode": "residual",
                    "direction_normalize": True,
                    "direction_scale": 4.0,
                    "direction_scale_learnable": True,
                }
            }
        }
        config = {
            "model_type": "nomad",
            "normalize": True,
            "direction_conditioning": True,
            "direction_conditioning_mode": "residual",
            "direction_normalize": True,
            "direction_scale": 4.0,
            "direction_scale_learnable": False,
        }

        with self.assertRaisesRegex(ValueError, "direction scale mismatch"):
            validate_checkpoint_action_scale(checkpoint, config, "mismatch.pth")


if __name__ == "__main__":
    unittest.main()
