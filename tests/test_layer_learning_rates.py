import unittest

import torch

from vnm_ros.training.optimizer import build_parameter_groups


class LayerLearningRatesTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.ModuleDict(
            {
                "encoder": torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                ),
                "head": torch.nn.Linear(4, 2),
            }
        )

    def test_builds_default_and_layer_groups(self):
        groups, summary = build_parameter_groups(
            self.model,
            {
                "learning_rate": 1e-4,
                "layer_learning_rates": {
                    "encoder": 2e-4,
                    "encoder.1": 3e-4,
                },
            },
        )

        rates = {group["name"]: group["lr"] for group in groups}
        counts = {item["name"]: item["parameter_tensors"] for item in summary}
        self.assertEqual(rates, {"default": 1e-4, "encoder": 2e-4, "encoder.1": 3e-4})
        self.assertEqual(counts, {"default": 2, "encoder": 2, "encoder.1": 2})

    def test_rejects_unknown_layer(self):
        with self.assertRaisesRegex(ValueError, "entry not found"):
            build_parameter_groups(
                self.model,
                {
                    "learning_rate": 1e-4,
                    "layer_learning_rates": {"missing": 2e-4},
                },
            )

    def test_rejects_frozen_layer(self):
        self.model["encoder"].requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "all matching parameters are frozen"):
            build_parameter_groups(
                self.model,
                {
                    "learning_rate": 1e-4,
                    "layer_learning_rates": {"encoder": 2e-4},
                },
            )


if __name__ == "__main__":
    unittest.main()
