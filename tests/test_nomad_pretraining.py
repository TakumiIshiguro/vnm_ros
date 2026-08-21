import os
import tempfile
import unittest

import torch
import yaml

from vnm_ros.models.model_loader import checkpoint_state_dict
from vnm_ros.models.model_loader import build_model
from vnm_ros.models.nomad_model import DirectionEncoder, NoMaDViNT
from vnm_ros.training.checkpoint import load_training_checkpoint, save_checkpoint


class NoMaDPretrainingTest(unittest.TestCase):
    def test_direction_encoder_produces_distinct_tokens(self):
        torch.manual_seed(0)
        encoder = DirectionEncoder(
            embedding_dim=16,
            input_dim=3,
            hidden_dim=8,
            latent_dim=4,
        )

        output = encoder(torch.eye(3))

        self.assertEqual(tuple(output.shape), (3, 16))
        self.assertGreater(torch.linalg.vector_norm(output[0] - output[1]), 0.0)
        self.assertGreater(torch.linalg.vector_norm(output[0] - output[2]), 0.0)

    def test_direction_token_replaces_goal_image(self):
        torch.manual_seed(0)
        conditioned = NoMaDViNT(
            context_size=1,
            obs_encoding_size=32,
            mha_num_attention_heads=4,
            mha_num_attention_layers=1,
            direction_encoder=DirectionEncoder(
                embedding_dim=32,
                hidden_dim=8,
                latent_dim=4,
            ),
        ).eval()
        observation = torch.randn(1, 6, 32, 32)
        first_goal = torch.randn(1, 3, 32, 32)
        second_goal = torch.randn(1, 3, 32, 32)
        goal_mask = torch.ones(1, dtype=torch.long)
        command = torch.tensor([[0.0, 1.0, 0.0]])

        with torch.no_grad():
            first = conditioned(
                observation,
                first_goal,
                input_goal_mask=goal_mask,
                cmd_dir=command,
            )
            second = conditioned(
                observation,
                second_goal,
                input_goal_mask=goal_mask,
                cmd_dir=command,
            )

        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)

    def test_residual_mode_preserves_pretrained_goal_masked_condition(self):
        torch.manual_seed(0)
        pretrained = NoMaDViNT(
            context_size=1,
            obs_encoding_size=32,
            mha_num_attention_heads=4,
            mha_num_attention_layers=1,
        ).eval()
        conditioned = NoMaDViNT(
            context_size=1,
            obs_encoding_size=32,
            mha_num_attention_heads=4,
            mha_num_attention_layers=1,
            direction_encoder=DirectionEncoder(
                embedding_dim=32,
                hidden_dim=8,
                latent_dim=4,
                zero_init_output=True,
            ),
            direction_conditioning_mode="residual",
        ).eval()
        conditioned.load_state_dict(pretrained.state_dict(), strict=False)
        observation = torch.randn(2, 6, 32, 32)
        goal = torch.randn(2, 3, 32, 32)
        goal_mask = torch.ones(2, dtype=torch.long)
        commands = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        with torch.no_grad():
            expected = pretrained(
                observation,
                goal,
                input_goal_mask=goal_mask,
            )
            actual = conditioned(
                observation,
                goal,
                input_goal_mask=goal_mask,
                cmd_dir=commands,
            )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_direction_scale_scales_residual(self):
        torch.manual_seed(0)
        conditioned = NoMaDViNT(
            context_size=1,
            obs_encoding_size=32,
            mha_num_attention_heads=4,
            mha_num_attention_layers=1,
            direction_encoder=DirectionEncoder(
                embedding_dim=32,
                hidden_dim=8,
                latent_dim=4,
            ),
            direction_conditioning_mode="residual",
        ).eval()
        observation = torch.randn(2, 6, 32, 32)
        goal = torch.randn(2, 3, 32, 32)
        goal_mask = torch.ones(2, dtype=torch.long)
        left = torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1)

        with torch.no_grad():
            conditioned.direction_scale = 0.0
            base = conditioned(
                observation,
                goal,
                input_goal_mask=goal_mask,
                cmd_dir=left,
            )
            conditioned.direction_scale = 1.0
            full = conditioned(
                observation,
                goal,
                input_goal_mask=goal_mask,
                cmd_dir=left,
            )
            conditioned.direction_scale = 0.25
            scaled = conditioned(
                observation,
                goal,
                input_goal_mask=goal_mask,
                cmd_dir=left,
            )

        torch.testing.assert_close(
            scaled - base,
            0.25 * (full - base),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_normalized_residual_has_fixed_norm(self):
        torch.manual_seed(0)
        direction_scale = 0.5
        conditioned = NoMaDViNT(
            context_size=1,
            obs_encoding_size=32,
            mha_num_attention_heads=4,
            mha_num_attention_layers=1,
            direction_encoder=DirectionEncoder(
                embedding_dim=32,
                hidden_dim=8,
                latent_dim=4,
            ),
            direction_conditioning_mode="residual",
            direction_scale=direction_scale,
            direction_normalize=True,
        ).eval()
        observation = torch.randn(3, 6, 32, 32)
        goal = torch.randn(3, 3, 32, 32)
        commands = torch.eye(3)

        with torch.no_grad():
            conditioned.direction_scale = 0.0
            base = conditioned(
                observation,
                goal,
                cmd_dir=commands,
            )
            conditioned.direction_scale = direction_scale
            actual = conditioned(
                observation,
                goal,
                cmd_dir=commands,
            )

        residual_norm = torch.linalg.vector_norm(actual - base, dim=-1)
        torch.testing.assert_close(
            residual_norm,
            torch.full_like(residual_norm, direction_scale),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_normalized_residual_builder_does_not_zero_initialize_output(self):
        config = {
            "model_type": "nomad",
            "context_size": 1,
            "obs_encoder": "efficientnet-b0",
            "encoding_size": 32,
            "mha_num_attention_heads": 4,
            "mha_num_attention_layers": 1,
            "mha_ff_dim_factor": 4,
            "direction_conditioning": True,
            "direction_conditioning_mode": "residual",
            "direction_normalize": True,
            "direction_scale": 0.5,
            "direction_num_commands": 3,
            "direction_hidden_dim": 8,
            "direction_latent_dim": 4,
            "down_dims": [8, 16],
            "cond_predict_scale": False,
        }

        model = build_model(config)
        output_layer = model.vision_encoder.direction_encoder.projector[-1]

        self.assertGreater(torch.linalg.vector_norm(output_layer.weight), 0.0)

    def test_rejects_invalid_direction_scale(self):
        with self.assertRaisesRegex(ValueError, "direction_scale"):
            NoMaDViNT(
                context_size=1,
                obs_encoding_size=32,
                mha_num_attention_heads=4,
                mha_num_attention_layers=1,
                direction_encoder=DirectionEncoder(
                    embedding_dim=32,
                    hidden_dim=8,
                    latent_dim=4,
                ),
                direction_conditioning_mode="residual",
                direction_scale=-0.1,
            )

    def test_learnable_scale_starts_at_initial_scale(self):
        encoder = DirectionEncoder(
            embedding_dim=32,
            hidden_dim=8,
            latent_dim=4,
            learnable_scale=True,
            initial_scale=4.0,
        )

        self.assertAlmostEqual(
            encoder.effective_scale().item(), 4.0, places=5
        )

    def test_learnable_scale_rejects_nonpositive_initial_scale(self):
        with self.assertRaisesRegex(ValueError, "initial_scale"):
            DirectionEncoder(
                embedding_dim=32,
                hidden_dim=8,
                latent_dim=4,
                learnable_scale=True,
                initial_scale=0.0,
            )

    def test_effective_scale_requires_learnable_scale(self):
        encoder = DirectionEncoder(embedding_dim=32, hidden_dim=8, latent_dim=4)

        with self.assertRaisesRegex(RuntimeError, "learnable_scale"):
            encoder.effective_scale()

    def test_learnable_scale_receives_gradients_from_residual_loss(self):
        torch.manual_seed(0)
        conditioned = NoMaDViNT(
            context_size=1,
            obs_encoding_size=32,
            mha_num_attention_heads=4,
            mha_num_attention_layers=1,
            direction_encoder=DirectionEncoder(
                embedding_dim=32,
                hidden_dim=8,
                latent_dim=4,
                learnable_scale=True,
                initial_scale=4.0,
            ),
            direction_conditioning_mode="residual",
            direction_normalize=True,
        )
        observation = torch.randn(2, 6, 32, 32)
        goal = torch.randn(2, 3, 32, 32)
        left = torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1)

        output = conditioned(observation, goal, cmd_dir=left)
        output.sum().backward()

        raw_scale = conditioned.direction_encoder.raw_scale
        self.assertIsNotNone(raw_scale.grad)
        self.assertNotEqual(float(raw_scale.grad.abs().sum()), 0.0)

    def test_builder_uses_learnable_scale_when_configured(self):
        config = {
            "model_type": "nomad",
            "context_size": 1,
            "obs_encoder": "efficientnet-b0",
            "encoding_size": 32,
            "mha_num_attention_heads": 4,
            "mha_num_attention_layers": 1,
            "mha_ff_dim_factor": 4,
            "direction_conditioning": True,
            "direction_conditioning_mode": "residual",
            "direction_normalize": True,
            "direction_scale": 4.0,
            "direction_scale_learnable": True,
            "direction_num_commands": 3,
            "direction_hidden_dim": 8,
            "direction_latent_dim": 4,
            "down_dims": [8, 16],
            "cond_predict_scale": False,
        }

        model = build_model(config)
        direction_encoder = model.vision_encoder.direction_encoder

        self.assertTrue(direction_encoder.learnable_scale)
        self.assertAlmostEqual(
            direction_encoder.effective_scale().item(), 4.0, places=5
        )

    def test_checkpoint_uses_ema_for_inference_and_raw_model_for_resume(self):
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.no_grad():
            model.weight.fill_(1.0)
        raw_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        ema_state = {
            key: torch.full_like(value, 2.0) for key, value in raw_state.items()
        }

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=0,
                best_validation_loss=1.0,
                config={"train": {"training": {"learning_rate": 1e-3}}},
                inference_state_dict=ema_state,
                training_state_dict=raw_state,
                ema_state_dict={"shadow_params": list(ema_state.values())},
            )
            checkpoint = torch.load(path, map_location="cpu")
            with open(os.path.splitext(path)[0] + ".yaml") as stream:
                saved_config = yaml.safe_load(stream)
            inference_state = checkpoint_state_dict(checkpoint)
            resumed_model = torch.nn.Linear(2, 1, bias=False)
            load_training_checkpoint(path, resumed_model, device="cpu")

        torch.testing.assert_close(
            inference_state["weight"], torch.full((1, 2), 2.0)
        )
        torch.testing.assert_close(
            resumed_model.weight, torch.full((1, 2), 1.0)
        )
        self.assertEqual(
            saved_config["train"]["training"]["learning_rate"], 1e-3
        )


if __name__ == "__main__":
    unittest.main()
