import os
import tempfile
import unittest

import torch
import yaml

from vnm_ros.models.model_loader import checkpoint_state_dict
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
