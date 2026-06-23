from typing import List

import numpy as np

from vnm_ros.utils.image_utils import to_numpy, transform_images


class NoMaDInference:
    def __init__(self, model, config: dict, device):
        self.model = model
        self.config = config
        self.device = device
        self.noise_scheduler = None

    def predict(self, context_images: List, goal_images: List):
        import torch

        obs_images = transform_images(context_images, self.config["image_size"]).to(
            self.device
        )
        goal_tensor = torch.cat(
            [
                transform_images(goal_image, self.config["image_size"]).to(self.device)
                for goal_image in goal_images
            ],
            dim=0,
        )

        goal_count = goal_tensor.shape[0]
        goal_mask = torch.zeros(goal_count, dtype=torch.long, device=self.device)
        with torch.no_grad():
            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_images.repeat(goal_count, 1, 1, 1),
                goal_img=goal_tensor,
                input_goal_mask=goal_mask,
            )
            distances = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            waypoints = self._sample_actions(obsgoal_cond)

        return to_numpy(distances), to_numpy(waypoints)

    def scale_waypoint(self, waypoint: np.ndarray, max_v: float, model_rate: float):
        return waypoint

    def _sample_actions(self, obsgoal_cond):
        import torch

        scheduler = self._noise_scheduler()
        sample_count = int(self.config.get("num_action_samples", 1))
        horizon = int(self.config["len_traj_pred"])
        goal_count = obsgoal_cond.shape[0]
        cond = obsgoal_cond.repeat_interleave(sample_count, dim=0)
        action = torch.randn(
            (goal_count * sample_count, horizon, 2), device=self.device
        )
        scheduler.set_timesteps(int(self.config.get("num_diffusion_iters", 10)))
        for timestep in scheduler.timesteps:
            noise_pred = self.model(
                "noise_pred_net",
                sample=action,
                timestep=timestep,
                global_cond=cond,
            )
            action = scheduler.step(
                model_output=noise_pred, timestep=timestep, sample=action
            ).prev_sample

        action = self._action_from_delta(action)
        action = action.reshape(goal_count, sample_count, horizon, 2)
        if self.config.get("action_sample_strategy", "first") == "mean":
            return action.mean(dim=1)
        return action[:, 0, :, :]

    def _noise_scheduler(self):
        if self.noise_scheduler is not None:
            return self.noise_scheduler
        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        except ImportError as exc:
            raise ImportError(
                "NoMaD inference requires diffusers. Install diffusers before "
                "using model_type: nomad."
            ) from exc
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(self.config.get("num_diffusion_iters", 10)),
            beta_schedule=self.config.get("beta_schedule", "squaredcos_cap_v2"),
            clip_sample=bool(self.config.get("clip_sample", True)),
            prediction_type=self.config.get("prediction_type", "epsilon"),
        )
        return self.noise_scheduler

    def _action_from_delta(self, normalized_delta):
        import torch

        action_stats = self.config.get("action_stats", {})
        min_values = action_stats.get("min")
        max_values = action_stats.get("max")
        if min_values is None or max_values is None:
            return torch.cumsum(normalized_delta, dim=1)

        min_tensor = torch.tensor(
            min_values, dtype=normalized_delta.dtype, device=normalized_delta.device
        )
        max_tensor = torch.tensor(
            max_values, dtype=normalized_delta.dtype, device=normalized_delta.device
        )
        delta = (normalized_delta + 1.0) * 0.5 * (max_tensor - min_tensor)
        delta = delta + min_tensor
        return torch.cumsum(delta, dim=1)
