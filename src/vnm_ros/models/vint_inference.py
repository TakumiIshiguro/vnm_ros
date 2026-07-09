from typing import List

import numpy as np

from vnm_ros.utils.image_utils import to_numpy, transform_images


class ViNTInference:
    def __init__(self, model, config: dict, device):
        self.model = model
        self.config = config
        self.device = device

    def predict(self, context_images: List, goal_images: List):
        import torch

        batch_obs = []
        batch_goal = []
        for goal_image in goal_images:
            batch_obs.append(transform_images(context_images, self.config["image_size"]))
            batch_goal.append(transform_images(goal_image, self.config["image_size"]))

        batch_obs = torch.cat(batch_obs, dim=0).to(self.device)
        batch_goal = torch.cat(batch_goal, dim=0).to(self.device)

        with torch.no_grad():
            distances, waypoints = self.model(batch_obs, batch_goal)

        return to_numpy(distances), to_numpy(waypoints)

    def predict_explore(self, context_images: List, cmd_dir=None):
        import torch

        obs = transform_images(context_images, self.config["image_size"]).to(self.device)
        cmd_dir_tensor = self._cmd_dir_tensor(cmd_dir, batch_size=1, dtype=obs.dtype)
        with torch.no_grad():
            _, waypoints = self.model(obs, goal_img=None, cmd_dir=cmd_dir_tensor)
        return to_numpy(waypoints)

    def scale_waypoint(self, waypoint: np.ndarray, max_v: float, model_rate: float):
        if self.config.get("normalize", True):
            waypoint = waypoint.copy()
            waypoint[..., :2] *= max_v / model_rate
        return waypoint

    def _cmd_dir_tensor(self, cmd_dir, batch_size: int, dtype):
        import torch

        if not self.config.get("direction_conditioning", False):
            return None
        if cmd_dir is None:
            cmd_dir = [1.0, 0.0, 0.0]
        cmd_dir_tensor = torch.as_tensor(cmd_dir, dtype=dtype, device=self.device)
        if cmd_dir_tensor.ndim == 1:
            cmd_dir_tensor = cmd_dir_tensor.reshape(1, -1)
        if cmd_dir_tensor.shape[0] == 1 and batch_size > 1:
            cmd_dir_tensor = cmd_dir_tensor.repeat(batch_size, 1)
        return cmd_dir_tensor
