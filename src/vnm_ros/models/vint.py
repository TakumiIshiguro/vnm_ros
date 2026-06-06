from typing import List

import numpy as np

from vnm_ros.models.model_loader import load_model
from vnm_ros.utils.image_utils import to_numpy, transform_images


class VNMModel:
    def __init__(self, config: dict, checkpoint_path: str):
        import torch

        device_name = config.get("device", "auto")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.config = config
        self.model = load_model(checkpoint_path, config, self.device)

    @property
    def context_size(self) -> int:
        return int(self.config["context_size"])

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

    def scale_waypoint(self, waypoint: np.ndarray, max_v: float, model_rate: float):
        if self.config.get("normalize", True):
            waypoint = waypoint.copy()
            waypoint[:2] *= max_v / model_rate
        return waypoint

