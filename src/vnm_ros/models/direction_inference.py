from typing import List, Sequence

import numpy as np

from vnm_ros.utils.image_utils import to_numpy, transform_images


class DirectionViNTInference:
    def __init__(self, model, config: dict, device):
        self.model = model
        self.config = config
        self.device = device

    def predict(self, context_images: List, cmd_dir: Sequence[float]):
        import torch

        obs = transform_images(context_images, self.config["image_size"]).to(self.device)
        cmd = torch.tensor([cmd_dir], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            distances, waypoints = self.model(obs, cmd)

        return to_numpy(distances), to_numpy(waypoints)

    def scale_waypoint(self, waypoint: np.ndarray, max_v: float, model_rate: float):
        if self.config.get("normalize", True):
            waypoint = waypoint.copy()
            waypoint[:2] *= max_v / model_rate
        return waypoint
