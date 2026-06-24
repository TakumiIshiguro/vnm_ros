from typing import List

from vnm_ros.models.direction_inference import DirectionViNTInference
from vnm_ros.models.model_loader import load_model
from vnm_ros.models.nomad_inference import NoMaDInference
from vnm_ros.models.vint_inference import ViNTInference


class VNMModel:
    def __init__(self, config: dict, checkpoint_path: str):
        import torch

        device_name = config.get("device", "auto")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.config = config
        self.model = load_model(checkpoint_path, config, self.device)
        self.model_type = config["model_type"]
        self.inference = self._build_inference()

    @property
    def context_size(self) -> int:
        return int(self.config["context_size"])

    def predict(self, context_images: List, goal_images: List):
        return self.inference.predict(context_images, goal_images)

    def predict_direction(self, context_images: List, cmd_dir):
        return self.inference.predict(context_images, cmd_dir)

    def scale_waypoint(self, waypoint, max_v: float, model_rate: float):
        return self.inference.scale_waypoint(waypoint, max_v, model_rate)

    def _build_inference(self):
        if self.model_type == "vint":
            return ViNTInference(self.model, self.config, self.device)
        if self.model_type == "direction_vint":
            return DirectionViNTInference(self.model, self.config, self.device)
        if self.model_type == "nomad":
            return NoMaDInference(self.model, self.config, self.device)
        raise ValueError(f"Unsupported model_type for inference: {self.model_type}")
