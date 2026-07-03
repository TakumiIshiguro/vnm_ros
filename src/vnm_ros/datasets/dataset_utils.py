import math
import os
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from vnm_ros.utils.image_utils import center_crop_resize


def numeric_image_files(directory: str) -> List[str]:
    files = [
        name
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(files, key=lambda name: int(os.path.splitext(name)[0]))


def load_image(path: str, image_size: Sequence[int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != tuple(image_size):
        image = center_crop_resize(image, image_size)
    return TF.to_tensor(image)


def yaw_to_rotation(yaw: float) -> np.ndarray:
    return np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )


def to_local_coords(positions: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    return (positions - origin).dot(yaw_to_rotation(yaw))


def angle_to_sin_cos(actions: np.ndarray) -> np.ndarray:
    result = np.zeros((actions.shape[0], 4), dtype=np.float32)
    result[:, :2] = actions[:, :2]
    result[:, 2] = np.cos(actions[:, 2])
    result[:, 3] = np.sin(actions[:, 2])
    return result
