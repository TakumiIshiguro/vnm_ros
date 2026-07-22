import math
import os
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from vnm_ros.utils.image_utils import preprocess_image


def numeric_image_files(directory: str) -> List[str]:
    files = [
        name
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(files, key=lambda name: int(os.path.splitext(name)[0]))


def load_image(
    path: str,
    image_size: Sequence[int],
    center_crop: bool = False,
) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = preprocess_image(image, image_size, center_crop=center_crop)
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


def majority_cmd_dir_label(cmd_dirs: np.ndarray):
    cmd_dirs = np.asarray(cmd_dirs)
    if cmd_dirs.ndim != 2 or cmd_dirs.shape[1] < 3:
        raise ValueError(f"cmd_dirs must have shape [N, >=3], got {cmd_dirs.shape}")
    labels = np.argmax(cmd_dirs[:, :3], axis=1)
    valid = np.sum(cmd_dirs[:, :3], axis=1) > 0.0
    labels = np.where(valid, labels, 0)
    counts = np.bincount(labels, minlength=3)
    winners = np.flatnonzero(counts == counts.max())
    if len(winners) != 1:
        return None
    return int(winners[0])


def cmd_dir_one_hot(label: int) -> np.ndarray:
    result = np.zeros(3, dtype=np.float32)
    result[int(label)] = 1.0
    return result
