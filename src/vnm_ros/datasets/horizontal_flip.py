import copy
import math

import numpy as np


def horizontal_flip_trajectory(trajectory: dict, source_trajectory: str = "") -> dict:
    flipped = copy.deepcopy(trajectory)

    positions = np.asarray(trajectory["position"]).copy()
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"position must have shape [N, >=2], got {positions.shape}")
    positions[:, 1] *= -1
    flipped["position"] = positions

    yaw = np.asarray(trajectory["yaw"]).copy()
    yaw = (-yaw + math.pi) % (2.0 * math.pi) - math.pi
    flipped["yaw"] = yaw.astype(np.asarray(trajectory["yaw"]).dtype, copy=False)

    if "cmd_dir" in trajectory:
        cmd_dir = np.asarray(trajectory["cmd_dir"]).copy()
        if cmd_dir.ndim != 2 or cmd_dir.shape[1] < 3:
            raise ValueError(f"cmd_dir must have shape [N, >=3], got {cmd_dir.shape}")
        cmd_dir[:, [1, 2]] = cmd_dir[:, [2, 1]]
        flipped["cmd_dir"] = cmd_dir

    metadata = dict(trajectory.get("metadata", {}))
    metadata["augmentation"] = "horizontal_flip"
    if source_trajectory:
        metadata["source_trajectory"] = source_trajectory
    flipped["metadata"] = metadata
    return flipped
