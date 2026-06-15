# SPDX-License-Identifier: MIT
# Copyright (c) 2023 Dhruv Shah, Ajay Sridhar, Nitish Dashora,
# Kyle Stachowicz, Kevin Black, Noriaki Hirose, Sergey Levine

"""Base model adapted from visualnav-transformer under the MIT License.

See THIRD_PARTY_NOTICES.md for source and copyright information.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        self.num_action_params = 4 if self.learn_angle else 2

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        return torch.flatten(z, 1)

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
