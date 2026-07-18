# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Dhruv Shah, Ajay Sridhar, Arjun Bhorkar,
# Noriaki Hirose, Sergey Levine

"""GNM model adapted from visualnav-transformer under the MIT License.

See THIRD_PARTY_NOTICES.md for source and copyright information.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vnm_ros.models.base_model import BaseModel


def _make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetEncoder(nn.Module):
    def __init__(self, num_images: int = 1, width_mult: float = 1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = (
            _make_divisible(last_channel * max(1.0, width_mult), 8)
            if width_mult > 1.0
            else last_channel
        )
        features = [ConvBNReLU(3 * num_images, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
        direction_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(context_size, len_traj_pred, learn_angle)
        self.direction_encoder = direction_encoder
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )

        stacked_mobilenet = MobileNetEncoder(num_images=2 + self.context_size)
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.dist_predictor = nn.Sequential(nn.Linear(32, 1))
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params)
        )

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor = None,
        cmd_dir: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        if self.direction_encoder is not None and cmd_dir is not None:
            goal_encoding = self._direction_encoding(obs_img, cmd_dir)
        else:
            if goal_img is None:
                raise ValueError("GNM requires goal_img when cmd_dir is not provided")
            obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
            goal_encoding = self.goal_mobilenet(obs_goal_input)
            goal_encoding = self.flatten(goal_encoding)
            goal_encoding = self.compress_goal(goal_encoding)

        z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)
        dist_pred = self.dist_predictor(z)
        action_pred = self.action_predictor(z)
        action_pred = action_pred.reshape(
            (
                action_pred.shape[0],
                self.len_trajectory_pred,
                self.num_action_params,
            )
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )
        return dist_pred, action_pred

    def _direction_encoding(self, obs_img: torch.Tensor, cmd_dir: torch.Tensor):
        batch_size = obs_img.shape[0]
        device = obs_img.device
        dtype = obs_img.dtype
        cmd_dir = cmd_dir.to(device=device, dtype=dtype)
        if cmd_dir.ndim == 1:
            cmd_dir = cmd_dir.reshape(1, -1)
        if cmd_dir.shape[0] == 1 and batch_size > 1:
            cmd_dir = cmd_dir.repeat(batch_size, 1)
        if cmd_dir.shape[0] != batch_size:
            raise ValueError(
                f"cmd_dir batch size {cmd_dir.shape[0]} does not match image batch {batch_size}"
            )
        return self.direction_encoder(cmd_dir)
