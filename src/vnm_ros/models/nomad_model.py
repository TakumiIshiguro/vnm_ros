# SPDX-License-Identifier: MIT
# Copyright (c) 2023 Dhruv Shah, Ajay Sridhar, Catherine Glossop,
# Sergey Levine

"""NoMaD model components adapted from visualnav-transformer.

The diffusion action head depends on the external `diffusion_policy` package.
It is imported lazily so ViNT-only use does not require NoMaD dependencies.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from vnm_ros.models.self_attention import PositionalEncoding


class NoMaD(nn.Module):
    def __init__(self, vision_encoder, noise_pred_net, dist_pred_net):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net

    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            return self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs.get("input_goal_mask"),
            )
        if func_name == "noise_pred_net":
            return self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=kwargs["global_cond"],
            )
        if func_name == "dist_pred_net":
            return self.dist_pred_net(kwargs["obsgoal_cond"])
        raise NotImplementedError(f"Unsupported NoMaD function: {func_name}")


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(embedding_dim // 16, 1),
        )

    def forward(self, x):
        return self.network(x.reshape((-1, self.embedding_dim)))


class NoMaDViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 3,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 256,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 4,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        if obs_encoder.split("-")[0] != "efficientnet":
            raise NotImplementedError(
                f"Unsupported NoMaD observation encoder: {obs_encoder}"
            )
        self.obs_encoder = replace_bn_with_gn(
            EfficientNet.from_name(obs_encoder, in_channels=3)
        )
        self.num_obs_features = self.obs_encoder._fc.in_features

        self.goal_encoder = replace_bn_with_gn(
            EfficientNet.from_name("efficientnet-b0", in_channels=6)
        )
        self.num_goal_features = self.goal_encoder._fc.in_features

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size
            )
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(
                self.num_goal_features, self.goal_encoding_size
            )
        else:
            self.compress_goal_enc = nn.Identity()

        self.positional_encoding = PositionalEncoding(
            self.obs_encoding_size, max_seq_len=self.context_size + 2
        )
        sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(
            sa_layer, num_layers=mha_num_attention_layers
        )

        goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        goal_mask[:, -1] = True
        no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.register_buffer("goal_mask", goal_mask)
        self.register_buffer("no_mask", no_mask)
        self.register_buffer("all_masks", torch.cat([no_mask, goal_mask], dim=0))
        avg_pool_mask = torch.cat(
            [
                1 - no_mask.float(),
                (1 - goal_mask.float())
                * ((self.context_size + 2) / (self.context_size + 1)),
            ],
            dim=0,
        )
        self.register_buffer("avg_pool_mask", avg_pool_mask)

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        device = obs_img.device
        obsgoal_img = torch.cat(
            [obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1
        )
        goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)

        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.cat(obs_img, dim=0)
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape(
            (self.context_size + 1, -1, self.obs_encoding_size)
        )
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)

        if input_goal_mask is not None:
            goal_mask_index = input_goal_mask.to(device).long()
            src_key_padding_mask = torch.index_select(
                self.all_masks.to(device), 0, goal_mask_index
            )
        else:
            goal_mask_index = None
            src_key_padding_mask = None

        obs_encoding = self.positional_encoding(obs_encoding)
        obs_encoding_tokens = self.sa_encoder(
            obs_encoding, src_key_padding_mask=src_key_padding_mask
        )
        if goal_mask_index is not None:
            avg_mask = torch.index_select(
                self.avg_pool_mask.to(device), 0, goal_mask_index
            ).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        return torch.mean(obs_encoding_tokens, dim=1)


def build_conditional_unet1d(
    input_dim: int, global_cond_dim: int, down_dims, cond_predict_scale: bool
):
    try:
        from diffusion_policy.model.diffusion.conditional_unet1d import (
            ConditionalUnet1D,
        )
    except ImportError as exc:
        raise ImportError(
            "NoMaD requires diffusion_policy and its dependencies. "
            "Install diffusion_policy plus packages such as einops before "
            "using model_type: nomad."
        ) from exc
    return ConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        down_dims=down_dims,
        cond_predict_scale=cond_predict_scale,
    )


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16):
    return replace_submodules(
        root_module=root_module,
        predicate=lambda module: isinstance(module, nn.BatchNorm2d),
        func=lambda module: nn.GroupNorm(
            num_groups=module.num_features // features_per_group,
            num_channels=module.num_features,
        ),
    )


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
):
    if predicate(root_module):
        return func(root_module)
    bn_list = [
        key.split(".")
        for key, module in root_module.named_modules(remove_duplicate=True)
        if predicate(module)
    ]
    for *parent, key in bn_list:
        parent_module = root_module
        if parent:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(key)] = func(parent_module[int(key)])
        else:
            setattr(parent_module, key, func(getattr(parent_module, key)))
    return root_module
