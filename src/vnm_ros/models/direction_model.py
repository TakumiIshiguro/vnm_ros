from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from vnm_ros.models.base_model import BaseModel
from vnm_ros.models.self_attention import MultiLayerDecoder


class DirectionViNT(BaseModel):
    """ViNT-style policy conditioned on a direction one-hot vector.

    The command token replaces ViNT's goal-image token. The input sequence is:
    previous/current image tokens followed by one direction token.
    """

    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        cmd_dir_dim: Optional[int] = 3,
        cmd_hidden_size: Optional[int] = 128,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super().__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.cmd_dir_dim = cmd_dir_dim

        if obs_encoder.split("-")[0] != "efficientnet":
            raise NotImplementedError(
                f"Unsupported DirectionViNT observation encoder: {obs_encoder}"
            )

        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.num_obs_features = self.obs_encoder._fc.in_features
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size
            )
        else:
            self.compress_obs_enc = nn.Identity()

        self.cmd_encoder = nn.Sequential(
            nn.Linear(self.cmd_dir_dim, cmd_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cmd_hidden_size, self.obs_encoding_size),
            nn.LayerNorm(self.obs_encoding_size),
            nn.ReLU(inplace=True),
        )

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(nn.Linear(32, 1))
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params)
        )

    def forward(
        self, obs_img: torch.Tensor, cmd_dir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cmd_dir.dim() == 1:
            cmd_dir = cmd_dir.unsqueeze(0)
        if cmd_dir.shape[-1] != self.cmd_dir_dim:
            raise ValueError(
                f"cmd_dir must have shape [batch, {self.cmd_dir_dim}], "
                f"got {tuple(cmd_dir.shape)}"
            )

        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.cat(obs_img, dim=0)
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape(
            (self.context_size + 1, -1, self.obs_encoding_size)
        )
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        cmd_dir = cmd_dir.to(device=obs_encoding.device, dtype=obs_encoding.dtype)
        cmd_encoding = self.cmd_encoder(cmd_dir).unsqueeze(1)

        final_repr = self.decoder(torch.cat((obs_encoding, cmd_encoding), dim=1))
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)
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
