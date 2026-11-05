# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    def __init__(self, args, prev_dim=1152, hidden_dim=2048, out_dim=2048, dim=768, pred_dim=512):
        super(SSLHead, self).__init__()
        self.dim = dim
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.projection_head_mu = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
        )
        # No LayerNorm on logvar output (would constrain the variance head).
        self.projection_head_logvar = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.rotation_head = nn.Linear(out_dim, 4)

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, pred_dim, bias=False),
            nn.LayerNorm(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, out_dim, bias=True),
        )

        self.reconstruct_pre = nn.Linear(out_dim, dim * 3 * 3 * 3)
        self.reconstruct_head = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1),
        )

    def representation(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def reparam(self, mu, logvar):
        """z = mu + sigma * eps, where sigma = exp(0.5 * logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        b = x.size()[0]
        hidden_states_out = self.swinViT(x.contiguous())
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        encs = [enc0, enc1, enc2, enc3, dec4]
        out = self.representation(encs)
        mu = self.projection_head_mu(out)
        logvar = self.projection_head_logvar(out)
        logvar = torch.clamp(logvar, -10, 10)
        # Stochastic latent in training; deterministic (z=mu) at eval.
        z = self.reparam(mu, logvar) if self.training else mu

        x_rot = self.rotation_head(z)
        p = self.predictor(z)
        x_rec = self.reconstruct_pre(z).view(b, self.dim, 3, 3, 3)
        x_rec = self.reconstruct_head(x_rec)

        return x_rot, mu, logvar, p, z.detach(), x_rec
