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
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence


class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.recon_loss = torch.nn.L1Loss()
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        self.alpha3 = args.alpha3

    def forward(self, output_rot, target_rot,
                mu1, logvar1, mu2, logvar2,
                p1, p2, z1, z2,
                output_recons, target_recons):
        rot_loss = self.rot_loss(output_rot, target_rot)

        # ProbCL: q(z|x) = N(mu, diag(exp(logvar))) with std floor to avoid point-mass collapse
        scale1 = (0.5 * logvar1).exp().clamp(min=1e-2)
        scale2 = (0.5 * logvar2).exp().clamp(min=1e-2)
        dist1 = Normal(loc=mu1, scale=scale1)
        dist2 = Normal(loc=mu2, scale=scale2)
        # Stop-grad targets so each side only matches the other's frozen distribution
        dist1_sg = Normal(loc=mu1.detach(), scale=scale1.detach())
        dist2_sg = Normal(loc=mu2.detach(), scale=scale2.detach())

        # 1) Distribution Contrastive Loss (DCL): KL(q1||sg(q2)) + KL(q2||sg(q1))
        kl_12 = kl_divergence(dist1, dist2_sg)  # (B, D)
        kl_21 = kl_divergence(dist2, dist1_sg)  # (B, D)
        dcl = 0.5 * (kl_12.mean() + kl_21.mean()) * 10

        # 2) Similarity Contrastive Loss (SCL), stop-grad on z
        p1_norm = F.normalize(p1, dim=1)
        p2_norm = F.normalize(p2, dim=1)
        z1_norm = F.normalize(z1.detach(), dim=1)
        z2_norm = F.normalize(z2.detach(), dim=1)
        sim_1 = (p1_norm * z2_norm).sum(dim=1)
        sim_2 = (p2_norm * z1_norm).sum(dim=1)
        scl = 1 - 0.5 * (sim_1.mean() + sim_2.mean())

        # 3) Prior regularization KL(q || N(0, I))
        dist0 = Normal(loc=torch.zeros_like(mu1), scale=torch.ones_like(mu1))
        kl_1p = kl_divergence(dist1, dist0)  # (B, D)
        kl_2p = kl_divergence(dist2, dist0)  # (B, D)
        reg_loss = 0.5 * (kl_1p.mean() + kl_2p.mean())

        contrast_loss = self.alpha1 * dcl + self.alpha2 * scl + self.alpha3 * reg_loss
        recon_loss = self.recon_loss(output_recons, target_recons)
        total_loss = self.alpha * contrast_loss + self.beta * recon_loss + self.gamma * rot_loss
        return total_loss, (rot_loss, contrast_loss, recon_loss, dcl, scl, reg_loss)
