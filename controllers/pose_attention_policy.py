"""
Final temporal-spatial visual quadrotor controller.

Feature order:
    geometry             64
    linear velocity      64
    angular velocity     64
    previous actions     32
    spatial pose         64
    normalized pose       5

Total: 293 dimensions.
"""

import copy
import torch
from torch import nn

class Augmented(nn.Module):
    def __init__(self, base, pose):
        super().__init__()

        self.base = base.eval().requires_grad_(False)
        self.pose = pose.eval().requires_grad_(False)

        self.head = copy.deepcopy(base.head)

        first = self.head[0]

        assert isinstance(first, nn.Linear)
        assert first.in_features == 224

        # 224 original features + 64 pose features
        # + 5 supervised pose predictions.
        wide = nn.Linear(
            293, first.out_features
        ).to(device=first.weight.device, dtype=first.weight.dtype)

        # Preserve original controller at initialization.
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, :224].copy_(first.weight)
            wide.bias.copy_(first.bias)

        self.head[0] = wide
        self.head.requires_grad_(True)

    def features(self, c, h):
        c = c.reshape(-1, 32, 8)
        b = self.base

        z = torch.cat([
            b._branch_features(b.estimator.geometry, c),
            b._branch_features(b.estimator.linear_velocity, c),
            b._branch_features(b.estimator.angular_velocity, c),
            h.reshape(-1, 32),
        ], -1)

        f, p = self.pose(c[:, -1])

        valid = (
            c[:, -1].abs().sum(-1, keepdim=True) > 1e-6
        ).to(f.dtype)

        return torch.cat((
            z,
            f * valid,
            p / self.pose.scale * valid,
        ), -1)

    def forward(self, c, h):
        return self.head(self.features(c, h))
