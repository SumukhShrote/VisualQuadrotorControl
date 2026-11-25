"""
Spatial self-attention over ordered marker corners.

Returns 64 learned pose features and five physical predictions.
"""

import torch
from torch import nn

class PoseAttention(nn.Module):
    """
    Spatial self-attention over four ordered marker corners.

    Outputs:
        64-dimensional learned pose features
        5-dimensional supervised pose prediction

    The existing temporal attention branches remain unchanged.
    """

    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(9, 64)
        self.pos = nn.Parameter(torch.zeros(1, 4, 64))

        self.qkv = nn.Linear(64, 192)
        self.mix = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)

        self.ff = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        self.readout = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

        self.register_buffer(
            "scale",
            torch.tensor([.3, .3, .3, .25, .1]),
        )

    def forward(self, c):
        c = c.reshape(-1, 4, 2)
        ctr = c.mean(1, keepdim=True)

        edge = (
            (c - torch.roll(c, -1, 1))
            .square()
            .sum(-1)
            .clamp_min(1e-12)
            .sqrt()
        )

        size = edge.mean(1, keepdim=True).clamp_min(.002)

        shape = (c - ctr) / size[:, :, None]

        # Opposite edge-length differences expose
        # perspective distortion rather than only centering.
        persp = torch.stack((
            edge[:, 0] - edge[:, 2],
            edge[:, 1] - edge[:, 3],
        ), -1) / size * 20

        globals_ = torch.cat((
            (ctr[:, 0] - .5) * 4,
            torch.log(size / .078),
            persp,
        ), -1)

        z = torch.cat((
            shape,
            c - .5,
            globals_[:, None].expand(-1, 4, -1),
        ), -1)

        h = self.embed(z) + self.pos

        q, k, v = self.qkv(h).chunk(3, -1)

        attention = torch.softmax(
            q @ k.transpose(-1, -2) / 8.0,
            -1,
        )

        h = self.norm(h + self.mix(attention @ v))
        h = h + self.ff(h)

        features = self.readout(h.flatten(1))
        prediction = self.decoder(features) * self.scale

        return features, prediction
