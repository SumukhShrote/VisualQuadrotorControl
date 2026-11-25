"""
Multi-timescale supervised visual state estimation.

Extracted from the original supervised Transformer experiments.
Architecture and parameter names are preserved.
"""

import torch
from torch import nn

GROUPS = {
    "position_m": [0, 1, 2],
    "linear_velocity_mps": [3, 4, 5],
    "angular_velocity_radps": [6, 7, 8],
    "projected_gravity": [9, 10, 11],
    "marker_direction": [12, 13, 14],
    "marker_distance_m": [15],
}

TARGET_IDX = list(range(15)) + [19]

class TemporalAttentionBranch(nn.Module):
    """Independent attention encoder for one state group."""

    def __init__(
        self,
        window,
        output_dim,
        width=64,
        num_heads=4,
        num_layers=2,
        include_visibility=False,
    ):
        super().__init__()

        self.window = window

        self.include_visibility = bool(include_visibility)
        self.input_projection = nn.Linear(12 if include_visibility else 8, width)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, window, width)
        )

        nn.init.normal_(
            self.position_embedding,
            std=0.02,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=width * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.query = nn.Parameter(
            torch.zeros(1, 1, width)
        )

        nn.init.normal_(self.query, std=0.02)

        self.pool = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(width)

        self.output = nn.Sequential(
            nn.Linear(width, 96),
            nn.GELU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, history):
        # Strictly restrict each branch to its assigned
        # temporal window. No privileged inputs.
        x = history[:, -self.window:, :]
        if self.include_visibility:
            pairs = x.reshape(x.shape[0], self.window, 4, 2)
            validity = (~(pairs < -0.99).all(dim=-1)).to(dtype=x.dtype)
            x = torch.cat((x, validity), dim=-1)

        z = (
            self.input_projection(x)
            + self.position_embedding
        )

        z = self.encoder(z)

        query = self.query.expand(
            history.shape[0], -1, -1
        )

        pooled, _ = self.pool(
            query,
            z,
            z,
            need_weights=False,
        )

        # Preserve direct access to newest visual geometry.
        features = self.norm(
            pooled[:, 0, :] + z[:, -1, :]
        )

        return self.output(features)

class TemporalEstimator(nn.Module):
    """
    Supervised multi-timescale state estimator.

    Inputs:
        [batch, 32, 8] ArUco corner history.

    Outputs:
        [batch, 16] reconstructed physical state.

    Branches:
        Geometry:  8 frames -> position, gravity,
                    marker direction, hover distance.
        Linear:   32 frames -> linear velocity.
        Angular:   8 frames -> angular velocity.

    No actions, IMU, ground-truth state, or other
    privileged information enters the estimator.
    """

    def __init__(self, include_visibility=False):
        super().__init__()

        self.geometry = TemporalAttentionBranch(
            window=8,
            output_dim=10,
            include_visibility=include_visibility,
        )

        self.linear_velocity = TemporalAttentionBranch(
            window=32,
            output_dim=3,
            include_visibility=include_visibility,
        )

        self.angular_velocity = TemporalAttentionBranch(
            window=8,
            output_dim=3,
            include_visibility=include_visibility,
        )

    def forward(self, observations):
        if (
            observations.ndim != 3
            or observations.shape[1:] != (32, 8)
        ):
            raise RuntimeError(
                f"Expected [B,32,8], got "
                f"{tuple(observations.shape)}"
            )

        geom = self.geometry(observations)
        linear = self.linear_velocity(observations)
        angular = self.angular_velocity(observations)

        # Match the exact 16-D target ordering used by
        # our existing supervised reconstruction experiment.
        return torch.cat([
            geom[:, 0:3],    # Position
            linear,         # Linear velocity
            angular,        # Angular velocity
            geom[:, 3:6],   # Projected gravity
            geom[:, 6:9],   # Marker direction
            geom[:, 9:10],  # Hover distance
        ], dim=-1)
