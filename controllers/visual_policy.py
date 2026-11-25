"""
Temporal visual controllers.

Includes the original corners-only actor and the
action-history controller used by the final policy.
"""

import copy
import torch
from torch import nn

from networks.temporal_transformer import TemporalEstimator

K_VALUES = (0, 1, 8)

class Actor(nn.Module):
    """
    Strict corners-only actor.

    Action head consumes attention features directly.
    Privileged state is only an optional supervised target.
    """

    def __init__(self, estimator_type):
        super().__init__()

        self.estimator = estimator_type()

        self.policy = nn.Sequential(
            nn.Linear(192, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Tanh(),
        )

    @staticmethod
    def branch_features(branch, history):
        x = history[:, -branch.window:, :]

        z = branch.encoder(
            branch.input_projection(x)
            + branch.position_embedding
        )

        query = branch.query.expand(
            x.shape[0], -1, -1
        )

        pooled, _ = branch.pool(
            query,
            z,
            z,
            need_weights=False,
        )

        return branch.norm(
            pooled[:, 0, :] + z[:, -1, :]
        )

    def forward(self, corners256):
        history = corners256.reshape(-1, 32, 8)

        geometry = self.branch_features(
            self.estimator.geometry, history
        )

        linear = self.branch_features(
            self.estimator.linear_velocity, history
        )

        angular = self.branch_features(
            self.estimator.angular_velocity, history
        )

        features = torch.cat(
            (geometry, linear, angular),
            dim=-1,
        )

        actions = self.policy(features)

        geometry_state = (
            self.estimator.geometry.output(geometry)
        )

        linear_state = (
            self.estimator.linear_velocity.output(linear)
        )

        angular_state = (
            self.estimator.angular_velocity.output(angular)
        )

        state = torch.cat(
            (
                geometry_state[:, :3],
                linear_state,
                angular_state,
                geometry_state[:, 3:],
            ),
            dim=-1,
        )

        return actions, state

class DeployPolicy(nn.Module):
    """Export only the corners-to-actions interface."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, corners256):
        return self.actor(corners256)[0]

class Policy(nn.Module):
    """Visual controller with configurable action history.

    Model construction is independent of checkpoint loading.
    Existing checkpoint parameter names are preserved.
    """

    def __init__(self, k=8):
        super().__init__()

        assert k in K_VALUES

        self.k = int(k)

        base = Actor(TemporalEstimator)

        self.estimator = base.estimator
        self.head = copy.deepcopy(base.policy)

        old_linear = self.head[0]

        assert isinstance(old_linear, nn.Linear)
        assert old_linear.in_features == 192

        expanded = nn.Linear(
            224,
            old_linear.out_features,
        )

        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :192].copy_(
                old_linear.weight
            )
            expanded.bias.copy_(
                old_linear.bias
            )

        self.head[0] = expanded

        self._branch_features = Actor.branch_features

    def forward(self, corner_history, actions8):
            corners = corner_history.reshape(-1, 32, 8)

            assert actions8.ndim == 3

            g = self._branch_features(
                self.estimator.geometry,
                corners
            )

            v = self._branch_features(
                self.estimator.linear_velocity,
                corners
            )

            w = self._branch_features(
                self.estimator.angular_velocity,
                corners
            )

            visual = torch.cat([g, v, w], dim=-1)

            if self.k == 0:
                a = torch.zeros_like(actions8)

            elif self.k == 1:
                a = torch.cat([
                    torch.zeros_like(actions8[:, :-1]),
                    actions8[:, -1:]
                ], dim=1)

            else:
                a = actions8

            return self.head(
                torch.cat(
                    [visual, a.reshape(-1, 32)],
                    dim=-1
                )
            )
