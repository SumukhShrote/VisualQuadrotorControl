"""Corners-only causal-TCN actor with a privileged asymmetric critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from rsl_rl.modules import ActorCritic


class CausalTCNBlock(nn.Module):
    """One left-padded causal temporal-convolution block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 11,
    ):
        super().__init__()

        if kernel_size < 1:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}."
            )

        self.left_padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
        )

        self.residual = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=1,
            )
        )

        self.activation = nn.ELU()

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual(sequence)

        padded = F.pad(
            sequence,
            (self.left_padding, 0),
        )

        convolved = self.conv(padded)

        if convolved.shape != residual.shape:
            raise RuntimeError(
                "TCN residual shape mismatch: "
                f"convolved={tuple(convolved.shape)}, "
                f"residual={tuple(residual.shape)}."
            )

        return self.activation(
            convolved + residual
        )


class CornersAsymmetricActorCritic(ActorCritic):
    """Standard PPO actor-critic with asymmetric observations.

    Actor:
        32 x 8 corner history
        -> three causal kernel-11 TCN blocks
        -> newest 128-D temporal feature
        -> 256 -> 256 -> 4
        -> tanh action mean

    Critic:
        privileged 20-D physical observation
        -> ordinary ActorCritic critic MLP
        -> scalar value
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        history_length: int = 32,
        token_dim: int = 8,
        temporal_channels: int = 128,
        **kwargs,
    ):
        self.history_length = int(history_length)
        self.token_dim = int(token_dim)
        self.temporal_channels = int(temporal_channels)

        expected_actor_obs = (
            self.history_length * self.token_dim
        )

        if num_actor_obs != expected_actor_obs:
            raise ValueError(
                f"Expected {expected_actor_obs} actor observations, "
                f"got {num_actor_obs}."
            )

        if num_critic_obs != 20:
            raise ValueError(
                "The asymmetric critic must receive exactly 20 "
                f"privileged values, got {num_critic_obs}."
            )

        if num_actions != 4:
            raise ValueError(
                f"Expected four CTBR actions, got {num_actions}."
            )

        # Construct the ordinary RSL-RL critic, action distribution, and
        # temporary actor. The temporary actor is replaced below.
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        self.tcn = nn.Sequential(
            CausalTCNBlock(
                input_dim=self.token_dim,
                output_dim=self.temporal_channels,
                kernel_size=11,
            ),
            CausalTCNBlock(
                input_dim=self.temporal_channels,
                output_dim=self.temporal_channels,
                kernel_size=11,
            ),
            CausalTCNBlock(
                input_dim=self.temporal_channels,
                output_dim=self.temporal_channels,
                kernel_size=11,
            ),
        )

        self.actor = nn.Sequential(
            nn.Linear(
                self.temporal_channels,
                256,
            ),
            nn.ELU(),
            nn.Linear(
                256,
                256,
            ),
            nn.ELU(),
            nn.Linear(
                256,
                num_actions,
            ),
        )

        print(
            "[CornersAsymmetricActorCritic]",
            f"actor_obs={num_actor_obs}",
            f"critic_obs={num_critic_obs}",
            f"actions={num_actions}",
            f"history={self.history_length}",
            f"token_dim={self.token_dim}",
            f"temporal_channels={self.temporal_channels}",
            f"initial_noise_std={init_noise_std}",
            flush=True,
        )

    def encode_temporal_state(
        self,
        actor_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the corner history without privileged information."""

        if actor_observations.ndim != 2:
            raise RuntimeError(
                "Actor observations must have shape "
                f"[batch, {self.history_length * self.token_dim}], "
                f"got {tuple(actor_observations.shape)}."
            )

        expected_dim = (
            self.history_length * self.token_dim
        )

        if actor_observations.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Expected actor observation dimension {expected_dim}, "
                f"got {actor_observations.shape[-1]}."
            )

        sequence = actor_observations.reshape(
            actor_observations.shape[0],
            self.history_length,
            self.token_dim,
        )

        # Conv1D expects batch x channels x time.
        sequence = sequence.transpose(1, 2)

        temporal_features = self.tcn(
            sequence
        )

        # Three kernel-11 causal blocks have a 31-frame receptive field.
        return temporal_features[:, :, -1]

    def _actor_mean(
        self,
        actor_observations: torch.Tensor,
    ) -> torch.Tensor:
        temporal_state = self.encode_temporal_state(
            actor_observations
        )

        raw_action = self.actor(
            temporal_state
        )

        return torch.tanh(
            raw_action
        )

    def update_distribution(
        self,
        actor_observations: torch.Tensor,
    ):
        mean = self._actor_mean(
            actor_observations
        )

        if self.noise_std_type == "scalar":
            standard_deviation = self.std.expand_as(
                mean
            )
        elif self.noise_std_type == "log":
            standard_deviation = torch.exp(
                self.log_std
            ).expand_as(mean)
        else:
            raise ValueError(
                "Unsupported noise_std_type: "
                f"{self.noise_std_type!r}."
            )

        self.distribution = Normal(
            mean,
            standard_deviation,
        )

    def act_inference(
        self,
        actor_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return the deterministic corners-only action mean."""

        return self._actor_mean(
            actor_observations
        )
