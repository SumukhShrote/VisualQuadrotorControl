# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class QuadrotorTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 750
    save_interval = 50
    experiment_name = "CTBR_Hover_Teacher"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[256, 256, 256],
        # critic_hidden_dims=[256, 256, 256],
        # activation="elu",
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="tanh",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class QuadrotorStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 750
    save_interval = 50
    experiment_name = "CTBR_Hover_Student"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="tanh",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class VisionActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the custom Asymmetric VisionActorCritic."""
    class_name: str = "VisionActorCritic"

    use_gru: bool = True
    gru_hidden_size: int = 128

    rnn_type: str = "gru"
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 1

    imu_dropout_prob: float = 0.0
    bbox_loss_weight: float = 0.05
    conf_loss_weight: float = 0.005
    ppo_bbox_grad_scale: float = 0.05

    actor_hidden_dims: list[int] = [128, 128]
    critic_hidden_dims: list[int] = [256, 256]
    activation: str = "elu"

@configclass
class CornersTCNActorCriticCfg(RslRlPpoActorCriticCfg):
    """Corners-history actor with a privileged 20-D critic."""

    class_name: str = "CornersTCNActorCritic"

    history_length: int = 32
    token_dim: int = 8
    temporal_channels: int = 128

    actor_hidden_dims: list[int] = [256, 256]
    critic_hidden_dims: list[int] = [256, 256]
    activation: str = "elu"

@configclass
class CornersAnchoredPPOCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPO"
    dagger_anchor_coef: float = 100.0

@configclass
class QuadrotorAsymmetricPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Phase-III asymmetric PPO fine-tuning.

    Actor observation:
        256-D analytic corners history.

    Critic observation:
        20-D privileged teacher state.
    """

    seed = 42
    device = "cuda:0"

    num_steps_per_env = 64
    max_iterations = 100
    save_interval = 10

    experiment_name = "CTBR_Hover_Corners_Asymmetric_PPO"
    empirical_normalization = False
    resume = False

    obs_groups = {
        "policy": ["policy"],
        "critic": ["teacher"],
    }

    policy = CornersTCNActorCriticCfg(
        init_noise_std=0.03,
        noise_std_type="scalar",
        history_length=32,
        token_dim=8,
        temporal_channels=128,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )

    algorithm = CornersAnchoredPPOCfg(
        dagger_anchor_coef=100.0,

        value_loss_coef=1.0,
        use_clipped_value_loss=True,

        clip_param=0.02,
        entropy_coef=0.0,

        num_learning_epochs=1,
        num_mini_batches=4,

        learning_rate=3.0e-4,
        schedule="fixed",

        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )