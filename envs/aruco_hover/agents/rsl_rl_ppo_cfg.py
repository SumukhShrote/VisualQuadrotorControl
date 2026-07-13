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
class QuadrotorAsymmetricPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for fine-tuning the distilled student via Asymmetric PPO."""
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "CTBR_Hover_Asymmetric_PPO"
    empirical_normalization = False
    
    policy = VisionActorCriticCfg(
        # Low initial noise because the policy is already somewhat competent
        init_noise_std=0.2,
        use_gru=True,   # or False 
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # Tighter clipping to prevent destroying the distilled weights
        clip_param=0.1, 
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Lower learning rate for fine-tuning
        learning_rate=1.0e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )