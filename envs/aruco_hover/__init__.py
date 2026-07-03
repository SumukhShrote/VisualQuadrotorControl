# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Teacher Quacopter Hover Environment.
"""

import gymnasium as gym

from . import quadrotor_aruco_hover_env
from .quadrotor_aruco_hover_env import (
    ArucoHoverEnvCfg,
    ArucoHoverTeacherEnv,
    ArucoHoverStudentEnv,
    ArucoHoverStudentEnvCfg,
)
from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-QuadrotorHoverTeacher-v0",
    entry_point="envs.aruco_hover.quadrotor_aruco_hover_env:ArucoHoverTeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ArucoHoverEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-QuadrotorHoverStudent-v0",
    entry_point="envs.aruco_hover.quadrotor_aruco_hover_env:ArucoHoverStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ArucoHoverEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorStudentPPORunnerCfg,
    },
)

# Distillation student task (vision+IMU student, GT teacher)
gym.register(
    id="Isaac-QuadrotorHoverStudentDistill-v0",
    entry_point="envs.aruco_hover.quadrotor_aruco_hover_env:ArucoHoverStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ArucoHoverStudentEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_distillation_cfg.QuadrotorStudentDistillationRunnerCfg,
    },
)