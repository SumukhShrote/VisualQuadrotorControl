# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Teacher Quacopter Hover Environment.
"""

import gymnasium as gym

from . import quadrotor_aruco_hover_env
from .quadrotor_aruco_hover_env import ArucoHoverEnv
from .quadrotor_aruco_hover_env import ArucoHoverEnvCfg
from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-QuadrotorHover-v0",
    entry_point="envs.aruco_hover.quadrotor_teacher_aruco_hover_env:ArucoHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ArucoHoverEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
    },
)