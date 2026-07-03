# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tasks.utils import import_packages

# Register agent config as entry point so hydra_task_config can find it
from .rsl_rl_ppo_cfg import QuadrotorTeacherPPORunnerCfg, QuadrotorStudentPPORunnerCfg
from .rsl_rl_distillation_cfg import QuadrotorStudentDistillationRunnerCfg
from .vision_student_teacher import VisionStudentTeacher