# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys
import os
from datetime import datetime

import torch
import gymnasium as gym

from isaaclab.app import AppLauncher
from utils import cli_args  # isort: skip

# --------------------------------------------------------------------------------------
# 1. CLI + AppLauncher
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (visual).")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000,
                    help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-QuadrotorHover-v0",
                    help="Name of the task.")
parser.add_argument("--seed", type=int, default=None,
                    help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="RL Policy training iterations.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras for this visual training script
args_cli.enable_cameras = True
args_cli.headless = True  # still headless over SSH

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------------
# 2. Imports that require the app (omni.*, isaaclab.*, envs, etc.)
# --------------------------------------------------------------------------------------

import envs  # registers Isaac-QuadrotorHover-v0
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

# PyTorch perf flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------------------
# 3. Hydra-configured main
# --------------------------------------------------------------------------------------


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent (visual version, cameras/rendering enabled)."""

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed
    agent_cfg.seed = env_cfg.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    agent_cfg.device = env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    agent_run_name = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # create Isaac environment
    # IMPORTANT: always use rgb_array here to keep rendering fully active
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array",
    )

    # wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train_visual"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training (visual script).")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()

    print("\n\nPlease run the following line to evaluate the trained model: ")
    print(
        f"python -m rl.eval_rslrl --video --task {args_cli.task} "
        f"--num_envs 100 --experiment_name {agent_cfg.experiment_name} "
        f"--load_run {agent_run_name}"
    )
    print("\n Hydra args: ")
    print(*[f"'{arg}'" for arg in hydra_args])


if __name__ == "__main__":
    main()
    simulation_app.close()