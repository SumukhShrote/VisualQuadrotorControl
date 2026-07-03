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
parser.add_argument("--teacher_checkpoint",type=str,
                    default=None, help="Path to pretrained teacher .pt checkpoint for distillation. Ignored for normal PPO training.")
# Add to argparse section:
parser.add_argument("--use_gru",type=lambda x: x.lower() in ("true", "1", "yes"),default=None,
                    help="Override whether the student uses a GRU. If unset, uses cfg default.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Dynamically enable cameras only if we are training the Visual Student (Distillation)
if args_cli.teacher_checkpoint is not None:
    print("[INFO] Distillation mode detected: Enabling TiledCameras for Visual Student.")
    args_cli.enable_cameras = True
else:
    print("[INFO] PPO Teacher mode detected: Disabling cameras for high-speed training.")
    args_cli.enable_cameras = False

args_cli.headless = True  # Always run headless over SSH

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------------
# 2. Imports that require the app (omni.*, isaaclab.*, envs, etc.)
# --------------------------------------------------------------------------------------

import envs
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as opr

from envs.aruco_hover.agents.vision_student_teacher import VisionStudentTeacher
opr.VisionStudentTeacher = VisionStudentTeacher

from rl.aux_distillation import AuxDistillation
import rsl_rl.algorithms as rsl_rl_algorithms
rsl_rl_algorithms.Distillation = AuxDistillation   # patch the algorithms module
opr.Distillation = AuxDistillation                 # patch the runner's imported reference

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
    
    if args_cli.use_gru is not None:
        agent_cfg.policy.use_gru = args_cli.use_gru

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

    cfg_dict = agent_cfg.to_dict()

    # DAgger beta-mixing params — passed through to AuxDistillation.__init__
    # beta anneals 1.0→0.0 over 300 iters: teacher-only → student-only
    if args_cli.teacher_checkpoint is not None:
        cfg_dict["algorithm"]["beta_start"] = 1.0
        cfg_dict["algorithm"]["beta_end"] = 0.0
        cfg_dict["algorithm"]["beta_anneal_iters"] = 300

    runner = OnPolicyRunner(
        env,
        cfg_dict,
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # save resume path before creating a new log_dir
    if args_cli.teacher_checkpoint is not None:
        # Distillation: load pretrained teacher weights
        print(f"[INFO]: Loading teacher checkpoint from: {args_cli.teacher_checkpoint}")
        runner.load(args_cli.teacher_checkpoint)
    elif agent_cfg.resume:
        # Normal PPO resume
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if args_cli.video:
        import cv2, numpy as np
        from pathlib import Path

        drone_video_dir = Path(log_dir) / "videos" / "drone_pov"
        drone_video_dir.mkdir(parents=True, exist_ok=True)

        isaac_env = runner.env.env.unwrapped

        writer = [None]
        frame_count = [0]

        original_step = isaac_env.step

        def step_with_capture(action):
            result = original_step(action)
            if frame_count[0] >= args_cli.video_length:
                return result

            try:
                rgb = isaac_env._camera.data.output.get("rgb", None)
            except Exception:
                return result

            if rgb is None:
                return result

            # Debug print — only on first frame
            if frame_count[0] == 0:
                print(f"[CAM] dtype={rgb.dtype} min={rgb.min():.4f} max={rgb.max():.4f} shape={rgb.shape}")
            
            if frame_count[0] == 0:
                import os
                from PIL import Image
                save_dir = os.path.expanduser("~/AerialCatching/videos")
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(rgb[0].cpu().numpy()).save(os.path.join(save_dir, "frame0.png"))
                # Image.fromarray(rgb[32].cpu().numpy()).save(os.path.join(save_dir, "frame32.png"))
                print(f"[CAM] Saved frames to {save_dir}/frame0.png and frame32.png")

            raw = rgb[0].cpu().numpy()
            if raw.dtype != np.uint8:
                raw = (raw * 255.0).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

            if writer[0] is None:
                h, w = frame_bgr.shape[:2]
                fps = max(1, int(1.0 / (isaac_env.cfg.sim.dt * isaac_env.cfg.decimation)))
                out_path = str(drone_video_dir / "drone_pov.mp4")
                writer[0] = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )

            writer[0].write(frame_bgr)
            frame_count[0] += 1
            if frame_count[0] >= args_cli.video_length:
                writer[0].release()
                writer[0] = None
                print(f"[VIDEO] Drone POV saved to {drone_video_dir}/drone_pov.mp4")

            return result

        isaac_env.step = step_with_capture

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    # flush (replace the existing flush block)
    if args_cli.video and writer[0] is not None:
        writer[0].release()
        print(f"[VIDEO] Drone POV flushed.")

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