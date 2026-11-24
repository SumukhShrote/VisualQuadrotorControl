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
parser.add_argument("--imu_dropout_prob", type=float, default=0.0, help="Probability of zeroing IMU to force vision")
parser.add_argument("--bbox_loss_weight", type=float, default=0.0, help="Weight for bounding box MSE loss")
parser.add_argument("--conf_loss_weight", type=float, default=0.0, help="Weight for bounding box confidence BCE loss")
parser.add_argument("--finetune_checkpoint", type=str, default=None,
                    help="Path to a pre-trained .pt file to load weights from (bypasses RSL-RL resume logic).")
parser.add_argument("--ppo_bbox_grad_scale", type=float, default=None,
                    help="Scale of PPO gradient flowing through bbox features")
parser.add_argument("--critic_warmup_iterations", type=int, default=30,
    help="Number of PPO iterations with the corners actor frozen.",
)
parser.add_argument("--ppo_actor_lr", type=float, default=1.0e-5,
    help="Learning rate for TCN actor fine-tuning.",
)
parser.add_argument("--ppo_critic_lr", type=float, default=3.0e-4,
    help="Learning rate for privileged critic training.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Dynamically enable cameras based on the task name.
# Any task with "Student" or "Asymmetric" in the name requires cameras.
if "Student" in args_cli.task or "Asymmetric" in args_cli.task:
    print(f"[INFO] Visual task ({args_cli.task}) detected: Enabling TiledCameras.")
    args_cli.enable_cameras = True
else:
    print(f"[INFO] Blind task ({args_cli.task}) detected: Disabling cameras for high-speed training.")
    args_cli.enable_cameras = False

if args_cli.task == "Isaac-QuadrotorHoverAsymmetricPPO-v0":
    # Corners are computed analytically. RGB is only needed for videos.
    args_cli.enable_cameras = bool(args_cli.video)

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

from envs.aruco_hover.agents.vision_student_teacher import (
    VisionStudentTeacher,
    VisionActorCritic,
    VisionActorCriticRecurrent,
    CornersTCNActorCritic,
)

opr.VisionStudentTeacher = VisionStudentTeacher
opr.VisionActorCritic = VisionActorCritic
opr.VisionActorCriticRecurrent = VisionActorCriticRecurrent
opr.CornersTCNActorCritic = CornersTCNActorCritic

from rl.aux_distillation import AuxDistillation
from rl.aux_ppo import AuxPPO
from rl.corners_ppo import CornersAnchoredPPO
import rsl_rl.algorithms as rsl_rl_algorithms
rsl_rl_algorithms.Distillation = AuxDistillation   # patch the algorithms module
opr.Distillation = AuxDistillation                 # patch the runner's imported reference

rsl_rl_algorithms.PPO = AuxPPO
opr.PPO = AuxPPO

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
    
    if (
        args_cli.use_gru is not None
        and hasattr(agent_cfg.policy, "use_gru")
    ):
        agent_cfg.policy.use_gru = args_cli.use_gru

    policy_use_gru = bool(
        getattr(agent_cfg.policy, "use_gru", False)
    )

    requested_policy_class = agent_cfg.policy.class_name

    is_distillation_task = (
        "Distill" in args_cli.task
        or requested_policy_class == "VisionStudentTeacher"
    )

    is_corners_ppo = (
        requested_policy_class == "CornersTCNActorCritic"
    )

    # Only perform automatic GRU/feedforward selection for the old
    # RGB VisionActorCritic family.
    if (
        not is_distillation_task
        and not is_corners_ppo
        and requested_policy_class
        in {
            "VisionActorCritic",
            "VisionActorCriticRecurrent",
        }
    ):
        if policy_use_gru:
            agent_cfg.policy.class_name = (
                "VisionActorCriticRecurrent"
            )
            agent_cfg.policy.rnn_type = "gru"
            agent_cfg.policy.rnn_hidden_dim = (
                agent_cfg.policy.gru_hidden_size
            )

            if not hasattr(
                agent_cfg.policy,
                "rnn_num_layers",
            ):
                agent_cfg.policy.rnn_num_layers = 1
        else:
            agent_cfg.policy.class_name = (
                "VisionActorCritic"
            )

    if is_corners_ppo and policy_use_gru:
        raise ValueError(
            "CornersTCNActorCritic must not use an RSL-RL GRU."
        )
    
    if is_corners_ppo:
        env_cfg.visual_dr_enabled = False
        env_cfg.debug_vis = False
        env_cfg.enable_onboard_camera = bool(args_cli.video)

    print(
        f"[INFO] Policy selection: "
        f"task={args_cli.task}, "
        f"use_gru={policy_use_gru}, "
        f"class_name={agent_cfg.policy.class_name}"
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
    if is_corners_ppo and not args_cli.video:
        render_mode = None
    else:
        render_mode = "rgb_array"

    if args_cli.task != "Isaac-QuadrotorHoverAsymmetricPPO-v0":
        raise RuntimeError(
            "This trainer is restricted to corners-only asymmetric PPO."
        )

    # Enable the environment's existing performance-based curriculum.
    env_cfg.curriculum_enabled = True

    # Negative value selects adaptive curriculum rather than a fixed level.
    env_cfg.forced_curriculum_level = -1.0

    # Do not enable the fixed nominal environment.
    env_cfg.nominal_mode = False

    if (
        not env_cfg.curriculum_enabled
        or env_cfg.forced_curriculum_level >= 0.0
        or env_cfg.nominal_mode
    ):
        raise RuntimeError("Adaptive curriculum configuration failed")

    print(
        "[ADAPTIVE CURRICULUM]",
        "enabled=True",
        "forced_level=-1.0",
        "initial_level=0.0",
    )

    for name in (
        "curriculum_eval_window",
        "curriculum_alignment_threshold",
        "curriculum_proximity_threshold",
        "curriculum_step_size",
        "curriculum_demotion_factor",
    ):
        print(
            "[CURRICULUM PARAMETER]",
            name,
            getattr(env_cfg, name, "<not defined>"),
        )
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=render_mode,
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

    # OnPolicyRunner recognizes only algorithm.class_name == "PPO".
    # Select which concrete Python implementation that name resolves to.
    if is_corners_ppo:
        rsl_rl_algorithms.PPO = CornersAnchoredPPO
        opr.PPO = CornersAnchoredPPO

        print(
            "[INFO] PPO implementation selected: "
            "CornersAnchoredPPO"
        )
    else:
        rsl_rl_algorithms.PPO = AuxPPO
        opr.PPO = AuxPPO

        print(
            "[INFO] PPO implementation selected: AuxPPO"
        )
        

    if is_corners_ppo:
        agent_cfg.num_steps_per_env = 32
        agent_cfg.save_interval = 25

        agent_cfg.policy.init_noise_std = 0.08

        agent_cfg.algorithm.clip_param = 0.10
        agent_cfg.algorithm.num_learning_epochs = 2
        agent_cfg.algorithm.num_mini_batches = 64
        agent_cfg.algorithm.dagger_anchor_coef = 1.0

        print(
            "[LARGE-SCALE PPO CONFIG]",
            f"rollout_steps={agent_cfg.num_steps_per_env}",
            f"noise={agent_cfg.policy.init_noise_std}",
            f"clip={agent_cfg.algorithm.clip_param}",
            f"epochs={agent_cfg.algorithm.num_learning_epochs}",
            f"minibatches={agent_cfg.algorithm.num_mini_batches}",
            f"anchor={agent_cfg.algorithm.dagger_anchor_coef}",
            f"save_interval={agent_cfg.save_interval}",
        )

    cfg_dict = agent_cfg.to_dict()

    # DAgger beta-mixing params — passed through to AuxDistillation.__init__
    # beta anneals 1.0→0.0 over 300 iters: teacher-only → student-only
    if args_cli.teacher_checkpoint is not None:
        cfg_dict["algorithm"]["beta_start"] = 1.0
        cfg_dict["algorithm"]["beta_end"] = 0.0
        cfg_dict["algorithm"]["beta_anneal_iters"] = 325
    
    policy_class_name = cfg_dict.get(
        "policy",
        {},
    ).get(
        "class_name",
        "",
    )

    if policy_class_name in {
        "VisionActorCritic",
        "VisionActorCriticRecurrent",
    }:
        cfg_dict["policy"]["imu_dropout_prob"] = (
            args_cli.imu_dropout_prob
        )
        cfg_dict["policy"]["bbox_loss_weight"] = (
            args_cli.bbox_loss_weight
        )
        cfg_dict["policy"]["conf_loss_weight"] = (
            args_cli.conf_loss_weight
        )

        if args_cli.ppo_bbox_grad_scale is not None:
            cfg_dict["policy"]["ppo_bbox_grad_scale"] = (
                args_cli.ppo_bbox_grad_scale
            )
        else:
            cfg_dict["policy"]["ppo_bbox_grad_scale"] = getattr(
                agent_cfg.policy,
                "ppo_bbox_grad_scale",
                0.05,
            )

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
    
    elif (
        args_cli.finetune_checkpoint is not None
        and isinstance(
            runner.alg.policy,
            CornersTCNActorCritic,
        )
    ):
        print(
            "[INFO] Loading corners-TCN DAgger checkpoint:",
            args_cli.finetune_checkpoint,
        )

        loaded_dict = torch.load(
            args_cli.finetune_checkpoint,
            map_location=agent_cfg.device,
        )

        if "student_state_dict" not in loaded_dict:
            raise KeyError(
                "Corners DAgger checkpoint does not contain "
                "'student_state_dict'."
            )

        policy = runner.alg.policy

        expected_metadata = {
            "history_length": policy.history_length,
            "token_dim": policy.token_dim,
            "temporal_channels": policy.temporal_channels,
        }

        for key, expected_value in expected_metadata.items():
            checkpoint_value = loaded_dict.get(key, None)

            if checkpoint_value is None:
                raise KeyError(
                    f"Checkpoint is missing metadata field {key!r}."
                )

            if int(checkpoint_value) != int(expected_value):
                raise RuntimeError(
                    f"Checkpoint {key}={checkpoint_value}, "
                    f"but PPO policy expects {expected_value}."
                )

        checkpoint_student_state = loaded_dict[
            "student_state_dict"
        ]

        load_result = policy.student.load_state_dict(
            checkpoint_student_state,
            strict=True,
        )

        print(
            "[INFO] Strict corners student load result:",
            load_result,
        )

        # Verify every loaded tensor exactly.
        loaded_student_state = policy.student.state_dict()

        maximum_parameter_difference = 0.0

        for key, checkpoint_tensor in checkpoint_student_state.items():
            loaded_tensor = loaded_student_state[key]

            difference = (
                loaded_tensor
                - checkpoint_tensor.to(loaded_tensor.device)
            ).abs().max().item()

            maximum_parameter_difference = max(
                maximum_parameter_difference,
                difference,
            )

        print(
            "[CHECK] Maximum loaded-parameter difference:",
            maximum_parameter_difference,
        )

        if maximum_parameter_difference != 0.0:
            raise RuntimeError(
                "Corners PPO actor did not load the DAgger "
                "checkpoint exactly."
            )

        # Compare outputs against the original DAgger class.
        reference_policy = VisionStudentTeacher(
            num_student_obs=256,
            num_teacher_obs=20,
            num_actions=env.num_actions,
            student_hidden_dims=[256, 256],
            teacher_hidden_dims=[64, 64],
            activation="elu",
            init_noise_std=0.01,
            history_length=policy.history_length,
            token_dim=policy.token_dim,
            temporal_channels=policy.temporal_channels,
            use_gru=False,
        ).to(agent_cfg.device)

        reference_policy.student.load_state_dict(
            checkpoint_student_state,
            strict=True,
        )

        policy.eval()
        reference_policy.eval()

        with torch.no_grad():
            probe_observations = torch.randn(
                64,
                256,
                device=agent_cfg.device,
            )

            reference_actions = (
                reference_policy.act_inference(
                    probe_observations
                )
            )

            ppo_actions = policy.act_inference(
                probe_observations
            )

            maximum_output_difference = (
                reference_actions - ppo_actions
            ).abs().max().item()

        print(
            "[CHECK] DAgger/PPO actor output difference:",
            maximum_output_difference,
        )

        if maximum_output_difference > 1.0e-7:
            raise RuntimeError(
                "The PPO actor does not reproduce the offline "
                "DAgger actor."
            )
        
        if not hasattr(
                runner.alg,
                "set_dagger_reference",
            ):
                raise TypeError(
                    "The configured corners PPO algorithm does not "
                    "provide set_dagger_reference()."
                )

        runner.alg.set_dagger_reference(
                checkpoint_student_state
            )

        del reference_policy
        policy.train()

        print(
            "[INFO] Corners-TCN actor initialized exactly; "
            "critic remains randomly initialized."
        )

    elif args_cli.finetune_checkpoint is not None:
        print(f"[INFO]: Loading fine-tuning checkpoint from: {args_cli.finetune_checkpoint}")

        loaded_dict = torch.load(args_cli.finetune_checkpoint, map_location=agent_cfg.device)
        state_dict = loaded_dict.get("model_state_dict", loaded_dict)

        own_state = runner.alg.policy.state_dict()
        mapped_state_dict = {}
        policy_is_recurrent = bool(getattr(runner.alg.policy, "is_recurrent", False))

        has_student4 = any(k.startswith("student.4.") for k in state_dict.keys())

        def remap_key(key: str):
            # Preferred direct names
            if key.startswith("backbone."):
                return key
            if key.startswith("bbox_head."):
                return key
            if key.startswith("vision_student."):
                return "actor." + key[len("vision_student."):]
            if key.startswith("gru."):
                if policy_is_recurrent:
                    return "memory_a.rnn." + key[len("gru."):]
                return None

            # Fallback for ModuleList-based student.* keys
            if key.startswith("student.0."):
                return "backbone." + key[len("student.0."):]
            if key.startswith("student.1."):
                return None  # pool is not used in PPO classes
            if key.startswith("student.2."):
                return "bbox_head." + key[len("student.2."):]

            if key.startswith("student.3."):
                if has_student4:
                    # recurrent student layout: 0=backbone,1=pool,2=bbox,3=gru,4=vision_student
                    if policy_is_recurrent:
                        return "memory_a.rnn." + key[len("student.3."):]
                    return None
                else:
                    # feedforward student layout: 0=backbone,1=pool,2=bbox,3=vision_student
                    return "actor." + key[len("student.3."):]

            if key.startswith("student.4."):
                return "actor." + key[len("student.4."):]

            return None

        for key, value in state_dict.items():
            new_key = remap_key(key)
            if new_key is None:
                continue

            if new_key in own_state and own_state[new_key].shape == value.shape:
                mapped_state_dict[new_key] = value
            else:
                got_shape = tuple(value.shape) if hasattr(value, "shape") else ""
                exp_shape = tuple(own_state[new_key].shape) if new_key in own_state else "missing"
                print(f"[WARN] Skipping mismatched key: {new_key} (checkpoint {got_shape} vs model {exp_shape})")

        load_ok = runner.alg.policy.load_state_dict(mapped_state_dict, strict=False)
        print("[INFO]: Mapped and loaded Distillation weights into PPO policy!")
        print("[INFO]: load_state_dict return value:", load_ok)

        loaded_keys = set(mapped_state_dict.keys())
        model_keys = set(own_state.keys())

        important_missing = [
            k for k in sorted(model_keys - loaded_keys)
            if k.startswith("actor.")
            or k.startswith("backbone.")
            or k.startswith("bbox_head.")
            or k.startswith("memory_a.")
        ]

        print("[INFO]: Important missing keys:", important_missing)

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

    if is_corners_ppo:
        policy = runner.alg.policy

        # Read the actual Gaussian exploration parameter.
        # action_std is not available before the first rollout.
        def _policy_noise_std_mean(module):
            if hasattr(module, "std"):
                return float(
                    module.std.detach().mean().item()
                )

            if hasattr(module, "log_std"):
                return float(
                    module.log_std.detach().exp().mean().item()
                )

            raise RuntimeError(
                "Policy exposes neither std nor log_std."
            )


        # Isolated full-actor PPO pilot.
        runner.alg.dagger_anchor_coef = 1.0

        actual_std = float(_policy_noise_std_mean(policy))

        if abs(actual_std - 0.08) > 1e-5:
            raise RuntimeError(
                f"Expected PPO action std 0.08, got {actual_std}"
            )

        print(
            "[LARGE-SCALE PPO CHECK]",
            f"action_std={actual_std:.6f}",
            f"anchor={runner.alg.dagger_anchor_coef}",
        )

        print(
            "[PPO PILOT] DAgger anchor coefficient:",
            runner.alg.dagger_anchor_coef,
        )

        if not isinstance(
            policy,
            CornersTCNActorCritic,
        ):
            raise TypeError(
                "Expected CornersTCNActorCritic, got "
                f"{type(policy).__name__}."
            )

        if not isinstance(
            runner.alg,
            CornersAnchoredPPO,
        ):
            raise TypeError(
                "Expected CornersAnchoredPPO, got "
                f"{type(runner.alg).__name__}."
            )

        if runner.alg._dagger_reference_student is None:
            raise RuntimeError(
                "The frozen DAgger reference was not initialized."
            )

        if hasattr(policy, "std"):
            noise_parameters = [policy.std]
        elif hasattr(policy, "log_std"):
            noise_parameters = [policy.log_std]
        else:
            raise RuntimeError(
                "The policy has no action-noise parameter."
            )

        critic_parameters = list(
            policy.critic.parameters()
        )

        # =============================================================
        # Phase III-A: privileged critic warm-up
        # =============================================================

        # Freeze actor, critic, and noise first.
        for parameter in policy.parameters():
            parameter.requires_grad_(False)

        # Only the privileged critic is trainable.
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        actor_before_warmup = {
            key: value.detach().clone()
            for key, value
            in policy.student.state_dict().items()
        }

        noise_before_warmup = [
            parameter.detach().clone()
            for parameter in noise_parameters
        ]

        critic_before_warmup = {
            key: value.detach().clone()
            for key, value
            in policy.critic.state_dict().items()
        }

        runner.alg.optimizer = torch.optim.Adam(
            critic_parameters,
            lr=args_cli.ppo_critic_lr,
        )

        runner.alg.learning_rate = (
            args_cli.ppo_critic_lr
        )

        runner.alg.schedule = "fixed"

        # A random critic should not be clipped around its random
        # initial value prediction during warm-up.
        original_clipped_value_setting = (
            runner.alg.use_clipped_value_loss
        )

        runner.alg.use_clipped_value_loss = False

        print(
            "[PHASE III-A] Critic warm-up:",
            f"iterations={args_cli.critic_warmup_iterations}",
            f"critic_lr={args_cli.ppo_critic_lr}",
        )

        try:
            runner.learn(
                num_learning_iterations=(
                    args_cli.critic_warmup_iterations
                ),

                # Start complete episodes rather than randomizing
                # episode ages.
                init_at_random_ep_len=False,
            )
        finally:
            runner.alg.use_clipped_value_loss = (
                original_clipped_value_setting
            )

        maximum_actor_warmup_difference = 0.0

        actor_after_warmup = (
            policy.student.state_dict()
        )

        for key, before_tensor in (
            actor_before_warmup.items()
        ):
            difference = (
                actor_after_warmup[key]
                - before_tensor
            ).abs().max().item()

            maximum_actor_warmup_difference = max(
                maximum_actor_warmup_difference,
                difference,
            )

        maximum_noise_warmup_difference = max(
            (
                after.detach()
                - before
            ).abs().max().item()
            for after, before in zip(
                noise_parameters,
                noise_before_warmup,
            )
        )

        maximum_critic_warmup_difference = 0.0

        critic_after_warmup = (
            policy.critic.state_dict()
        )

        for key, before_tensor in (
            critic_before_warmup.items()
        ):
            difference = (
                critic_after_warmup[key]
                - before_tensor
            ).abs().max().item()

            maximum_critic_warmup_difference = max(
                maximum_critic_warmup_difference,
                difference,
            )

        print(
            "[CHECK] Warm-up actor parameter difference:",
            maximum_actor_warmup_difference,
        )

        print(
            "[CHECK] Warm-up noise parameter difference:",
            maximum_noise_warmup_difference,
        )

        print(
            "[CHECK] Warm-up critic parameter difference:",
            maximum_critic_warmup_difference,
        )

        if maximum_actor_warmup_difference != 0.0:
            raise RuntimeError(
                "The actor changed during critic warm-up."
            )

        if maximum_noise_warmup_difference != 0.0:
            raise RuntimeError(
                "The action-noise parameter changed during "
                "critic warm-up."
            )

        if maximum_critic_warmup_difference == 0.0:
            raise RuntimeError(
                "The critic did not change during warm-up."
            )

        # The first runner.learn() ended at warmup_iterations - 1.
        runner.current_learning_iteration = (
            args_cli.critic_warmup_iterations
        )

        # =============================================================
        # Phase III-B: one or more anchored PPO iterations
        # =============================================================

        # Freeze the complete pretrained actor.
        for parameter in policy.student.parameters():
            parameter.requires_grad_(False)

        if not isinstance(
            policy.actor[-1],
            torch.nn.Linear,
        ):
            raise TypeError(
                "Expected policy.actor[-1] to be the final "
                "Linear(256, 4) control layer."
            )

        # Initially permit only the final 256 -> 4 layer to adapt.
        trainable_actor_parameters = list(
            policy.student.parameters()
        )

        for parameter in trainable_actor_parameters:
            parameter.requires_grad_(True)

        # Keep exploration variance fixed for this pilot.
        for parameter in noise_parameters:
            parameter.requires_grad_(False)

        # Continue updating the critic.
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        trainable_actor_count = sum(
            parameter.numel()
            for parameter
            in trainable_actor_parameters
        )

        if trainable_actor_count <= 1028:
            raise RuntimeError(
                "Expected full TCN actor with more than "
                "1028 trainable parameters, got "
                f"{trainable_actor_count}."
            )

        runner.alg.optimizer = torch.optim.Adam(
            [
                {
                    "params": trainable_actor_parameters,
                    "lr": args_cli.ppo_actor_lr,
                    "name": "actor_full_tcn",
                },
                {
                    "params": critic_parameters,
                    "lr": args_cli.ppo_critic_lr,
                    "name": "critic",
                },
            ]
        )

        runner.alg.learning_rate = (
            args_cli.ppo_actor_lr
        )

        runner.alg.schedule = "fixed"

        assigned_parameter_ids = {
            id(parameter)
            for parameter in (
                trainable_actor_parameters
                + critic_parameters
            )
        }

        unassigned_trainable_parameters = [
            name
            for name, parameter
            in policy.named_parameters()
            if (
                parameter.requires_grad
                and id(parameter)
                not in assigned_parameter_ids
            )
        ]

        if unassigned_trainable_parameters:
            raise RuntimeError(
                "Unassigned trainable PPO parameters: "
                f"{unassigned_trainable_parameters}"
            )

        print(
            "[PHASE III-B] Anchored asymmetric PPO:",
            f"iterations={agent_cfg.max_iterations}",
            f"actor_lr={args_cli.ppo_actor_lr}",
            f"critic_lr={args_cli.ppo_critic_lr}",
            f"clip={runner.alg.clip_param}",
            f"anchor_coef={runner.alg.dagger_anchor_coef}",
            f"noise={_policy_noise_std_mean(policy):.6f}",
        )

        print(
            "[PHASE III-B] Trainable actor parameters:",
            trainable_actor_count,
        )

        runner.learn(
            num_learning_iterations=(
                agent_cfg.max_iterations
            ),

            # Continue from the actual post-warm-up environment
            # state rather than randomly aging episodes again.
            init_at_random_ep_len=False,
        )

    else:
        runner.learn(
            num_learning_iterations=(
                agent_cfg.max_iterations
            ),
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