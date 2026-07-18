# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect persistent DAgger samples for the corners-TCN student."""

import argparse
import glob
import hashlib
import os
import sys

from isaaclab.app import AppLauncher

from utils import cli_args  # isort: skip


# =============================================================================
# CLI AND ISAAC SIM LAUNCH
# =============================================================================

parser = argparse.ArgumentParser(
    description="Collect persistent DAgger samples."
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=256,
    help="Number of parallel environments.",
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-QuadrotorHoverStudentDistill-v0",
)

parser.add_argument(
    "--seed",
    type=int,
    default=0,
)

parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    required=True,
    help="Privileged teacher checkpoint.",
)

parser.add_argument(
    "--student_checkpoint",
    type=str,
    default=None,
    help="Offline DAgger student checkpoint from train_dagger.py.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
)

parser.add_argument(
    "--round_index",
    type=int,
    required=True,
)

parser.add_argument(
    "--curriculum_level",
    type=float,
    default=None,
    help=(
        "Freeze reset difficulty at a value in [0, 1]. "
        "For example: 0.0, 0.25, 0.5, 0.75, or 1.0."
    ),
)

parser.add_argument(
    "--teacher_probability",
    type=float,
    required=True,
)

parser.add_argument(
    "--controller_hold_steps",
    type=int,
    default=25,
    help=(
        "Number of consecutive policy steps for which each "
        "environment keeps its sampled teacher/student controller."
    ),
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=500_000,
)

parser.add_argument(
    "--shard_size",
    type=int,
    default=100_000,
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Analytic 256-D corner histories do not require rendered RGB cameras.
args_cli.enable_cameras = False
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# IMPORTS REQUIRING ISAAC SIM
# =============================================================================

import gymnasium as gym
import torch

import envs  # noqa: F401
import isaaclab_tasks  # noqa: F401

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as opr
import rsl_rl.algorithms as rsl_rl_algorithms

from envs.aruco_hover.agents.vision_student_teacher import (
    VisionStudentTeacher,
)

from rl.aux_distillation import AuxDistillation


# Make the custom classes visible to RSL-RL's class lookup.
opr.VisionStudentTeacher = VisionStudentTeacher

rsl_rl_algorithms.Distillation = AuxDistillation
opr.Distillation = AuxDistillation


def _sha256_file(path: str | None) -> str | None:
    """Return the SHA-256 digest of a checkpoint, or None."""

    if path is None:
        return None

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# MAIN
# =============================================================================

@hydra_task_config(
    args_cli.task,
    "rsl_rl_cfg_entry_point",
)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Collect observations and privileged teacher labels."""

    # -------------------------------------------------------------------------
    # Validate arguments
    # -------------------------------------------------------------------------
    if not 0.0 <= args_cli.teacher_probability <= 1.0:
        raise ValueError(
            "--teacher_probability must lie in [0, 1]."
        )

    if args_cli.num_samples <= 0:
        raise ValueError(
            "--num_samples must be positive."
        )

    if args_cli.shard_size <= 0:
        raise ValueError(
            "--shard_size must be positive."
        )
    
    if args_cli.controller_hold_steps <= 0:
        raise ValueError(
            "--controller_hold_steps must be positive."
        )

    if (
        args_cli.teacher_probability < 1.0
        and args_cli.student_checkpoint is None
    ):
        raise ValueError(
            "A student checkpoint is required whenever "
            "--teacher_probability is less than 1.0."
        )

    # -------------------------------------------------------------------------
    # Configure environment and device
    # -------------------------------------------------------------------------
    agent_cfg = cli_args.update_rsl_rl_cfg(
        agent_cfg,
        args_cli,
    )

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    # Fresh corners-only collection uses analytic projection only.
    if hasattr(env_cfg, "enable_onboard_camera"):
        env_cfg.enable_onboard_camera = False

    if hasattr(env_cfg, "visual_dr_enabled"):
        env_cfg.visual_dr_enabled = False

    if hasattr(env_cfg, "debug_vis"):
        env_cfg.debug_vis = False

    if args_cli.curriculum_level is not None:
        if not 0.0 <= args_cli.curriculum_level <= 1.0:
            raise ValueError(
                "--curriculum_level must lie in [0, 1]."
            )

        env_cfg.forced_curriculum_level = float(
            args_cli.curriculum_level
        )

        print(
            "[DAgger] Forced curriculum level:",
            env_cfg.forced_curriculum_level,
        )

        pos_start = torch.tensor(
            env_cfg.init_pos_noise_start,
            dtype=torch.float32,
        )

        pos_end = torch.tensor(
            env_cfg.init_pos_noise_end,
            dtype=torch.float32,
        )

        level = env_cfg.forced_curriculum_level

        active_pos_noise = (
            pos_start
            + level * (pos_end - pos_start)
        )

        active_yaw_noise = (
            env_cfg.init_yaw_noise_start
            + level
            * (
                env_cfg.init_yaw_noise_end
                - env_cfg.init_yaw_noise_start
            )
        )

        print(
            "[DAgger] Active reset ranges:",
            f"position=±{active_pos_noise.tolist()} m,",
            f"yaw=±{active_yaw_noise:.4f} rad",
        )

    env_cfg.sim.device = (
        args_cli.device
        if args_cli.device is not None
        else env_cfg.sim.device
    )

    agent_cfg.device = env_cfg.sim.device

    # The collector must construct the corners-TCN student.
    agent_cfg.policy.class_name = "VisionStudentTeacher"
    agent_cfg.policy.use_gru = False

    # -------------------------------------------------------------------------
    # Protect existing data from accidental overwrite
    # -------------------------------------------------------------------------
    os.makedirs(
        args_cli.output_dir,
        exist_ok=True,
    )

    round_pattern = os.path.join(
        args_cli.output_dir,
        f"round_{args_cli.round_index:02d}_shard_*.pt",
    )

    existing_round_files = glob.glob(
        round_pattern
    )

    if existing_round_files:
        raise FileExistsError(
            f"Round {args_cli.round_index} already contains "
            f"{len(existing_round_files)} shard files in "
            f"{args_cli.output_dir}. Remove those files or "
            f"choose another --round_index."
        )

    # -------------------------------------------------------------------------
    # Create environment and policy
    # -------------------------------------------------------------------------
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None,
    )

    env = RslRlVecEnvWrapper(
        env
    )

    cfg_dict = agent_cfg.to_dict()

    runner = OnPolicyRunner(
        env,
        cfg_dict,
        log_dir=None,
        device=agent_cfg.device,
    )

    print(
        "[INFO] Loading privileged teacher checkpoint:",
        args_cli.teacher_checkpoint,
    )

    # This is the same teacher-loading route used by your existing
    # distillation run.
    runner.load(
        args_cli.teacher_checkpoint
    )

    policy = runner.alg.policy

    # -------------------------------------------------------------------------
    # Load the latest offline student for DAgger rounds after round zero
    # -------------------------------------------------------------------------
    if args_cli.student_checkpoint is not None:
        print(
            "[INFO] Loading offline student checkpoint:",
            args_cli.student_checkpoint,
        )

        student_checkpoint = torch.load(
            args_cli.student_checkpoint,
            map_location=agent_cfg.device,
        )

        if "student_state_dict" not in student_checkpoint:
            raise KeyError(
                "Student checkpoint does not contain "
                "'student_state_dict'."
            )

        policy.student.load_state_dict(
            student_checkpoint["student_state_dict"],
            strict=True,
        )

        print("[INFO] Offline student loaded.")

    policy.eval()

    teacher_checkpoint_sha256 = _sha256_file(
        args_cli.teacher_checkpoint
    )

    student_checkpoint_sha256 = _sha256_file(
        args_cli.student_checkpoint
    )

    # -------------------------------------------------------------------------
    # Obtain initial observation groups
    # -------------------------------------------------------------------------
    _, extras = env.get_observations()

    obs_dict = extras.get(
        "observations",
        None,
    )

    if obs_dict is None:
        raise RuntimeError(
            "env.get_observations() did not return "
            "extras['observations']."
        )

    if "policy" not in obs_dict:
        raise KeyError(
            "Observation dictionary has no 'policy' group."
        )

    if "teacher" not in obs_dict:
        raise KeyError(
            "Observation dictionary has no 'teacher' group."
        )

    if obs_dict["policy"].shape[-1] != 256:
        raise RuntimeError(
            f"Expected policy observations with shape "
            f"(N, 256), got {obs_dict['policy'].shape}."
        )

    if obs_dict["teacher"].shape[-1] != 20:
        raise RuntimeError(
            f"Expected teacher observations with shape "
            f"(N, 20), got {obs_dict['teacher'].shape}."
        )

    print(
        "[DAgger] Policy observation shape:",
        obs_dict["policy"].shape,
    )

    print(
        "[DAgger] Teacher observation shape:",
        obs_dict["teacher"].shape,
    )

    # -------------------------------------------------------------------------
    # Collection buffers
    # -------------------------------------------------------------------------
    observation_buffer: list[torch.Tensor] = []
    action_buffer: list[torch.Tensor] = []

    samples_collected = 0
    buffered_samples = 0
    shard_index = 0

    candidate_samples_seen = 0
    fully_missing_history_count = 0
    partially_missing_history_count = 0

    teacher_mask = None
    controller_step = 0

    print(
        "[DAgger] Starting collection:",
        f"round={args_cli.round_index},",
        f"teacher_probability={args_cli.teacher_probability},",
        f"target_samples={args_cli.num_samples}",
    )

    # -------------------------------------------------------------------------
    # DAgger collection loop
    # -------------------------------------------------------------------------
    with torch.no_grad():
        while samples_collected < args_cli.num_samples:
            policy_obs = obs_dict["policy"]
            teacher_obs = obs_dict["teacher"]

            # The environment clips actions to [-1, 1].
            # Store and execute the same bounded action.
            teacher_action = torch.clamp(
                policy.evaluate(teacher_obs),
                min=-1.0,
                max=1.0,
            )

            student_action = torch.clamp(
                policy.act_inference(policy_obs),
                min=-1.0,
                max=1.0,
            )

            if teacher_action.shape[-1] != 4:
                raise RuntimeError(
                    f"Expected four teacher actions, got "
                    f"{teacher_action.shape}."
                )

            if student_action.shape[-1] != 4:
                raise RuntimeError(
                    f"Expected four student actions, got "
                    f"{student_action.shape}."
                )

            # Keep the selected controller for a contiguous temporal segment.
            # This allows the student to visit its own state distribution during
            # later DAgger rounds instead of being corrected every single step.
            if (
                teacher_mask is None
                or controller_step
                % args_cli.controller_hold_steps
                == 0
            ):
                teacher_mask = (
                    torch.rand(
                        policy_obs.shape[0],
                        1,
                        device=policy_obs.device,
                    )
                    < args_cli.teacher_probability
                )

            controller_step += 1

            executed_action = torch.where(
                teacher_mask,
                teacher_action,
                student_action,
            )

            if not torch.isfinite(teacher_action).all():
                raise RuntimeError(
                    "Teacher produced NaN or Inf actions."
                )

            if not torch.isfinite(student_action).all():
                raise RuntimeError(
                    "Student produced NaN or Inf actions."
                )

            if teacher_action.abs().max() > 1.00001:
                raise RuntimeError(
                    f"Teacher action was not clipped correctly: "
                    f"{teacher_action.abs().max().item()}"
                )

            # --------------------------------------------------------------
            # Retain every episode-safe 32-frame history.
            #
            # The environment resets its temporal buffer by repeating the
            # first post-reset corner observation over all 32 timestamps.
            # Therefore these samples do not cross episode boundaries.
            #
            # Fully missing histories remain valid training examples.
            # --------------------------------------------------------------
            corner_history = policy_obs.reshape(
                policy_obs.shape[0],
                32,
                4,
                2,
            )

            valid_corner = (
                (corner_history[..., 0] >= 0.0)
                & (corner_history[..., 0] <= 1.0)
                & (corner_history[..., 1] >= 0.0)
                & (corner_history[..., 1] <= 1.0)
            )

            history_has_marker = (
                valid_corner
                .any(dim=2)
                .any(dim=1)
            )

            missing_coordinate = policy_obs.eq(-1.0)

            fully_missing_history = ~history_has_marker

            partially_missing_history = (
                missing_coordinate.any(dim=1)
                & history_has_marker
            )

            candidate_samples_seen += int(
                policy_obs.shape[0]
            )

            fully_missing_history_count += int(
                fully_missing_history.sum().item()
            )

            partially_missing_history_count += int(
                partially_missing_history.sum().item()
            )

            remaining = (
                args_cli.num_samples
                - samples_collected
            )

            take_count = min(
                remaining,
                int(policy_obs.shape[0]),
            )

            if take_count > 0:
                selected_observations = policy_obs[
                    :take_count
                ]

                selected_teacher_actions = teacher_action[
                    :take_count
                ]

                observation_buffer.append(
                    selected_observations
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )

                # The teacher labels every visited state, regardless of which
                # controller generated the transition.
                action_buffer.append(
                    selected_teacher_actions
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )

                samples_collected += take_count
                buffered_samples += take_count

            should_save = (
                buffered_samples > 0
                and (
                    buffered_samples
                    >= args_cli.shard_size
                    or samples_collected
                    >= args_cli.num_samples
                )
            )

            if should_save:
                observations_to_save = torch.cat(
                    observation_buffer,
                    dim=0,
                )

                actions_to_save = torch.cat(
                    action_buffer,
                    dim=0,
                )

                shard_path = os.path.join(
                    args_cli.output_dir,
                    (
                        f"round_{args_cli.round_index:02d}"
                        f"_shard_{shard_index:04d}.pt"
                    ),
                )

                torch.save(
                    {
                        "observations": observations_to_save,
                        "teacher_actions": actions_to_save,

                        "schema_version": "corners_dagger_v1",
                        "task": str(args_cli.task),
                        "seed": int(args_cli.seed),

                        "round_index": int(
                            args_cli.round_index
                        ),

                        "shard_index": int(
                            shard_index
                        ),

                        "sample_count": int(
                            observations_to_save.shape[0]
                        ),

                        "teacher_probability": float(
                            args_cli.teacher_probability
                        ),

                        "student_probability": float(
                            1.0
                            - args_cli.teacher_probability
                        ),

                        "curriculum_level":
                            args_cli.curriculum_level,

                        "init_pos_noise_start":
                            list(env_cfg.init_pos_noise_start),

                        "init_pos_noise_end":
                            list(env_cfg.init_pos_noise_end),

                        "init_yaw_noise_start":
                            float(env_cfg.init_yaw_noise_start),

                        "init_yaw_noise_end":
                            float(env_cfg.init_yaw_noise_end),

                        "history_length": 32,
                        "token_dim": 8,

                        "controller_hold_steps": int(
                            args_cli.controller_hold_steps
                        ),

                        "observation_dtype": "float32",
                        "action_dtype": "float32",
                        "action_range": [-1.0, 1.0],

                        "all_histories_retained": True,

                        "teacher_checkpoint": os.path.abspath(
                            args_cli.teacher_checkpoint
                        ),

                        "teacher_checkpoint_sha256":
                            teacher_checkpoint_sha256,

                        "student_checkpoint": (
                            os.path.abspath(
                                args_cli.student_checkpoint
                            )
                            if args_cli.student_checkpoint is not None
                            else None
                        ),

                        "student_checkpoint_sha256":
                            student_checkpoint_sha256,

                        "candidate_samples_seen_cumulative":
                            candidate_samples_seen,

                        "fully_missing_history_count_cumulative":
                            fully_missing_history_count,

                        "partially_missing_history_count_cumulative":
                            partially_missing_history_count,
                    },
                    shard_path,
                )

                print(
                    f"[DAgger] Saved {shard_path}: "
                    f"{observations_to_save.shape[0]} samples; "
                    f"total={samples_collected}/"
                    f"{args_cli.num_samples}"
                )

                observation_buffer.clear()
                action_buffer.clear()

                buffered_samples = 0
                shard_index += 1

            if samples_collected >= args_cli.num_samples:
                break
                
            # Step the simulator with the mixed behavior policy.
            _, _, _, extras = env.step(
                executed_action
            )

            obs_dict = extras.get(
                "observations",
                None,
            )

            if obs_dict is None:
                raise RuntimeError(
                    "env.step() did not return "
                    "extras['observations']."
                )

    env.close()

    fully_missing_percentage = (
        100.0
        * fully_missing_history_count
        / max(candidate_samples_seen, 1)
    )

    partially_missing_percentage = (
        100.0
        * partially_missing_history_count
        / max(candidate_samples_seen, 1)
    )

    print()
    print("=" * 72)
    print("DAGGER COLLECTION SUMMARY")
    print("=" * 72)

    print(
        f"Accepted samples:            "
        f"{samples_collected}"
    )

    print(
        f"Candidate samples seen:      "
        f"{candidate_samples_seen}"
    )

    print(
        f"Fully missing histories:     "
        f"{fully_missing_history_count} "
        f"({fully_missing_percentage:.3f}%)"
    )

    print(
        f"Partially missing histories: "
        f"{partially_missing_history_count} "
        f"({partially_missing_percentage:.3f}%)"
    )

    print(
        f"Shards:                      "
        f"{shard_index}"
    )

    print(
        f"Curriculum level:            "
        f"{args_cli.curriculum_level}"
    )

    print(
        f"Teacher probability:         "
        f"{args_cli.teacher_probability}"
    )

    print(
        f"Student probability:         "
        f"{1.0 - args_cli.teacher_probability}"
    )

    print(
        f"Controller hold steps:       "
        f"{args_cli.controller_hold_steps}"
    )

    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()