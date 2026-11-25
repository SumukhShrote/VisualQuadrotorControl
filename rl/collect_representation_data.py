# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect instrumented DAgger samples for the corners-TCN student.

This v2 collector preserves per-row controller, episode, outcome,
visibility, reward, and teacher-student disagreement provenance.
"""

import argparse
import glob
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
    help="Offline DAgger student checkpoint from train_tcn_dagger.py.",
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
    "--mixing_mode",
    type=str,
    choices=("segment", "episode", "episode_balanced"),
    default="episode",
    help=(
        "Controller-assignment mode. 'segment' reproduces the existing "
        "global hold-based switching behavior. 'episode' samples one "
        "controller per environment at episode reset and keeps it for "
        "the complete episode."
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


parser.add_argument(
    "--torchscript_policy",
    type=str,
    required=True,
    help="Frozen corners-to-actions TorchScript policy.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# The corners policy uses analytic projection and does not consume RGB.
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
        and ((args_cli.student_checkpoint is None and not args_cli.torchscript_policy) and not args_cli.torchscript_policy)
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

    # Corners are projected analytically. Rendering and visual randomization
    # cannot affect the 256-D student observation and only reduce throughput.
    env_cfg.enable_onboard_camera = False
    env_cfg.visual_dr_enabled = False
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
    env_cfg.eval_mode = True
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array",
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

    transformer_policy = torch.jit.load(
        args_cli.torchscript_policy,
        map_location="cpu",
    ).eval()

    transformer_device = None

    print(
        "[TRANSFORMER] Loaded:",
        args_cli.torchscript_policy,
        flush=True,
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
    # Collection buffers and per-environment provenance state
    # -------------------------------------------------------------------------
    row_buffers: dict[str, list[torch.Tensor]] = {
        "observations": [],
        "teacher_observations": [],
        "teacher_actions": [],
        "student_actions": [],
        "executed_actions": [],
        "controller_is_teacher": [],
        "environment_ids": [],
        "episode_ids": [],
        "episode_steps": [],
        "collection_steps": [],
        "rewards": [],
        "dones": [],
        "terminated": [],
        "timeouts": [],
        "termination_reason_codes": [],
        "too_far": [],
        "too_tilted": [],
        "too_low": [],
        "termination_bitmasks": [],
        "newest_visible_corner_count": [],
        "visible_corner_count_over_history": [],
        "teacher_student_l2": [],
        "teacher_student_linf": [],
    }

    samples_collected = 0
    buffered_samples = 0
    shard_index = 0

    candidate_samples_seen = 0
    rejected_no_marker_count = 0

    accepted_teacher_count = 0
    accepted_student_count = 0
    accepted_done_count = 0
    accepted_timeout_count = 0
    accepted_termination_count = 0

    target_teacher_rows = int(
        round(
            args_cli.num_samples
            * args_cli.teacher_probability
        )
    )

    target_student_rows = (
        args_cli.num_samples
        - target_teacher_rows
    )

    num_envs = int(obs_dict["policy"].shape[0])
    provenance_device = obs_dict["policy"].device

    environment_ids = torch.arange(
        num_envs,
        dtype=torch.int64,
        device=provenance_device,
    )

    episode_ids = torch.zeros(
        num_envs,
        dtype=torch.int64,
        device=provenance_device,
    )

    episode_steps = torch.zeros(
        num_envs,
        dtype=torch.int64,
        device=provenance_device,
    )

    teacher_mask = None
    controller_step = 0
    collection_step = 0

    print(
        "[DAgger] Starting collection:",
        f"round={args_cli.round_index},",
        f"teacher_probability={args_cli.teacher_probability},",
        f"mixing_mode={args_cli.mixing_mode},",
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
            teacher_action = torch.clamp(
                policy.evaluate(teacher_obs),
                min=-1.0,
                max=1.0,
            )

            # The PPO teacher and Transformer receive observations
            # from the same pre-step simulation state.
            if transformer_device != policy_obs.device:
                transformer_policy = transformer_policy.to(
                    policy_obs.device
                )
                transformer_device = policy_obs.device

            assert policy_obs.shape[-1] == 256

            student_action = torch.clamp(
                transformer_policy(policy_obs),
                min=-1.0,
                max=1.0,
            )

            if controller_step < 5:
                print(
                    "[ACTION PAIR]",
                    "step=", controller_step,
                    "teacher=",
                    teacher_action[0].detach().cpu().tolist(),
                    "transformer=",
                    student_action[0].detach().cpu().tolist(),
                    flush=True,
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
            # Controller assignment
            #
            # segment:
            #   Reproduces the v1 global hold-based switching behavior.
            #
            # episode:
            #   Samples once initially and again only for environments that
            #   finished an episode on the previous step.
            # --------------------------------------------------------------
            should_sample_all = teacher_mask is None

            if (
                args_cli.mixing_mode == "segment"
                and teacher_mask is not None
                and controller_step
                % args_cli.controller_hold_steps
                == 0
            ):
                should_sample_all = True

            if should_sample_all:
                teacher_mask = (
                    torch.rand(
                        num_envs,
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

            # --------------------------------------------------------------
            # Corner visibility and disagreement at the pre-step state
            # --------------------------------------------------------------
            corner_history = policy_obs.reshape(
                num_envs,
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

            any_corner_visible_per_frame = (
                valid_corner.any(dim=2)
            )

            history_has_marker = (
                any_corner_visible_per_frame.any(dim=1)
            )

            newest_visible_corner_count = (
                valid_corner[:, -1, :]
                .sum(dim=1)
                .to(torch.int16)
            )

            visible_corner_count_over_history = (
                valid_corner
                .sum(dim=(1, 2))
                .to(torch.int16)
            )

            action_difference = (
                teacher_action - student_action
            )

            teacher_student_l2 = (
                torch.linalg.vector_norm(
                    action_difference,
                    ord=2,
                    dim=-1,
                )
            )

            teacher_student_linf = (
                action_difference
                .abs()
                .amax(dim=-1)
            )

            candidate_samples_seen += num_envs

            rejected_no_marker_count += int(
                (~history_has_marker).sum().item()
            )

            candidate_valid_indices = torch.nonzero(
                history_has_marker,
                as_tuple=False,
            ).flatten()

            remaining = (
                args_cli.num_samples
                - samples_collected
            )

            if args_cli.mixing_mode == "episode_balanced":
                candidate_is_teacher = teacher_mask[
                    candidate_valid_indices,
                    0,
                ]

                teacher_candidates = (
                    candidate_valid_indices[
                        candidate_is_teacher
                    ]
                )

                student_candidates = (
                    candidate_valid_indices[
                        ~candidate_is_teacher
                    ]
                )

                teacher_rows_remaining = max(
                    target_teacher_rows
                    - accepted_teacher_count,
                    0,
                )

                student_rows_remaining = max(
                    target_student_rows
                    - accepted_student_count,
                    0,
                )

                selected_teacher_indices = (
                    teacher_candidates[
                        :teacher_rows_remaining
                    ]
                )

                selected_student_indices = (
                    student_candidates[
                        :student_rows_remaining
                    ]
                )

                valid_indices = torch.cat(
                    (
                        selected_teacher_indices,
                        selected_student_indices,
                    ),
                    dim=0,
                )

                if valid_indices.numel() > 0:
                    valid_indices = torch.sort(
                        valid_indices
                    ).values

            else:
                valid_indices = (
                    candidate_valid_indices[
                        :remaining
                    ]
                )

            selected = None

            if (
                remaining > 0
                and valid_indices.numel() > 0
            ):
                valid_indices = valid_indices[
                    :remaining
                ]

                selected_count = int(
                    valid_indices.numel()
                )

                selected = {
                    "observations": (
                        policy_obs[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "teacher_observations": (
                        teacher_obs[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "teacher_actions": (
                        teacher_action[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "student_actions": (
                        student_action[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "executed_actions": (
                        executed_action[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "controller_is_teacher": (
                        teacher_mask[
                            valid_indices,
                            0,
                        ]
                        .detach()
                        .clone()
                    ),
                    "environment_ids": (
                        environment_ids[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "episode_ids": (
                        episode_ids[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "episode_steps": (
                        episode_steps[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "collection_steps": torch.full(
                        (selected_count,),
                        collection_step,
                        dtype=torch.int64,
                        device=policy_obs.device,
                    ),
                    "newest_visible_corner_count": (
                        newest_visible_corner_count[
                            valid_indices
                        ]
                        .detach()
                        .clone()
                    ),
                    "visible_corner_count_over_history": (
                        visible_corner_count_over_history[
                            valid_indices
                        ]
                        .detach()
                        .clone()
                    ),
                    "teacher_student_l2": (
                        teacher_student_l2[valid_indices]
                        .detach()
                        .clone()
                    ),
                    "teacher_student_linf": (
                        teacher_student_linf[
                            valid_indices
                        ]
                        .detach()
                        .clone()
                    ),
                    "valid_indices": valid_indices,
                }

            # --------------------------------------------------------------
            # Execute the mixed action, then attach the post-step outcome to
            # the corresponding pre-step observation and actions.
            # --------------------------------------------------------------
            _, step_rewards, dones, extras = env.step(
                executed_action
            )

            dones = (
                dones
                .reshape(-1)
                .to(dtype=torch.bool)
            )

            step_rewards = (
                step_rewards
                .reshape(-1)
                .to(dtype=torch.float32)
            )

            raw_timeouts = extras.get(
                "time_outs",
                None,
            )

            provenance = extras.get(
                "termination_provenance",
                None,
            )

            if provenance is None:
                raise RuntimeError(
                    "env.step() did not return exact "
                    "termination_provenance."
                )

            too_far = (
                provenance["too_far"]
                .reshape(-1)
                .to(
                    device=dones.device,
                    dtype=torch.bool,
                )
            )

            too_tilted = (
                provenance["too_tilted"]
                .reshape(-1)
                .to(
                    device=dones.device,
                    dtype=torch.bool,
                )
            )

            too_low = (
                provenance["too_low"]
                .reshape(-1)
                .to(
                    device=dones.device,
                    dtype=torch.bool,
                )
            )

            provenance_timeouts = (
                provenance["timeout"]
                .reshape(-1)
                .to(
                    device=dones.device,
                    dtype=torch.bool,
                )
            )

            termination_bitmasks = (
                provenance["bitmask"]
                .reshape(-1)
                .to(
                    device=dones.device,
                    dtype=torch.int16,
                )
            )

            if raw_timeouts is None:
                timeouts = provenance_timeouts
            else:
                wrapper_timeouts = (
                    raw_timeouts
                    .reshape(-1)
                    .to(
                        device=dones.device,
                        dtype=torch.bool,
                    )
                )

                if not torch.equal(
                    wrapper_timeouts,
                    provenance_timeouts,
                ):
                    raise RuntimeError(
                        "Wrapper timeout flags disagree with "
                        "environment termination provenance."
                    )

                timeouts = wrapper_timeouts

            terminated = (
                too_far
                | too_tilted
                | too_low
            )

            expected_dones = (
                terminated
                | timeouts
            )

            if not torch.equal(
                dones,
                expected_dones,
            ):
                raise RuntimeError(
                    "Done flags disagree with exact environment "
                    "termination provenance."
                )

            expected_bitmasks = (
                too_far.to(torch.int16)
                | (too_tilted.to(torch.int16) << 1)
                | (too_low.to(torch.int16) << 2)
                | (timeouts.to(torch.int16) << 3)
            )

            if not torch.equal(
                termination_bitmasks,
                expected_bitmasks,
            ):
                raise RuntimeError(
                    "Termination bitmask disagrees with "
                    "individual cause flags."
                )

            termination_reason_codes = torch.zeros(
                num_envs,
                dtype=torch.int8,
                device=dones.device,
            )

            termination_reason_codes[timeouts] = 1
            termination_reason_codes[terminated] = 2

            if selected is not None:
                valid_indices = selected.pop(
                    "valid_indices"
                )

                selected_rewards = (
                    step_rewards[valid_indices]
                    .detach()
                    .clone()
                )

                selected_dones = (
                    dones[valid_indices]
                    .detach()
                    .clone()
                )

                selected_terminated = (
                    terminated[valid_indices]
                    .detach()
                    .clone()
                )

                selected_timeouts = (
                    timeouts[valid_indices]
                    .detach()
                    .clone()
                )

                selected_reasons = (
                    termination_reason_codes[
                        valid_indices
                    ]
                    .detach()
                    .clone()
                )

                selected_too_far = (
                    too_far[valid_indices]
                    .detach()
                    .clone()
                )

                selected_too_tilted = (
                    too_tilted[valid_indices]
                    .detach()
                    .clone()
                )

                selected_too_low = (
                    too_low[valid_indices]
                    .detach()
                    .clone()
                )

                selected_bitmasks = (
                    termination_bitmasks[
                        valid_indices
                    ]
                    .detach()
                    .clone()
                )

                selected["rewards"] = selected_rewards
                selected["dones"] = selected_dones
                selected["terminated"] = selected_terminated
                selected["timeouts"] = selected_timeouts
                selected[
                    "termination_reason_codes"
                ] = selected_reasons

                selected["too_far"] = selected_too_far
                selected["too_tilted"] = selected_too_tilted
                selected["too_low"] = selected_too_low
                selected[
                    "termination_bitmasks"
                ] = selected_bitmasks

                for key, tensor in selected.items():
                    if key in {
                        "observations",
                        "teacher_observations",
                        "teacher_actions",
                        "student_actions",
                        "executed_actions",
                        "rewards",
                        "teacher_student_l2",
                        "teacher_student_linf",
                    }:
                        tensor = tensor.to(
                            dtype=torch.float32
                        )

                    row_buffers[key].append(
                        tensor.detach().cpu()
                    )

                accepted_count = int(
                    selected["observations"].shape[0]
                )

                selected_teacher_mask = selected[
                    "controller_is_teacher"
                ]

                accepted_teacher_count += int(
                    selected_teacher_mask.sum().item()
                )

                accepted_student_count += int(
                    (
                        ~selected_teacher_mask
                    ).sum().item()
                )

                accepted_done_count += int(
                    selected_dones.sum().item()
                )

                accepted_timeout_count += int(
                    selected_timeouts.sum().item()
                )

                accepted_termination_count += int(
                    selected_terminated.sum().item()
                )

                samples_collected += accepted_count
                buffered_samples += accepted_count

            # --------------------------------------------------------------
            # Advance per-environment episode provenance.
            # The action just executed belongs to the old episode ID/step.
            # --------------------------------------------------------------
            episode_steps += 1

            done_indices = torch.nonzero(
                dones,
                as_tuple=False,
            ).flatten()

            if done_indices.numel() > 0:
                episode_ids[done_indices] += 1
                episode_steps[done_indices] = 0

                if args_cli.mixing_mode in (
                    "episode",
                    "episode_balanced",
                ):
                    teacher_mask[done_indices] = (
                        torch.rand(
                            done_indices.numel(),
                            1,
                            device=policy_obs.device,
                        )
                        < args_cli.teacher_probability
                    )

            collection_step += 1

            obs_dict = extras.get(
                "observations",
                None,
            )

            if obs_dict is None:
                raise RuntimeError(
                    "env.step() did not return "
                    "extras['observations']."
                )

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
                tensors_to_save = {
                    key: torch.cat(
                        chunks,
                        dim=0,
                    )
                    for key, chunks
                    in row_buffers.items()
                }

                saved_rows = int(
                    tensors_to_save[
                        "observations"
                    ].shape[0]
                )

                if saved_rows != buffered_samples:
                    raise RuntimeError(
                        "Buffered-row accounting mismatch: "
                        f"buffered={buffered_samples}, "
                        f"tensor_rows={saved_rows}."
                    )

                for key, tensor in tensors_to_save.items():
                    if tensor.shape[0] != saved_rows:
                        raise RuntimeError(
                            f"Row-count mismatch for {key}: "
                            f"{tensor.shape[0]} versus "
                            f"{saved_rows}."
                        )

                shard_path = os.path.join(
                    args_cli.output_dir,
                    (
                        f"round_{args_cli.round_index:02d}"
                        f"_shard_{shard_index:04d}.pt"
                    ),
                )

                payload = {
                    **tensors_to_save,

                    "schema_version": 4,
                    "probe_target_semantics": (
                        "teacher_observations_at_same_pre_step_state_as_observations"
                    ),
                    "round_index": args_cli.round_index,

                    "teacher_probability":
                        args_cli.teacher_probability,

                    "student_probability":
                        1.0
                        - args_cli.teacher_probability,

                    "mixing_mode":
                        args_cli.mixing_mode,

                    "accepted_row_balancing": (
                        args_cli.mixing_mode
                        == "episode_balanced"
                    ),

                    "target_teacher_rows":
                        target_teacher_rows,

                    "target_student_rows":
                        target_student_rows,

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

                    "controller_hold_steps":
                        args_cli.controller_hold_steps,

                    "teacher_checkpoint":
                        args_cli.teacher_checkpoint,

                    "student_checkpoint":
                        args_cli.student_checkpoint,

                    "observation_dtype": "float32",

                    "invalid_history_filter":
                        "any_corner_visible_at_least_once",

                    "provenance_alignment": (
                        "pre_step_observation_and_actions_"
                        "with_post_step_reward_and_outcome"
                    ),

                    "termination_reason_names": {
                        0: "none",
                        1: "timeout",
                        2: "non_timeout_termination",
                    },

                    "termination_bit_names": {
                        1: "too_far",
                        2: "too_tilted",
                        4: "too_low",
                        8: "timeout",
                    },

                    "termination_bitmask_semantics": (
                        "bitwise_or; multiple causes may be active"
                    ),

                    "candidate_samples_seen":
                        candidate_samples_seen,

                    "rejected_no_marker_count":
                        rejected_no_marker_count,
                }

                torch.save(
                    payload,
                    shard_path,
                )

                print(
                    f"[DAgger-v2] Saved {shard_path}: "
                    f"{saved_rows} samples; "
                    f"total={samples_collected}/"
                    f"{args_cli.num_samples}"
                )

                for chunks in row_buffers.values():
                    chunks.clear()

                buffered_samples = 0
                shard_index += 1

            if samples_collected >= args_cli.num_samples:
                break

    env.close()

    rejection_percentage = (
        100.0
        * rejected_no_marker_count
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
        f"All-invalid histories:       "
        f"{rejected_no_marker_count} "
        f"({rejection_percentage:.3f}%)"
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
        f"Controller hold steps:       "
        f"{args_cli.controller_hold_steps}"
    )
    print(
        f"Mixing mode:                 "
        f"{args_cli.mixing_mode}"
    )
    print(
        f"Target teacher rows:         "
        f"{target_teacher_rows}"
    )
    print(
        f"Target student rows:         "
        f"{target_student_rows}"
    )
    print(
        f"Accepted teacher rows:       "
        f"{accepted_teacher_count}"
    )
    print(
        f"Accepted student rows:       "
        f"{accepted_student_count}"
    )
    print(
        f"Realized teacher fraction:   "
        f"{accepted_teacher_count / max(samples_collected, 1):.6f}"
    )
    print(
        f"Accepted done rows:          "
        f"{accepted_done_count}"
    )
    print(
        f"Accepted timeout rows:       "
        f"{accepted_timeout_count}"
    )
    print(
        f"Accepted terminated rows:    "
        f"{accepted_termination_count}"
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()