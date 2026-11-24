#!/usr/bin/env python3
"""Native RSL-RL PPO fine-tuning for the K8 DAgger Transformer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--rollout_steps", type=int, default=8)
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--minibatches", type=int, default=2)
parser.add_argument("--level", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=7400)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--init_std", type=float, default=0.03)
parser.add_argument("--desired_kl", type=float, default=0.01)
parser.add_argument("--anchor_coef", type=float, default=20.0)
parser.add_argument("--out_dir", type=str, default="")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

assert args.num_envs > 0 and args.rollout_steps > 0
assert args.iterations > 0 and args.epochs > 0
assert args.minibatches > 0
assert args.num_envs * args.rollout_steps >= args.minibatches
assert 0 <= args.level <= 1
assert 0 < args.init_std < 0.2

args.headless = True
args.enable_cameras = False

app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.algorithms import PPO as NativePPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from envs.aruco_hover.quadrotor_aruco_hover_env import (
    ArucoHoverStudentEnv,
    ArucoHoverStudentEnvCfg,
)

from rl.corners_asymmetric_ppo.config import (
    CornersAsymmetricPPORunnerCfg,
)

ROOT = Path.cwd()
BASE = ROOT / "experiments/state_bottleneck"

SOURCE = BASE / "launch_action_history_study.py"

ROUND2 = (
    BASE
    / "k8_onpolicy_dagger_round2_20261120T225553Z"
    / "finetuned_k8"
)

REFERENCE_PT = ROUND2 / "policy.pt"
REFERENCE_TS = ROUND2 / "policy.ts"

for p in (SOURCE, REFERENCE_PT, REFERENCE_TS):
    assert p.is_file(), f"Missing input: {p}"


def get_policy_class():
    spec = importlib.util.spec_from_file_location(
        "dagger_k8_policy_for_rsl",
        SOURCE,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.Policy


class K8HistoryEnv(ArucoHoverStudentEnv):
    """Original hover simulator with causal action history."""

    def __init__(self, cfg, render_mode=None):
        self._k8_actions = None

        super().__init__(
            cfg=cfg,
            render_mode=render_mode,
        )

        self._k8_actions = torch.zeros(
            self.num_envs,
            8,
            4,
            device=self.device,
            dtype=torch.float32,
        )

        self._history_checks = 0

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)

        if self._k8_actions is not None:
            assert actions.shape == (
                self.num_envs, 4
            )

            assert torch.isfinite(actions).all()

            self._k8_actions = torch.roll(
                self._k8_actions,
                shifts=-1,
                dims=1,
            )

            self._k8_actions[:, -1] = (
                actions.detach()
            )

    def _reset_idx(self, env_ids):
        if self._k8_actions is not None:
            if env_ids is None:
                self._k8_actions.zero_()
            else:
                self._k8_actions[env_ids] = 0.0

        super()._reset_idx(env_ids)

        if self._k8_actions is not None:
            if env_ids is None:
                self._k8_actions.zero_()
            else:
                self._k8_actions[env_ids] = 0.0

    def _get_observations(self):
        obs = super()._get_observations()

        if self._k8_actions is None:
            return obs

        corners = obs["policy"]
        privileged = obs["teacher"]

        assert corners.shape == (
            self.num_envs, 256
        )

        assert privileged.shape == (
            self.num_envs, 20
        )

        error = (
            self._k8_actions[:, -1]
            - privileged[:, 15:19]
        ).abs().max().item()

        if error > 1e-5:
            raise RuntimeError(
                "Executed-action history mismatch: "
                f"{error}"
            )

        self._history_checks += 1

        obs["policy"] = torch.cat(
            [
                corners,
                self._k8_actions.reshape(
                    self.num_envs, 32
                ),
            ],
            dim=-1,
        )

        # Compatible with RSL-RL versions that require
        # the privileged group to be named "critic".
        obs["critic"] = privileged

        return obs


class K8TransformerActorCritic(ActorCritic):
    """RSL-RL PPO actor-critic with bounded CTBR actions."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        activation="elu",
        init_noise_std=0.03,
        noise_std_type="scalar",
        **kwargs,
    ):
        assert (
            num_actor_obs,
            num_critic_obs,
            num_actions,
        ) == (288, 20, 4)

        kwargs.pop("history_length", None)
        kwargs.pop("token_dim", None)
        kwargs.pop("temporal_channels", None)

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(
                actor_hidden_dims
            ),
            critic_hidden_dims=list(
                critic_hidden_dims
            ),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        assert noise_std_type == "scalar"

        # Initialize the asymmetric critic from the
        # trained privileged-state PPO teacher.
        #
        # Only critic weights are transferred.
        # The visual actor remains initialized from
        # the Round-2 K8 DAgger checkpoint.

        teacher_path = (
            Path("logs/rsl_rl/CTBR_Hover_Teacher/")
            / "2026-06-25_12-41-52"
            / "model_749.pt"
        )

        teacher_checkpoint = torch.load(
            teacher_path,
            map_location="cpu",
            weights_only=True,
        )

        teacher_state = (
            teacher_checkpoint["model_state_dict"]
        )

        critic_state = {
            key.removeprefix("critic."): value
            for key, value in teacher_state.items()
            if key.startswith("critic.")
        }

        # Strict loading verifies every parameter shape
        # and catches architectural incompatibility.
        self.critic.load_state_dict(
            critic_state,
            strict=True,
        )

        print(
            "[PASS] Pretrained teacher critic loaded",
            flush=True,
        )

        print(
            "[CRITIC] Source:",
            teacher_path,
            flush=True,
        )

        Policy = get_policy_class()

        self.actor = Policy(8)

        checkpoint = torch.load(
            REFERENCE_PT,
            map_location="cpu",
            weights_only=False,
        )

        assert checkpoint["k"] == 8

        self.actor.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.actor.eval()

        # Optimize the complete pretrained Transformer.
        for p in self.actor.parameters():
            p.requires_grad_(True)

        self.distribution = None

        print(
            "[ACTOR] Loaded Round 2 K8 Transformer",
            flush=True,
        )

        print(
            "[ACTOR] Entire Transformer trainable",
            flush=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)

        # Preserve deterministic Transformer behavior.
        # Gradients remain enabled for trainable parameters.
        self.actor.eval()

        return self

    def _mean_action(self, observations):
        assert observations.shape[-1] == 288

        return self.actor(
            observations[:, :256],
            observations[:, 256:].reshape(
                -1, 8, 4
            ),
        )

    def act_inference(self, observations):
        return self._mean_action(observations)

    def update_distribution(self, observations):
        # During native PPO optimization, remember the
        # actual actor observations used in this minibatch.
        #
        # Rollout collection occurs under inference_mode,
        # so it does not populate this cache.

        if torch.is_grad_enabled():
            self._anchor_batch_obs = (
                observations.detach().clone()
            )

        bounded_mean = self._mean_action(
            observations
        )

        # Convert the pretrained bounded mean into
        # the corresponding pre-tanh Gaussian mean.
        raw_mean = torch.atanh(
            bounded_mean.clamp(
                -0.999999,
                0.999999,
            )
        )

        std = (
            self.std.clamp(
                min=1e-4,
                max=0.3,
            ).expand_as(raw_mean)
        )

        self.distribution = Normal(
            raw_mean,
            std,
        )

    def act(self, observations, **kwargs):
        self.update_distribution(observations)

        return torch.tanh(
            self.distribution.sample()
        )

    def get_actions_log_prob(self, actions):
        bounded = actions.clamp(
            -0.999999,
            0.999999,
        )

        z = torch.atanh(bounded)

        log_jacobian = 2.0 * (
            math.log(2.0)
            - z
            - F.softplus(-2.0 * z)
        )

        return (
            self.distribution.log_prob(z)
            - log_jacobian
        ).sum(dim=-1)

    @property
    def action_mean(self):
        # Pre-tanh parameters for RSL-RL's KL.
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        # Gaussian entropy proxy. Entropy coefficient
        # is zero in this experiment.
        return self.distribution.entropy().sum(
            dim=-1
        )


def export_policy(model, filename):
    inference = copy.deepcopy(
        model.actor
    ).cpu().eval()

    examples = (
        torch.zeros(2, 256),
        torch.zeros(2, 8, 4),
    )

    with torch.inference_mode():
        traced = torch.jit.trace(
            inference,
            examples,
            check_trace=True,
        )

        torch.testing.assert_close(
            traced(*examples),
            inference(*examples),
            atol=1e-5,
            rtol=1e-5,
        )

        traced.save(str(filename))



def install_kl_guard(runner, policy, args, run):
    """Constrain native PPO actor updates using rollout observations."""

    import csv
    import math
    import torch

    assert float(args.anchor_coef) == 0.0
    assert int(args.iterations) <= 10
    assert hasattr(runner.alg, "storage")
    assert hasattr(policy, "std")

    reference_path = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments/state_bottleneck"
        / "precision_no_anchor_control_20261122T021745Z"
        / "model/policy_initial.ts"
    )

    assert reference_path.is_file(), reference_path

    device = next(policy.actor.parameters()).device

    reference = torch.jit.load(
        str(reference_path),
        map_location=device,
    ).eval()

    for parameter in reference.parameters():
        parameter.requires_grad_(False)


    # SMALL-ACTOR-STEP PPO
    #
    # Reduce actual actor learning rates 50x.
    # Preserve the privileged critic learning rate.
    #
    # This hook is registered after the original
    # differential-LR hook.

    actor_parameter_ids = {
        id(p) for p in policy.actor.parameters()
    }

    actor_parameter_ids.add(id(policy.std))

    lr_reported = False

    def reduce_actor_learning_rates(
        optimizer,
        step_args,
        step_kwargs,
    ):
        nonlocal lr_reported

        actor_groups = 0
        critic_groups = 0

        for group in optimizer.param_groups:

            is_actor_group = any(
                id(p) in actor_parameter_ids
                for p in group["params"]
            )

            if is_actor_group:
                group["lr"] *= 0.02
                actor_groups += 1
            else:
                critic_groups += 1

            if not lr_reported:
                print(
                    "[ACTUAL LR]",
                    group.get("name", "unnamed"),
                    f"lr={group['lr']:.9e}",
                    flush=True,
                )

        assert actor_groups >= 2
        assert critic_groups >= 1

        lr_reported = True

    runner.alg.optimizer.register_step_pre_hook(
        reduce_actor_learning_rates
    )

    print(
        "[PASS] Actor optimizer steps reduced 50x",
        flush=True,
    )

    native_update = runner.alg.update
    iteration = 0

    output_csv = run / "trust_guard.csv"

    def gaussian_kl(mu0, std0, mu1, std1):
        return (
            torch.log(std1 / std0)
            + (
                std0.square()
                + (mu0 - mu1).square()
            ) / (2 * std1.square())
            - 0.5
        ).sum(dim=-1)

    def current_actions(observations):
        return policy.actor(
            observations[:, :256],
            observations[:, 256:].reshape(-1, 8, 4),
        )

    @torch.no_grad()
    def evaluate_change(
        observations,
        previous_actions,
        previous_std,
        reference_actions,
        reference_std,
    ):
        actions = torch.cat(
            [
                current_actions(chunk)
                for chunk in observations.split(1024)
            ],
            dim=0,
        )

        std = (
            policy.std.detach()
            .flatten()
            .to(device)
            .clamp_min(1e-6)
        )

        mu_previous = torch.atanh(
            previous_actions.clamp(-0.999999, 0.999999)
        )

        mu_current = torch.atanh(
            actions.clamp(-0.999999, 0.999999)
        )

        mu_reference = torch.atanh(
            reference_actions.clamp(-0.999999, 0.999999)
        )

        kl_previous = gaussian_kl(
            mu_previous,
            previous_std,
            mu_current,
            std,
        )

        kl_reference = gaussian_kl(
            mu_reference,
            reference_std,
            mu_current,
            std,
        )

        return {
            "kl_prev": kl_previous.mean().item(),
            "kl_ref": kl_reference.mean().item(),
            "kl_prev_p99": torch.quantile(
                kl_previous, 0.99
            ).item(),
            "action_mae": (
                actions - previous_actions
            ).abs().mean().item(),
        }

    def guarded_update(*update_args, **update_kwargs):
        nonlocal iteration

        # Sample real observations from the current
        # PPO rollout, before storage is cleared.
        storage = runner.alg.storage

        observations = storage.observations

        assert isinstance(
            observations, torch.Tensor
        ), type(observations)

        assert observations.ndim == 3
        assert observations.shape[-1] == 288

        flat = observations.reshape(-1, 288)

        stride = max(1, len(flat) // 8192)

        probe = (
            flat[::stride][:8192]
            .detach()
            .clone()
        )

        assert torch.isfinite(probe).all()

        with torch.no_grad():

            before = {
                name: tensor.detach().clone()
                for name, tensor
                in policy.actor.state_dict().items()
            }

            previous_std = (
                policy.std.detach().flatten().clone()
                .to(device)
            )

            previous_actions = torch.cat(
                [
                    current_actions(chunk)
                    for chunk in probe.split(1024)
                ],
                dim=0,
            )

            reference_actions = torch.cat(
                [
                    reference(
                        chunk[:, :256],
                        chunk[:, 256:].reshape(-1, 8, 4),
                    )
                    for chunk in probe.split(1024)
                ],
                dim=0,
            )

            reference_std = torch.full_like(
                previous_std, 0.03
            )

        # Perform the original native RSL-RL PPO update.
        # The privileged critic is updated normally.
        result = native_update(
            *update_args,
            **update_kwargs,
        )

        with torch.no_grad():

            proposed = {
                name: tensor.detach().clone()
                for name, tensor
                in policy.actor.state_dict().items()
            }

            proposed_std = (
                policy.std.detach().flatten().clone()
                .to(device)
            )

            def interpolate(alpha):

                mixed = {}

                for name, original in before.items():
                    updated = proposed[name]

                    if original.is_floating_point():
                        mixed[name] = torch.lerp(
                            original, updated, alpha
                        )
                    else:
                        mixed[name] = original

                policy.actor.load_state_dict(
                    mixed, strict=True
                )

                policy.std.copy_(
                    torch.lerp(
                        previous_std,
                        proposed_std,
                        alpha,
                    ).reshape_as(policy.std)
                )

            def measure():
                return evaluate_change(
                    probe,
                    previous_actions,
                    previous_std,
                    reference_actions,
                    reference_std,
                )

            def safe(metrics):
                return (
                    all(
                        math.isfinite(value)
                        for value in metrics.values()
                    )
                    and metrics["kl_prev"] <= 0.002
                                        and metrics["kl_prev_p99"] <= 0.02
                    and metrics["action_mae"] <= 0.002
                )

            raw_metrics = measure()

            if safe(raw_metrics):
                alpha = 1.0
                metrics = raw_metrics

            else:
                interpolate(0.0)
                baseline_metrics = measure()

                if not safe(baseline_metrics):
                    raise RuntimeError(
                        "No safe update available: "
                        + repr(baseline_metrics)
                    )

                low = 0.0
                high = 1.0
                metrics = baseline_metrics

                # Backtracking/bisection over the entire
                # Transformer parameter displacement.
                for _ in range(14):

                    midpoint = (low + high) / 2

                    interpolate(midpoint)

                    candidate = measure()

                    if safe(candidate):
                        low = midpoint
                        metrics = candidate
                    else:
                        high = midpoint

                alpha = low
                interpolate(alpha)

            assert safe(metrics), metrics

            iteration += 1

            row = {
                "iteration": iteration,
                "alpha": alpha,
                "raw_kl": raw_metrics["kl_prev"],
                "raw_action_mae": raw_metrics["action_mae"],
                **metrics,
            }

            with output_csv.open(
                "a", newline=""
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=list(row),
                )

                if file.tell() == 0:
                    writer.writeheader()

                writer.writerow(row)

            print(
                "[TRUST GUARD]",
                f"iteration={iteration}",
                f"alpha={alpha:.6f}",
                f"raw_kl={raw_metrics['kl_prev']:.4f}",
                f"kl_prev={metrics['kl_prev']:.6f}",
                f"kl_ref={metrics['kl_ref']:.6f}",
                f"p99={metrics['kl_prev_p99']:.6f}",
                f"action_mae={metrics['action_mae']:.6f}",
                flush=True,
            )

        return result

    runner.alg.update = guarded_update

    print(
        "[PASS] Native PPO trust guard installed",
        flush=True,
    )

def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or "cuda:0"

    env_cfg = ArucoHoverStudentEnvCfg()

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = device

    env_cfg.forced_curriculum_level = args.level

    env_cfg.eval_mode = False
    env_cfg.nominal_mode = False

    env_cfg.enable_onboard_camera = False
    env_cfg.visual_dr_enabled = False
    env_cfg.debug_vis = False

    # K8_PRECISION_ANCHOR_PILOT
    # Evaluator tight condition:
    # position <= 0.10 m AND speed <= 0.20 m/s.

    env_cfg.tight_hover_reward_v1 = True

    env_cfg.w_tight_hover = 12.0
    env_cfg.tight_pos_sigma = 0.10
    env_cfg.tight_lin_speed_sigma = 0.15

    env_cfg.w_stable_hover = 2.0
    env_cfg.stable_pos_error = 0.10
    env_cfg.stable_lin_speed = 0.20

    print(
        "[PRECISION REWARD]",
        "tight_weight=12.0",
        "pos_sigma=0.10",
        "speed_sigma=0.15",
        "stable_position=0.10",
        "stable_speed=0.20",
        flush=True,
    )

    env = K8HistoryEnv(
        cfg=env_cfg,
        render_mode=None,
    )

    try:
        # Declare the additional action-history observations
        # to the RSL-RL wrapper.

        # The original environment declares a legacy
        # observation space with policy=20.
        #
        # Its student observation implementation actually
        # produces corners256 and privileged20.
        #
        # Our K8HistoryEnv augments corners256 with
        # eight previous executed CTBR commands (32).
        #
        # Declare the resulting RSL-RL observation groups
        # explicitly rather than relying on stale metadata.

        original_space = env.single_observation_space

        print(
            "[OBS SPACE] Original declaration:",
            original_space,
            flush=True,
        )

        actor_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(288,),
            dtype=np.float32,
        )

        privileged_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32,
        )

        env.single_observation_space = gym.spaces.Dict(
            {
                "policy": actor_space,
                "teacher": privileged_space,
                "critic": privileged_space,
            }
        )

        env.observation_space = (
            gym.vector.utils.batch_space(
                env.single_observation_space,
                env.num_envs,
            )
        )

        print(
            "[PASS] Corrected observation declaration: "
            "policy=288 teacher=20 critic=20",
            flush=True,
        )

        wrapped = RslRlVecEnvWrapper(env)

        obs, extras = wrapped.get_observations()

        assert obs.shape == (
            args.num_envs, 288
        )

        assert extras["observations"][
            "teacher"
        ].shape == (args.num_envs, 20)

        assert extras["observations"][
            "critic"
        ].shape == (args.num_envs, 20)

        print(
            "[PASS] RSL-RL environment shapes: "
            "actor=288 critic=20",
            flush=True,
        )

        agent_cfg = CornersAsymmetricPPORunnerCfg()

        agent_cfg.seed = args.seed
        agent_cfg.device = device

        agent_cfg.num_steps_per_env = (
            args.rollout_steps
        )

        agent_cfg.max_iterations = (
            args.iterations
        )

        agent_cfg.save_interval = 1

        agent_cfg.experiment_name = (
            "K8_Transformer_DAgger_Asymmetric_RSL_PPO"
        )

        agent_cfg.run_name = ""
        agent_cfg.resume = False

        agent_cfg.obs_groups = {
            "policy": ["policy"],
            "critic": ["teacher"],
        }

        agent_cfg.policy.class_name = (
            "K8TransformerActorCritic"
        )

        agent_cfg.policy.init_noise_std = (
            args.init_std
        )

        agent_cfg.policy.noise_std_type = (
            "scalar"
        )

        # Match the pretrained teacher critic exactly.
        agent_cfg.policy.critic_hidden_dims = [64, 64]
        agent_cfg.policy.activation = "tanh"

        agent_cfg.algorithm.class_name = "PPO"

        agent_cfg.algorithm.num_learning_epochs = (
            args.epochs
        )

        agent_cfg.algorithm.num_mini_batches = (
            args.minibatches
        )

        agent_cfg.algorithm.learning_rate = (
            args.learning_rate
        )

        agent_cfg.algorithm.clip_param = 0.1
        agent_cfg.algorithm.schedule = "adaptive"

        agent_cfg.algorithm.desired_kl = (
            args.desired_kl
        )

        agent_cfg.algorithm.entropy_coef = 0.0
        agent_cfg.algorithm.max_grad_norm = 0.5

        agent_cfg.algorithm.gamma = 0.99
        agent_cfg.algorithm.lam = 0.95

        # Register custom actor with the existing
        # OnPolicyRunner. PPO remains native RSL-RL.
        runner_module.K8TransformerActorCritic = (
            K8TransformerActorCritic
        )

        runner_module.PPO = NativePPO

        stamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        run = (
            Path(args.out_dir)
            if args.out_dir
            else BASE / f"k8_rsl_ppo_{stamp}"
        )

        run.mkdir(
            parents=True,
            exist_ok=False,
        )

        print(
            "[DEBUG] Entering OnPolicyRunner initialization",
            flush=True,
        )

        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=str(run),
            device=device,
        )

        print(
            "[DEBUG] OnPolicyRunner initialized successfully",
            flush=True,
        )

        policy = runner.alg.policy

        assert isinstance(
            policy,
            K8TransformerActorCritic,
        )

        assert type(runner.alg) is NativePPO

        # Frozen reference is the original successful
        # Round 2 DAgger TorchScript policy.
        frozen_dagger = torch.jit.load(
            str(REFERENCE_TS),
            map_location=device,
        ).eval()

        for p in frozen_dagger.parameters():
            p.requires_grad_(False)

        policy._anchor_batch_obs = None
        policy._anchor_history = []
        policy._grad_diagnostics = []

        print(
            "[ANCHOR] Frozen Round 2 DAgger reference loaded",
            flush=True,
        )

        # K8_FULL_MODEL_PARAMETER_GROUPS
        groups = []
        included = set()

        def group(name, parameters, ratio):
            params = [
                p for p in parameters
                if p.requires_grad
            ]

            assert params, name

            for p in params:
                assert id(p) not in included, name
                included.add(id(p))

            groups.append({
                "params": params,
                "lr": args.learning_rate * ratio,
                "ratio": ratio,
                "name": name,
            })

        group(
            "visual_estimator",
            policy.actor.estimator.parameters(),
            0.1,
        )

        group(
            "policy_head",
            policy.actor.head.parameters(),
            1.0,
        )

        group(
            "privileged_critic",
            policy.critic.parameters(),
            10.0,
        )

        remaining = [
            p for p in policy.parameters()
            if p.requires_grad
            and id(p) not in included
        ]

        if remaining:
            group(
                "action_std",
                remaining,
                0.1,
            )

        expected = {
            id(p)
            for p in policy.parameters()
            if p.requires_grad
        }

        assert included == expected

        runner.alg.optimizer = torch.optim.Adam(
            groups,
            eps=1e-5,
        )

        # Preserve relative learning rates when the
        # native PPO adaptive-KL scheduler changes LR.
        def sync_group_lrs(
            optimizer,
            step_args,
            step_kwargs,
        ):
            # Preserve the existing differential learning
            # rates and native PPO adaptive-KL scheduling.
            base_lr = min(
                float(runner.alg.learning_rate),
                float(args.learning_rate),
            )

            for item in optimizer.param_groups:
                item["lr"] = (
                    base_lr * item["ratio"]
                )

            if args.anchor_coef <= 0:
                return

            batch = policy._anchor_batch_obs

            if batch is None:
                raise RuntimeError(
                    "Anchor minibatch missing during PPO step"
                )

            # Native PPO already computed its gradient.
            # Add d/dtheta of:
            #
            # anchor_coef * E[
            #   ||pi_theta(o) - pi_DAgger(o)||_2^2
            # ]
            #
            # on the SAME on-policy minibatch.
            #
            # Microbatching controls memory use.
            total = batch.shape[0]

            assert total > 0

            mean_anchor_loss = 0.0
            mean_action_mse = 0.0

            # K8_GRAD_AUDIT
            #
            # Native RSL-RL has already computed and
            # clipped the PPO gradient at this point.
            #
            # Save the actor portion so that we can
            # separately measure the anchor gradient.

            actor_parameters = [
                p
                for p in policy.actor.parameters()
                if p.requires_grad
            ]

            ppo_actor_grads = [
                (
                    p.grad.detach().clone()
                    if p.grad is not None
                    else torch.zeros_like(p)
                )
                for p in actor_parameters
            ]

            with torch.enable_grad():

                for chunk in batch.split(2048):
                    corners = chunk[:, :256]
                    history = chunk[:, 256:].reshape(
                        -1, 8, 4
                    )

                    # Actor stays in eval mode to disable
                    # dropout, but gradients are enabled.
                    current = policy.actor(
                        corners,
                        history,
                    )

                    with torch.no_grad():
                        reference = frozen_dagger(
                            corners,
                            history,
                        )

                    action_mse = (
                        (current - reference)
                        .square()
                        .sum(dim=-1)
                        .mean()
                    )

                    fraction = (
                        chunk.shape[0] / total
                    )

                    anchor_loss = (
                        args.anchor_coef
                        * fraction
                        * action_mse
                    )

                    if not torch.isfinite(anchor_loss):
                        raise RuntimeError(
                            "Nonfinite reference-policy loss"
                        )

                    anchor_loss.backward()

                    mean_anchor_loss += (
                        anchor_loss.detach().item()
                    )

                    mean_action_mse += (
                        fraction
                        * action_mse.detach().item()
                    )

            # Measure the gradients separately.
            #
            # ppo_actor_grads contains the actor PPO
            # gradient AFTER native PPO gradient clipping.
            #
            # The difference between the current gradient
            # and the saved PPO gradient is the gradient
            # contributed by the anchor regularizer.

            ppo_sq = torch.zeros(
                (), device=device
            )

            anchor_sq = torch.zeros(
                (), device=device
            )

            grad_dot = torch.zeros(
                (), device=device
            )

            for p, ppo_grad in zip(
                actor_parameters,
                ppo_actor_grads,
            ):
                combined_grad = (
                    p.grad.detach()
                    if p.grad is not None
                    else torch.zeros_like(p)
                )

                anchor_grad = (
                    combined_grad - ppo_grad
                )

                ppo_sq += ppo_grad.square().sum()
                anchor_sq += anchor_grad.square().sum()

                grad_dot += (
                    ppo_grad * anchor_grad
                ).sum()

            ppo_norm = torch.sqrt(ppo_sq)
            anchor_norm = torch.sqrt(anchor_sq)

            cosine = (
                grad_dot
                / (
                    ppo_norm * anchor_norm
                    + 1e-12
                )
            )

            ratio = (
                anchor_norm
                / (ppo_norm + 1e-12)
            )

            policy._grad_diagnostics.append({
                "ppo_grad_norm": ppo_norm.item(),
                "anchor_grad_norm": anchor_norm.item(),
                "anchor_ppo_ratio": ratio.item(),
                "gradient_cosine": cosine.item(),
                "base_lr": float(
                    runner.alg.learning_rate
                ),
            })

            # The native PPO gradient was already clipped.
            # Clip the combined PPO + anchor gradient again.
            nn.utils.clip_grad_norm_(
                [
                    p
                    for item in optimizer.param_groups
                    for p in item["params"]
                    if p.grad is not None
                ],
                0.5,
            )

            policy._anchor_history.append(
                (
                    mean_anchor_loss,
                    mean_action_mse,
                )
            )

            # Prevent accidentally reusing a stale minibatch.
            policy._anchor_batch_obs = None

        runner.alg.optimizer.register_step_pre_hook(
            sync_group_lrs
        )

        print(
            "[PASS] Full-model optimizer with native PPO",
            flush=True,
        )

        for item in groups:
            print(
                "[LR]",
                item["name"],
                item["lr"],
                flush=True,
            )

        # Verify exact pretrained-policy initialization.
        with torch.no_grad():
            reference = torch.jit.load(
                str(REFERENCE_TS),
                map_location=device,
            ).eval()

            expected = reference(
                obs[:, :256],
                obs[:, 256:].reshape(
                    -1, 8, 4
                ),
            )

            # Diagnostic: does Transformer training mode
            # explain the action discrepancy?

            policy.actor.train(True)

            train_actions = policy.act_inference(obs)

            train_error = (
                expected - train_actions
            ).abs().max().item()

            policy.actor.eval()

            actual = policy.act_inference(obs)

            error = (
                expected - actual
            ).abs().max().item()

            print(
                "[ACTOR MODE DIAGNOSTIC]",
                f"train_mode_maxdiff={train_error:.8f}",
                f"eval_mode_maxdiff={error:.8f}",
                flush=True,
            )

            if error >= 1e-4:
                raise RuntimeError(
                    "RSL-RL actor still differs from the "
                    "original DAgger checkpoint in eval mode: "
                    f"maxdiff={error}"
                )

            # RSL-RL training must not reactivate dropout.
            policy.train()

            assert not policy.actor.training, (
                "PPO training mode reactivated Transformer dropout"
            )

            after_train = policy.act_inference(obs)

            mode_error = (
                expected - after_train
            ).abs().max().item()

            assert mode_error < 1e-4, (
                f"Actor changed after policy.train(): {mode_error}"
            )

            policy.eval()

            print(
                "[PASS] Deterministic actor remains "
                "unchanged across RSL-RL mode switches",
                flush=True,
            )

            # Check the stochastic policy interface.
            actions = policy.act(obs)

            log_prob = (
                policy.get_actions_log_prob(
                    actions
                )
            )

            assert actions.shape == (
                args.num_envs, 4
            )

            assert (actions.abs() <= 1).all()
            assert torch.isfinite(log_prob).all()

            assert torch.isfinite(
                policy.action_mean
            ).all()

            assert torch.isfinite(
                policy.action_std
            ).all()

        print(
            "[PASS] Pretrained actor equivalence: "
            f"maxdiff={error:.6g}",
            flush=True,
        )

        print(
            "[PASS] Bounded actions and log probabilities",
            flush=True,
        )

        print(
            "[PASS] Native RSL-RL PPO selected",
            flush=True,
        )

        # K8_OVERNIGHT_BENCHMARK
        import csv
        import re
        import shutil
        import subprocess
        import sys

        def find_benchmark():
            known = [
                BASE / "evaluate_action_history.py",
                BASE
                / "action_history_study_20261120T200813Z"
                / "evaluate_action_history.py",
            ]

            matches = [
                p for p in known
                if p.is_file()
            ]

            if not matches:
                matches = sorted(
                    BASE.rglob(
                        "evaluate_action_history.py"
                    )
                )

            if not matches:
                raise RuntimeError(
                    "Cannot find evaluate_action_history.py. "
                    "Overnight training was NOT started."
                )

            return matches[0]

        benchmark = (
            find_benchmark()
            if args.iterations >= 25
            else None
        )

        best = {"holds": -1}

        def benchmark_checkpoint(ts_file, label):
            benchmark = (
                ROOT
                / "experiments/state_bottleneck"
                / "action_history_study_20261120T200813Z"
                / "evaluate_action_history.py"
            )

            if not benchmark.is_file():
                raise FileNotFoundError(
                    f"Benchmark evaluator missing: {benchmark}"
                )

            print(
                "[BENCHMARK ENABLED]",
                benchmark,
                flush=True,
            )

            scores = []

            for level in (0, 1):
                target = (
                    run
                    / "benchmark"
                    / f"{label}_L{level}"
                )

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                log = (
                    target.parent
                    / f"{label}_L{level}.log"
                )

                command = [
                    sys.executable,
                    "-u",
                    str(benchmark),
                    "--task",
                    "Isaac-QuadrotorHoverStudentDistill-v0",
                    "--policy",
                    str(ts_file),
                    "--variant",
                    f"k8_{label}",
                    "--level",
                    str(level),
                    "--num-envs",
                    "64",
                    "--steps",
                    "500",
                    "--seed",
                    "6300",
                    "--device",
                    ("cuda:1" if str(args.device) == "cuda:0" else "cuda:0"),
                    "--out",
                    str(target),
                ]

                print(
                    "[BENCHMARK START]",
                    label,
                    f"L{level}",
                    flush=True,
                )

                with log.open("w") as output:
                    process = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                    )

                summary = target / "summary.json"

                if (
                    process.returncode != 0
                    or not summary.is_file()
                ):
                    tail = "\\n".join(
                        log.read_text(
                            errors="replace"
                        ).splitlines()[-40:]
                    )

                    raise RuntimeError(
                        f"Benchmark {label} L{level} failed:"
                        f"\\n{tail}"
                    )

                data = json.loads(
                    summary.read_text()
                )

                assert data["num_envs"] == 64
                assert (
                    data["completed"]
                    + data["censored"]
                    == 64
                )

                row = {
                    "label": label,
                    "level": level,
                    "survival": int(
                        data[
                            "full_horizon_survival_count"
                        ]
                    ),
                    "holds": int(
                        data[
                            "final_two_second_tight_hold_count"
                        ]
                    ),
                    "position_rms_m": float(
                        data["mean_position_rms_m"]
                    ),
                    "speed_rms_mps": float(
                        data["mean_speed_rms_mps"]
                    ),
                }

                scores.append(row)

                print(
                    f"[BENCHMARK] {label} L{level}: "
                    f"survival={row['survival']}/64 "
                    f"holds={row['holds']}/64 "
                    f"pos={row['position_rms_m']:.4f}m "
                    f"speed={row['speed_rms_mps']:.4f}m/s",
                    flush=True,
                )

            results_csv = (
                run / "benchmark_history.csv"
            )

            with results_csv.open(
                "a",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(scores[0]),
                )

                if f.tell() == 0:
                    writer.writeheader()

                writer.writerows(scores)

            survival = sum(
                x["survival"] for x in scores
            )

            holds = sum(
                x["holds"] for x in scores
            )

            # Preserve the best screened policy,
            # requiring at least 95% survival.
            if (
                survival >= 122
                and holds > best["holds"]
            ):
                best["holds"] = holds

                shutil.copy2(
                    ts_file,
                    run / "best_policy.ts",
                )

                (
                    run / "best_metrics.json"
                ).write_text(
                    json.dumps(
                        {
                            "checkpoint": label,
                            "survival": survival,
                            "holds": holds,
                            "scores": scores,
                        },
                        indent=2,
                    )
                )

                print(
                    f"[NEW BEST] {label}: "
                    f"survival={survival}/128 "
                    f"holds={holds}/128",
                    flush=True,
                )

        original_rsl_save = runner.save

        def save_with_benchmark(
            path,
            *save_args,
            **save_kwargs,
        ):
            original_rsl_save(
                path,
                *save_args,
                **save_kwargs,
            )

            ts_path = Path(path).with_suffix(".ts")

            temporary = Path(
                str(ts_path) + ".tmp"
            )

            export_policy(
                policy,
                temporary,
            )

            temporary.replace(ts_path)

            print(
                "[CHECKPOINT READY]",
                ts_path,
                flush=True,
            )

            anchor_stats = policy._anchor_history

            if anchor_stats:
                avg_loss = sum(
                    x[0] for x in anchor_stats
                ) / len(anchor_stats)

                avg_mse = sum(
                    x[1] for x in anchor_stats
                ) / len(anchor_stats)

                print(
                    "[ANCHOR]",
                    f"loss={avg_loss:.6f}",
                    f"action_mse={avg_mse:.8f}",
                    flush=True,
                )

                anchor_stats.clear()

            grad_data = policy._grad_diagnostics

            if grad_data:
                import csv

                def average(key):
                    return sum(
                        x[key] for x in grad_data
                    ) / len(grad_data)

                row = {
                    "checkpoint": Path(path).stem,
                    "optimizer_steps": len(grad_data),
                    "ppo_grad_norm": average(
                        "ppo_grad_norm"
                    ),
                    "anchor_grad_norm": average(
                        "anchor_grad_norm"
                    ),
                    "anchor_ppo_ratio": average(
                        "anchor_ppo_ratio"
                    ),
                    "gradient_cosine": average(
                        "gradient_cosine"
                    ),
                    "base_lr_mean": average(
                        "base_lr"
                    ),
                    "base_lr_min": min(
                        x["base_lr"]
                        for x in grad_data
                    ),
                    "base_lr_max": max(
                        x["base_lr"]
                        for x in grad_data
                    ),
                }

                diagnostic_path = (
                    run / "gradient_diagnostics.csv"
                )

                with diagnostic_path.open(
                    "a", newline=""
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(row),
                    )

                    if f.tell() == 0:
                        writer.writeheader()

                    writer.writerow(row)

                print(
                    "[GRAD DIAG]",
                    f"checkpoint={row['checkpoint']}",
                    f"ppo={row['ppo_grad_norm']:.6f}",
                    f"anchor={row['anchor_grad_norm']:.6f}",
                    f"ratio={row['anchor_ppo_ratio']:.3f}",
                    f"cosine={row['gradient_cosine']:.3f}",
                    f"lr={row['base_lr_mean']:.3e}",
                    flush=True,
                )

                grad_data.clear()

            # Native RSL-RL filenames use model_0.pt
            # after the first update, model_1.pt after the
            # second update, etc.
            #
            # Avoid the broken regex used overnight.
            stem = Path(path).stem

            if (
                stem.startswith("model_")
                and stem[6:].isdigit()
                and args.iterations >= 5
            ):
                completed = int(stem[6:]) + 1

                if completed in (5, 10):
                    label = f"iter_{completed:04d}"

                    # RSL-RL may save the last iteration
                    # twice. Do not repeat its evaluation.
                    already_done = (
                        run
                        / "benchmark"
                        / f"{label}_L1"
                        / "summary.json"
                    )

                    if not already_done.is_file():
                        benchmark_checkpoint(
                            ts_path,
                            label,
                        )

                        import csv

                        history_csv = (
                            run / "benchmark_history.csv"
                        )

                        with history_csv.open(
                            newline=""
                        ) as file:
                            rows = [
                                row
                                for row in csv.DictReader(file)
                                if row["label"] == label
                            ]

                        assert len(rows) == 2, rows

                        survival = sum(
                            int(row["survival"])
                            for row in rows
                        )

                        holds = sum(
                            int(row["holds"])
                            for row in rows
                        )

                        print(
                            f"[PRECISION GATE] "
                            f"iteration={completed} "
                            f"survival={survival}/128 "
                            f"tight_holds={holds}/128 "
                            f"baseline_holds=63/128",
                            flush=True,
                        )

                        # Initial DAgger result: 63/128.
                        # Give PPO three iterations before
                        # applying the regression gate.
                        #
                        # A large regression terminates the
                        # pilot instead of wasting transitions.
                        if completed >= 3:
                            if (
                                survival < 122
                                or holds < 55
                            ):
                                print(
                                    "[EARLY STOP] Precision "
                                    "or survival regressed",
                                    flush=True,
                                )

                                raise RuntimeError(
                                    "Precision safety gate failed "
                                    f"at iteration {completed}: "
                                    f"survival={survival}/128, "
                                    f"holds={holds}/128"
                                )

        runner.save = save_with_benchmark

        manifest = {
            "anchor_coef": args.anchor_coef,
            "precision_reward": {
                "w_tight_hover": 12.0,
                "tight_pos_sigma": 0.10,
                "tight_lin_speed_sigma": 0.15,
                "w_stable_hover": 2.0,
                "stable_pos_error": 0.10,
                "stable_lin_speed": 0.20,
            },
            "baseline_checkpoint": str(REFERENCE_PT),
            "baseline_sha256": hashlib.sha256(
                REFERENCE_TS.read_bytes()
            ).hexdigest(),
            "actor_inputs": (
                "256 corners + 32 previous-action values"
            ),
            "critic_inputs": "privileged 20D",
            "algorithm": "native RSL-RL PPO",
            "visual_encoder_frozen": False,
            "tight_hover_reward": True,
            "args": vars(args),
        }

        (run / "manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                default=str,
            )
        )

        export_policy(
            policy,
            run / "policy_initial.ts",
        )

        print("[RUN]", run, flush=True)

        # Initial evaluation must work before
        # committing to overnight PPO training.
        if args.iterations >= 25:
            benchmark_checkpoint(
                run / "policy_initial.ts",
                "initial",
            )

        install_kl_guard(runner, policy, args, run)
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=False,
        )

        export_policy(
            policy,
            run / "policy_final.ts",
        )

        runner.save(
            str(run / "final_rsl_rl.pt")
        )

        print(
            "[PASS] RSL-RL PPO integration completed",
            flush=True,
        )

        print(
            "[HISTORY CHECKS]",
            env._history_checks,
            flush=True,
        )

        print(
            "[FINAL POLICY]",
            run / "policy_final.ts",
            flush=True,
        )

    except BaseException:
        import traceback
        import sys

        print(
            "\n========== ACTUAL PYTHON FAILURE ==========",
            flush=True,
        )

        traceback.print_exc()

        sys.stdout.flush()
        sys.stderr.flush()

        raise

    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
