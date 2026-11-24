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

        # Protect the learned visual representation
        # during this initial fine-tuning experiment.
        for p in self.actor.estimator.parameters():
            p.requires_grad_(False)

        self.distribution = None

        print(
            "[ACTOR] Loaded Round 2 K8 Transformer",
            flush=True,
        )

        print(
            "[ACTOR] Visual encoder frozen",
            flush=True,
        )

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

    env_cfg.tight_hover_reward_v1 = True

    env = K8HistoryEnv(
        cfg=env_cfg,
        render_mode=None,
    )

    try:
        print(
            "[DEBUG 1] Environment initialized successfully",
            flush=True,
        )

        print(
            "[DEBUG 2] Environment observation attributes:",
            flush=True,
        )

        for attr in (
            "single_observation_space",
            "observation_space",
            "num_observations",
        ):
            try:
                value = getattr(env, attr)
                print(
                    f"  {attr}: {value}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"  {attr}: ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )

        print(
            "[DEBUG 3] Entering observation-space configuration",
            flush=True,
        )

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

        print(
            "[DEBUG 5] RSL-RL environment wrapper created",
            flush=True,
        )

        print(
            "[DEBUG 6] Requesting initial observations",
            flush=True,
        )

        obs, extras = wrapped.get_observations()

        print(
            "[DEBUG 7] Received observations",
            flush=True,
        )

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

        agent_cfg.save_interval = max(
            1,
            args.iterations,
        )

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
            "[DEBUG 8] Constructing OnPolicyRunner",
            flush=True,
        )

        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=str(run),
            device=device,
        )

        policy = runner.alg.policy

        assert isinstance(
            policy,
            K8TransformerActorCritic,
        )

        assert type(runner.alg) is NativePPO

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

            actual = policy.act_inference(obs)

            error = (
                expected - actual
            ).abs().max().item()

            assert error < 1e-4, error

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

        manifest = {
            "baseline_checkpoint": str(REFERENCE_PT),
            "baseline_sha256": hashlib.sha256(
                REFERENCE_TS.read_bytes()
            ).hexdigest(),
            "actor_inputs": (
                "256 corners + 32 previous-action values"
            ),
            "critic_inputs": "privileged 20D",
            "algorithm": "native RSL-RL PPO",
            "visual_encoder_frozen": True,
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

        print(
            "[DEBUG 9] Starting native PPO learning",
            flush=True,
        )

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

        print(
            "[DEBUG FAILURE] Exception caught before cleanup",
            flush=True,
        )

        traceback.print_exc()
        raise

    finally:
        print(
            "[DEBUG CLEANUP] Closing environment",
            flush=True,
        )

        env.close()

        print(
            "[DEBUG CLEANUP] Closing Isaac Sim",
            flush=True,
        )

        app.close()


if __name__ == "__main__":
    main()
