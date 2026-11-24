"""Structured corners-only PPO with on-policy auxiliary state supervision."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--init", choices=["ppo400", "distill"], required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--mini_batches", type=int, default=4)
parser.add_argument("--aux_samples", type=int, default=1024)
parser.add_argument("--aux_coef", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

simulation_app = AppLauncher(args).app

import torch
from torch import nn

import envs

from rl.corners_asymmetric_ppo.environment import (
    CornersAsymmetricPPOEnv,
    CornersAsymmetricPPOEnvCfg,
)
from rl.corners_asymmetric_ppo.config import (
    CornersAsymmetricPPORunnerCfg,
)
from experiments.state_bottleneck.structured_init import (
    StructuredCornersActorCritic,
    STATE_INDICES,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.algorithms import PPO as NativePPO
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module


ROOT = Path("/home/sumukh/AerialCatching")

INITIAL = ROOT / (
    "experiments/state_bottleneck/results/"
    f"structured_init_{args.init}.pt"
)

OUT = ROOT / (
    "experiments/state_bottleneck/structured_ppo_runs"
) / args.tag


class OnPolicyStatePPO(NativePPO):
    """
    Native PPO followed by one auxiliary state-supervision update.

    Auxiliary examples are sampled from the same rollout,
    using aligned actor and critic observation indices.
    """

    def update(self, *update_args, **update_kwargs):
        storage = self.storage

        if not hasattr(storage, "observations"):
            raise RuntimeError(
                "Unsupported PPO storage: no actor observations"
            )

        if not hasattr(storage, "critic_observations"):
            raise RuntimeError(
                "Unsupported PPO storage: no critic observations"
            )

        actor_obs = storage.observations
        critic_obs = storage.critic_observations

        if actor_obs.shape[-1] != 256:
            raise RuntimeError("Expected 256-D corner observations")

        if critic_obs.shape[-1] != 20:
            raise RuntimeError("Expected 20-D privileged observations")

        if actor_obs.shape[:2] != critic_obs.shape[:2]:
            raise RuntimeError("Actor/critic rollout slots misaligned")

        steps = int(
            getattr(storage, "step", actor_obs.shape[0])
        )

        if not 0 < steps <= actor_obs.shape[0]:
            raise RuntimeError("Invalid rollout storage position")

        flat_actor = actor_obs[:steps].reshape(-1, 256)
        flat_critic = critic_obs[:steps].reshape(-1, 20)

        indices = torch.randperm(
            flat_actor.shape[0],
            device=flat_actor.device,
        )[:min(args.aux_samples, flat_actor.shape[0])]

        # Both use the same selected time/environment indices.
        # Clone before native PPO potentially clears storage.
        x = flat_actor[indices].detach().clone()

        y = (
            flat_critic[indices][:, STATE_INDICES]
            .detach()
            .clone()
        )

        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite visual observations")

        if not torch.isfinite(y).all():
            raise RuntimeError("Non-finite privileged targets")

        policy = self.policy

        if not hasattr(policy, "encode_structured"):
            raise RuntimeError("Incorrect PPO policy architecture")

        # Preserve native PPO behavior.
        result = super().update(
            *update_args, **update_kwargs
        )

        normalized_target = (
            y - policy.state_target_mean
        ) / policy.state_target_std

        _, predicted, _ = policy.encode_structured(x)

        if predicted.shape != normalized_target.shape:
            raise RuntimeError("State prediction shape mismatch")

        squared_error = (
            predicted - normalized_target
        ).square()

        weights = torch.ones(
            13, device=squared_error.device
        )

        # Velocity and projected gravity receive greater weight.
        weights[3:9] = 2.0

        loss = args.aux_coef * (
            squared_error * weights
        ).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        state_grad = sum(
            float(p.grad.detach().norm())
            for p in policy.state_head.parameters()
            if p.grad is not None
        )

        tcn_grad = sum(
            float(p.grad.detach().norm())
            for p in policy.tcn.parameters()
            if p.grad is not None
        )

        if (
            not torch.isfinite(loss).item()
            or state_grad <= 0
            or tcn_grad <= 0
        ):
            raise RuntimeError(
                "Auxiliary gradients invalid: "
                f"state={state_grad} tcn={tcn_grad}"
            )

        nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=1.0
        )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        print(
            "[AUX ONPOLICY]",
            f"samples={len(x)}",
            f"state_mse={squared_error.mean().item():.6f}",
            f"state_grad={state_grad:.6f}",
            f"tcn_grad={tcn_grad:.6f}",
            flush=True,
        )

        return result


def main():
    if (
        args.iterations < 1
        or args.num_envs < 1
        or args.steps < 1
        or args.aux_samples < 1
        or args.aux_coef <= 0
    ):
        raise ValueError("Invalid training configuration")

    if OUT.exists():
        raise FileExistsError(
            f"Run directory already exists: {OUT}"
        )

    if not INITIAL.is_file():
        raise FileNotFoundError(INITIAL)

    OUT.mkdir(parents=True)

    torch.manual_seed(args.seed)

    env_cfg = CornersAsymmetricPPOEnvCfg()

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"

    env_cfg.curriculum_enabled = False
    env_cfg.forced_curriculum_level = 0.0

    env_cfg.eval_mode = False
    env_cfg.nominal_mode = False

    env_cfg.enable_onboard_camera = False
    env_cfg.visual_dr_enabled = False
    env_cfg.debug_vis = False

    train_cfg = CornersAsymmetricPPORunnerCfg()

    train_cfg.seed = args.seed
    train_cfg.device = "cuda:0"
    train_cfg.max_iterations = args.iterations
    train_cfg.num_steps_per_env = args.steps
    train_cfg.save_interval = 10
    train_cfg.run_name = args.tag

    train_cfg.policy.class_name = (
        "StructuredCornersActorCritic"
    )

    train_cfg.algorithm.learning_rate = 1e-4
    train_cfg.algorithm.clip_param = 0.1
    train_cfg.algorithm.desired_kl = 0.005
    train_cfg.algorithm.num_learning_epochs = 2
    train_cfg.algorithm.num_mini_batches = args.mini_batches
    train_cfg.algorithm.schedule = "adaptive"

    environment = None

    try:
        environment = CornersAsymmetricPPOEnv(
            cfg=env_cfg,
            render_mode=None,
        )

        wrapped = RslRlVecEnvWrapper(environment)

        runner_module.StructuredCornersActorCritic = (
            StructuredCornersActorCritic
        )

        runner_module.PPO = OnPolicyStatePPO

        runner = OnPolicyRunner(
            wrapped,
            train_cfg.to_dict(),
            log_dir=str(OUT),
            device="cuda:0",
        )

        policy = runner.alg.policy

        if not isinstance(policy, StructuredCornersActorCritic):
            raise RuntimeError("Wrong structured policy class")

        checkpoint = torch.load(
            INITIAL,
            map_location="cpu",
            weights_only=False,
        )

        result = nn.Module.load_state_dict(
            policy,
            checkpoint["model_state_dict"],
            strict=True,
        )

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Initialization weights mismatch")

        optimizer_parameters = {
            id(p)
            for group in runner.alg.optimizer.param_groups
            for p in group["params"]
        }

        if not all(
            id(p) in optimizer_parameters
            for p in policy.state_head.parameters()
        ):
            raise RuntimeError(
                "PPO optimizer omits the state head"
            )

        print(
            "[LOAD]",
            f"initialization={args.init}",
            f"environments={args.num_envs}",
            f"rollout_steps={args.steps}",
            f"iterations={args.iterations}",
            flush=True,
        )

        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=True,
        )

        final_checkpoint = OUT / "model_after_training.pt"

        runner.save(str(final_checkpoint))

        print(
            "[DONE]",
            args.init,
            final_checkpoint,
            flush=True,
        )

    finally:
        if environment is not None:
            environment.close()

        simulation_app.close()


if __name__ == "__main__":
    main()
