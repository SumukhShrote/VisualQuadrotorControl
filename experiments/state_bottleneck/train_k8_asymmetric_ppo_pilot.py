#!/usr/bin/env python3
"""Conservative PPO fine-tuning of the K8 DAgger Transformer."""

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
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rollout_steps", type=int, default=8)
parser.add_argument("--updates", type=int, default=1)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--level", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=7400)
parser.add_argument("--actor_encoder_lr", type=float, default=3e-6)
parser.add_argument("--actor_head_lr", type=float, default=1e-5)
parser.add_argument("--critic_lr", type=float, default=3e-4)
parser.add_argument("--anchor_coef", type=float, default=0.25)
parser.add_argument("--init_log_std", type=float, default=-3.5)
parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip", type=float, default=0.10)
parser.add_argument("--out_dir", type=str, default="")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

assert args.num_envs > 0
assert args.rollout_steps > 0
assert args.updates > 0
assert args.epochs > 0
assert args.minibatches > 0
assert args.num_envs * args.rollout_steps >= args.minibatches
assert 0 <= args.level <= 1

args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from envs.aruco_hover.quadrotor_aruco_hover_env import (
    ArucoHoverStudentEnv,
    ArucoHoverStudentEnvCfg,
)

ROOT = Path.cwd()
BASE = ROOT / "experiments/state_bottleneck"

ROUND2 = (
    BASE
    / "k8_onpolicy_dagger_round2_20261120T225553Z"
    / "finetuned_k8"
)

SOURCE = BASE / "launch_action_history_study.py"
BASE_PT = ROUND2 / "policy.pt"
BASE_TS = ROUND2 / "policy.ts"

for path in (SOURCE, BASE_PT, BASE_TS):
    assert path.is_file(), f"Missing: {path}"


def load_policy_class():
    spec = importlib.util.spec_from_file_location(
        "k8_policy_for_ppo",
        SOURCE,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.Policy


def observations(obs):
    assert isinstance(obs, dict), type(obs)

    corners = obs["policy"].detach()
    privileged = obs["teacher"].detach()

    assert corners.ndim == 2
    assert corners.shape[1] == 256

    assert privileged.ndim == 2
    assert privileged.shape[1] == 20

    return corners, privileged


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def raw_mean(policy, corners, history):
    action = policy(corners, history)

    assert action.shape[-1] == 4

    return torch.atanh(
        action.clamp(-0.999, 0.999)
    )


def action_log_prob(z, mean, std):
    normal = Normal(mean, std)

    # Tanh change-of-variables correction.
    jacobian = (
        2.0 * (
            math.log(2.0)
            - z
            - F.softplus(-2.0 * z)
        )
    ).sum(dim=-1)

    return (
        normal.log_prob(z).sum(dim=-1)
        - jacobian
    )


def check_history(history, privileged, label):
    error = (
        history[:, -1]
        - privileged[:, 15:19]
    ).abs().max().item()

    if error > 1e-5:
        raise RuntimeError(
            f"{label}: executed-action history "
            f"misalignment: {error}"
        )


def save_checkpoint(
    actor,
    critic,
    update,
    run,
    manifest,
    baseline_hash,
):
    folder = run / f"update_{update:03d}"

    folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    torch.save(
        {
            "actor_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in actor.state_dict().items()
            },
            "critic_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in critic.state_dict().items()
            },
            "k": 8,
            "update": update,
            "config": manifest,
            "baseline_sha256": baseline_hash,
        },
        folder / "checkpoint.pt",
    )

    inference_actor = copy.deepcopy(
        actor
    ).cpu().eval()

    example = (
        torch.zeros(2, 256),
        torch.zeros(2, 8, 4),
    )

    with torch.inference_mode():
        traced = torch.jit.trace(
            inference_actor,
            example,
            check_trace=True,
        )

        torch.testing.assert_close(
            traced(*example),
            inference_actor(*example),
            atol=1e-5,
            rtol=1e-5,
        )

        traced.save(
            str(folder / "policy.ts")
        )

    print(
        "[SAVED]",
        folder / "policy.ts",
        flush=True,
    )


def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_name = args.device or "cuda:0"
    device = torch.device(device_name)

    cfg = ArucoHoverStudentEnvCfg()

    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = device_name

    cfg.forced_curriculum_level = args.level

    cfg.eval_mode = False
    cfg.nominal_mode = False

    cfg.enable_onboard_camera = False
    cfg.visual_dr_enabled = False
    cfg.debug_vis = False

    assert hasattr(
        cfg,
        "tight_hover_reward_v1",
    )

    cfg.tight_hover_reward_v1 = True

    print(
        "[ENV] Tight-hover reward enabled",
        flush=True,
    )

    env = ArucoHoverStudentEnv(cfg=cfg)

    try:
        Policy = load_policy_class()

        actor = Policy(8).to(device).eval()

        saved = torch.load(
            BASE_PT,
            map_location="cpu",
            weights_only=False,
        )

        assert int(saved["k"]) == 8

        actor.load_state_dict(
            saved["model_state_dict"],
            strict=True,
        )

        actor.eval()

        frozen = copy.deepcopy(
            actor
        ).to(device).eval()

        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

        reference = torch.jit.load(
            str(BASE_TS),
            map_location=device,
        ).eval()

        critic = Critic().to(device)

        optimizer = torch.optim.Adam(
            [
                {
                    "params": actor.estimator.parameters(),
                    "lr": args.actor_encoder_lr,
                },
                {
                    "params": actor.head.parameters(),
                    "lr": args.actor_head_lr,
                },
                {
                    "params": critic.parameters(),
                    "lr": args.critic_lr,
                },
            ],
            eps=1e-5,
        )

        reset_result = env.reset()

        obs = (
            reset_result[0]
            if isinstance(reset_result, tuple)
            else reset_result
        )

        corners, privileged = observations(obs)

        n = corners.shape[0]
        assert n == args.num_envs

        history = torch.zeros(
            n, 8, 4,
            device=device,
        )

        check_history(
            history,
            privileged,
            "Initial reset",
        )

        with torch.inference_mode():
            current_actions = actor(
                corners,
                history,
            )

            baseline_actions = reference(
                corners,
                history,
            )

            error = (
                current_actions
                - baseline_actions
            ).abs().max().item()

            assert error < 1e-4, error

            assert torch.isfinite(
                current_actions
            ).all()

        print(
            "[PASS] Initialized PPO actor matches "
            f"Round 2 DAgger: maxdiff={error:.3g}",
            flush=True,
        )

        print(
            "[PASS] Previous-action history aligned",
            flush=True,
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        run = (
            Path(args.out_dir)
            if args.out_dir
            else BASE / f"k8_asymmetric_ppo_pilot_{timestamp}"
        )

        run.mkdir(
            parents=True,
            exist_ok=False,
        )

        baseline_hash = hashlib.sha256(
            BASE_TS.read_bytes()
        ).hexdigest()

        manifest = {
            "baseline": str(BASE_PT),
            "baseline_sha256": baseline_hash,
            "seed": args.seed,
            "level": args.level,
            "num_envs": n,
            "rollout_steps": args.rollout_steps,
            "updates": args.updates,
            "epochs": args.epochs,
            "minibatches": args.minibatches,
            "actor_encoder_lr": args.actor_encoder_lr,
            "actor_head_lr": args.actor_head_lr,
            "critic_lr": args.critic_lr,
            "anchor_coef": args.anchor_coef,
            "init_log_std": args.init_log_std,
            "tight_hover_reward_v1": True,
        }

        (run / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        print("[RUN]", run, flush=True)

        save_checkpoint(
            actor,
            critic,
            0,
            run,
            manifest,
            baseline_hash,
        )

        std = torch.full(
            (4,),
            math.exp(args.init_log_std),
            device=device,
        )

        gamma = args.gamma
        lam = args.gae_lambda

        metrics_file = run / "metrics.jsonl"

        for update in range(
            1,
            args.updates + 1,
        ):
            rollout = {
                name: []
                for name in (
                    "corners",
                    "history",
                    "privileged",
                    "z",
                    "logp",
                    "value",
                    "reward",
                    "done",
                )
            }

            episode_ends = 0

            # Collect fresh on-policy experience.
            with torch.inference_mode():

                for step in range(args.rollout_steps):

                    check_history(
                        history,
                        privileged,
                        f"Update {update}, step {step}",
                    )

                    mean = raw_mean(
                        actor,
                        corners,
                        history,
                    )

                    z = (
                        mean
                        + std * torch.randn_like(mean)
                    )

                    action = torch.tanh(z)

                    logp = action_log_prob(
                        z,
                        mean,
                        std,
                    )

                    value = critic(privileged)

                    result = env.step(action)

                    assert len(result) == 5

                    (
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        extras,
                    ) = result

                    done = (
                        terminated | truncated
                    ).reshape(-1).bool()

                    reward = reward.reshape(-1).float()

                    assert torch.isfinite(reward).all()

                    samples = {
                        "corners": corners,
                        "history": history,
                        "privileged": privileged,
                        "z": z,
                        "logp": logp,
                        "value": value,
                        "reward": reward,
                        "done": done,
                    }

                    for name, value_tensor in samples.items():
                        rollout[name].append(
                            value_tensor.detach().clone()
                        )

                    # Only previous executed commands
                    # may enter the next policy observation.
                    next_history = torch.roll(
                        history,
                        shifts=-1,
                        dims=1,
                    )

                    next_history[:, -1] = action

                    # Isaac Lab automatically resets done envs.
                    next_history[done] = 0.0

                    (
                        next_corners,
                        next_privileged,
                    ) = observations(next_obs)

                    check_history(
                        next_history,
                        next_privileged,
                        f"After step {step}",
                    )

                    corners = next_corners
                    privileged = next_privileged
                    history = next_history

                    episode_ends += int(
                        done.sum().item()
                    )

                bootstrap = critic(privileged)

                advantages = torch.zeros(
                    args.rollout_steps,
                    n,
                    device=device,
                )

                gae = torch.zeros(
                    n,
                    device=device,
                )

                for t in reversed(
                    range(args.rollout_steps)
                ):
                    nonterminal = (
                        1.0
                        - rollout["done"][t].float()
                    )

                    next_value = (
                        bootstrap
                        if t == args.rollout_steps - 1
                        else rollout["value"][t + 1]
                    )

                    delta = (
                        rollout["reward"][t]
                        + gamma * next_value * nonterminal
                        - rollout["value"][t]
                    )

                    gae = (
                        delta
                        + gamma * lam * nonterminal * gae
                    )

                    advantages[t] = gae

                returns = (
                    advantages
                    + torch.stack(rollout["value"])
                )

            flat = {
                name: torch.stack(items).flatten(0, 1)
                for name, items in rollout.items()
            }

            flat["advantages"] = advantages.reshape(-1)
            flat["returns"] = returns.reshape(-1)

            adv = flat["advantages"]

            flat["advantages"] = (
                (adv - adv.mean())
                / (
                    adv.std(unbiased=False)
                    + 1e-8
                )
            )

            batch_size = n * args.rollout_steps

            totals = {
                "policy": 0.0,
                "value": 0.0,
                "anchor": 0.0,
                "kl": 0.0,
            }

            num_minibatches = 0

            for epoch in range(args.epochs):

                permutation = torch.randperm(
                    batch_size,
                    device=device,
                )

                for indices in torch.tensor_split(
                    permutation,
                    args.minibatches,
                ):
                    if indices.numel() == 0:
                        continue

                    mean = raw_mean(
                        actor,
                        flat["corners"][indices],
                        flat["history"][indices],
                    )

                    logp = action_log_prob(
                        flat["z"][indices],
                        mean,
                        std,
                    )

                    log_ratio = (
                        logp
                        - flat["logp"][indices]
                    ).clamp(-20.0, 20.0)

                    ratio = log_ratio.exp()

                    advantage = flat["advantages"][indices]

                    policy_loss = -torch.minimum(
                        ratio * advantage,
                        ratio.clamp(
                            1.0 - args.clip,
                            1.0 + args.clip,
                        ) * advantage,
                    ).mean()

                    predicted_value = critic(
                        flat["privileged"][indices]
                    )

                    value_loss = (
                        0.5
                        * (
                            predicted_value
                            - flat["returns"][indices]
                        ).square().mean()
                    )

                    with torch.no_grad():
                        original_action = frozen(
                            flat["corners"][indices],
                            flat["history"][indices],
                        )

                    anchor_loss = (
                        torch.tanh(mean)
                        - original_action
                    ).square().mean()

                    loss = (
                        policy_loss
                        + 0.5 * value_loss
                        + args.anchor_coef * anchor_loss
                    )

                    assert torch.isfinite(loss)

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        list(actor.parameters())
                        + list(critic.parameters()),
                        1.0,
                    )

                    optimizer.step()

                    with torch.no_grad():
                        approx_kl = (
                            ratio - 1.0 - log_ratio
                        ).mean().item()

                    totals["policy"] += policy_loss.item()
                    totals["value"] += value_loss.item()
                    totals["anchor"] += anchor_loss.item()
                    totals["kl"] += approx_kl

                    num_minibatches += 1

            row = {
                "update": update,
                "transitions": update * batch_size,
                "mean_reward": flat["reward"].mean().item(),
                "episode_end_events": episode_ends,
            }

            for name, total in totals.items():
                row[name] = total / max(
                    num_minibatches,
                    1,
                )

            with metrics_file.open("a") as f:
                f.write(
                    json.dumps(row) + "\n"
                )

            print(
                f"[PPO] update={update:03d} "
                f"transitions={row['transitions']} "
                f"reward={row['mean_reward']:.4f} "
                f"done={episode_ends} "
                f"KL={row['kl']:.5f} "
                f"anchor={row['anchor']:.6f}",
                flush=True,
            )

            if (
                update % args.save_every == 0
                or update == args.updates
            ):
                save_checkpoint(
                    actor,
                    critic,
                    update,
                    run,
                    manifest,
                    baseline_hash,
                )

        print(
            "[PPO COMPLETE]",
            run,
            flush=True,
        )

    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
