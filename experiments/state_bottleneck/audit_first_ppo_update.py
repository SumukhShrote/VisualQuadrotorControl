#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=6300)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app = AppLauncher(args).app

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from envs.aruco_hover.quadrotor_aruco_hover_env import (
    ArucoHoverStudentEnv,
    ArucoHoverStudentEnvCfg,
)

ROOT = Path.cwd()

RUN = (
    ROOT
    / "experiments/state_bottleneck"
    / "precision_no_anchor_control_20261122T021745Z"
    / "model"
)

device = args.device or "cuda:0"

torch.manual_seed(args.seed)

baseline = torch.jit.load(
    str(RUN / "policy_initial.ts"),
    map_location=device,
).eval()

updated = torch.jit.load(
    str(RUN / "model_0.ts"),
    map_location=device,
).eval()

checkpoint = torch.load(
    RUN / "model_0.pt",
    map_location="cpu",
    weights_only=False,
)

state = checkpoint["model_state_dict"]

std0 = torch.full(
    (4,), 0.03, device=device
)

if "std" in state:
    std1 = state["std"].detach().to(device).flatten()
    assert std1.numel() == 4
    assert (std1 > 0).all()
else:
    print(
        "[WARNING] Trained std not found; assuming 0.03",
        flush=True,
    )
    std1 = std0.clone()

print("[STD] original:", std0.tolist(), flush=True)
print("[STD] updated:", std1.tolist(), flush=True)

cfg = ArucoHoverStudentEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.sim.device = device

cfg.eval_mode = True
cfg.nominal_mode = False
cfg.forced_curriculum_level = args.level
cfg.enable_onboard_camera = False
cfg.visual_dr_enabled = False
cfg.debug_vis = False

env = ArucoHoverStudentEnv(cfg=cfg)

records = {
    "action_delta_max": [],
    "action_delta_l2": [],
    "gaussian_kl": [],
    "initial_near_limit": [],
    "updated_near_limit": [],
    "logprob_error_original": [],
    "logprob_error_updated": [],
    "sample_saturation_original": [],
    "sample_saturation_updated": [],
}


def log_jacobian(z):
    return (
        2.0 * (
            math.log(2.0)
            - z
            - F.softplus(-2.0 * z)
        )
    ).sum(dim=-1)


def logprob_roundtrip_error(raw_mean, std):
    z = raw_mean + std * torch.randn_like(raw_mean)

    action = torch.tanh(z)

    reconstructed = torch.atanh(
        action.clamp(-0.999999, 0.999999)
    )

    distribution = Normal(raw_mean, std)

    actual = (
        distribution.log_prob(z).sum(dim=-1)
        - log_jacobian(z)
    )

    recovered = (
        distribution.log_prob(reconstructed).sum(dim=-1)
        - log_jacobian(reconstructed)
    )

    error = (actual - recovered).abs()

    saturated = (
        action.abs() >= 0.999999
    ).any(dim=-1).float()

    return error, saturated


def record(name, tensor):
    records[name].append(
        tensor.detach().cpu().reshape(-1)
    )


def report(name):
    values = torch.cat(records[name]).float()

    p50 = torch.quantile(values, 0.50).item()
    p95 = torch.quantile(values, 0.95).item()
    p99 = torch.quantile(values, 0.99).item()

    print(
        f"{name:<28} "
        f"mean={values.mean().item():.6f} "
        f"p50={p50:.6f} "
        f"p95={p95:.6f} "
        f"p99={p99:.6f} "
        f"max={values.max().item():.6f}"
    )


try:
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset

    history = torch.zeros(
        args.num_envs,
        8,
        4,
        device=device,
    )

    with torch.inference_mode():

        for step in range(args.steps):
            corners = obs["policy"]
            privileged = obs["teacher"]

            assert corners.shape == (
                args.num_envs, 256
            )
            assert privileged.shape == (
                args.num_envs, 20
            )

            error = (
                history[:, -1]
                - privileged[:, 15:19]
            ).abs().max().item()

            assert error < 1e-5, (
                f"History misalignment: {error}"
            )

            a0 = baseline(corners, history)
            a1 = updated(corners, history)

            assert a0.shape == a1.shape == (
                args.num_envs, 4
            )

            delta = a1 - a0

            record(
                "action_delta_max",
                delta.abs().max(dim=-1).values,
            )

            record(
                "action_delta_l2",
                torch.linalg.norm(delta, dim=-1),
            )

            record(
                "initial_near_limit",
                (a0.abs() >= 0.99).any(dim=-1).float(),
            )

            record(
                "updated_near_limit",
                (a1.abs() >= 0.99).any(dim=-1).float(),
            )

            raw0 = torch.atanh(
                a0.clamp(-0.999999, 0.999999)
            )

            raw1 = torch.atanh(
                a1.clamp(-0.999999, 0.999999)
            )

            # Exact Gaussian KL for the pre-tanh
            # distributions, using recorded std values.
            # Tanh preserves KL in exact arithmetic.
            kl = (
                torch.log(std1 / std0)
                + (
                    std0.square()
                    + (raw0 - raw1).square()
                ) / (2.0 * std1.square())
                - 0.5
            ).sum(dim=-1)

            record("gaussian_kl", kl)

            error0, saturation0 = (
                logprob_roundtrip_error(raw0, std0)
            )

            error1, saturation1 = (
                logprob_roundtrip_error(raw1, std1)
            )

            record("logprob_error_original", error0)
            record("logprob_error_updated", error1)

            record(
                "sample_saturation_original",
                saturation0,
            )

            record(
                "sample_saturation_updated",
                saturation1,
            )

            # Advance with the ORIGINAL DAgger policy.
            # This audits behavior on actual reference-
            # policy rollout observations.
            (
                next_obs,
                reward,
                terminated,
                truncated,
                extras,
            ) = env.step(a0)

            done = (
                terminated | truncated
            ).reshape(-1).bool()

            history = torch.roll(
                history,
                shifts=-1,
                dims=1,
            )

            history[:, -1] = a0
            history[done] = 0.0

            obs = next_obs

            if (step + 1) % 100 == 0:
                print(
                    f"[AUDIT] {step+1}/{args.steps}",
                    flush=True,
                )

    print()
    print("=" * 100)
    print(
        f"FIRST PPO UPDATE — REAL OBSERVATION AUDIT "
        f"LEVEL {args.level}"
    )
    print("=" * 100)

    for name in records:
        report(name)

    print()
    print("[PASS] Real-observation audit completed")

finally:
    env.close()
    app.close()
