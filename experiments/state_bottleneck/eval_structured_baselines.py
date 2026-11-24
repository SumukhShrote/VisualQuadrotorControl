"""Evaluate original PPO400 and structured visual initializations."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["original", "ppo400", "distill"],
    required=True,
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=11300)
parser.add_argument("--level", type=float, default=0.0)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from torch import nn

# Initialize environment registration before importing the PPO
# environment. This avoids the circular package import.
import envs  # noqa: F401

from rl.corners_asymmetric_ppo.environment import (
    CornersAsymmetricPPOEnv,
    CornersAsymmetricPPOEnvCfg,
)

from experiments.state_bottleneck.structured_init import (
    PPO400,
    ROOT,
    build_policy,
)

RESULTS = ROOT / "experiments/state_bottleneck/results"


def load_model(mode, device):
    structured = mode != "original"
    model = build_policy(device, structured=structured)

    if mode == "original":
        path = PPO400
    else:
        path = RESULTS / f"structured_init_{mode}.pt"

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    # Use native PyTorch loader, not RSL-RL's bool-returning
    # load_state_dict override.
    nn.Module.load_state_dict(
        model,
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print("[LOAD] mode=", mode, flush=True)
    print("[LOAD] checkpoint=", path, flush=True)

    return model


def main():
    device = torch.device("cuda:0")

    cfg = CornersAsymmetricPPOEnvCfg()

    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = str(device)

    cfg.curriculum_enabled = False
    cfg.forced_curriculum_level = args.level

    cfg.eval_mode = False
    cfg.nominal_mode = False

    cfg.enable_onboard_camera = False
    cfg.visual_dr_enabled = False
    cfg.debug_vis = False

    env = None

    try:
        env = CornersAsymmetricPPOEnv(
            cfg=cfg,
            render_mode=None,
        )

        observations, _ = env.reset()

        # Evaluation must begin at the start of a complete episode.
        if not hasattr(env, "episode_length_buf"):
            raise RuntimeError("Cannot inspect episode timers")

        horizon = int(env.max_episode_length)

        print(
            "[HORIZON] max_episode_length=",
            horizon,
            flush=True,
        )

        if not 498 <= horizon <= 502:
            raise RuntimeError(
                f"Unexpected episode horizon: {horizon}"
            )

        initial_timers = env.episode_length_buf.detach().clone()

        print(
            "[TIMER BEFORE] min=",
            int(initial_timers.min().item()),
            "max=",
            int(initial_timers.max().item()),
            "nonzero=",
            int((initial_timers != 0).sum().item()),
            flush=True,
        )

        # Remove training-time timer staggering from evaluation.
        # Physical state and observation histories are unchanged.
        env.episode_length_buf.zero_()

        print(
            "[TIMER AFTER] max=",
            int(env.episode_length_buf.max().item()),
            flush=True,
        )

        if "policy" not in observations:
            raise RuntimeError(
                f"Missing policy observations: "
                f"{list(observations)}"
            )

        corners = observations["policy"]

        if corners.shape != (args.num_envs, 256):
            raise RuntimeError(
                f"Unexpected observation shape: "
                f"{tuple(corners.shape)}"
            )

        model = load_model(args.mode, device)

        # Verify parity against original PPO400 on the
        # actual simulator observations, not just offline data.
        reference = None

        if args.mode == "ppo400":
            reference = load_model("original", device)

        finished = torch.zeros(
            args.num_envs,
            dtype=torch.bool,
            device=device,
        )

        timed_out = torch.zeros_like(finished)

        episode_steps = torch.zeros(
            args.num_envs,
            dtype=torch.long,
            device=device,
        )

        episode_returns = torch.zeros(
            args.num_envs,
            device=device,
        )

        max_action_difference = 0.0

        for step in range(550):

            with torch.no_grad():
                corners = observations["policy"]

                actions = model.act_inference(corners)

                if actions.shape != (args.num_envs, 4):
                    raise RuntimeError(
                        f"Unexpected action shape: "
                        f"{tuple(actions.shape)}"
                    )

                if not torch.isfinite(actions).all():
                    raise RuntimeError("Non-finite actions")

                if reference is not None:
                    original_actions = (
                        reference.act_inference(corners)
                    )

                    difference = (
                        actions - original_actions
                    ).abs().max().item()

                    max_action_difference = max(
                        max_action_difference,
                        difference,
                    )

                    if difference > 1e-7:
                        raise RuntimeError(
                            f"Action parity failed: {difference}"
                        )

            (
                observations,
                rewards,
                terminated,
                truncated,
                _,
            ) = env.step(actions)

            rewards = rewards.reshape(-1)
            terminated = terminated.reshape(-1).bool()
            truncated = truncated.reshape(-1).bool()

            active = ~finished

            episode_returns[active] += rewards[active]
            episode_steps[active] += 1

            newly_finished = active & (
                terminated | truncated
            )

            timed_out[newly_finished] = (
                truncated[newly_finished]
                & ~terminated[newly_finished]
            )

            finished |= newly_finished

            if finished.all():
                break

        if not finished.all():
            raise RuntimeError(
                f"Only {finished.sum().item()}/"
                f"{args.num_envs} episodes completed"
            )

        # A timeout is only a success after a full episode.
        timeout_lengths = episode_steps[timed_out]

        if timeout_lengths.numel():
            print(
                "[TIMEOUT LENGTHS] min=",
                int(timeout_lengths.min().item()),
                "max=",
                int(timeout_lengths.max().item()),
                flush=True,
            )

            if (timeout_lengths < horizon - 1).any():
                raise RuntimeError(
                    "Premature timeouts detected. "
                    "Survival results are invalid."
                )

        successes = int(timed_out.sum().item())
        failures = args.num_envs - successes

        print()
        print("=" * 70)
        print("[RESULT] mode=", args.mode)
        print("[RESULT] seed=", args.seed)
        print("[RESULT] curriculum_level=", args.level)

        print(
            f"[EVAL] Initial episodes: "
            f"timeouts={successes}, "
            f"deaths={failures}, "
            f"total={args.num_envs}"
        )

        print(
            "[RESULT] mean_initial_length=",
            episode_steps.float().mean().item()
        )

        print(
            "[RESULT] mean_initial_return=",
            episode_returns.mean().item()
        )

        if reference is not None:
            print(
                "[PARITY] max_action_difference=",
                max_action_difference,
            )

        print("[DONE]", args.mode)
        print("=" * 70, flush=True)

    finally:
        if env is not None:
            env.close()

        simulation_app.close()


if __name__ == "__main__":
    main()
