from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from dreamer import Dreamer
from dreamer_envs import make_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record DreamerV3 rollout and open-loop prediction videos."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument(
        "--constant-action",
        type=float,
        nargs=4,
        default=None,
        metavar=("THRUST", "ROLL", "PITCH", "YAW"),
    )
    return parser.parse_args()


def compose_config(args: argparse.Namespace):
    config_dir = str((Path(__file__).parent / "configs").resolve())

    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
    ):
        return compose(
            config_name="configs",
            overrides=[
                "env=isaaclab_quadrotor_rgb",
                "model=size12M",
                "model.rep_loss=dreamer",
                "model.compile=false",
                f"seed={args.seed}",
                f"device={args.device}",
                "env.env_num=1",
                "env.eval_episode_num=0",
            ],
        )


def add_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    origin_y: int = 24,
) -> np.ndarray:
    result = np.ascontiguousarray(frame.copy())

    for index, text in enumerate(lines):
        y = origin_y + index * 22

        cv2.putText(
            result,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return result


def resize(frame: np.ndarray, scale: int) -> np.ndarray:
    height, width = frame.shape[:2]

    return cv2.resize(
        frame,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )


def write_video(
    path: Path,
    frames: list[np.ndarray],
    fps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))


@torch.no_grad()
def collect_episode(
    envs,
    agent: Dreamer,
    *,
    max_frames: int,
    constant_action: list[float] | None = None,
) -> torch.Tensor:
    device = agent.device

    done = torch.ones(
        envs.env_num,
        dtype=torch.bool,
        device=device,
    )

    agent_state = agent.get_initial_state(envs.env_num)
    action = agent_state["prev_action"].clone()

    cache = []

    for _ in range(max_frames):
        transition, step_done = envs.step(
            action.detach(),
            done,
        )

        transition = transition.to(
            device,
            non_blocking=True,
        )
        done = step_done.to(device)

        # Match the repository evaluation convention: store the action
        # associated with the current transition before computing the next one.
        transition["action"] = action.detach().clone()
        cache.append(transition.clone())

        action, agent_state = agent.act(
            transition.clone(),
            agent_state,
            eval=True,
        )
        if constant_action is not None:
            action = torch.tensor(
                constant_action,
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0).expand(
                envs.env_num,
                -1,
            ).clone()

        if bool(done[0]) and len(cache) > 1:
            break

    if not cache:
        raise RuntimeError("No transitions were collected.")

    return torch.stack(cache, dim=1)


def to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    return (
        np.clip(array, 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_float32_matmul_precision("high")
    config = compose_config(args)

    print("[INFO] Algorithm:", config.model.rep_loss)
    print("[INFO] Task:", config.env.task)
    print("[INFO] Device:", config.device)
    print("[INFO] Creating Isaac Lab environment")

    envs = None

    try:
        envs, _, obs_space, act_space = make_envs(
            config.env
        )

        print("[INFO] Observation space:", obs_space)
        print("[INFO] Action space:", act_space)

        agent = Dreamer(
            config.model,
            obs_space,
            act_space,
        ).to(config.device)

        checkpoint = torch.load(
            args.checkpoint,
            map_location=config.device,
            weights_only=False,
        )

        agent.load_state_dict(
            checkpoint["agent_state_dict"],
            strict=True,
        )

        # Ensure inference copies share the newly loaded parameters.
        agent.clone_and_freeze()
        agent.eval()

        if agent.rep_loss != "dreamer":
            raise RuntimeError(
                f"Expected DreamerV3, got {agent.rep_loss!r}."
            )

        print("[PASS] Loaded reconstruction-based DreamerV3")

        episodes = []

        for attempt in range(args.attempts):
            rollout = collect_episode(
                envs,
                agent,
                max_frames=args.max_frames,
                constant_action=args.constant_action,
            )
            episodes.append(rollout)

            score = float(
                rollout["reward"][0, :, 0].sum().cpu()
            )

            print(
                f"[INFO] Attempt {attempt + 1}: "
                f"length={rollout.shape[1]}, "
                f"score={score:.4f}"
            )

        # Use the longest uninterrupted episode for open-loop prediction.
        rollout = max(
            episodes,
            key=lambda item: item.shape[1],
        )

        sequence_length = int(rollout.shape[1])

        if sequence_length < 6:
            raise RuntimeError(
                "Dreamer open-loop prediction needs at least "
                f"6 frames; longest episode had {sequence_length}."
            )

        rewards = (
            rollout["reward"][0, :, 0]
            .detach()
            .cpu()
            .numpy()
        )
        actions = (
            rollout["action"][0]
            .detach()
            .cpu()
            .numpy()
        )
        actual_images = (
            rollout["image"][0]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------------
        # Actual onboard-policy rollout
        # --------------------------------------------------------------
        rollout_frames = []

        for step, frame in enumerate(actual_images):
            frame = resize(
                to_uint8(frame),
                args.scale,
            )

            action = actions[step]

            frame = add_text(
                frame,
                [
                    f"Actual simulated onboard RGB | step {step}",
                    f"reward={rewards[step]:+.4f}",
                    (
                        "CTBR="
                        f"[{action[0]:+.2f}, "
                        f"{action[1]:+.2f}, "
                        f"{action[2]:+.2f}, "
                        f"{action[3]:+.2f}]"
                    ),
                ],
            )
            rollout_frames.append(frame)

        rollout_path = (
            args.output_dir
            / "dreamerv3_policy_rollout.mp4"
        )
        write_video(
            rollout_path,
            rollout_frames,
            args.fps,
        )

        # --------------------------------------------------------------
        # Decoder/open-loop world-model prediction
        # --------------------------------------------------------------
        initial = agent.get_initial_state(1)

        prediction = agent.video_pred(
            rollout.clone(),
            (
                initial["stoch"],
                initial["deter"],
            ),
        )

        # Shape: (B, T, 3*H, W, C), stacked as:
        # ground truth / model prediction / prediction error.
        prediction = (
            prediction[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        prediction_frames = []
        panel_height = actual_images.shape[1] * args.scale

        for step, frame in enumerate(prediction):
            frame = resize(
                to_uint8(frame),
                args.scale,
            )

            frame = add_text(
                frame,
                [f"GROUND TRUTH | step {step}"],
                origin_y=24,
            )
            frame = add_text(
                frame,
                [
                    (
                        "DREAMER RECONSTRUCTION"
                        if step < 5
                        else "DREAMER OPEN-LOOP PREDICTION"
                    )
                ],
                origin_y=panel_height + 24,
            )
            frame = add_text(
                frame,
                ["ABSOLUTE-STYLE ERROR VISUALIZATION"],
                origin_y=2 * panel_height + 24,
            )

            prediction_frames.append(frame)

        prediction_path = (
            args.output_dir
            / "dreamerv3_open_loop_prediction.mp4"
        )
        write_video(
            prediction_path,
            prediction_frames,
            args.fps,
        )

        np.savez_compressed(
            args.output_dir / "dreamerv3_rollout_data.npz",
            image=actual_images,
            action=actions,
            reward=rewards,
            is_first=(
                rollout["is_first"][0]
                .detach()
                .cpu()
                .numpy()
            ),
            is_last=(
                rollout["is_last"][0]
                .detach()
                .cpu()
                .numpy()
            ),
            is_terminal=(
                rollout["is_terminal"][0]
                .detach()
                .cpu()
                .numpy()
            ),
        )

        summary = {
            "checkpoint": str(
                args.checkpoint.resolve()
            ),
            "algorithm": agent.rep_loss,
            "task": str(config.env.task),
            "sequence_length": sequence_length,
            "score": float(rewards.sum()),
            "first_five_frames": (
                "posterior conditioning/reconstruction"
            ),
            "remaining_frames": (
                "open-loop latent prediction using actions"
            ),
            "policy_rollout_video": str(
                rollout_path.resolve()
            ),
            "open_loop_video": str(
                prediction_path.resolve()
            ),
        }

        (
            args.output_dir
            / "dreamerv3_video_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print()
        print("[PASS] Video validation completed")
        print("[PASS] Rollout:", rollout_path)
        print("[PASS] Open loop:", prediction_path)
        print(
            "[PASS] Sequence length:",
            sequence_length,
        )
        print(
            "[PASS] Episode score:",
            float(rewards.sum()),
        )

    finally:
        if envs is not None:
            envs.close()


if __name__ == "__main__":
    main()
