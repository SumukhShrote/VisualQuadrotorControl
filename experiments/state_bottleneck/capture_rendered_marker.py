"""Rendered RGB measurement pilot. No policy training."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, choices=(64, 256), required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--capture-every", type=int, default=5)
parser.add_argument("--seed", type=int, default=13721)
parser.add_argument("--output", type=str, required=True)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
args.headless = True
args.enable_cameras = True

simulation_app = AppLauncher(args).app

import cv2
import numpy as np
import torch
from torch import nn

# Registration must precede the PPO environment import.
import envs

from rl.corners_asymmetric_ppo.environment import (
    CornersAsymmetricPPOEnv,
    CornersAsymmetricPPOEnvCfg,
)
from experiments.state_bottleneck.structured_init import (
    build_policy,
    PPO400,
)


def convert_rgb(frame):
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    image = np.asarray(frame)

    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected RGB image shape: {image.shape}")

    image = image[..., :3]

    if image.dtype != np.uint8:
        maximum = float(np.nanmax(image))

        if maximum <= 1.01:
            image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def detect_aruco(image):
    """Detect marker patterns without using privileged geometry."""

    if not hasattr(cv2, "aruco"):
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detections = []

    dictionaries = (
        "DICT_4X4_50",
        "DICT_5X5_50",
        "DICT_6X6_250",
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_36h11",
    )

    for name in dictionaries:
        identifier = getattr(cv2.aruco, name, None)

        if identifier is None:
            continue

        dictionary = cv2.aruco.getPredefinedDictionary(identifier)

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary
            )

        if ids is not None:
            for points, marker_id in zip(corners, ids.reshape(-1)):
                detections.append(
                    (
                        name,
                        int(marker_id),
                        points.reshape(4, 2),
                    )
                )

    return detections


def corner_error(reference, detections, resolution):
    """Match detected corner ordering to analytical reference for scoring."""

    target = reference.reshape(4, 2)

    if not ((target >= 0) & (target <= 1)).all():
        return float("nan")

    target = target * resolution
    best = float("inf")

    for _, _, corners in detections:
        for ordered in (corners, corners[::-1]):
            for shift in range(4):
                candidate = np.roll(ordered, shift, axis=0)

                error = np.linalg.norm(
                    candidate - target, axis=1
                ).mean()

                best = min(best, float(error))

    return best if np.isfinite(best) else float("nan")


def draw_overlay(image, reference, detections):
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    target = reference.reshape(4, 2)

    # Green: analytically projected corners (evaluation reference).
    if ((target >= 0) & (target <= 1)).all():
        for u, v in target:
            cv2.circle(
                canvas,
                (int(round(u * width)), int(round(v * height))),
                3,
                (0, 255, 0),
                1,
            )

    # Red: corners actually detected in rendered RGB.
    for _, _, points in detections[:5]:
        cv2.polylines(
            canvas,
            [np.round(points).astype(np.int32)],
            True,
            (0, 0, 255),
            1,
        )

    return canvas


def main():
    output = Path(args.output)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing capture: {output}"
        )

    output.mkdir(parents=True)

    cfg = CornersAsymmetricPPOEnvCfg()

    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = "cuda:0"

    cfg.curriculum_enabled = False
    cfg.forced_curriculum_level = 0.0

    # Crucial: enable actual rendered camera observations.
    cfg.enable_onboard_camera = True

    cfg.camera.width = args.resolution
    cfg.camera.height = args.resolution
    cfg.camera.data_types = ["rgb"]

    cfg.visual_dr_enabled = False
    cfg.corner_noise_px = 0.0
    cfg.corner_dropout_prob = 0.0
    cfg.debug_vis = False

    environment = None

    try:
        environment = CornersAsymmetricPPOEnv(
            cfg=cfg,
            render_mode=None,
        )

        observations, _ = environment.reset()

        camera = getattr(environment, "_camera", None)

        if camera is None:
            raise RuntimeError(
                "RGB camera was not created. "
                "Check the camera prim and _setup_scene."
            )

        print(
            "[CAMERA] resolution=",
            args.resolution,
            "num_envs=",
            args.num_envs,
            flush=True,
        )

        # Use our previously validated visual PPO400 controller.
        policy = build_policy(
            torch.device("cuda:0"),
            structured=False,
        )

        checkpoint = torch.load(
            PPO400,
            map_location="cpu",
            weights_only=False,
        )

        nn.Module.load_state_dict(
            policy,
            checkpoint["model_state_dict"],
            strict=True,
        )

        policy.eval()

        print("[POLICY] PPO400 loaded", flush=True)

        all_images = []
        all_corners = []
        all_states = []
        all_steps = []

        resolutions = sorted(set((64, args.resolution)))

        detection_counts = {r: [] for r in resolutions}
        detection_errors = {r: [] for r in resolutions}

        for step in range(args.steps):
            with torch.inference_mode():
                actions = policy.act_inference(
                    observations["policy"]
                )

            (
                observations,
                rewards,
                terminated,
                truncated,
                extras,
            ) = environment.step(actions)

            if (step + 1) % args.capture_every != 0:
                continue

            camera_output = camera.data.output

            if "rgb" not in camera_output:
                raise RuntimeError(
                    "Camera has no RGB output. "
                    f"Available outputs: {list(camera_output)}"
                )

            rgb = camera_output["rgb"]

            images = np.stack([
                convert_rgb(rgb[i])
                for i in range(args.num_envs)
            ])

            if images.shape[1:3] != (
                args.resolution,
                args.resolution,
            ):
                raise RuntimeError(
                    f"Unexpected image shape: {images.shape}"
                )

            corners = (
                observations["policy"][:, -8:]
                .detach()
                .cpu()
                .numpy()
            )

            if "teacher" not in observations:
                raise RuntimeError(
                    f"Missing privileged labels: "
                    f"{list(observations)}"
                )

            states = (
                observations["teacher"]
                .detach()
                .cpu()
                .numpy()
            )

            if states.shape != (args.num_envs, 20):
                raise RuntimeError(
                    f"Unexpected teacher labels: {states.shape}"
                )

            all_images.append(images)
            all_corners.append(corners)
            all_states.append(states)
            all_steps.append(step + 1)

            if len(all_images) == 1:
                print(
                    "[RGB]",
                    "shape=", images.shape,
                    "min=", int(images.min()),
                    "max=", int(images.max()),
                    "channel_std=",
                    images.std(axis=(0, 1, 2)).round(2),
                    flush=True,
                )

            for env_id, image in enumerate(images):
                for resolution in resolutions:

                    if resolution == args.resolution:
                        resized = image
                    else:
                        resized = cv2.resize(
                            image,
                            (resolution, resolution),
                            interpolation=cv2.INTER_AREA,
                        )

                    detections = detect_aruco(resized)

                    detection_counts[resolution].append(
                        len(detections)
                    )

                    detection_errors[resolution].append(
                        corner_error(
                            corners[env_id],
                            detections,
                            resolution,
                        )
                    )

                    if env_id == 0 and len(all_images) in (1, 3, 8):

                        filename = (
                            output
                            / f"step_{step+1:03d}_{resolution}"
                        )

                        cv2.imwrite(
                            str(filename) + "_rgb.png",
                            cv2.cvtColor(
                                resized,
                                cv2.COLOR_RGB2BGR,
                            ),
                        )

                        cv2.imwrite(
                            str(filename) + "_overlay.png",
                            draw_overlay(
                                resized,
                                corners[env_id],
                                detections,
                            ),
                        )

            print(
                "[CAPTURE]",
                f"step={step+1}",
                f"images={len(all_images)*args.num_envs}",
                flush=True,
            )

        np.savez_compressed(
            output / "rendered_observations.npz",
            rgb=np.stack(all_images),
            analytic_corners=np.stack(all_corners),
            teacher_observations=np.stack(all_states),
            policy_steps=np.asarray(all_steps),
            resolution=np.int32(args.resolution),
        )

        for resolution in resolutions:
            counts = np.asarray(
                detection_counts[resolution]
            )

            errors = np.asarray(
                detection_errors[resolution]
            )

            errors = errors[np.isfinite(errors)]

            print(
                "[DETECTION]",
                f"resolution={resolution}",
                f"frames={len(counts)}",
                f"detected={int((counts > 0).sum())}",
                f"detection_fraction={(counts > 0).mean():.4f}",
                f"mean_corner_error_px="
                f"{errors.mean() if len(errors) else float('nan'):.5f}",
                flush=True,
            )

        print(
            "[DONE]",
            output / "rendered_observations.npz",
            flush=True,
        )

    finally:
        if environment is not None:
            environment.close()

        simulation_app.close()


if __name__ == "__main__":
    main()
