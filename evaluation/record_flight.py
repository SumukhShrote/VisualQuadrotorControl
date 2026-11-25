"""Record visual quadrotor policies across curriculum levels."""

import argparse
import json
import shutil
import subprocess
import hashlib
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=40001)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--camera-offset", nargs=3, type=float,
        default=[-0.55, -0.85, 0.45],
        metavar=("X", "Y", "Z"),
    )

    parser.add_argument(
        "--levels", nargs="+", type=float,
        default=[0., .25, .5, .75, 1.]
    )

    parser.add_argument(
        "--domains", nargs="+",
        choices=["original", "expanded"],
        default=["original", "expanded"]
    )

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = True
    args.num_envs = 1

    policy_path = Path(args.policy).resolve(strict=True)

    if not shutil.which("ffmpeg"):
        parser.error("FFmpeg is required")

    launcher = AppLauncher(args)
    env = None

    try:
        import cv2
        import numpy as np
        import torch
        import gymnasium as gym
        import envs

        from isaaclab_tasks.utils import parse_env_cfg
        from envs.aruco_hover.quadrotor_aruco_hover_env import (
            ArucoHoverTeacherEnv,
        )
        from rl import pose_attention_runtime as runtime

        runtime.configure(args)
        runtime.patch_reset(ArucoHoverTeacherEnv)
        runtime.RESET["scale"] = 1.0

        policy = torch.jit.load(
            str(policy_path), map_location=args.device
        ).eval()

        cfg = parse_env_cfg(
            runtime.TASK,
            device=args.device,
            num_envs=1,
            use_fabric=True,
        )

        cfg.enable_onboard_camera = False
        cfg.visual_dr_enabled = False
        cfg.debug_vis = False
        cfg.episode_length_s = args.duration
        cfg.viewer.resolution = (args.width, args.height)
        cfg.viewer.origin_type = "world"

        if hasattr(cfg, "eval_mode"):
            cfg.eval_mode = True

        env = gym.make(
            runtime.TASK,
            cfg=cfg,
            render_mode="rgb_array",
        )

        dt = env.unwrapped.step_dt
        stride = max(1, round(1 / (args.fps * dt)))
        fps = 1 / (stride * dt)
        horizon = env.unwrapped.max_episode_length - 1

        records = []

        import isaaclab.sim as sim_utils
        from isaaclab.markers import (
            VisualizationMarkers,
            VisualizationMarkersCfg,
        )

        hover_visual = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/RecordingHoverTarget",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.018,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),
                        ),
                    ),
                },
            )
        )

        hover_visual.set_visibility(True)

        for domain in args.domains:
            output = runtime.OUT / f"{domain}_curriculum.mp4"

            writer = subprocess.Popen([
                "ffmpeg", "-hide_banner",
                "-loglevel", "error", "-n",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{args.width}x{args.height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output),
            ], stdin=subprocess.PIPE)

            try:
                for level in args.levels:
                    seed = args.seed + round(level * 100)

                    obs, history = runtime.reset(
                        env, level, domain == "expanded", seed
                    )

                    hover_visual.visualize(
                        translations=env.unwrapped._hover_point_w
                    )

                    robot = getattr(env.unwrapped, "_robot", None)

                    if robot is None:
                        assets = env.unwrapped.scene.articulations
                        assert len(assets) == 1, list(assets)
                        robot = next(iter(assets.values()))

                    camera_offset = np.asarray(
                        args.camera_offset, dtype=float
                    )

                    def follow_camera():
                        position = (
                            robot.data.root_pos_w[0]
                            .detach().cpu().numpy()
                        )

                        hover = (
                            env.unwrapped._hover_point_w[0]
                            .detach().cpu().numpy()
                        )

                        marker = (
                            env.unwrapped._marker_point_w[0]
                            .detach().cpu().numpy()
                        )

                        # Horizontal direction from drone to marker.
                        forward = marker - position
                        forward[2] = 0.0
                        forward /= max(np.linalg.norm(forward), 1e-6)

                        up = np.array([0., 0., 1.])
                        lateral = np.cross(up, forward)

                        # Follow from behind, to the side and above.
                        eye = (
                            position
                            + camera_offset[0] * forward
                            + camera_offset[1] * lateral
                            + camera_offset[2] * up
                        )

                        # Frame both the drone and the marker.
                        target = position + 0.45 * (marker - position)

                        env.unwrapped.sim.set_camera_view(
                            eye.tolist(),
                            target.tolist(),
                            camera_prim_path=cfg.viewer.cam_prim_path,
                        )

                    follow_camera()
                    env.unwrapped.sim.render()

                    print(
                        "[FOLLOW CAMERA]",
                        "robot=", robot.data.root_pos_w[0].tolist(),
                        "offset=", camera_offset.tolist(),
                        "camera=", cfg.viewer.cam_prim_path,
                        flush=True,
                    )

                    frames = 0
                    terminated = False

                    print(
                        f"[RECORDING] {domain} L={level} seed={seed}",
                        flush=True,
                    )

                    with torch.no_grad():
                        for step in range(horizon):

                            if step % stride == 0:
                                state = obs["teacher"][0]

                                error = float(state[:3].norm())
                                speed = float(state[3:6].norm())

                                follow_camera()
                                env.unwrapped.sim.render()
                                image = env.render()

                                if image is None:
                                    raise RuntimeError(
                                        "RGB rendering unavailable"
                                    )

                                if torch.is_tensor(image):
                                    image = image.cpu().numpy()

                                image = np.asarray(image)

                                if image.ndim != 3:
                                    raise RuntimeError(
                                        f"Invalid image: {image.shape}"
                                    )

                                if image.dtype != np.uint8:
                                    if image.max() <= 1:
                                        image = image * 255
                                    image = image.clip(0, 255).astype(np.uint8)

                                image = cv2.cvtColor(
                                    image[:, :, :3],
                                    cv2.COLOR_RGB2BGR
                                )

                                image = cv2.resize(
                                    image, (args.width, args.height)
                                )

                                cv2.rectangle(
                                    image, (10, 10), (760, 125),
                                    (25, 25, 25), -1
                                )

                                labels = [
                                    f"{domain.upper()} | Curriculum {level:.2f}",
                                    f"Time: {step * dt:.2f} s",
                                    f"Position error: {error:.3f} m",
                                    f"Speed: {speed:.3f} m/s",
                                ]

                                for i, label in enumerate(labels):
                                    cv2.putText(
                                        image, label,
                                        (22, 33 + 27*i),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        .65, (255, 255, 255), 1,
                                        cv2.LINE_AA,
                                    )

                                corners = (
                                    obs["policy"][0, -8:]
                                    .reshape(4, 2).cpu().numpy()
                                )

                                inset = np.full(
                                    (220, 220, 3), 35, np.uint8
                                )

                                visible = (
                                    np.isfinite(corners).all()
                                    and np.abs(corners).sum() > 1e-6
                                    and (corners >= 0).all()
                                    and (corners <= 1).all()
                                )

                                if visible:
                                    pts = (
                                        corners * 219
                                    ).astype(np.int32)

                                    cv2.polylines(
                                        inset, [pts], True,
                                        (255, 220, 60), 2
                                    )

                                    for i, point in enumerate(pts):
                                        x, y = map(int, point)

                                        cv2.circle(
                                            inset, (x, y), 4,
                                            (60, 220, 255), -1
                                        )

                                        cv2.putText(
                                            inset, str(i),
                                            (x+5, y-5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            .45, (255,255,255), 1
                                        )

                                x0 = args.width - 240

                                image[50:270, x0:x0+220] = inset

                                cv2.putText(
                                    image, "Policy corner input",
                                    (x0, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    .55, (255,255,255), 1
                                )

                                writer.stdin.write(
                                    np.ascontiguousarray(image).tobytes()
                                )

                                frames += 1

                            action = policy(
                                obs["policy"], history
                            ).clamp(-1, 1)

                            obs, _, term, trunc, _ = env.step(action)

                            done = term | trunc

                            history = runtime.shift(
                                history, action, done
                            )

                            if bool(done[0]):
                                terminated = bool(term[0])
                                break

                    result = {
                        "domain": domain,
                        "level": level,
                        "seed": seed,
                        "steps": step + 1,
                        "frames": frames,
                        "terminated": terminated,
                        "full_horizon": (
                            step + 1 == horizon and not terminated
                        ),
                    }

                    records.append(result)
                    print(json.dumps(result), flush=True)

            finally:
                writer.stdin.close()
                status = writer.wait()

                if status:
                    raise RuntimeError(
                        f"FFmpeg failed with status {status}"
                    )

            print("[VIDEO COMPLETE]", output)

        manifest = {
            "policy": str(policy_path),
            "policy_sha256": hashlib.sha256(
                policy_path.read_bytes()
            ).hexdigest(),
            "fps": fps,
            "episodes": records,
        }

        (runtime.OUT / "video_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

        print("[PASS] Recording complete")

    finally:
        if env is not None:
            env.close()

        launcher.app.close()


if __name__ == "__main__":
    main()
