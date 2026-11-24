import argparse
import sys
from isaaclab.app import AppLauncher

import math
import collections
from pathlib import Path

from utils import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-QuadrotorHover-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--save_prefix", type=str, default="", help="Prefix for saving files.")
parser.add_argument("--follow_robot", type=int, default=-1, help="Follow robot index.")
parser.add_argument("--debug_bbox", action="store_true", default=False, help="Draw GT bounding box on onboard camera video for debugging.")
parser.add_argument("--debug_corners", action="store_true", default=False, help="Draw the latest four projected corners directly from the 32-step policy observation.")
parser.add_argument("--use_gru", type=lambda x: x.lower() == "true", default=False, help="Use GRU temporal memory in student policy — must match the trained checkpoint.")
parser.add_argument("--student_checkpoint", type=str, default=None)
parser.add_argument("--rsl_checkpoint", type=str, default=None)
parser.add_argument(
    "--rgb_ablation",
    choices=("normal", "neutral"),
    default="normal",
)
parser.add_argument("--curriculum_level", type=float, default=None, help=("Freeze evaluation reset difficulty at a value in [0, 1]."))

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# The current student environment still constructs a TiledCamera even though
# the corners policy does not consume RGB. Keep camera support enabled until
# camera construction is made optional in the environment.
# RGB rendering is needed only when recording video.
args_cli.enable_cameras = bool(
    args_cli.video
)

args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import time
import yaml
import cv2
import numpy as np
import torch

import gymnasium as gym
import envs  # registers task

from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.dict import print_dict
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import rsl_rl.runners.on_policy_runner as opr
from rl.corners_ppo import CornersAnchoredPPO
import rsl_rl.algorithms as rsl_rl_algorithms
from envs.aruco_hover.agents.vision_student_teacher import VisionStudentTeacher, VisionActorCritic, CornersTCNActorCritic
opr.VisionStudentTeacher = VisionStudentTeacher
opr.VisionActorCritic = VisionActorCritic
opr.CornersTCNActorCritic = CornersTCNActorCritic

from controllers.corners_asymmetric_policy import (
    CornersAsymmetricActorCritic,
)

opr.CornersAsymmetricActorCritic = (
    CornersAsymmetricActorCritic
)

from isaaclab.utils.io import load_yaml


# =============================================================================
#  ONBOARD CAMERA RECORDER
# =============================================================================

class OnboardCameraRecorder:
    """
    Records the onboard TiledCamera RGB feed to an MP4, optionally overlaying
    the vision network's bbox prediction.

    _last_marker is a raw 4-D learned vector — NOT guaranteed [0,1].
    We interpret the first two values as (cx, cy) and use sigmoid to map them
    to [0,1] image coords. We draw a fixed-size crosshair since w/h are
    not reliably meaningful without supervised bbox training.
    """

    def __init__(self, output_path: str, img_h: int = 128, img_w: int = 128,
                 fps: int = 50, upscale: int = 4):
        self.img_h   = img_h
        self.img_w   = img_w
        self.out_h   = img_h * upscale
        self.out_w   = img_w * upscale
        self.fps     = fps
        self.upscale = upscale

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.out_w, self.out_h)
        )
        print(f"[OnboardRecorder] Writing to: {output_path}  "
              f"({self.out_w}x{self.out_h} @ {fps} fps)")

    def write_frame(
        self,
        rgb_nhwc: torch.Tensor,
        feature_map: torch.Tensor | None = None,
        env_idx: int = 0,
        bbox_gt: tuple | None = None,
        bbox_pred: tuple | None = None,
        corners_obs=None,
    ):
        """
        Args:
            rgb_nhwc:    (N, H, W, 3) uint8 from TiledCamera
            feature_map: (N, C, H', W') float32 from backbone
            env_idx:     which env to record
            bbox_gt:     (cx_n, cy_n, w_n, h_n) normalized [0,1] — drawn in GREEN
            bbox_pred:   (cx_n, cy_n, w_n, h_n, conf) normalized [0,1] — student's own
                         prediction from _last_marker (post-Sigmoid). Drawn in ORANGE.
                         conf is the network's own visibility confidence estimate.
        """
        import cv2
        import numpy as np

        # --- base frame ---------------------------------------------------------
        frame = rgb_nhwc[env_idx].cpu().numpy()       # (H, W, 3) uint8 RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.upscale > 1:
            frame_bgr = cv2.resize(frame_bgr, (self.out_w, self.out_h),
                                interpolation=cv2.INTER_NEAREST)

        # --- heatmap overlay ----------------------------------------------------
        if feature_map is not None:
            feat = feature_map[env_idx].cpu().float()    # (C, H', W')
            heat = feat.mean(0).numpy()                  # (H', W')

            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

            heat_resized = cv2.resize(heat, (self.out_w, self.out_h),
                                    interpolation=cv2.INTER_CUBIC)
            heat_color = cv2.applyColorMap(
                (heat_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            frame_bgr = cv2.addWeighted(frame_bgr, 0.6, heat_color, 0.4, 0)

            max_idx = np.unravel_index(heat_resized.argmax(), heat_resized.shape)
            py, px = int(max_idx[0]), int(max_idx[1])
            cv2.drawMarker(frame_bgr, (px, py), (255, 255, 255),
                        cv2.MARKER_CROSS, markerSize=20,
                        thickness=2, line_type=cv2.LINE_AA)
            cv2.putText(frame_bgr,
                        f"max act: {heat.max():.2f}  ({px},{py})",
                        (4, self.out_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # --- bbox overlay (drawn AFTER upscale at full output resolution) -------
        if bbox_gt is not None:
            cx_n, cy_n, w_n, h_n = bbox_gt
            cx_px = int(cx_n * self.out_w)
            cy_px = int(cy_n * self.out_h)
            w_px  = max(8, int(w_n * self.out_w))
            h_px  = max(8, int(h_n * self.out_h))
            cv2.rectangle(frame_bgr,
                        (cx_px - w_px // 2, cy_px - h_px // 2),
                        (cx_px + w_px // 2, cy_px + h_px // 2),
                        (0, 255, 0), 2)
            cv2.drawMarker(frame_bgr, (cx_px, cy_px), (0, 255, 0),
                        cv2.MARKER_CROSS, markerSize=30, thickness=2,
                        line_type=cv2.LINE_AA)
            cv2.putText(frame_bgr,
                        f"cx={cx_n:.2f} cy={cy_n:.2f} w={w_n:.3f}",
                        (4, self.out_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
        
        # --- predicted bbox overlay (student's own output, drawn in ORANGE) -----
        # BGR orange = (0, 140, 255)
        if bbox_pred is not None:
            cx_p, cy_p, w_p, h_p, conf_p = bbox_pred
            cx_px = int(cx_p * self.out_w)
            cy_px = int(cy_p * self.out_h)
            w_px  = max(8, int(w_p * self.out_w))
            h_px  = max(8, int(h_p * self.out_h))
            cv2.rectangle(frame_bgr,
                        (cx_px - w_px // 2, cy_px - h_px // 2),
                        (cx_px + w_px // 2, cy_px + h_px // 2),
                        (0, 140, 255), 2)
            cv2.drawMarker(frame_bgr, (cx_px, cy_px), (0, 140, 255),
                        cv2.MARKER_TILTED_CROSS, markerSize=24, thickness=2,
                        line_type=cv2.LINE_AA)
            cv2.putText(frame_bgr,
                        f"pred cx={cx_p:.2f} cy={cy_p:.2f} conf={conf_p:.2f}",
                        (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 140, 255), 2, cv2.LINE_AA)
        
        # ----------------------------------------------------------------------
        # Projected-corner overlay from the exact policy observation
        # ----------------------------------------------------------------------
        if corners_obs is not None:
            corners_array = np.asarray(
                corners_obs,
                dtype=np.float32,
            ).reshape(4, 2)

            # Preserve an image without projected-corner drawings for the zoom inset.
            zoom_source = frame_bgr.copy()

            # BGR colors.
            corner_colors = [
                (255, 0, 0),      # 0: blue
                (0, 255, 255),    # 1: yellow
                (0, 0, 255),      # 2: red
                (255, 0, 255),    # 3: magenta
            ]

            # Position labels away from the center of the marker.
            label_offsets = [
                (-18, -8),
                (8, -8),
                (8, 18),
                (-18, 18),
            ]

            pixel_points = [None, None, None, None]

            for corner_index, (u_n, v_n) in enumerate(corners_array):
                valid = (
                    np.isfinite(u_n)
                    and np.isfinite(v_n)
                    and 0.0 <= u_n <= 1.0
                    and 0.0 <= v_n <= 1.0
                )

                if not valid:
                    continue

                x_px = int(round(float(u_n) * (self.out_w - 1)))
                y_px = int(round(float(v_n) * (self.out_h - 1)))

                pixel_points[corner_index] = (x_px, y_px)

            all_corners_valid = all(
                point is not None
                for point in pixel_points
            )

            # ------------------------------------------------------------------
            # Small overlay on the complete onboard image
            # ------------------------------------------------------------------
            if all_corners_valid:
                polygon = np.asarray(
                    pixel_points,
                    dtype=np.int32,
                ).reshape(-1, 1, 2)

                cv2.polylines(
                    frame_bgr,
                    [polygon],
                    isClosed=True,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            for corner_index, point in enumerate(pixel_points):
                if point is None:
                    continue

                color = corner_colors[corner_index]

                # Small cross instead of large overlapping circles.
                cv2.drawMarker(
                    frame_bgr,
                    point,
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=8,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )

                label_x = point[0] + label_offsets[corner_index][0]
                label_y = point[1] + label_offsets[corner_index][1]

                cv2.putText(
                    frame_bgr,
                    str(corner_index),
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            valid_points = [
                point
                for point in pixel_points
                if point is not None
            ]

            # ------------------------------------------------------------------
            # Magnified marker inset
            # ------------------------------------------------------------------
            if valid_points:
                point_array = np.asarray(
                    valid_points,
                    dtype=np.float32,
                )

                min_x = float(point_array[:, 0].min())
                max_x = float(point_array[:, 0].max())
                min_y = float(point_array[:, 1].min())
                max_y = float(point_array[:, 1].max())

                span_x = max_x - min_x
                span_y = max_y - min_y
                marker_span = max(span_x, span_y)

                center_x = 0.5 * (min_x + max_x)
                center_y = 0.5 * (min_y + max_y)

                # Crop an area roughly three times larger than the marker.
                crop_size = int(
                    max(
                        48,
                        marker_span * 3.0,
                    )
                )

                crop_size = min(
                    crop_size,
                    self.out_w,
                    self.out_h,
                )

                x0 = int(round(center_x - crop_size / 2))
                y0 = int(round(center_y - crop_size / 2))

                x0 = max(
                    0,
                    min(x0, self.out_w - crop_size),
                )

                y0 = max(
                    0,
                    min(y0, self.out_h - crop_size),
                )

                x1 = x0 + crop_size
                y1 = y0 + crop_size

                zoom_crop = zoom_source[
                    y0:y1,
                    x0:x1,
                ].copy()

                inset_size = min(
                    240,
                    self.out_w // 2,
                    self.out_h // 2,
                )

                zoom_image = cv2.resize(
                    zoom_crop,
                    (inset_size, inset_size),
                    interpolation=cv2.INTER_NEAREST,
                )

                scale_x = inset_size / float(x1 - x0)
                scale_y = inset_size / float(y1 - y0)

                zoom_points = [None, None, None, None]

                for corner_index, point in enumerate(pixel_points):
                    if point is None:
                        continue

                    zoom_x = int(
                        round((point[0] - x0) * scale_x)
                    )

                    zoom_y = int(
                        round((point[1] - y0) * scale_y)
                    )

                    zoom_point = (zoom_x, zoom_y)
                    zoom_points[corner_index] = zoom_point

                if all(
                    point is not None
                    for point in zoom_points
                ):
                    zoom_polygon = np.asarray(
                        zoom_points,
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)

                    cv2.polylines(
                        zoom_image,
                        [zoom_polygon],
                        isClosed=True,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

                for corner_index, point in enumerate(zoom_points):
                    if point is None:
                        continue

                    color = corner_colors[corner_index]

                    cv2.circle(
                        zoom_image,
                        point,
                        radius=5,
                        color=color,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

                    cv2.circle(
                        zoom_image,
                        point,
                        radius=8,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                    cv2.putText(
                        zoom_image,
                        str(corner_index),
                        (
                            point[0] + label_offsets[corner_index][0],
                            point[1] + label_offsets[corner_index][1],
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                # Show the marker size in both output and original camera pixels.
                raw_span_x = span_x / float(self.upscale)
                raw_span_y = span_y / float(self.upscale)

                cv2.rectangle(
                    zoom_image,
                    (0, 0),
                    (inset_size - 1, inset_size - 1),
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    zoom_image,
                    "TCN CORNER ZOOM",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    zoom_image,
                    (
                        f"sensor span: "
                        f"{raw_span_x:.1f} x {raw_span_y:.1f}px"
                    ),
                    (6, inset_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Place inset in upper-right corner.
                inset_x = self.out_w - inset_size - 8
                inset_y = 8

                frame_bgr[
                    inset_y:inset_y + inset_size,
                    inset_x:inset_x + inset_size,
                ] = zoom_image

            valid_count = sum(
                point is not None
                for point in pixel_points
            )

            cv2.putText(
                frame_bgr,
                f"policy corners: {valid_count}/4",
                (5, self.out_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # print(f"[DEBUG WRITE] bbox_gt arg={bbox_gt}, frame shape={frame_bgr.shape}")
        self.writer.write(frame_bgr)   # <-- now happens AFTER all drawing

    def close(self):
        self.writer.release()
        print("[OnboardRecorder] Video saved.")


# =============================================================================
#  MAIN
# =============================================================================

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (args_cli.num_envs if args_cli.num_envs is not None
                              else env_cfg.scene.num_envs)

    # -------------------------------------------------------------------------
    # Resolve output/checkpoint paths
    # -------------------------------------------------------------------------
    if args_cli.student_checkpoint is not None:
        # Offline DAgger checkpoints do not live inside a normal RSL-RL run.
        resume_path = None

        policy_path = os.path.dirname(
            os.path.abspath(args_cli.student_checkpoint)
        )
        log_dir = policy_path

        print(
            "[INFO] Evaluating offline DAgger student:",
            args_cli.student_checkpoint,
        )
    else:
        log_root_path = os.path.abspath(
            os.path.join(
                "logs",
                "rsl_rl",
                agent_cfg.experiment_name,
            )
        )

        print(
            f"[INFO] Loading experiment from directory: "
            f"{log_root_path}"
        )

        if args_cli.rsl_checkpoint is not None:
            resume_path = os.path.abspath(args_cli.rsl_checkpoint)
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(resume_path)
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )

        log_dir = os.path.dirname(resume_path)
        policy_path = log_dir

        print("Resume path:", resume_path)
        print("Log dir:", log_dir)

    env_cfg.eval_mode = True
    env_cfg.seed = args_cli.seed

    # Evaluation uses analytic corners. Replicator textures, tiled RGB
    # cameras, and debug visualization are unnecessary unless recording.
    env_cfg.visual_dr_enabled = False
    env_cfg.debug_vis = False
    env_cfg.enable_onboard_camera = bool(
        args_cli.video
    )

    env_cfg.viewer.resolution = (
        1920,
        1080,
    )

    if args_cli.curriculum_level is not None:
        if not 0.0 <= args_cli.curriculum_level <= 1.0:
            raise ValueError(
                "--curriculum_level must lie in [0, 1]."
            )

        env_cfg.forced_curriculum_level = float(
            args_cli.curriculum_level
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
            "[EVAL] Forced curriculum level:",
            level,
        )

        print(
            "[EVAL] Active reset ranges:",
            f"position=±{active_pos_noise.tolist()} m,",
            f"yaw=±{active_yaw_noise:.4f} rad",
        )
    env_cfg.sim.device = (args_cli.device if args_cli.device is not None
                          else env_cfg.sim.device)
    agent_cfg.device = env_cfg.sim.device

    if os.path.exists(os.path.join(log_dir, "params/env.yaml")):
        with open(os.path.join(log_dir, "params/env.yaml")) as f:
            hydra_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        for key in ("pos_distance_reward_scale", "pos_error_reward_scale",
                    "lin_vel_reward_scale", "ang_vel_reward_scale",
                    "hover_distance"):
            if key in hydra_cfg:
                setattr(env_cfg, key, hydra_cfg[key])

    robot_index_prefix = ""
    camera_mode = os.environ.get("EVAL_CAMERA_MODE", "follow")

    if camera_mode == "follow":
        if args_cli.follow_robot >= 0:
            env_cfg.viewer.eye    = (-0.5, 0.5, 0.5)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.env_index   = args_cli.follow_robot
            env_cfg.viewer.asset_name  = "robot"
            robot_index_prefix = f"_robot_{args_cli.follow_robot}"
    elif camera_mode == "topdown":
        env_cfg.viewer.eye    = (0.5, 0.0, 8.0)
        env_cfg.viewer.lookat = (0.5, 0.0, 1.5)
        env_cfg.viewer.origin_type = "world"
        robot_index_prefix = "_topdown"
    elif camera_mode == "observer":
        env_cfg.viewer.eye    = (-2.0, -3.0, 2.0)
        env_cfg.viewer.lookat = (1.5, 0.0, 1.5)
        env_cfg.viewer.origin_type = "world"
        robot_index_prefix = "_observer"
    
    record_env_idx = args_cli.follow_robot if args_cli.follow_robot >= 0 else 0

    env_cfg.viewer.resolution = (1920, 1080)

    save_prefix       = args_cli.save_prefix
    video_name        = save_prefix + "_eval_video" + robot_index_prefix
    video_folder_path = os.path.join(policy_path, "videos", "eval")

    # ---- Existing third-person viewport video (unchanged) ------------------
    video_kwargs = {
        "video_folder": video_folder_path,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "name_prefix":  video_name,
    }
    render_mode = (
        "rgb_array"
        if args_cli.video
        else None
    )

    print(
        "[EVAL STARTUP] Creating environment...",
        flush=True,
    )

    envs = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=render_mode,
    )

    print(
        "[EVAL STARTUP] Environment created.",
        flush=True,
    )

    if args_cli.video:
        video_fps = getattr(env_cfg, "policy_rate_hz", None)

        if video_fps is None:
            sim_cfg = getattr(env_cfg, "sim", None)
            sim_dt = getattr(sim_cfg, "dt", None)
            decimation = getattr(env_cfg, "decimation", None)

            if sim_dt is not None and decimation is not None:
                video_fps = 1.0 / (
                    float(sim_dt) * float(decimation)
                )

        if video_fps is None:
            video_fps = 50.0
            print(
                "[WARN] Could not infer the environment video FPS. "
                "Using 50 FPS."
            )

        envs.metadata = dict(
            getattr(envs, "metadata", {}) or {}
        )
        envs.metadata["render_fps"] = int(
            round(float(video_fps))
        )

        print(
            "[INFO] Third-person video FPS: "
            f"{envs.metadata['render_fps']}"
        )

        envs = gym.wrappers.RecordVideo(
            envs,
            **video_kwargs,
        )

    device = envs.unwrapped.device
    # ---- Onboard camera recorder -------------------------------------------
    onboard_recorder = None

    if (
        args_cli.video
        and getattr(envs.unwrapped, "_camera", None) is not None
    ):
        cam_h = env_cfg.camera.height
        cam_w = env_cfg.camera.width

        onboard_video_path = os.path.join(
            video_folder_path,
            (
                save_prefix
                + "_onboard_camera"
                + robot_index_prefix
                + ".mp4"
            ),
        )

        onboard_recorder = OnboardCameraRecorder(
            output_path=onboard_video_path,
            img_h=cam_h,
            img_w=cam_w,
            fps=env_cfg.policy_rate_hz,
            upscale=8,
        )

    elif args_cli.video:
        print(
            "[INFO] This environment has no onboard camera. "
            "Skipping onboard-camera MP4; third-person video remains enabled."
        )

    # ---- Load RL policy ----------------------------------------------------
    envs = RslRlVecEnvWrapper(envs)
    # ---- Construct policy --------------------------------------------------
    agent_cfg_dict = agent_cfg.to_dict()

    if (
        "policy" in agent_cfg_dict
        and "use_gru" in agent_cfg_dict["policy"]
    ):
        agent_cfg_dict["policy"]["use_gru"] = False

    if args_cli.rsl_checkpoint is not None:
        import pathlib as _rgb_pathlib

        params_path = (
            _rgb_pathlib.Path(args_cli.rsl_checkpoint).resolve().parent
            / "params" / "agent.yaml"
        )

        if not params_path.is_file():
            raise FileNotFoundError(params_path)

        saved_cfg = yaml.safe_load(params_path.read_text())

        agent_cfg_dict["policy"] = dict(saved_cfg["policy"])
        agent_cfg_dict["algorithm"] = dict(saved_cfg["algorithm"])

        if agent_cfg_dict["policy"].get("class_name") != "VisionActorCritic":
            raise RuntimeError(
                "Expected the historical feedforward VisionActorCritic"
            )

        print(
            "[HISTORICAL CONFIG]",
            "policy=", agent_cfg_dict["policy"]["class_name"],
            "algorithm=", agent_cfg_dict["algorithm"]["class_name"],
            "use_gru=", agent_cfg_dict["policy"].get("use_gru"),
            flush=True,
        )

    policy_class_name = agent_cfg_dict.get(
        "policy",
        {},
    ).get(
        "class_name",
        "",
    )

    if policy_class_name == "CornersTCNActorCritic":
        # algorithm.class_name remains "PPO"; this changes the
        # concrete implementation used to construct the runner.
        rsl_rl_algorithms.PPO = CornersAnchoredPPO
        opr.PPO = CornersAnchoredPPO

        print(
            "[INFO] Evaluation PPO implementation: "
            "CornersAnchoredPPO"
        )

    ppo_runner = OnPolicyRunner(
        envs,
        agent_cfg_dict,
        log_dir=None,
        device=agent_cfg.device,
    )

    if args_cli.student_checkpoint is None:
        if resume_path is None:
            raise RuntimeError(
                "No RSL-RL checkpoint path was resolved."
            )

        print(
            f"[INFO] Loading RSL-RL model weights for inference: "
            f"{resume_path}"
        )

        checkpoint = torch.load(
            resume_path,
            map_location=agent_cfg.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"RSL-RL checkpoint {resume_path} does not contain "
                "'model_state_dict'. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        load_result = ppo_runner.alg.policy.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        print(
            "[INFO] RSL-RL model loaded strictly; "
            "optimizer state intentionally skipped during evaluation."
        )
        print("[INFO] load_state_dict result:", load_result)

        if "iter" in checkpoint:
            ppo_runner.current_learning_iteration = int(
                checkpoint["iter"]
            )

    else:
        print(
            "[INFO] Loading offline DAgger student:",
            args_cli.student_checkpoint,
        )

        checkpoint = torch.load(
            args_cli.student_checkpoint,
            map_location=agent_cfg.device,
        )

        if "student_state_dict" not in checkpoint:
            raise KeyError(
                "Offline checkpoint does not contain "
                "'student_state_dict'."
            )

        ppo_runner.alg.policy.student.load_state_dict(
            checkpoint["student_state_dict"],
            strict=True,
        )

        print("[INFO] Offline student loaded successfully.")

    agent = ppo_runner.get_inference_policy(
        device=envs.unwrapped.device
    )

    if args_cli.rsl_checkpoint is not None:
        visual_policy = ppo_runner.alg.policy

        if not hasattr(visual_policy, "backbone"):
            raise RuntimeError("Loaded policy has no MobileNet backbone")

        rgb_seen = {"count": 0}

        def rgb_input_hook(module, inputs):
            x = inputs[0]

            if x.ndim != 4 or x.shape[1] != 3:
                raise RuntimeError(
                    f"Expected normalized RGB [N,3,H,W], got {tuple(x.shape)}"
                )

            if rgb_seen["count"] == 0:
                print(
                    "[RGB INPUT]",
                    "shape=", tuple(x.shape),
                    "std=", round(float(x.std().item()), 6),
                    "mode=", args_cli.rgb_ablation,
                    flush=True,
                )

                if float(x.std().item()) < 1e-4:
                    raise RuntimeError(
                        "RGB input has negligible variation. "
                        "Check camera observations."
                    )

            rgb_seen["count"] += 1

            if args_cli.rgb_ablation == "neutral":
                return (torch.zeros_like(x),) + tuple(inputs[1:])

            return None

        visual_policy.backbone.register_forward_pre_hook(
            rgb_input_hook
        )

        print(
            "[ABLATION] mode=", args_cli.rgb_ablation,
            "checkpoint=", args_cli.rsl_checkpoint,
            flush=True,
        )

    # Direct reference to Isaac Lab env for camera access
    # envs is now RslRlVecEnvWrapper -> RecordVideo -> ArucoHoverStudentEnv
    isaac_env = envs.unwrapped   # ArucoHoverStudentEnv

    # # ---- ONNX export -------------------------------------------------------
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # try:
    #     export_policy_as_onnx(
    #         ppo_runner.alg.policy,
    #         normalizer=ppo_runner.obs_normalizer,
    #         path=export_model_dir,
    #         filename="teacher_policy.onnx",
    #     )
    # except AttributeError as e:
    #     print(f"[WARN] ONNX export skipped: {e}")

    # ---- Initial observations ----------------------------------------------
    obs, extras = envs.get_observations()

    obs_dict = extras.get(
        "observations",
        None,
    )

    if obs_dict is None:
        raise RuntimeError(
            "envs.get_observations() did not return "
            "extras['observations']."
        )

    policy_observations = obs_dict["policy"]

    # A separately loaded corners-only student consumes the flattened
    # 32-frame x 8-coordinate history. Native RSL-RL policies consume
    # the observation dimension defined by their environment/task.
    if args_cli.student_checkpoint is not None:
        if policy_observations.shape[-1] != 256:
            raise RuntimeError(
                "Expected a 256-D corners history for the offline "
                f"student, got {policy_observations.shape}."
            )
    else:
        print(
            "[INFO] Native RSL-RL policy observation dimension:",
            policy_observations.shape[-1],
        )

    full_state = obs_dict.get(
        "full_state",
        None,
    )

    print(
        "[INFO] Initial policy shape:",
        obs_dict["policy"].shape,
    )

    print(
        "[INFO] Full-state shape:",
        full_state.shape,
    )

    print(
        "[INFO] Initial full state, env 0:",
        full_state[0].detach().cpu().tolist(),
    )

    max_steps       = int(env_cfg.episode_length_s * env_cfg.policy_rate_hz)
    full_state_size = obs_dict["full_state"].shape[1]

    full_states = torch.zeros((args_cli.num_envs, max_steps, full_state_size),
                              dtype=torch.float32).to(device)
    rewards     = torch.zeros((args_cli.num_envs, max_steps),
                              dtype=torch.float32).to(device)
    actions_log = torch.zeros((args_cli.num_envs, max_steps, 4),
                              dtype=torch.float32).to(device)

    steps = 0
    times = []

    # Per-environment episode tracking.
    running_episode_returns = torch.zeros(
        args_cli.num_envs,
        device=device,
        dtype=torch.float32,
    )

    running_episode_lengths = torch.zeros(
        args_cli.num_envs,
        device=device,
        dtype=torch.long,
    )

    completed_episode_returns = []
    completed_episode_lengths = []

    number_of_deaths = 0
    number_of_timeouts = 0
    number_of_completed_episodes = 0
    
    # ------------------------------------------------------------------
    # Initial-episode paired-evaluation records
    #
    # Each environment contributes exactly one record: the episode that
    # began at evaluation startup. Episodes created by automatic resets
    # are deliberately excluded.
    # ------------------------------------------------------------------

    initial_full_state = (
        obs_dict["full_state"]
        .detach()
        .cpu()
        .clone()
    )

    initial_policy_observation = (
        obs_dict["policy"]
        .detach()
        .cpu()
        .clone()
    )

    first_episode_recorded = torch.zeros(
        args_cli.num_envs,
        device=device,
        dtype=torch.bool,
    )

    first_episode_survived = torch.zeros(
        args_cli.num_envs,
        device=device,
        dtype=torch.bool,
    )

    first_episode_died = torch.zeros(
        args_cli.num_envs,
        device=device,
        dtype=torch.bool,
    )

    first_episode_returns = torch.full(
        (args_cli.num_envs,),
        float("nan"),
        device=device,
        dtype=torch.float32,
    )

    first_episode_lengths = torch.full(
        (args_cli.num_envs,),
        -1,
        device=device,
        dtype=torch.long,
    )

    first_episode_termination_steps = torch.full(
        (args_cli.num_envs,),
        -1,
        device=device,
        dtype=torch.long,
    )

    # Raw action statistics before clipping.
    raw_action_value_count = 0
    raw_action_exceed_count = 0

    raw_action_vector_count = 0
    raw_action_vector_exceed_count = 0

    maximum_raw_action_magnitude = 0.0

    with torch.no_grad():
        while simulation_app.is_running():

            while steps < max_steps:
                full_states[:, steps, :] = obs_dict["full_state"]
                # Corner-history extraction is only valid for the separately
                # loaded 256-D corners student. Native RSL-RL policies, such as
                # the 20-D teacher, bypass all corner-specific diagnostics.
                latest_policy_corners = None
                corners_overlay = None

                if args_cli.student_checkpoint is not None:
                    policy_history = obs_dict["policy"].reshape(
                        args_cli.num_envs,
                        32,
                        8,
                    )

                    latest_policy_corners = policy_history[
                        record_env_idx,
                        -1,
                    ].reshape(4, 2)

                    corners_overlay = (
                        latest_policy_corners
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if args_cli.debug_corners and steps < 10:
                        print(
                            f"\n[CORNER DEBUG] step={steps}"
                        )

                        for corner_index in range(4):
                            u_value = float(
                                latest_policy_corners[
                                    corner_index,
                                    0,
                                ].item()
                            )

                            v_value = float(
                                latest_policy_corners[
                                    corner_index,
                                    1,
                                ].item()
                            )

                            print(
                                f"  corner {corner_index}: "
                                f"u={u_value:.5f}, "
                                f"v={v_value:.5f}"
                            )

                # ---- Policy forward pass -----------------------------------
                # agent() calls act_inference -> update_distribution -> _encode
                # which stashes _last_marker on ppo_runner.alg.policy
                start = time.time()

                raw_actions = agent(
                    obs_dict["policy"]
                )

                times.append(
                    time.time() - start
                )

                if not torch.isfinite(raw_actions).all():
                    raise RuntimeError(
                        f"Student produced non-finite actions at step {steps}: "
                        f"{raw_actions}"
                    )

                # Count individual action components outside [-1, 1].
                exceeds_limit = raw_actions.abs() > 1.0

                raw_action_exceed_count += int(
                    exceeds_limit.sum().item()
                )

                raw_action_value_count += raw_actions.numel()

                # Count environment action vectors where at least one component
                # is outside [-1, 1].
                vector_exceeds_limit = exceeds_limit.any(
                    dim=-1
                )

                raw_action_vector_exceed_count += int(
                    vector_exceeds_limit.sum().item()
                )

                raw_action_vector_count += raw_actions.shape[0]

                maximum_raw_action_magnitude = max(
                    maximum_raw_action_magnitude,
                    float(raw_actions.abs().max().item()),
                )

                # Match the clipping performed by _pre_physics_step().
                actions = torch.clamp(
                    raw_actions,
                    min=-1.0,
                    max=1.0,
                )

                actions_log[:, steps] = actions

                if steps < 10:
                    print(
                        f"\n[EVAL] step={steps}, "
                        f"raw={raw_actions[record_env_idx].detach().cpu().tolist()}, "
                        f"executed={actions[record_env_idx].detach().cpu().tolist()}"
                    )


                # ---- Read onboard RGB --------------------------------------
                # rgb: (N, H, W, 3) uint8 — confirmed from debug output
                # This is the frame from the PREVIOUS step (sensor lag of 1),
                # which is acceptable for video overlay purposes.
                rgb_nhwc = None

                if onboard_recorder is not None:
                    if getattr(isaac_env, "_camera", None) is None:
                        raise RuntimeError(
                            "Video recording was requested, but the "
                            "onboard camera was not created."
                        )

                    try:
                        rgb_data = (
                            isaac_env
                            ._camera
                            .data
                            .output
                            .get("rgb", None)
                        )

                        if (
                            rgb_data is not None
                            and rgb_data.ndim == 4
                        ):
                            rgb_nhwc = rgb_data

                    except Exception as error:
                        print(
                            f"[WARN] Camera read failed at "
                            f"step {steps}: {error}"
                        )

                # ---- Write onboard frame -----------------------------------
                if onboard_recorder is not None and rgb_nhwc is not None:
                    #feature_map = getattr(ppo_runner.alg.policy, "_last_feature_map", None)
                    feature_map = None  # Disable feature map overlay for now
                    bbox_tuple = None
                    bbox_raw = None

                    # Compute and draw the bbox independently of corner debugging.
                    if args_cli.debug_bbox:
                        bbox_raw = (
                            isaac_env._compute_bbox_gt()
                        )

                        candidate = tuple(
                            bbox_raw[
                                record_env_idx
                            ].detach().cpu().tolist()
                        )

                        # Layout:
                        # [cx, cy, width, height, visible]
                        if (
                            candidate[0] >= 0.0
                            and candidate[4] > 0.5
                        ):
                            bbox_tuple = candidate[:4]

                        elif steps < 5:
                            print(
                                f"[DEBUG DRAW] step {steps}: "
                                f"marker bbox not visible"
                            )

                    # Compare the exact policy corners with the separately calculated bbox.
                    if (
                        args_cli.debug_corners
                        and steps < 10
                    ):
                        valid_corner_mask = (
                            (latest_policy_corners[:, 0] >= 0.0)
                            & (latest_policy_corners[:, 0] <= 1.0)
                            & (latest_policy_corners[:, 1] >= 0.0)
                            & (latest_policy_corners[:, 1] <= 1.0)
                        )

                        if (
                            valid_corner_mask.all()
                            and bbox_raw is not None
                            and bbox_raw[
                                record_env_idx,
                                4,
                            ] > 0.5
                        ):
                            corner_center = (
                                latest_policy_corners.mean(
                                    dim=0
                                )
                            )

                            bbox_center = bbox_raw[
                                record_env_idx,
                                0:2,
                            ]

                            center_error = (
                                corner_center
                                - bbox_center
                            )

                            print(
                                "[PROJECTION CHECK] "
                                f"corner center="
                                f"{corner_center.detach().cpu().tolist()}, "
                                f"bbox center="
                                f"{bbox_center.detach().cpu().tolist()}, "
                                f"difference="
                                f"{center_error.detach().cpu().tolist()}"
                            )

                        if bbox_raw is not None and steps < 5:
                            print(
                                "[DEBUG] selected bbox:",
                                bbox_raw[
                                    record_env_idx
                                ].detach().cpu().tolist(),
                            )
                            
                    onboard_recorder.write_frame(
                        rgb_nhwc=rgb_nhwc,
                        feature_map=None,
                        env_idx=record_env_idx,
                        bbox_gt=bbox_tuple,
                        bbox_pred=None,
                        corners_obs=(
                            corners_overlay
                            if args_cli.debug_corners
                            else None
                        ),
                    )

                # ---- Step environment --------------------------------------
                obs, reward, dones, extras = envs.step(
                    actions
                )

                reward = reward.detach()

                dones = dones.to(
                    device=device,
                    dtype=torch.bool,
                )

                # Accumulate the return and length of each currently running episode.
                running_episode_returns += reward
                running_episode_lengths += 1

                rewards[:, steps] = reward

                # --------------------------------------------------------------
                # Determine which terminations were timeouts.
                # --------------------------------------------------------------
                timeout_mask = extras.get(
                    "time_outs",
                    None,
                )

                if timeout_mask is None:
                    timeout_mask = extras.get(
                        "truncated",
                        None,
                    )

                if timeout_mask is not None:
                    timeout_mask = timeout_mask.to(
                        device=device,
                        dtype=torch.bool,
                    )
                else:
                    # Fallback for environments that do not explicitly return
                    # a timeout mask.
                    timeout_mask = (
                        dones
                        & (running_episode_lengths >= max_steps)
                    )

                # Any termination that was not a timeout is counted as a death.
                death_mask = dones & ~timeout_mask

                # ------------------------------------------------------
                # Record the first episode only.
                #
                # running_episode_returns and running_episode_lengths
                # still contain the completed episode values here. They
                # are reset later in the existing done_indices block.
                # ------------------------------------------------------

                first_done_mask = (
                    dones
                    & ~first_episode_recorded
                )

                first_done_indices = torch.nonzero(
                    first_done_mask,
                    as_tuple=False,
                ).squeeze(-1)

                if first_done_indices.numel() > 0:
                    first_episode_survived[
                        first_done_indices
                    ] = timeout_mask[
                        first_done_indices
                    ]

                    first_episode_died[
                        first_done_indices
                    ] = death_mask[
                        first_done_indices
                    ]

                    first_episode_returns[
                        first_done_indices
                    ] = running_episode_returns[
                        first_done_indices
                    ]

                    first_episode_lengths[
                        first_done_indices
                    ] = running_episode_lengths[
                        first_done_indices
                    ]

                    # The initial episode started at evaluation step zero,
                    # so its termination step is identical to its recorded
                    # episode length.
                    first_episode_termination_steps[
                        first_done_indices
                    ] = running_episode_lengths[
                        first_done_indices
                    ]

                    first_episode_recorded[
                        first_done_indices
                    ] = True

                number_of_timeouts += int(
                    timeout_mask.sum().item()
                )

                number_of_deaths += int(
                    death_mask.sum().item()
                )

                number_of_completed_episodes += int(
                    dones.sum().item()
                )

                # Store the return and length of completed episodes.
                done_indices = torch.nonzero(
                    dones,
                    as_tuple=False,
                ).flatten()

                if done_indices.numel() > 0:
                    completed_episode_returns.extend(
                        running_episode_returns[
                            done_indices
                        ].detach().cpu().tolist()
                    )

                    completed_episode_lengths.extend(
                        running_episode_lengths[
                            done_indices
                        ].detach().cpu().tolist()
                    )

                    # The Isaac Lab environment resets completed environments
                    # automatically, so reset our external counters too.
                    running_episode_returns[
                        done_indices
                    ] = 0.0

                    running_episode_lengths[
                        done_indices
                    ] = 0

                # IMPORTANT:
                # Do not call envs.unwrapped._get_observations() here.
                # It would shift the 32-step TCN history a second time.
                obs_dict = extras.get(
                    "observations",
                    None,
                )

                if obs_dict is None:
                    raise RuntimeError(
                        "envs.step() did not return extras['observations']."
                    )

                info = extras

                if steps == 0:
                    print(
                        "\n[DEBUG] extras keys:",
                        list(extras.keys()),
                    )

                steps += 1

                print(
                    f"Step: {steps}/{max_steps}",
                    end="\r",
                )

            print()

            # ---- Finalise onboard video ------------------------------------
            if onboard_recorder is not None:
                onboard_recorder.close()

            # ---- Save tensors ----------------------------------------------
            print("Full states shape: ", full_states.shape)
            torch.save(full_states,
                       os.path.join(policy_path,
                                    save_prefix + "eval_full_states.pt"))
            torch.save(rewards,
                       os.path.join(policy_path,
                                    save_prefix + "eval_rewards.pt"))

            print("Final Info:\n", info)

            mean_inference_time = (
                float(np.mean(times))
                if len(times) > 0
                else float("nan")
            )

                        # ----------------------------------------------------------
            # Save paired initial-episode records.
            # ----------------------------------------------------------

            missing_initial_records = torch.nonzero(
                ~first_episode_recorded,
                as_tuple=False,
            ).squeeze(-1)

            if missing_initial_records.numel() > 0:
                raise RuntimeError(
                    "Evaluation ended before all initial episodes "
                    "terminated. Missing environment IDs: "
                    f"{missing_initial_records.detach().cpu().tolist()}"
                )

            initial_timeout_count = int(
                first_episode_survived.sum().item()
            )

            initial_death_count = int(
                first_episode_died.sum().item()
            )

            if (
                initial_timeout_count
                + initial_death_count
                != args_cli.num_envs
            ):
                raise RuntimeError(
                    "Initial-episode outcome count is inconsistent: "
                    f"timeouts={initial_timeout_count}, "
                    f"deaths={initial_death_count}, "
                    f"num_envs={args_cli.num_envs}."
                )

            record_directory = Path(
                "results/historical_evaluations/initial_episode_records"
            )

            record_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            record_prefix = (
                str(args_cli.save_prefix)
                if args_cli.save_prefix
                else "evaluation"
            )

            # Prevent a save prefix containing directories from creating
            # unexpected nested paths.
            record_prefix = Path(record_prefix).name

            checkpoint_source = (
                str(resume_path)
                if resume_path is not None
                else str(args_cli.student_checkpoint)
            )

            initial_episode_record = {
                "schema_version": 1,
                "task": str(args_cli.task),
                "seed": int(args_cli.seed),
                "curriculum_level": (lambda value: None if value is None else float(value))(
                    args_cli.curriculum_level
                ),
                "num_envs": int(args_cli.num_envs),
                "max_steps": int(max_steps),
                "checkpoint_source": checkpoint_source,
                "save_prefix": record_prefix,

                "env_id": torch.arange(
                    args_cli.num_envs,
                    dtype=torch.long,
                ),

                "initial_full_state": initial_full_state,
                "initial_policy_observation": (
                    initial_policy_observation
                ),

                "recorded": (
                    first_episode_recorded
                    .detach()
                    .cpu()
                ),

                "survived": (
                    first_episode_survived
                    .detach()
                    .cpu()
                ),

                "died": (
                    first_episode_died
                    .detach()
                    .cpu()
                ),

                "episode_return": (
                    first_episode_returns
                    .detach()
                    .cpu()
                ),

                "episode_length": (
                    first_episode_lengths
                    .detach()
                    .cpu()
                ),

                "termination_step": (
                    first_episode_termination_steps
                    .detach()
                    .cpu()
                ),
            }

            initial_episode_record_path = (
                record_directory
                / f"{record_prefix}_initial_episodes.pt"
            )

            torch.save(
                initial_episode_record,
                initial_episode_record_path,
            )

            print(
                "[EVAL] Saved initial-episode records:",
                initial_episode_record_path,
            )

            print(
                "[EVAL] Initial episodes:",
                f"timeouts={initial_timeout_count},",
                f"deaths={initial_death_count},",
                f"total={args_cli.num_envs}",
            )

            if completed_episode_returns:
                mean_completed_return = float(
                    np.mean(completed_episode_returns)
                )

                mean_completed_length = float(
                    np.mean(completed_episode_lengths)
                )
            else:
                mean_completed_return = float("nan")
                mean_completed_length = float("nan")

            death_percentage = (
                100.0
                * number_of_deaths
                / max(number_of_completed_episodes, 1)
            )

            timeout_percentage = (
                100.0
                * number_of_timeouts
                / max(number_of_completed_episodes, 1)
            )

            raw_value_exceed_percentage = (
                100.0
                * raw_action_exceed_count
                / max(raw_action_value_count, 1)
            )

            raw_vector_exceed_percentage = (
                100.0
                * raw_action_vector_exceed_count
                / max(raw_action_vector_count, 1)
            )

            print()
            print("=" * 72)
            print("STUDENT-ONLY EVALUATION SUMMARY")
            print("=" * 72)

            print(
                f"Initial episodes:         "
                f"{args_cli.num_envs}"
            )

            print(
                f"Initial deaths:           "
                f"{initial_death_count} "
                f"({100.0 * initial_death_count / args_cli.num_envs:.2f}%)"
            )

            print(
                f"Initial timeouts:         "
                f"{initial_timeout_count} "
                f"({100.0 * initial_timeout_count / args_cli.num_envs:.2f}%)"
            )

            print(
                f"Completed episodes:       "
                f"{number_of_completed_episodes}"
            )

            print(
                f"Deaths:                   "
                f"{number_of_deaths} "
                f"({death_percentage:.2f}%)"
            )

            print(
                f"Timeouts:                 "
                f"{number_of_timeouts} "
                f"({timeout_percentage:.2f}%)"
            )

            print(
                f"Mean completed return:    "
                f"{mean_completed_return:.4f}"
            )

            print(
                f"Mean completed length:    "
                f"{mean_completed_length:.2f} steps"
            )

            print(
                f"Mean completed duration:  "
                f"{mean_completed_length / env_cfg.policy_rate_hz:.2f} s"
            )

            print(
                f"Raw values outside range: "
                f"{raw_action_exceed_count}/"
                f"{raw_action_value_count} "
                f"({raw_value_exceed_percentage:.3f}%)"
            )

            print(
                f"Action vectors clipped:   "
                f"{raw_action_vector_exceed_count}/"
                f"{raw_action_vector_count} "
                f"({raw_vector_exceed_percentage:.3f}%)"
            )

            print(
                f"Maximum |raw action|:     "
                f"{maximum_raw_action_magnitude:.4f}"
            )

            print(
                f"Average inference time:   "
                f"{mean_inference_time * 1000.0:.3f} ms"
            )

            print("=" * 72)

            # --------------------------------------------------------------
            # Exact-target stationary-hold diagnostics
            # --------------------------------------------------------------
            nominal_pos_noise = np.asarray(
                getattr(
                    env_cfg,
                    "nominal_pos_noise",
                    [float("nan")] * 3,
                ),
                dtype=np.float64,
            )

            exact_reset_requested = (
                bool(getattr(env_cfg, "eval_mode", False))
                and bool(getattr(env_cfg, "nominal_mode", False))
                and np.allclose(nominal_pos_noise, 0.0)
                and abs(
                    float(
                        getattr(
                            env_cfg,
                            "nominal_yaw_noise",
                            float("nan"),
                        )
                    )
                ) <= 1.0e-12
                and abs(
                    float(
                        getattr(
                            env_cfg,
                            "nominal_vel_noise",
                            float("nan"),
                        )
                    )
                ) <= 1.0e-12
                and abs(
                    float(
                        getattr(
                            env_cfg,
                            "nominal_ang_vel_noise",
                            float("nan"),
                        )
                    )
                ) <= 1.0e-12
            )

            if (
                exact_reset_requested
                and steps > 0
                and full_state_size >= 18
            ):
                settle_steps = int(
                    round(
                        2.0
                        * float(env_cfg.policy_rate_hz)
                    )
                )
                settle_steps = min(
                    settle_steps,
                    max(steps - 1, 0),
                )

                state_post = full_states[
                    :,
                    settle_steps:steps,
                    :,
                ]
                actions_post = actions_log[
                    :,
                    settle_steps:steps,
                    :,
                ]

                distance_post = state_post[:, :, 17]

                lin_speed_post = torch.linalg.vector_norm(
                    state_post[:, :, 7:10],
                    dim=-1,
                )

                ang_speed_post = torch.linalg.vector_norm(
                    state_post[:, :, 10:13],
                    dim=-1,
                )

                action_norm_post = torch.linalg.vector_norm(
                    actions_post,
                    dim=-1,
                )

                initial_z = initial_full_state[
                    :,
                    2,
                ].to(
                    device=device,
                    dtype=full_states.dtype,
                )

                z_shift = (
                    full_states[:, :steps, 2]
                    - initial_z.unsqueeze(1)
                )

                rms_distance_per_env = torch.sqrt(
                    torch.mean(
                        distance_post.square(),
                        dim=1,
                    )
                )

                rms_speed_per_env = torch.sqrt(
                    torch.mean(
                        lin_speed_post.square(),
                        dim=1,
                    )
                )

                rms_body_rate_per_env = torch.sqrt(
                    torch.mean(
                        ang_speed_post.square(),
                        dim=1,
                    )
                )

                rms_action_norm_per_env = torch.sqrt(
                    torch.mean(
                        action_norm_post.square(),
                        dim=1,
                    )
                )

                final_distance = full_states[
                    :,
                    steps - 1,
                    17,
                ]

                maximum_excursion = full_states[
                    :,
                    :steps,
                    17,
                ].amax(dim=1)

                final_z_shift = z_shift[:, -1]
                maximum_upward_excursion = z_shift.amax(dim=1)

                fraction_within_010 = (
                    distance_post <= 0.10
                ).float().mean(dim=1)

                fraction_within_015 = (
                    distance_post <= 0.15
                ).float().mean(dim=1)

                fraction_within_025 = (
                    distance_post <= 0.25
                ).float().mean(dim=1)

                print()
                print("=" * 72)
                print("EXACT-TARGET STATIONARY-HOLD DIAGNOSTICS")
                print("=" * 72)
                print(
                    f"Settling period ignored:       "
                    f"{settle_steps / float(env_cfg.policy_rate_hz):.2f} s"
                )
                print(
                    f"Mean post-settle RMS distance: "
                    f"{rms_distance_per_env.mean().item():.4f} m"
                )
                print(
                    f"Mean final distance:           "
                    f"{final_distance.mean().item():.4f} m"
                )
                print(
                    f"Mean maximum excursion:        "
                    f"{maximum_excursion.mean().item():.4f} m"
                )
                print(
                    f"Mean post-settle RMS speed:    "
                    f"{rms_speed_per_env.mean().item():.4f} m/s"
                )
                print(
                    f"Mean post-settle RMS body rate:"
                    f" {rms_body_rate_per_env.mean().item():.4f} rad/s"
                )
                print(
                    f"Mean post-settle RMS action:   "
                    f"{rms_action_norm_per_env.mean().item():.4f}"
                )
                print(
                    f"Mean final vertical shift:     "
                    f"{final_z_shift.mean().item():+.4f} m"
                )
                print(
                    f"Mean maximum upward excursion: "
                    f"{maximum_upward_excursion.mean().item():+.4f} m"
                )
                print(
                    f"Mean fraction within 0.10 m:   "
                    f"{100.0 * fraction_within_010.mean().item():.2f}%"
                )
                print(
                    f"Mean fraction within 0.15 m:   "
                    f"{100.0 * fraction_within_015.mean().item():.2f}%"
                )
                print(
                    f"Mean fraction within 0.25 m:   "
                    f"{100.0 * fraction_within_025.mean().item():.2f}%"
                )
                print("=" * 72)

            # ---- Plots (unchanged) -----------------------------------------
            if args_cli.follow_robot >= 0:
                import matplotlib.pyplot as plt
                import isaaclab.utils.math as isaac_math_utils

                quad_pos = full_states[
                    args_cli.follow_robot, :steps, 0:3
                ].cpu().numpy()

                quad_vel = full_states[
                    args_cli.follow_robot, :steps, 7:10
                ].cpu().numpy()

                quad_ang_vel = full_states[
                    args_cli.follow_robot, :steps, 10:13
                ].cpu().numpy()

                marker_dist = full_states[
                    args_cli.follow_robot, :steps, 17
                ].cpu().numpy()

                quad_euler = isaac_math_utils.euler_xyz_from_quat(
                    full_states[
                        args_cli.follow_robot,
                        :steps,
                        3:7,
                    ]
                )
                quad_roll  = quad_euler[0].cpu().numpy()
                quad_pitch = quad_euler[1].cpu().numpy()
                quad_yaw   = quad_euler[2].cpu().numpy()

                T = steps

                x = np.arange(T) * (
                    1.0 / env_cfg.policy_rate_hz
                )

                save_path = os.path.join(video_folder_path, video_name + ".png")
                fig = plt.figure(figsize=(10, 10))

                plt.subplot(4, 3, 1);  plt.plot(x, quad_pos[:, 0],     label="Quad X");     plt.legend(loc="best")
                plt.subplot(4, 3, 2);  plt.plot(x, quad_pos[:, 1],     label="Quad Y");     plt.legend(loc="best")
                plt.subplot(4, 3, 3)
                plt.plot(x, quad_pos[:, 2],  label="Quad Z")
                plt.plot(x, marker_dist,     label="Marker dist", linestyle="--")
                plt.axhline(1.5, color="gray", linestyle=":", label="Target 1.5 m")
                plt.legend(loc="best")
                plt.subplot(4, 3, 4);  plt.plot(x, quad_vel[:, 0],     label="Quad Vel X"); plt.legend(loc="best")
                plt.subplot(4, 3, 5);  plt.plot(x, quad_vel[:, 1],     label="Quad Vel Y"); plt.legend(loc="best")
                plt.subplot(4, 3, 6);  plt.plot(x, quad_vel[:, 2],     label="Quad Vel Z"); plt.legend(loc="best")
                plt.subplot(4, 3, 7);  plt.plot(x, quad_roll,          label="Quad Roll");  plt.legend(loc="best")
                plt.subplot(4, 3, 8);  plt.plot(x, quad_pitch,         label="Quad Pitch"); plt.legend(loc="best")
                plt.subplot(4, 3, 9);  plt.plot(x, quad_yaw,           label="Quad Yaw");   plt.legend(loc="best")
                plt.subplot(4, 3, 10); plt.plot(x, quad_ang_vel[:, 0], label="Ang Vel X");  plt.legend(loc="best")
                plt.subplot(4, 3, 11); plt.plot(x, quad_ang_vel[:, 1], label="Ang Vel Y");  plt.legend(loc="best")
                plt.subplot(4, 3, 12); plt.plot(x, quad_ang_vel[:, 2], label="Ang Vel Z");  plt.legend(loc="best")

                plt.tight_layout()
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
                plt.close(fig)

                fig = plt.figure(figsize=(10, 10))
                actions_np = actions_log.cpu().numpy()
                act_labels = ["Thrust (norm.)", "Roll rate", "Pitch rate", "Yaw rate"]
                for i, label in enumerate(act_labels):
                    plt.subplot(2, 2, i + 1)
                    plt.plot(
                        x,
                        actions_np[
                            args_cli.follow_robot,
                            :steps,
                            i,
                        ],
                        label=label,
                    )
                    plt.legend(loc="best")
                plt.tight_layout()
                save_path = os.path.join(video_folder_path,
                                         video_name + "_actions.png")
                plt.savefig(save_path)
                print(f"Saved actions plot to {save_path}")
                plt.close(fig)

            envs.close()
            simulation_app.close()
            break

if __name__ == "__main__":
    main()
    simulation_app.close()