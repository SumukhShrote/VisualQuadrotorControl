'''import argparse
import sys
from isaaclab.app import AppLauncher

import math           
import collections    

# local imports
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

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = args_cli.video
args_cli.headless = True  # make False to see the simulation

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import time
import yaml

import gymnasium as gym
import envs  # registers Isaac-QuadrotorHover-v0

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
from envs.aruco_hover.agents.vision_student_teacher import VisionStudentTeacher
opr.VisionStudentTeacher = VisionStudentTeacher

from isaaclab.utils.io import load_yaml

import numpy as np
import torch


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print("Resume path: ", resume_path)
    log_dir = os.path.dirname(resume_path)
    print("Log dir: ", log_dir)

    policy_path = log_dir

    env_cfg.eval_mode = True
    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # Restore reward scalars from saved params/env.yaml so eval matches training
    if os.path.exists(os.path.join(log_dir, "params/env.yaml")):
        with open(os.path.join(log_dir, "params/env.yaml")) as f:
            hydra_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

        for key in ("pos_distance_reward_scale", "pos_error_reward_scale",
                    "lin_vel_reward_scale", "ang_vel_reward_scale",
                    "hover_distance"):
            if key in hydra_cfg:
                setattr(env_cfg, key, hydra_cfg[key])

    robot_index_prefix = ""

    camera_mode = os.environ.get(
        "EVAL_CAMERA_MODE",
        "follow",
    )

    if camera_mode == "follow":

        if args_cli.follow_robot >= 0:
            env_cfg.viewer.eye = (-0.5, 0.5, 0.5)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.env_index = args_cli.follow_robot
            env_cfg.viewer.asset_name = "robot"

            robot_index_prefix = (
                f"_robot_{args_cli.follow_robot}"
            )

    elif camera_mode == "topdown":

        env_cfg.viewer.eye = (0.5, 0.0, 8.0)
        env_cfg.viewer.lookat = (0.5, 0.0, 1.5)
        env_cfg.viewer.origin_type = "world"

        robot_index_prefix = "_topdown"

    elif camera_mode == "observer":

        env_cfg.viewer.eye = (-2.0, -3.0, 2.0)
        env_cfg.viewer.lookat = (1.5, 0.0, 1.5)
        env_cfg.viewer.origin_type = "world"

        robot_index_prefix = "_observer"

    env_cfg.viewer.resolution = (1920, 1080)
    print("\n\nUpdated env cfg: ", env_cfg)

    save_prefix = args_cli.save_prefix
    video_name = save_prefix + "_eval_video" + robot_index_prefix
    video_folder_path = os.path.join(policy_path, "videos", "eval")

    video_kwargs = {
        "video_folder": video_folder_path,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "name_prefix": video_name
    }
    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    device = envs.unwrapped.device

    # Load RL policy
    envs = RslRlVecEnvWrapper(envs)  # This calls Reset!!
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(envs, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    agent = ppo_runner.get_inference_policy(device=envs.unwrapped.device)

    # DEBUG
    print(f"[DEBUG] agent type: {type(agent)}")
    print(f"[DEBUG] policy type: {type(ppo_runner.alg.policy)}")
    print(f"[DEBUG] has _last_marker before step: {hasattr(ppo_runner.alg.policy, '_last_marker')}")

    # Also check camera access
    isaac_env = envs_wrapped.unwrapped
    print(f"[DEBUG] has _camera: {hasattr(isaac_env, '_camera')}")
    if hasattr(isaac_env, '_camera') and isaac_env._camera is not None:
        rgb = isaac_env._camera.data.output.get("rgb", None)
        print(f"[DEBUG] rgb shape before first step: {rgb.shape if rgb is not None else None}")

    # Export ONNX for Jetson Orin NX TensorRT conversion
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_onnx(
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="teacher_policy.onnx",
        )
    except AttributeError as e:
        print(f"[WARN] ONNX export skipped (custom recurrent policy not supported by exporter): {e}")

    # RslRlVecEnvWrapper returns a tuple: (obs, extras)
    obs, extras = envs.get_observations()
    obs_dict = extras.get("observations", {})
    full_state = obs_dict.get("full_state", None)
    print("Starting obs: ", full_state)

    max_steps = int(env_cfg.episode_length_s * env_cfg.policy_rate_hz)

    full_state_size = obs_dict["full_state"].shape[1]
    full_states  = torch.zeros((args_cli.num_envs, max_steps, full_state_size), dtype=torch.float32).to(device)
    rewards      = torch.zeros((args_cli.num_envs, max_steps), dtype=torch.float32).to(device)
    actions_log  = torch.zeros((args_cli.num_envs, max_steps, 4), dtype=torch.float32).to(device)

    steps = 0
    done = False
    done_count = 0
    times = []

    with torch.no_grad():
        while simulation_app.is_running():
            print(f"[EVAL] policy obs shape: {obs_dict['policy'].shape}")
            print(f"[EVAL] obs keys available: {list(obs_dict.keys())}")
            # Also add this:
            print(f"[EVAL] task: {args_cli.task}")
            print(f"[EVAL] env type: {type(envs.unwrapped)}")
            while steps < max_steps and not done:
                full_states[:, steps, :] = obs_dict["full_state"]

                start = time.time()
                actions = agent(obs_dict["policy"])
                actions_log[:, steps] = actions
                times.append(time.time() - start)

                # Wrapper returns 4 values: obs, reward, dones, extras
                obs, reward, dones, extras = envs.step(actions)
                done_count += dones.sum().item()
                obs_dict = extras.get("observations", envs.unwrapped._get_observations())
                info = extras
                rewards[:, steps] = reward.detach()
                steps += 1
                print("Step: ", steps)
                

            print("Full states shape: ", full_states.shape)
            torch.save(full_states, os.path.join(policy_path, save_prefix + "eval_full_states.pt"))
            torch.save(rewards, os.path.join(policy_path, save_prefix + "eval_rewards.pt"))

            print("Final Info: \n\n", info, "\n")
            print("\nAverage inference time: ", np.mean(times))

            # State slices for quadrotor hover (18-dim full_state):
            # [0:3]   quad position (world)
            # [3:7]   quaternion [x,y,z,w]
            # [7:10]  linear velocity (body)
            # [10:13] angular velocity (body)
            # [13:17] last action [thrust, roll_rate, pitch_rate, yaw_rate]
            # [17]    distance to ArUco marker
            quad_pos     = full_states[args_cli.follow_robot, :-1, 0:3].cpu().numpy()
            quad_vel     = full_states[args_cli.follow_robot, :-1, 7:10].cpu().numpy()
            quad_ang_vel = full_states[args_cli.follow_robot, :-1, 10:13].cpu().numpy()
            marker_dist  = full_states[args_cli.follow_robot, :-1, 17].cpu().numpy()

            if args_cli.follow_robot >= 0:
                import matplotlib.pyplot as plt
                import isaaclab.utils.math as isaac_math_utils

                quad_euler = isaac_math_utils.euler_xyz_from_quat(full_states[args_cli.follow_robot, :-1, 3:7])
                quad_roll  = quad_euler[0].cpu().numpy()
                quad_pitch = quad_euler[1].cpu().numpy()
                quad_yaw   = quad_euler[2].cpu().numpy()

                T = rewards.shape[1] - 1
                x = np.arange(T) * (1 / env_cfg.policy_rate_hz)

                save_path = video_folder_path + "/" + video_name + ".png"

                fig = plt.figure(figsize=(10, 10))

                plt.subplot(4, 3, 1)
                plt.plot(x, quad_pos[:, 0], label="Quad X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 2)
                plt.plot(x, quad_pos[:, 1], label="Quad Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 3)
                plt.plot(x, quad_pos[:, 2], label="Quad Z")
                plt.plot(x, marker_dist, label="Marker dist", linestyle="--")
                plt.axhline(1.5, color="gray", linestyle=":", label="Target 1.5 m")
                plt.legend(loc="best")
                plt.subplot(4, 3, 4)
                plt.plot(x, quad_vel[:, 0], label="Quad Vel X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 5)
                plt.plot(x, quad_vel[:, 1], label="Quad Vel Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 6)
                plt.plot(x, quad_vel[:, 2], label="Quad Vel Z")
                plt.legend(loc="best")
                plt.subplot(4, 3, 7)
                plt.plot(x, quad_roll, label="Quad Roll")
                plt.legend(loc="best")
                plt.subplot(4, 3, 8)
                plt.plot(x, quad_pitch, label="Quad Pitch")
                plt.legend(loc="best")
                plt.subplot(4, 3, 9)
                plt.plot(x, quad_yaw, label="Quad Yaw")
                plt.legend(loc="best")
                plt.subplot(4, 3, 10)
                plt.plot(x, quad_ang_vel[:, 0], label="Quad Ang Vel X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 11)
                plt.plot(x, quad_ang_vel[:, 1], label="Quad Ang Vel Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 12)
                plt.plot(x, quad_ang_vel[:, 2], label="Quad Ang Vel Z")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
                plt.close(fig)

                # Plot actions
                fig = plt.figure(figsize=(10, 10))
                actions_np = actions_log.cpu().numpy()
                act_labels = ["Thrust (norm.)", "Roll rate", "Pitch rate", "Yaw rate"]
                for i, label in enumerate(act_labels):
                    plt.subplot(2, 2, i + 1)
                    plt.plot(x, actions_np[args_cli.follow_robot, :-1, i], label=label)
                    plt.legend(loc="best")
                plt.tight_layout()
                save_path = video_folder_path + "/" + video_name + "_actions.png"
                plt.savefig(save_path)
                print(f"Saved actions plot to {save_path}")
                plt.close(fig)

            envs.close()
            simulation_app.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
'''

import argparse
import sys
from isaaclab.app import AppLauncher

import math
import collections

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
parser.add_argument("--debug_bbox_pred", action="store_true", default=False, help="Draw student-predicted bounding box (from policy._last_marker) on onboard camera video for debugging.")
parser.add_argument("--use_gru", type=lambda x: x.lower() == "true", default=False, help="Use GRU temporal memory in student policy — must match the trained checkpoint.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = args_cli.video
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
from envs.aruco_hover.agents.vision_student_teacher import VisionStudentTeacher
opr.VisionStudentTeacher = VisionStudentTeacher

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

    def write_frame(self, rgb_nhwc: torch.Tensor,
                feature_map: torch.Tensor | None = None,
                env_idx: int = 0,
                bbox_gt: tuple | None = None,
                bbox_pred: tuple | None = None):
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

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run,
                                      agent_cfg.load_checkpoint)
    print("Resume path: ", resume_path)
    log_dir = os.path.dirname(resume_path)
    print("Log dir: ", log_dir)
    policy_path = log_dir

    env_cfg.eval_mode = True
    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.seed = args_cli.seed
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
    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    device = envs.unwrapped.device

    # ---- Onboard camera recorder -------------------------------------------
    # Camera resolution comes from env cfg — always in sync
    cam_h = env_cfg.camera.height   # 128
    cam_w = env_cfg.camera.width    # 128

    onboard_video_path = os.path.join(
        video_folder_path,
        save_prefix + "_onboard_camera" + robot_index_prefix + ".mp4",
    )
    onboard_recorder = OnboardCameraRecorder(
        output_path=onboard_video_path,
        img_h=cam_h,
        img_w=cam_w,
        fps=env_cfg.policy_rate_hz,  # 50 — matches policy step rate
        upscale=4,                    # 128 -> 512 pixels
    )

    # ---- Load RL policy ----------------------------------------------------
    envs = RslRlVecEnvWrapper(envs)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # Forward use_gru flag so VisionStudentTeacher is constructed with the right architecture
    agent_cfg_dict = agent_cfg.to_dict()
    if "policy" in agent_cfg_dict:
        agent_cfg_dict["policy"]["use_gru"] = args_cli.use_gru
    ppo_runner = OnPolicyRunner(envs, agent_cfg_dict,
                                log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    agent = ppo_runner.get_inference_policy(device=envs.unwrapped.device)

    # Direct reference to Isaac Lab env for camera access
    # envs is now RslRlVecEnvWrapper -> RecordVideo -> ArucoHoverStudentEnv
    isaac_env = envs.unwrapped   # ArucoHoverStudentEnv

    # ---- ONNX export -------------------------------------------------------
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_onnx(
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="teacher_policy.onnx",
        )
    except AttributeError as e:
        print(f"[WARN] ONNX export skipped: {e}")

    # ---- Initial observations ----------------------------------------------
    obs, extras  = envs.get_observations()
    obs_dict     = extras.get("observations", {})
    full_state   = obs_dict.get("full_state", None)
    print("Starting obs: ", full_state)

    max_steps       = int(env_cfg.episode_length_s * env_cfg.policy_rate_hz)
    full_state_size = obs_dict["full_state"].shape[1]

    full_states = torch.zeros((args_cli.num_envs, max_steps, full_state_size),
                              dtype=torch.float32).to(device)
    rewards     = torch.zeros((args_cli.num_envs, max_steps),
                              dtype=torch.float32).to(device)
    actions_log = torch.zeros((args_cli.num_envs, max_steps, 4),
                              dtype=torch.float32).to(device)

    steps      = 0
    done       = False
    done_count = 0
    times      = []

    with torch.no_grad():
        while simulation_app.is_running():

            while steps < max_steps and not done:
                full_states[:, steps, :] = obs_dict["full_state"]

                # ---- Policy forward pass -----------------------------------
                # agent() calls act_inference -> update_distribution -> _encode
                # which stashes _last_marker on ppo_runner.alg.policy
                start   = time.time()
                actions = agent(obs_dict["policy"])
                actions_log[:, steps] = actions
                times.append(time.time() - start)

                # ---- Read marker prediction --------------------------------
                # _last_marker: (N, 5) — [cx, cy, w, h, confidence], all post-Sigmoid,
                # set as a side effect of agent() calling update_distribution() -> _encode
                marker = None
                if hasattr(ppo_runner.alg.policy, "_last_marker"):
                    marker = ppo_runner.alg.policy._last_marker  # (N, 5)

                # ---- Read onboard RGB --------------------------------------
                # rgb: (N, H, W, 3) uint8 — confirmed from debug output
                # This is the frame from the PREVIOUS step (sensor lag of 1),
                # which is acceptable for video overlay purposes.
                rgb_nhwc = None
                try:
                    rgb_data = isaac_env._camera.data.output.get("rgb", None)
                    if rgb_data is not None and rgb_data.ndim == 4:
                        rgb_nhwc = rgb_data  # (N, H, W, 3) uint8, no alpha
                except Exception as e:
                    print(f"[WARN] Camera read failed at step {steps}: {e}")

                # ---- Write onboard frame -----------------------------------
                if rgb_nhwc is not None:
                    #feature_map = getattr(ppo_runner.alg.policy, "_last_feature_map", None)
                    feature_map = None  # Disable feature map overlay for now
                    bbox_tuple = None
                    if args_cli.debug_bbox:
                        _e = envs.unwrapped
                        bbox_raw = _e._compute_bbox_gt()
                        if steps < 5:
                            print(f"[DEBUG] record_env_idx={record_env_idx}, "
                                  f"num_envs={args_cli.num_envs}, follow_robot={args_cli.follow_robot}")
                            print(f"[DEBUG] bbox_raw full tensor: {bbox_raw}")
                        candidate = tuple(bbox_raw[record_env_idx].cpu().tolist())
                        # -1.0 sentinel means marker wasn't in front of the camera this frame
                        if candidate[0] != -1.0:
                            bbox_tuple = candidate[:4]  # GT bbox is (cx, cy, w, h, visible) -> drop visible flag
                        else:
                            print(f"[DEBUG DRAW] step {steps}: marker not visible, skipping GT box")

                    # ---- Predicted bbox overlay (student's own belief) ----------
                    bbox_pred_tuple = None
                    if args_cli.debug_bbox_pred and marker is not None:
                        pred_candidate = tuple(marker[record_env_idx].cpu().tolist())  # (cx, cy, w, h, conf)
                        conf = pred_candidate[4]
                        if steps < 5:
                            print(f"[DEBUG] predicted bbox: cx={pred_candidate[0]:.3f} "
                                  f"cy={pred_candidate[1]:.3f} w={pred_candidate[2]:.3f} "
                                  f"h={pred_candidate[3]:.3f} conf={conf:.3f}")
                        # Only draw when the network itself is confident the marker is visible
                        if conf > 0.5:
                            bbox_pred_tuple = pred_candidate
                        else:
                            print(f"[DEBUG DRAW] step {steps}: predicted confidence {conf:.2f} <= 0.5, skipping pred box")

                    onboard_recorder.write_frame(
                        rgb_nhwc=rgb_nhwc,
                        feature_map=feature_map,
                        env_idx=record_env_idx,
                        bbox_gt=bbox_tuple,
                        bbox_pred=bbox_pred_tuple,
                    )

                # ---- Step environment --------------------------------------
                obs, reward, dones, extras = envs.step(actions)
                # # Add INSIDE the eval loop, right after obs_dict is updated (after envs.step):
                # if args_cli.debug_bbox and steps < 5:
                #     bbox_gt = obs_dict.get("bbox_gt", None)
                #     print(f"[DEBUG] step {steps} bbox_gt: {bbox_gt[0].cpu().tolist() if bbox_gt is not None else None}")
                done_count += dones.sum().item()
                obs_dict    = extras.get("observations",
                                         envs.unwrapped._get_observations())
                info        = extras
                rewards[:, steps] = reward.detach()
                steps += 1
                print(f"Step: {steps}", end="\r")

            print()

            # ---- Finalise onboard video ------------------------------------
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
            print("\nAverage inference time: ", np.mean(times))

            # ---- Plots (unchanged) -----------------------------------------
            if args_cli.follow_robot >= 0:
                import matplotlib.pyplot as plt
                import isaaclab.utils.math as isaac_math_utils

                quad_pos     = full_states[args_cli.follow_robot, :-1, 0:3].cpu().numpy()
                quad_vel     = full_states[args_cli.follow_robot, :-1, 7:10].cpu().numpy()
                quad_ang_vel = full_states[args_cli.follow_robot, :-1, 10:13].cpu().numpy()
                marker_dist  = full_states[args_cli.follow_robot, :-1, 17].cpu().numpy()

                quad_euler = isaac_math_utils.euler_xyz_from_quat(
                    full_states[args_cli.follow_robot, :-1, 3:7]
                )
                quad_roll  = quad_euler[0].cpu().numpy()
                quad_pitch = quad_euler[1].cpu().numpy()
                quad_yaw   = quad_euler[2].cpu().numpy()

                T = rewards.shape[1] - 1
                x = np.arange(T) * (1.0 / env_cfg.policy_rate_hz)

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
                    plt.plot(x, actions_np[args_cli.follow_robot, :-1, i], label=label)
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