import argparse
import sys
from isaaclab.app import AppLauncher

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

    # Export ONNX for Jetson Orin NX TensorRT conversion
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="teacher_policy.onnx",
    )

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