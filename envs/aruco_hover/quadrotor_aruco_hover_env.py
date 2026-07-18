# Copyright (c) 2024, VI-Hover Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Quadrotor ArUco Hover Environment (Teacher + Student)
=====================================================

Task
----
Hover stably at a fixed distance (1.5 m) in front of a vertical wall
bearing an ArUco 6x6 marker, keeping the camera (body-x axis) pointed
directly at the marker.

Control Interface
-----------------
CTBR Collective Thrust + Body Rates  [thrust, roll_rate, pitch_rate, yaw_rate]
Actions in [-1, 1], decoded inside _pre_physics_step.

Observation Definitions
-----------------------

1) Teacher / Privileged Observation (17-dim, ground-truth state)
   Used by the teacher policy and as privileged input for asymmetric RL:

   [0:3]   pos_error_b   hover point minus drone position, expressed in body frame
   [3:6]   lin_vel_b     linear velocity in body frame
   [6:9]   ang_vel_b     angular velocity in body frame
   [9:12]  grav_vec_b    projected gravity unit vector in body frame
   [12:16] last_action   previous [thrust, roll_rate, pitch_rate, yaw_rate]
   [16]    dist_scalar   Euclidean distance to hover point (curriculum proxy)

   Total: 17 dims.

2) Student Policy Observation Groups (visual + IMU)
   Used by the visual student policy:

   - images:   Onboard RGB camera, shape (num_envs, 3, H, W), values in [0, 1].
               Full-resolution forward-facing view of the ArUco marker and wall.

   - policy:   IMU-like vector, shape (num_envs, 10):
               [0:3]   lin_vel_b     linear velocity in body frame
               [3:6]   ang_vel_b     angular velocity in body frame
               [6:10]  last_action   previous [thrust, roll_rate, pitch_rate, yaw_rate]

   - privileged: Same 17-D teacher observation as above, available during training
                 for the critic or distillation, but not required at deployment.

3) Full-State Logging Vector (18-dim)
   Returned under the key "full_state" for evaluation and plotting:

   [0:3]   quadrotor position in world frame
   [3:7]   orientation quaternion [x, y, z, w]
   [7:10]  linear velocity in body frame
   [10:13] angular velocity in body frame
   [13:17] last_action [thrust, roll_rate, pitch_rate, yaw_rate]
   [17]    distance to ArUco hover point

   Total: 18 dims.
"""

from __future__ import annotations

import math
import torch
import numpy as np
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply_inverse,
    subtract_frame_transforms,
    matrix_from_quat,
)
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg

# Robot asset
from configs.quadrotor_asset import QUADROTOR_CFG

import os
from utils.assets import MODELS_PATH
import random

USD_MARKER_PATH = os.path.join(
    MODELS_PATH,
    "aruco_marker",
    "aruco_marker.usd",
)
ARENA_SIZE = 3.0
WALL_HEIGHT = 3.0
WALL_THICKNESS = 0.05

import omni

def get_replicator():
    """Enable and import omni.replicator.core safely without Isaac Lab helper."""
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled("omni.replicator.core", True)
    import omni.replicator.core as rep
    return rep
    
# =============================================================================
#  DEBUG WINDOW
# =============================================================================

class ArucoHoverEnvWindow(BaseEnvWindow):
    """Minimal UI window shown in the Isaac Lab viewport."""

    def __init__(self, env, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# =============================================================================
#  CONFIGURATION DATACLASS
# =============================================================================

@configclass
class ArucoHoverEnvCfg(DirectRLEnvCfg): 
    """
    Single source of truth for all hyper-parameters.
    Never hard-code magic numbers in the env class put them here.
    """

    # ---------- Episode / timing --------------------------------------------
    episode_length_s: float = 10.0

    sim_rate_hz: int = 500          # PhysX step rate
    policy_rate_hz: int = 50        # Policy query rate (10x decimation)
    pd_loop_rate_hz: int = 250      # Inner CTBR attitude-rate PD loop

    decimation: int = sim_rate_hz // policy_rate_hz           # 10
    pd_loop_decimation: int = sim_rate_hz // pd_loop_rate_hz  # 2

    # ---------- Action / observation sizes ----------------------------------
    action_space: int = 4      # [collective_thrust, roll_rate, pitch_rate, yaw_rate]
    # observation_space: int = 17
    observation_space: int = 20      # was 17 — +3 for marker direction vector
    state_space: int = 0

    num_actions: int = action_space
    num_observations: int = observation_space

    ui_window_class_type = ArucoHoverEnvWindow
    debug_vis: bool = True

    # ---------- Simulation --------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / sim_rate_hz,
        render_interval=decimation,
        render=sim_utils.RenderCfg(
           rendering_mode="performance",
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Ground plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ---------- Scene ----------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=4.0,      # wider than default; wall extends +-1 m
        replicate_physics=True,
    )

    # ---------- Robot (Crazyflie 2.1 (For now)) ----------
    robot: ArticulationCfg = QUADROTOR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Physical parameters Crazyflie 2.1 defaults
    mass: float = 0.041         # kg
    Ixx = 1.5417e-5             # kg.m^2
    Iyy = 1.5904e-5
    Izz = 2.869e-5
    thrust_to_weight: float = 2.2
    arm_length: float = 0.043    # m, centre-to-motor
    k_eta: float = 2.3e-8        # thrust coefficient [N / (rad/s)^2]
    k_m: float = 7.8e-10         # reaction-torque coefficient
    tau_m: float = 0.005         # motor first-order time constant [s]
    motor_speed_min: float = 0.0
    motor_speed_max: float = 2500.0
    init_motor_speed: float = 1788.53  # rad/s near hover equilibrium

    # ---------- CTBR inner-loop PD gains ----------
    control_mode: str = "CTBR"
    body_rate_scale_xy: float = 6.2832   # +- 2*pi rad/s for roll/pitch
    body_rate_scale_z: float  = 1.5708   # +- pi/2 rad/s for yaw
    kp_omega: float = 75.0
    kd_omega: float = 10.0

    # ---------- Hover goal geometry ----------
    # Wall face placed at x = wall_x_pos (local env frame).
    # Drone hovers at (wall_x_pos - hover_distance, 0, marker_height).
    wall_x_pos: float = 2.0        # [m] wall face x position in local env frame
    hover_distance: float = 1.5    # [m] desired standoff
    marker_height: float = 1.5     # [m] marker centre height above ground

    # ---------- Curriculum for init noise (yaw / position) ----------
    curriculum_enabled: bool = True
    init_yaw_noise_start: float = 0.15
    init_yaw_noise_end: float = 1.2
    init_pos_noise_start: list = [0.15, 0.15, 0.15]
    init_pos_noise_end: list = [0.4, 0.4, 0.4]
    # Negative means normal automatic behavior.
    # A value in [0, 1] freezes the reset distribution at that
    # curriculum difficulty, which is useful for DAgger collection.
    forced_curriculum_level: float = -1.0
    init_vel_noise: float = 0.1
    init_ang_vel_noise: float = 0.1
    # init_pos_noise_start: list = [0.0, 0.0, 0.0]
    # init_pos_noise_end: list = [0.0, 0.0, 0.0]
    # init_vel_noise: float = 0.0
    # init_ang_vel_noise: float = 0.0

    # Performance-based curriculum thresholds
    # curriculum_step_size: float = 0.05        # fraction of [start, end] range advanced per promotion
    # curriculum_alignment_threshold: float = 0.85   # mean alignment_score needed to advance
    # curriculum_proximity_threshold: float = 0.7    # mean (gated) proximity reward needed to advance
    curriculum_demotion_factor: float = 0.5   # how much to step back on regression (vs step forward)
    # curriculum_eval_window: int = 20          # number of resets to average performance over before judging

    curriculum_alignment_threshold: float = 0.90   # was 0.85
    curriculum_proximity_threshold: float = 0.80   # was 0.70
    curriculum_eval_window: int = 40               # was 20 — more stable signal
    curriculum_step_size: float = 0.03            # was 0.05 — slower progression

    # ---------- Domain randomisation (multiplicative, fraction of nominal) ----------
    dr_dict: dict = {
        "mass":             0.15,   # +-15%
        "inertia":          0.10,   # +-10%
        "thrust_to_weight": 0.10,
        "k_eta":            0.10,
        "tau_m":            0.15,
        "kp_omega":         0.10,
        "kd_omega":         0.10,
    }

    # ---------- Visual Domain Randomisation ----------
    # Applied per-reset to make the student robust to lighting/color variation
    visual_dr_enabled: bool = True
    wall_color_range: float = 0.3       # +-30% around white on background regions
    lighting_intensity_range: tuple = (800.0, 3000.0)   # dome light range
    lighting_color_temp_range: tuple = (0.6, 1.0)       # warm to cool white
    num_distractor_boxes: int = 4       # random colored boxes in background

    # ---------- Reward weights ----------
    w_alive: float = 0.02            # survival bonus per step
    w_proximity: float = 3.0         # Gaussian proximity scale
    proximity_sigma: float = 1.0    # Gaussian std [m]
    w_lin_vel: float = 0.05          # linear velocity penalty weight
    w_ang_vel: float = 0.02          # angular velocity penalty weight
    w_orientation: float = 1.0       # facing-error (1 - cos theta) weight
    alignment_power: int = 2         # exponent for the gating term
    w_action_rate: float = 0.10      # action-delta (jerk) penalty weight

    # ---------- Termination thresholds ----------
    max_pos_error: float = 4.0      # [m] from hover point
    max_tilt_deg: float = 75.0      # [deg] tilt limit
    min_height: float = 0.10        # [m] below this -> terminated

    # ---------- Onboard RGB camera ----------
    # camera: CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/Camera",
    #     update_period=1.0 / policy_rate_hz,   # same as policy rate
    #     width=64,
    #     height=64,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 20.0),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 0.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),      # facing +Z in ROS convention
    #         convention="ros",
    #     ),
    # )

    # The corners policy does not require an RGB sensor.
    # Enable this only when recording onboard video.
    enable_onboard_camera: bool = True

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",  # changed: body/Camera not Robot/Camera
        update_period=1.0 / policy_rate_hz,
        width=64,
        height=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),   # identity — no extra rotation
            convention="world",
        ),
    )

    # Number of consecutive RGB frames supplied to the student policy.
    visual_history_length: int = 32
    # Start clean. Add real-detector noise later.
    corner_noise_px: float = 0.0
    corner_dropout_prob: float = 0.0
    marker_size: float = 0.10

    # ---------- Misc ----------
    scale_reward_with_time: bool = True  # multiply rewards by step_dt
    action_history_length: int = 1       # reserved for DAgger extension
    seed: int = 0

@configclass
class ArucoHoverStudentEnvCfg(ArucoHoverEnvCfg):
    """
    Corners-TCN student and DAgger environment.

    Automatic curriculum is disabled. A fixed difficulty may be selected
    through forced_curriculum_level.

    The lateral-position and yaw ranges are deliberately restricted so a
    corners-only policy normally receives an observable marker.
    """

    curriculum_enabled: bool = False
    forced_curriculum_level: float = -1.0

    # Depth and height remain strongly randomized.
    # Lateral displacement is restricted to preserve marker visibility.
    init_pos_noise_start: list = [0.15, 0.05, 0.15]
    init_pos_noise_end: list = [0.40, 0.15, 0.40]

    # Approximately ±2.9 degrees to ±10.3 degrees.
    init_yaw_noise_start: float = 0.05
    init_yaw_noise_end: float = 0.18

    init_vel_noise: float = 0.10
    init_ang_vel_noise: float = 0.10

    visual_history_length: int = 32

class ArucoHoverTeacherEnv(DirectRLEnv):
    """
    Vectorised Isaac Lab environment for the ArUco marker hover task.

    Key design decisions
    --------------------
    * Goal is a FIXED hover point no trajectory tracking.
    * Wall is a kinematic RigidObject (does not fall; collisions still register).
    * CTBR control with a sub-decimation inner PD loop mirrors your reference env.
    * Domain randomisation applied per-reset for sim-to-real robustness.
    """

    cfg: ArucoHoverEnvCfg

    # ---------- __init__ ----------
    def __init__(self, cfg: ArucoHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        torch.manual_seed(self.cfg.seed)

        # Hover target in world frame — set properly in _reset_idx once
        # env_origins are available from the terrain.
        self._hover_point_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._marker_point_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ---------- Actions / control buffers ----------
        self._actions          = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_action  = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_omega_err = torch.zeros(self.num_envs, 3, device=self.device)
        self.pd_loop_counter   = 0

        # Wrench buffer: [thrust_z, Mx, My, Mz]
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        # External force (N, 1, 3) and torque (N, 1, 3) in body frame
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Motor state
        self._motor_speeds     = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)

        # ---------- Per-env randomisable physical parameters ----------
        self._robot_mass       = self.cfg.mass             * torch.ones(self.num_envs, device=self.device)
        self._thrust_to_weight = self.cfg.thrust_to_weight * torch.ones(self.num_envs, device=self.device)
        self._k_eta            = self.cfg.k_eta            * torch.ones(self.num_envs, device=self.device)
        self._tau_m            = self.cfg.tau_m            * torch.ones(self.num_envs, device=self.device)
        self._arm_length       = self.cfg.arm_length       * torch.ones(self.num_envs, device=self.device)
        self._k_m              = self.cfg.k_m              * torch.ones(self.num_envs, device=self.device)
        self._kp_omega         = self.cfg.kp_omega         * torch.ones(self.num_envs, device=self.device)
        self._kd_omega         = self.cfg.kd_omega         * torch.ones(self.num_envs, device=self.device)

        # ---------- Gravity & inertia ----------
        self._gravity_magnitude = torch.tensor(
            self.sim.cfg.gravity, device=self.device
        ).norm()
        self._grav_unit_w = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)

        # Inertia tensor: (N, 3, 3) diagonal
        self.inertia_tensor = torch.diag(
            torch.tensor([self.cfg.Ixx, self.cfg.Iyy, self.cfg.Izz],
                         device=self.device)
        ).unsqueeze(0).expand(self.num_envs, -1, -1).clone()

        # ---------- Build motor allocation matrices ----------
        self._build_motor_allocation()

        # ---------- Episode reward logging ----------
        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in ["alive", "proximity", "lin_vel", "ang_vel",
                      "orientation", "action_rate"]
        }

        # ---------- Robot body index ----------
        self._body_id = self._robot.find_bodies("body")[0]

        self.set_debug_vis(self.cfg.debug_vis)

        # ---------- Performance-based curriculum state ----------
        initial_curriculum_level = (
            float(self.cfg.forced_curriculum_level)
            if self.cfg.forced_curriculum_level >= 0.0
            else 0.0
        )

        self._curriculum_level = torch.tensor(
            initial_curriculum_level,
            device=self.device,
            dtype=torch.float32,
        )
        self._curriculum_alignment_buf = torch.zeros(self.cfg.curriculum_eval_window, device=self.device)
        self._curriculum_proximity_buf = torch.zeros(self.cfg.curriculum_eval_window, device=self.device)
        self._curriculum_buf_idx = 0
        self._curriculum_buf_filled = False

    # ---------- MOTOR ALLOCATION ----------
    def _build_motor_allocation(self):
        """
        Build the 4x4 thrust-to-wrench matrix f_to_TM for all envs.

        Plus-configuration (standard Crazyflie 2.1):
          Motor 0: (+L/sqrt2, +L/sqrt2, 0)  CW  -> dir = +1
          Motor 1: (+L/sqrt2, -L/sqrt2, 0)  CCW -> dir = -1
          Motor 2: (-L/sqrt2, -L/sqrt2, 0)  CW  -> dir = +1
          Motor 3: (-L/sqrt2, +L/sqrt2, 0)  CCW -> dir = -1

        Row layout of f_to_TM:
          Row 0: thrust (sum of motor forces)
          Rows 1-2: roll/pitch moments (cross product of position x z-hat)
          Row 3: yaw moment (reaction torques, sign from rotor direction)
        """
        r2o2 = math.sqrt(2.0) / 2.0
        L = self.cfg.arm_length
        dirs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)

        rotor_pos = torch.tensor([
            [ r2o2,  r2o2, 0.0],
            [ r2o2, -r2o2, 0.0],
            [-r2o2, -r2o2, 0.0],
            [-r2o2,  r2o2, 0.0],
        ], device=self.device) * L  # (4, 3)

        kappa = self.cfg.k_m / self.cfg.k_eta   # torque-to-thrust ratio

        # cross(rotor_pos, z_hat) gives the moment arm for each motor
        z_hat = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(4, -1)
        cross = torch.linalg.cross(rotor_pos, z_hat)  # (4, 3)

        # Build single-env 4x4 matrix
        f_to_TM_single = torch.cat([
            torch.ones(1, 4, device=self.device),           # thrust row
            cross[:, :2].T,                                  # Mx, My (2, 4)
            (kappa * dirs).unsqueeze(0),                     # Mz (1, 4)
        ], dim=0)  # (4, 4)

        self.f_to_TM = f_to_TM_single.unsqueeze(0).expand(
            self.num_envs, -1, -1
        ).clone()
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        # Cache rotor directions for DR rebuilds
        self._rotor_dirs = dirs.unsqueeze(0).expand(self.num_envs, -1).clone()

    def _rebuild_motor_allocation(self, env_ids: torch.Tensor):
        """Recompute f_to_TM for envs whose k_eta or arm_length was randomised."""
        n = len(env_ids)
        r2o2 = math.sqrt(2.0) / 2.0
        L    = self._arm_length[env_ids]  # (n,)

        rotor_pos = torch.tensor([
            [ r2o2,  r2o2, 0.0],
            [ r2o2, -r2o2, 0.0],
            [-r2o2, -r2o2, 0.0],
            [-r2o2,  r2o2, 0.0],
        ], device=self.device).unsqueeze(0) * L.view(n, 1, 1)  # (n, 4, 3)

        kappa = self._k_m[env_ids] / self._k_eta[env_ids]  # (n,)
        dirs  = self._rotor_dirs[env_ids]                   # (n, 4)

        z_hat = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device
        ).expand(n, 4, -1)
        cross = torch.linalg.cross(rotor_pos, z_hat)  # (n, 4, 3)

        self.f_to_TM[env_ids] = torch.cat([
            torch.ones(n, 1, 4, device=self.device),
            cross[:, :, :2].permute(0, 2, 1),                # (n, 2, 4)
            (kappa.view(n, 1, 1) * dirs.view(n, 1, 4)),      # (n, 1, 4)
        ], dim=1)  # (n, 4, 4)

        self.TM_to_f[env_ids] = torch.linalg.inv(self.f_to_TM[env_ids])

    def _build_material_pool(self) -> list:
        if not self.cfg.visual_dr_enabled:
            return []

        try:
            rep = get_replicator()
        except Exception as e:
            print(f"[WARN] Replicator unavailable: {e}")
            return []

        import omni.usd
        stage = omni.usd.get_context().get_stage()

        texture_dir = os.path.join(MODELS_PATH, "textures", "generated")
        try:
            self._texture_files_cache = sorted([
                os.path.join(texture_dir, f)
                for f in os.listdir(texture_dir) if f.endswith(".png")
            ])
        except FileNotFoundError:
            print(f"[WARN] Texture dir not found: {texture_dir}")
            return []

        if not self._texture_files_cache:
            print("[WARN] No textures found in generated dir")
            return []

        dummy_prim = stage.GetPrimAtPath("/World/envs/env_0/WallPX")
        if not dummy_prim.IsValid():
            print("[WARN] WallPX not found, cannot build material pool")
            return []

        pool = []
        for tex_path in self._texture_files_cache:
            mat_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl",
                bind_prims=[dummy_prim],
                count=1,
                project_uvw=True,
            )
            rep.functional.modify.attribute(
                mat_prims, "diffuse_texture", [tex_path]
            )
            rep.functional.modify.attribute(
                mat_prims, "texture_scale", [(10.0, 10.0)]
            )
            pool.append(mat_prims[0].GetPath().pathString)

        print(f"[INFO] Material pool ready: {len(pool)} materials")
        return pool

    def _randomize_wall_textures(self, env_ids: torch.Tensor | None):
        if not self.cfg.visual_dr_enabled:
            return
        if not hasattr(self, '_wall_material_pool') or not self._wall_material_pool:
            return

        import omni.usd
        from pxr import UsdShade

        stage = omni.usd.get_context().get_stage()
        n_mats = len(self._wall_material_pool)
        wall_names = ["WallPX", "WallNX", "WallPY", "WallNY"]
        env_indices = env_ids.tolist() if env_ids is not None else range(self.num_envs)

        for env_idx in env_indices:
            mat_path = self._wall_material_pool[random.randint(0, n_mats - 1)]
            material = UsdShade.Material(stage.GetPrimAtPath(mat_path))
            for wall_name in wall_names:
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/{wall_name}")
                if prim.IsValid():
                    if prim.IsInstanceable():
                        prim.SetInstanceable(False)
                    UsdShade.MaterialBindingAPI(prim).Bind(
                        material, UsdShade.Tokens.strongerThanDescendants
                    )
                                    
    def _apply_aruco_texture(self):
        """Apply ArUco PNG to wall via USD OmniPBR — called once after scene setup."""
        import omni.usd
        from pxr import UsdShade, Sdf

        stage = omni.usd.get_context().get_stage()
        texture_path = os.path.join(MODELS_PATH, "textures", "aruco_6x6_marker.png")

        for i in range(min(self.num_envs, 1)):  # instanceable — only need env_0
            wall_path = f"/World/envs/env_{i}/Wall"
            wall_prim = stage.GetPrimAtPath(wall_path)
            if not wall_prim.IsValid():
                print(f"[WARN] Wall prim not found at {wall_path}")
                return

            mat_path = f"/World/Looks/ArucoMarkerMat"
            material = UsdShade.Material.Define(stage, mat_path)
            shader   = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("OmniPBR")
            shader.CreateInput("diffuse_texture",    Sdf.ValueTypeNames.Asset).Set(texture_path)
            shader.CreateInput("project_uvw",        Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("roughness_constant", Sdf.ValueTypeNames.Float).Set(0.8)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI(wall_prim).Bind(material)
            print(f"[INFO] ArUco texture applied: {texture_path}")
    
    def _randomise_visuals(self, env_ids: torch.Tensor):
        """
        Per-reset visual DR for student policy robustness:
        1. Randomise dome light intensity and color temperature
        2. Randomise wall background color (ArUco itself stays fixed via USD)
        3. Place distractor colored boxes in the background
        """
        if not self.cfg.visual_dr_enabled:
            return

        try:
            import omni.usd
            from pxr import UsdLux, Sdf, Gf, UsdGeom, UsdShade
        except ImportError:
            return

        stage = omni.usd.get_context().get_stage()

        # 1. Randomise dome light
        light_prim = stage.GetPrimAtPath("/World/Light")
        if light_prim.IsValid():
            intensity = float(torch.empty(1).uniform_(
                *self.cfg.lighting_intensity_range
            ).item())
            temp = float(torch.empty(1).uniform_(
                *self.cfg.lighting_color_temp_range
            ).item())
            light = UsdLux.DomeLight(light_prim)
            light.GetIntensityAttr().Set(intensity)
            r = float(torch.empty(1).uniform_(temp, 1.0).item())
            b = float(torch.empty(1).uniform_(temp, 1.0).item())
            light.GetColorAttr().Set(Gf.Vec3f(r, 1.0, b))

        # 2. Randomise wall material color (background tint — ArUco PNG is on top)
        mat_prim = stage.GetPrimAtPath("/World/Looks/ArucoMarkerMat/Shader")
        if mat_prim.IsValid():
            shader = UsdShade.Shader(mat_prim)
            tint_input = shader.GetInput("diffuse_tint")
            if not tint_input:
                tint_input = shader.CreateInput(
                    "diffuse_tint", Sdf.ValueTypeNames.Color3f
                )
            tint = float(torch.empty(1).uniform_(
                1.0 - self.cfg.wall_color_range, 1.0
            ).item())
            tint_input.Set(Gf.Vec3f(tint, tint, tint))

    def _setup_scene(self):
        """
        Spawn:
          1. Crazyflie drone (Articulation)
          2. ArUco marker wall (kinematic RigidObject)
          3. Flat terrain + clone environments
          4. Dome lighting
        """
        # 1. Drone
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # 2. ArUco wall (USD marker asset)
        marker_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Marker",
            spawn=sim_utils.UsdFileCfg(
                usd_path=USD_MARKER_PATH,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    self.cfg.wall_x_pos - 0.03,
                    0.0,
                    self.cfg.marker_height,
                ),
                rot=(0.7071, 0.0, 0.7071, 0.0),
            ),
        )
        self._marker = RigidObject(marker_cfg)
        self.scene.rigid_objects["marker"] = self._marker

        # 2b. Enclosure walls
        wall_px_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/WallPX",
            spawn=sim_utils.CuboidCfg(
                size=(WALL_THICKNESS, ARENA_SIZE, WALL_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.8, 0.8),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    self.cfg.wall_x_pos,
                    0.0,
                    WALL_HEIGHT / 2,
                ),
            ),
        )
        self._wall_px = RigidObject(wall_px_cfg)
        self.scene.rigid_objects["wall_px"] = self._wall_px

        wall_nx_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/WallNX",
            spawn=sim_utils.CuboidCfg(
                size=(WALL_THICKNESS, ARENA_SIZE, WALL_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.8, 0.8),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    -1.0,
                    0.0,
                    WALL_HEIGHT / 2,
                ),
            ),
        )
        self._wall_nx = RigidObject(wall_nx_cfg)
        self.scene.rigid_objects["wall_nx"] = self._wall_nx

        wall_py_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/WallPY",
            spawn=sim_utils.CuboidCfg(
                size=(ARENA_SIZE, WALL_THICKNESS, WALL_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.8, 0.8),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    0.5,
                    ARENA_SIZE / 2,
                    WALL_HEIGHT / 2,
                ),
            ),
        )
        self._wall_py = RigidObject(wall_py_cfg)
        self.scene.rigid_objects["wall_py"] = self._wall_py

        wall_ny_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/WallNY",
            spawn=sim_utils.CuboidCfg(
                size=(ARENA_SIZE, WALL_THICKNESS, WALL_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.8, 0.8),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    0.5,
                    -ARENA_SIZE / 2,
                    WALL_HEIGHT / 2,
                ),
            ),
        )
        self._wall_ny = RigidObject(wall_ny_cfg)
        self.scene.rigid_objects["wall_ny"] = self._wall_ny

        # 3. Terrain clone
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(
            global_prim_paths=[self.cfg.terrain.prim_path]
        )

        # 4. Lighting
        light_intensity = float(np.random.uniform(500, 5000))
        light_color = (
            float(np.random.uniform(0.85, 1.0)),
            float(np.random.uniform(0.85, 1.0)),
            float(np.random.uniform(0.85, 1.0)),
        )
        light_cfg = sim_utils.DomeLightCfg(
            intensity=light_intensity,
            color=light_color,
        )
        light_cfg.func("/World/Light", light_cfg)

        # Optional: self._apply_aruco_texture()
        self._wall_material_pool = self._build_material_pool()

    # ---------- CONTROL HELPERS ----------

    def _compute_motor_speeds_des(self) -> torch.Tensor:
        """
        Invert the wrench allocation:
            f_des = TM_to_f @ wrench_des
            speed = sign(f) * sqrt(|f| / k_eta)
        Clamped to [motor_speed_min, motor_speed_max].
        """
        f_des = torch.bmm(
            self.TM_to_f, self._wrench_des.unsqueeze(-1)
        ).squeeze(-1)  # (N, 4)

        speed_sq  = f_des / self._k_eta.unsqueeze(-1)
        speeds    = torch.sign(speed_sq) * torch.sqrt(speed_sq.abs())
        return speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)

    def _get_moment_from_ctbr(self, actions: torch.Tensor) -> torch.Tensor:
        """
        CTBR PD inner loop.

        Desired body rates decoded from normalised actions:
            omega_des_x = actions[:,1] * body_rate_scale_xy
            omega_des_y = actions[:,2] * body_rate_scale_xy
            omega_des_z = actions[:,3] * body_rate_scale_z

        Returns commanded moment (N, 3) in body frame via:
            tau = I * (-kp * omega_err - kd * domega_err/dt)
        """
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_des[:, 0] = actions[:, 1] * self.cfg.body_rate_scale_xy
        omega_des[:, 1] = actions[:, 2] * self.cfg.body_rate_scale_xy
        omega_des[:, 2] = actions[:, 3] * self.cfg.body_rate_scale_z

        omega     = self._robot.data.root_ang_vel_b  # current body rates (N, 3)
        omega_err = omega - omega_des

        # Finite-difference derivative of the error (crude but fast)
        omega_dot_err = (
            omega_err - self._previous_omega_err
        ) / self.cfg.pd_loop_rate_hz
        self._previous_omega_err = omega_err.clone()

        kp = self._kp_omega.view(self.num_envs, 1)  # per-env gain (after DR)
        kd = self._kd_omega.view(self.num_envs, 1)
        omega_dot_des = -kp * omega_err - kd * omega_dot_err  # (N, 3)

        # tau = I * alpha_des
        moment = torch.bmm(
            self.inertia_tensor, omega_dot_des.unsqueeze(-1)
        ).squeeze(-1)  # (N, 3)
        return moment

    # ---------- ISAAC LAB MANDATORY OVERRIDES ----------

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Called once per policy step (every `decimation` physics steps).
        Decodes the 4-dim normalised action into a desired wrench.
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Collective thrust: map [-1, 1] -> [0, m*g*T2W]
        robot_weight = self._robot_mass * self._gravity_magnitude  # (N,)
        max_thrust   = robot_weight * self._thrust_to_weight        # (N,)
        self._wrench_des[:, 0] = (
            (self._actions[:, 0] + 1.0) / 2.0
        ) * max_thrust

        # Attitude moments from CTBR PD
        self._wrench_des[:, 1:] = self._get_moment_from_ctbr(self._actions)

        # Desired motor speeds (cached; updated again in _apply_action)
        self._motor_speeds_des = self._compute_motor_speeds_des()
        self.pd_loop_counter = 0

    def _apply_action(self):
        """
        Called every physics step.
        Re-runs the CTBR PD at pd_loop_rate_hz, then steps motor dynamics.
        """
        # Re-compute inner-loop moments at the higher PD frequency
        if self.pd_loop_counter % self.cfg.pd_loop_decimation == 0:
            self._wrench_des[:, 1:] = self._get_moment_from_ctbr(self._actions)
            self._motor_speeds_des  = self._compute_motor_speeds_des()
        self.pd_loop_counter += 1

        # First-order motor dynamics: alpha = exp(-dt / tau_m)
        alpha = torch.exp(
            -self.physics_dt / self._tau_m
        ).unsqueeze(-1)  # (N, 1)
        self._motor_speeds = (
            alpha * self._motor_speeds
            + (1.0 - alpha) * self._motor_speeds_des
        ).clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)

        # Reconstruct actual wrench from motor speeds
        motor_forces = self._k_eta.unsqueeze(-1) * self._motor_speeds ** 2
        wrench = torch.bmm(
            self.f_to_TM, motor_forces.unsqueeze(-1)
        ).squeeze(-1)  # (N, 4)

        # Apply to the physics body
        self._thrust[:, 0, 2] = wrench[:, 0]   # z-component of thrust in body
        self._moment[:, 0, :]  = wrench[:, 1:]  # body moments
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )
    
    def _get_curriculum_noise(
        self,
    ) -> tuple[torch.Tensor, float]:
        """
        Return reset position and yaw ranges for the active difficulty.

        Priority:
        1. forced_curriculum_level when it is in [0, 1]
        2. maximum difficulty when automatic curriculum is disabled
        3. current performance-based curriculum level
        """

        forced_level = float(
            self.cfg.forced_curriculum_level
        )

        if forced_level >= 0.0:
            level = max(
                0.0,
                min(1.0, forced_level),
            )
        elif not self.cfg.curriculum_enabled:
            level = 1.0
        else:
            level = float(
                self._curriculum_level.item()
            )

        pos_start = torch.tensor(
            self.cfg.init_pos_noise_start,
            device=self.device,
            dtype=torch.float32,
        )

        pos_end = torch.tensor(
            self.cfg.init_pos_noise_end,
            device=self.device,
            dtype=torch.float32,
        )

        pos_noise = (
            pos_start
            + level * (pos_end - pos_start)
        )

        yaw_noise = (
            self.cfg.init_yaw_noise_start
            + level
            * (
                self.cfg.init_yaw_noise_end
                - self.cfg.init_yaw_noise_start
            )
        )

        return pos_noise, float(yaw_noise)
        
    def _update_curriculum(self, env_ids: torch.Tensor):
        """
        Performance-gated curriculum update. Called once per reset batch.
        Tracks rolling mean of (alignment, proximity) episode-average reward
        components; promotes difficulty when both exceed threshold over a
        window of resets, demotes (partially) if performance regresses hard.
        """
        if (
            not self.cfg.curriculum_enabled
            or self.cfg.forced_curriculum_level >= 0.0
        ):
            return

        # Per-episode means for the envs that just reset, normalised the same
        # way as the logged Episode_Reward/* extras (divide by episode length).
        mean_alignment = (
            self._episode_sums["orientation"][env_ids].mean() / self.max_episode_length_s
        ) / max(self.cfg.w_orientation, 1e-6)   # undo the w_orientation scale -> back to [0,1] alignment_score
        mean_proximity = (
            self._episode_sums["proximity"][env_ids].mean() / self.max_episode_length_s
        ) / max(self.cfg.w_proximity, 1e-6)     # undo the w_proximity scale -> rough [0,1] proxy

        # Push into rolling buffers
        idx = self._curriculum_buf_idx % self.cfg.curriculum_eval_window
        self._curriculum_alignment_buf[idx] = mean_alignment
        self._curriculum_proximity_buf[idx] = mean_proximity
        self._curriculum_buf_idx += 1
        if self._curriculum_buf_idx >= self.cfg.curriculum_eval_window:
            self._curriculum_buf_filled = True

        if not self._curriculum_buf_filled:
            return   # not enough data yet to judge performance

        avg_alignment = self._curriculum_alignment_buf.mean()
        avg_proximity = self._curriculum_proximity_buf.mean()

        if (avg_alignment > self.cfg.curriculum_alignment_threshold
                and avg_proximity > self.cfg.curriculum_proximity_threshold):
            # Promote: advance level, reset window so next judgment uses fresh data
            self._curriculum_level = torch.clamp(
                self._curriculum_level + self.cfg.curriculum_step_size, 0.0, 1.0
            )
            self._curriculum_buf_filled = False
            self._curriculum_buf_idx = 0
        elif avg_alignment < self.cfg.curriculum_alignment_threshold * 0.5:
            # Regression: performance collapsed at current difficulty, step back
            self._curriculum_level = torch.clamp(
                self._curriculum_level - self.cfg.curriculum_step_size * self.cfg.curriculum_demotion_factor,
                0.0, 1.0,
            )
            self._curriculum_buf_filled = False
            self._curriculum_buf_idx = 0

    def _get_observations(self) -> dict:
        """
        Returns a dict with key "policy" containing the 20-dim observation.

        All quantities are in the BODY frame to make the obs position-invariant.
        [0:3]   pos_err_b        hover point minus drone position, body frame
        [3:6]   lin_vel_b
        [6:9]   ang_vel_b
        [9:12]  grav_vec_b
        [12:15] marker_dir_b     unit vector toward the ArUco marker, body frame  <-- NEW
        [15:19] last_action
        [19]    dist_scalar
        """
        pos_w = self._robot.data.root_pos_w    # (N, 3) world pos
        ori_w = self._robot.data.root_quat_w   # (N, 4) wxyz quaternion
        lv_w  = self._robot.data.root_lin_vel_w
        av_b  = self._robot.data.root_ang_vel_b

        # [0:3] Position error rotated into body frame (hover point, NOT marker)
        pos_err_w = self._hover_point_w - pos_w
        pos_err_b = quat_apply_inverse(ori_w, pos_err_w)

        # [3:6] Linear velocity in body frame
        lin_vel_b = quat_apply_inverse(ori_w, lv_w)

        # [9:12] Projected gravity vector in body frame
        grav_b = quat_apply_inverse(ori_w, self._grav_unit_w)

        # ---- NEW: direction-to-marker vector, body frame --------------------
        # Marker world position is fixed per-env: hover point + hover_distance
        # along local +X (this matches how the marker is spawned in _setup_scene,
        # at wall_x_pos - 0.03, i.e. hover_distance ahead of the hover point).
        marker_dir_w = self._marker_point_w - pos_w           # (N, 3) drone -> marker
        marker_dist  = torch.linalg.norm(marker_dir_w, dim=-1, keepdim=True).clamp(min=1e-6)
        marker_dir_w_unit = marker_dir_w / marker_dist
        marker_dir_b = quat_apply_inverse(ori_w, marker_dir_w_unit)  # (N, 3) unit vector, body frame

        # [16] Scalar distance to HOVER POINT (curriculum proxy, unchanged semantics)
        dist = torch.linalg.norm(pos_err_w, dim=-1, keepdim=True)

        obs = torch.cat([
            pos_err_b,              # 3  -> [0:3]
            lin_vel_b,              # 3  -> [3:6]
            av_b,                   # 3  -> [6:9]
            grav_b,                 # 3  -> [9:12]
            marker_dir_b,           # 3  -> [12:15]  NEW
            self._previous_action,  # 4  -> [15:19]
            dist,                   # 1  -> [19]
        ], dim=-1)  # (N, 20)

        # Full world-frame state for logging/eval (unchanged, still 18-D)
        full_state = torch.cat([
            pos_w,
            ori_w,
            lv_w,
            av_b,
            self._previous_action,
            dist,
        ], dim=-1)  # (N, 18)

        return {"policy": obs, "full_state": full_state}

    def _get_rewards(self) -> torch.Tensor:
        """
        Reward terms with alignment-gated proximity:

        r_t = w_alive
            + alignment_score^power * w_prox * exp( -||delta_p||^2 / (2*sigma^2) )
            - w_vlin * ||v_body||
            - w_vang * ||omega_body||
            + w_ori * alignment_score                      # BONUS, not penalty
            - w_rate * ||a_t - a_{t-1}||

        alignment_score = max(0, cos_theta)   where cos_theta is between
        body-X axis and the unit vector pointing at the ArUco marker.
        Gating means proximity reward is ~zero unless the drone is looking
        at the marker; the bonus term independently rewards good alignment
        even before the drone has closed the distance.
        """
        pos_w = self._robot.data.root_pos_w
        ori_w = self._robot.data.root_quat_w
        lv_w  = self._robot.data.root_lin_vel_w
        av_b  = self._robot.data.root_ang_vel_b

        # --------------Term 1: Survival bonus-----------------------------------
        r_alive = torch.full(
            (self.num_envs,), self.cfg.w_alive, device=self.device
        )

        # --------------Shared: alignment to ACTUAL marker direction-------------
        # (NOT the static world +X target used previously — this now tracks the
        #  true marker position, so curriculum-randomised yaw/position still
        #  produces a correct alignment signal.)
        marker_dir_w = self._marker_point_w - pos_w                    # (N, 3)
        marker_dist  = torch.linalg.norm(marker_dir_w, dim=-1, keepdim=True).clamp(min=1e-6)
        marker_dir_w_unit = marker_dir_w / marker_dist                 # (N, 3)

        R = matrix_from_quat(ori_w)             # (N, 3, 3)
        body_x_w = R[:, :, 0]                   # (N, 3) body-X axis in world frame

        cos_theta = (body_x_w * marker_dir_w_unit).sum(dim=-1).clamp(-1.0, 1.0)  # (N,)
        alignment_score = cos_theta.clamp(min=0.0) ** self.cfg.alignment_power  # (N,) in [0, 1]

        # --------------Term 2: Gaussian proximity reward, GATED by alignment----
        pos_err_w  = self._hover_point_w - pos_w          # (N, 3) — still hover point, not marker
        dist_sq    = (pos_err_w ** 2).sum(dim=-1)
        sigma_sq   = self.cfg.proximity_sigma ** 2
        r_proximity_raw = self.cfg.w_proximity * torch.exp(
            -dist_sq / (2.0 * sigma_sq)
        )
        r_proximity = alignment_score * r_proximity_raw   # zero unless facing marker

        # --------------Term 3: Linear velocity penalty---------------------------
        lin_vel_b = quat_apply_inverse(ori_w, lv_w)
        p_lin_vel = self.cfg.w_lin_vel * torch.linalg.norm(lin_vel_b, dim=-1)

        # --------------Term 4: Angular velocity penalty---------------------------
        p_ang_vel = self.cfg.w_ang_vel * torch.linalg.norm(av_b, dim=-1)

        # --------------Term 5: Orientation BONUS (was penalty)-------------------
        r_orientation = self.cfg.w_orientation * alignment_score   # positive bonus, no longer (1 - cos)

        # --------------Term 6: Action-rate (jerk) penalty------------------------
        action_delta  = self._actions - self._previous_action
        p_action_rate = self.cfg.w_action_rate * torch.linalg.norm(
            action_delta, dim=-1
        )

        # --------------Aggregate---------------------------------------------------
        dt = self.step_dt if self.cfg.scale_reward_with_time else 1.0

        rewards = {
            "alive":        r_alive        * dt,
            "proximity":    r_proximity    * dt,
            "lin_vel":     -p_lin_vel      * dt,
            "ang_vel":     -p_ang_vel      * dt,
            "orientation":  r_orientation  * dt,   # NOTE: now positive, sign flipped from before
            "action_rate": -p_action_rate  * dt,
        }

        for k, v in rewards.items():
            self._episode_sums[k] += v

        self._previous_action = self._actions.clone()

        return sum(rewards.values())

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Termination conditions (all vectorised):
          - dist from hover point > max_pos_error
          - tilt angle > max_tilt_deg  (catastrophic flip)
          - height < min_height  (hit the ground)
          - episode length exceeded (timeout)
        """
        pos_w = self._robot.data.root_pos_w
        ori_w = self._robot.data.root_quat_w

        # Distance from hover point
        dist    = torch.linalg.norm(self._hover_point_w - pos_w, dim=-1)
        too_far = dist > self.cfg.max_pos_error

        # Tilt: project gravity into body frame; large tilt = cos approaches -1
        grav_b     = quat_apply_inverse(ori_w, self._grav_unit_w)
        tilt_cos   = (-grav_b[:, 2]).clamp(-1.0, 1.0)
        tilt_rad   = torch.acos(tilt_cos)
        too_tilted = tilt_rad > math.radians(self.cfg.max_tilt_deg)

        # Ground collision
        too_low = pos_w[:, 2] < self.cfg.min_height

        died    = too_far | too_tilted | too_low
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        return died, timeout

    def _domain_randomise(self, env_ids: torch.Tensor):
        """
        Multiplicative parameter noise applied per episode reset.
        Each parameter p is drawn as:
            p_rand = p_nominal * (1 + U(-range, +range))
        Only parameters listed with nonzero range in cfg.dr_dict are randomised.
        """
        n  = len(env_ids)
        dr = self.cfg.dr_dict

        def _rand(scale: float) -> torch.Tensor:
            """U(-scale, +scale) multiplier, shape (n,)."""
            return 1.0 + (torch.rand(n, device=self.device) * 2.0 - 1.0) * scale

        if dr.get("mass", 0.0) > 0.0:
            self._robot_mass[env_ids] = self.cfg.mass * _rand(dr["mass"])

        if dr.get("thrust_to_weight", 0.0) > 0.0:
            self._thrust_to_weight[env_ids] = (
                self.cfg.thrust_to_weight * _rand(dr["thrust_to_weight"])
            )

        if dr.get("k_eta", 0.0) > 0.0:
            self._k_eta[env_ids] = self.cfg.k_eta * _rand(dr["k_eta"])

        if dr.get("tau_m", 0.0) > 0.0:
            self._tau_m[env_ids] = self.cfg.tau_m * _rand(dr["tau_m"])

        if dr.get("kp_omega", 0.0) > 0.0:
            self._kp_omega[env_ids] = self.cfg.kp_omega * _rand(dr["kp_omega"])

        if dr.get("kd_omega", 0.0) > 0.0:
            self._kd_omega[env_ids] = self.cfg.kd_omega * _rand(dr["kd_omega"])

        if dr.get("inertia", 0.0) > 0.0:
            # Scale diagonal inertia independently per env
            diag_scale = _rand(dr["inertia"]).view(n, 1, 1)
            base = torch.diag(
                torch.tensor(
                    [self.cfg.Ixx, self.cfg.Iyy, self.cfg.Izz],
                    device=self.device,
                )
            ).unsqueeze(0)  # (1, 3, 3)
            self.inertia_tensor[env_ids] = base * diag_scale

        # Rebuild motor allocation matrix if thrust coefficient changed
        if dr.get("k_eta", 0.0) > 0.0:
            self._rebuild_motor_allocation(env_ids)

    # --------------DEBUG VISUALISATION
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create a green sphere marker at each env's hover target."""
        if debug_vis and not hasattr(self, "_hover_marker"):
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/HoverTarget",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),
                        ),
                    )
                },
            )
            self._hover_marker = VisualizationMarkers(marker_cfg)
        if not hasattr(self, "_vis_offset"):
            self._vis_offset = torch.tensor(
                [0.0, 0.0, 0.20], device=self.device
            ).unsqueeze(0).expand(self.num_envs, -1)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_hover_marker"):
            self._hover_marker.visualize(self._hover_point_w + self._vis_offset)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        Reset a subset of environments:
          1. Log episode stats
          2. Compute per-env hover target from terrain origins
          3. Randomise initial drone state
          4. Initialise motors near hover equilibrium
          5. Apply domain randomisation
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # --------------1. Episode logging-------------------------------------
        self._update_curriculum(env_ids)   # NEW — must run before sums are zeroed below

        extras: dict = {}
        for key in self._episode_sums:
            avg = self._episode_sums[key][env_ids].mean()
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/died"] = (
            self.reset_terminated[env_ids].sum().item()
        )
        extras["Episode_Termination/timeout"] = (
            self.reset_time_outs[env_ids].sum().item()
        )
        if self.cfg.forced_curriculum_level >= 0.0:
            active_curriculum_level = float(
                self.cfg.forced_curriculum_level
            )
        elif not self.cfg.curriculum_enabled:
            active_curriculum_level = 1.0
        else:
            active_curriculum_level = float(
                self._curriculum_level.item()
            )

        extras["Curriculum/level"] = (
            active_curriculum_level
        )
        self.extras.setdefault("log", {}).update(extras)

        # --------------2. Compute world-frame hover point for each env--------------------------
        # env_origins is (num_envs, 3) and offset by the terrain grid.
        local_hover = torch.tensor(
            [self.cfg.wall_x_pos - self.cfg.hover_distance,
             0.0,
             self.cfg.marker_height],
            device=self.device,
        )
        self._hover_point_w[env_ids] = (
                    self._terrain.env_origins[env_ids] + local_hover
                )
        local_marker = torch.tensor(
            [self.cfg.wall_x_pos - 0.03,   # matches marker spawn offset in _setup_scene
            0.0,
            self.cfg.marker_height],
            device=self.device,
        )
        self._marker_point_w[env_ids] = (
            self._terrain.env_origins[env_ids] + local_marker
        )

        # --------------3. Robot initial state-----------------------------------
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Stagger episode lengths to avoid simultaneous mass-resets
        if (
            len(env_ids) == self.num_envs
            and not getattr(self.cfg, "eval_mode", False)
        ):
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf,
                high=int(self.max_episode_length),
            )

        n = len(env_ids)
        default_root = self._robot.data.default_root_state[env_ids].clone()

        # Position = hover point + uniform noise
        curriculum_pos_noise, curriculum_yaw_noise = self._get_curriculum_noise()
        pos_noise = (
            torch.rand(n, 3, device=self.device) * 2.0 - 1.0
        ) * curriculum_pos_noise

        default_root[:, 0:3] = self._hover_point_w[env_ids] + pos_noise

        # Orientation: yaw rotated pi (facing -x) + small yaw noise
        base_yaw  = torch.full((n,), 0.0, device=self.device)
        yaw_noise = (
            torch.rand(n, device=self.device) * 2.0 - 1.0
        ) * curriculum_yaw_noise
        yaw_total = base_yaw + yaw_noise
        # wxyz pure-yaw quaternion: w=cos(y/2), x=0, y=0, z=sin(y/2)
        default_root[:, 3] = torch.cos(yaw_total / 2.0)
        default_root[:, 4] = 0.0
        default_root[:, 5] = 0.0
        default_root[:, 6] = torch.sin(yaw_total / 2.0)

        # Linear velocity noise
        vel_noise = (
            torch.rand(n, 3, device=self.device) * 2.0 - 1.0
        ) * torch.tensor(self.cfg.init_vel_noise, device=self.device)
        default_root[:, 7:10] = vel_noise

        # Angular velocity noise
        ang_noise = (
            torch.rand(n, 3, device=self.device) * 2.0 - 1.0
        ) * torch.tensor(self.cfg.init_ang_vel_noise, device=self.device)
        default_root[:, 10:13] = ang_noise

        self._robot.write_root_pose_to_sim(
            default_root[:, :7], env_ids=env_ids
        )
        self._robot.write_root_velocity_to_sim(
            default_root[:, 7:], env_ids=env_ids
        )

        # --------------4. Motor initialisation near hover equilibrium--------------------------
        robot_weight   = self._robot_mass[env_ids] * self._gravity_magnitude
        per_motor_force = robot_weight / 4.0
        hover_speed_sq  = per_motor_force / self._k_eta[env_ids]
        hover_speed     = torch.sqrt(hover_speed_sq.clamp(min=0.0))
        self._motor_speeds[env_ids] = hover_speed.unsqueeze(-1).expand(-1, 4)

        # --------------5. Reset control state-----------------------------------
        self._actions[env_ids]          = 0.0
        self._previous_action[env_ids]  = 0.0
        self._previous_omega_err[env_ids] = 0.0

        # --------------6. Domain randomisation-----------------------------------
        self._domain_randomise(env_ids)

        # --------------7. Visual domain randomisation (Phase 2 student robustness) --------------
        self._randomise_visuals(env_ids)          # lights, ArUco tint, etc.
        self._randomize_wall_textures(env_ids)    # per-env wall textures via Replicator


class ArucoHoverStudentEnv(ArucoHoverTeacherEnv):
    """Visual + IMU student environment for ArUco hover.

    Observations:
      - images:     (N, 3, H, W) RGB from onboard camera, float32 in [0, 1]
      - policy:     IMU-like vector [lin_vel_b, ang_vel_b, last_action] with noise/bias
      - privileged: same 17-D teacher observation (for critic/distillation)
      - full_state: 18-D logging vector (pos, quat, vel, last_action, dist)
    """

    cfg: ArucoHoverEnvCfg  # shared config

    def __init__(self, cfg: ArucoHoverEnvCfg, render_mode: str | None = None, **kwargs):
        # Reuse all teacher initialization: physics, rewards, DR, etc.
        super().__init__(cfg, render_mode, **kwargs)

        # Teacher env sets up robot, walls, DR, etc.
        #super().__init__(cfg, render_mode, **kwargs)

        # Camera will be created in _setup_scene and managed by the scene
        #self._camera = None

        # IMU bias (accel[3] + gyro[3]) per env, sampled at reset
        self._imu_bias = torch.zeros(self.num_envs, 6, device=self.device)
        
        # ------------------------------------------------------------------
        # 32-step visual-corner history
        # ------------------------------------------------------------------
        K = self.cfg.visual_history_length

        # Four corners × (u, v) = 8 values per timestamp.
        self._corner_history = torch.zeros(
            self.num_envs,
            K,
            8,
            device=self.device,
            dtype=torch.float32,
        )

        self._corner_history_initialized = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

        half = self.cfg.marker_size / 2.0

        # Marker lies in the world Y-Z plane.
        self._marker_corner_offsets = torch.tensor(
            [
                [0.0, -half, -half],
                [0.0,  half, -half],
                [0.0,  half,  half],
                [0.0, -half,  half],
            ],
            device=self.device,
            dtype=torch.float32,
        )
    
    def _setup_scene(self):
        """Set up the shared scene and optionally create the RGB camera."""

        # Create robot, marker, walls, terrain, and lighting.
        super()._setup_scene()

        self._camera = None

        # Analytic corner projection does not require a rendered RGB camera.
        if not self.cfg.enable_onboard_camera:
            print(
                "[INFO] Onboard RGB camera disabled. "
                "Using analytic marker-corner observations only."
            )
            return

        self._camera = TiledCamera(
            self.cfg.camera
        )

        self.scene.sensors[
            "onboard_camera"
        ] = self._camera

        import omni.usd
        from pxr import Sdf

        stage = omni.usd.get_context().get_stage()

        src = "/World/envs/env_0/Robot/Camera"
        dst = "/World/envs/env_0/Robot/body/Camera"

        src_prim = stage.GetPrimAtPath(src)

        dst_parent = stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/body"
        )

        if (
            src_prim.IsValid()
            and dst_parent.IsValid()
        ):
            edit = Sdf.BatchNamespaceEdit()

            edit.Add(
                Sdf.NamespaceEdit.Rename(
                    Sdf.Path(src),
                    Sdf.Path(dst),
                )
            )

            stage.GetRootLayer().Apply(
                edit
            )

            print(
                "[INFO] Camera reparented to "
                "Robot/body/Camera"
            )
        else:
            print(
                "[WARN] Camera reparent failed — "
                f"src valid: {src_prim.IsValid()}, "
                f"dst parent valid: {dst_parent.IsValid()}"
            )

    # --------------------------------------------------------------------------
    # Resets
    # --------------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Let teacher handle None / full reset logic
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # IMU bias: accel bias (m/s^2), gyro bias (rad/s)
        accel_bias_std = 0.1
        gyro_bias_std  = 0.02

        accel_bias = torch.randn(len(env_ids), 3, device=self.device) * accel_bias_std
        gyro_bias  = torch.randn(len(env_ids), 3, device=self.device) * gyro_bias_std
        self._imu_bias[env_ids] = torch.cat([accel_bias, gyro_bias], dim=-1)

        # This method may be called during parent initialization before the history
        # buffers are created, hence the hasattr guard.
        if hasattr(self, "_corner_history"):
            self._corner_history[env_ids] = 0.0
            self._corner_history_initialized[env_ids] = False

    # --------------------------------------------------------------------------
    # Observation helpers
    # --------------------------------------------------------------------------

    def _compute_imu_obs(self) -> torch.Tensor:
        """IMU-like observation with realistic noise and bias.

        Returns [lin_vel_b, ang_vel_b, last_action] with:
          - per-step Gaussian noise
          - per-episode Gaussian bias (stored in self._imu_bias)
        """
        ori_w  = self._robot.data.root_quat_w
        lv_w   = self._robot.data.root_lin_vel_w
        av_b   = self._robot.data.root_ang_vel_b

        lin_vel_b = quat_apply_inverse(ori_w, lv_w)
        imu_clean = torch.cat([lin_vel_b, av_b], dim=-1)          # (N, 6)

        noise_std = torch.tensor([0.05]*3 + [0.01]*3, device=self.device)
        imu_noisy = imu_clean + self._imu_bias + torch.randn_like(imu_clean) * noise_std

        grav_b = quat_apply_inverse(ori_w, self._grav_unit_w)     # (N, 3)

        # # Noisy position error — bridges teacher/student info gap
        # pos_w     = self._robot.data.root_pos_w
        # pos_err_w = self._hover_point_w - pos_w
        # pos_err_b = quat_apply_inverse(ori_w, pos_err_w)          # (N, 3)
        # pos_err_noisy = pos_err_b + torch.randn_like(pos_err_b) * 0.1  # 10cm std noise

        return torch.cat([
            imu_noisy,          # 6
            grav_b,             # 3
            # pos_err_noisy,      # 3  
            self._previous_action,  # 4
        ], dim=-1)              # (N, 13)

    def _compute_teacher_obs(self) -> torch.Tensor:
        """Reconstruct the 20-D privileged teacher observation.

        Same semantics as your original teacher obs:
          [0:3]   pos_error_b
          [3:6]   lin_vel_b
          [6:9]   ang_vel_b
          [9:12]  grav_vec_b
          [12:15] marker_dir_b
          [15:19] last_action
          [19]    dist_scalar
        """
        pos_w = self._robot.data.root_pos_w
        ori_w = self._robot.data.root_quat_w
        lv_w  = self._robot.data.root_lin_vel_w
        av_b  = self._robot.data.root_ang_vel_b

        # Position error in body frame
        pos_err_w = self._hover_point_w - pos_w
        pos_err_b = quat_apply_inverse(ori_w, pos_err_w)

        # Linear velocity in body frame
        lin_vel_b = quat_apply_inverse(ori_w, lv_w)

        # Gravity unit vector in body frame
        grav_b = quat_apply_inverse(ori_w, self._grav_unit_w)

        # Distance scalar
        dist = torch.linalg.norm(self._hover_point_w - pos_w, dim=-1, keepdim=True)

        marker_pos_w = self._hover_point_w.clone()
        marker_pos_w[:, 0] += self.cfg.hover_distance 
        marker_dir_w = marker_pos_w - pos_w
        marker_dir_w = marker_dir_w / torch.linalg.norm(marker_dir_w, dim=-1, keepdim=True).clamp(min=1e-6)
        marker_dir_b = quat_apply_inverse(ori_w, marker_dir_w)

        teacher_obs = torch.cat(
            [
                pos_err_b,                # 3
                lin_vel_b,                # 3
                av_b,                     # 3
                grav_b,                   # 3
                marker_dir_b,             # 3 
                self._previous_action,    # 4
                dist,                     # 1
            ],
            dim=-1,
        )  # Now 20 dims
        return teacher_obs
    
    def _compute_bbox_gt(self) -> torch.Tensor:
        """
        Project the ArUco marker into the onboard camera image.
        Returns (N, 4): [cx_norm, cy_norm, w_norm, h_norm].
        When the marker is not in front of the camera (depth <= MIN_DEPTH),
        all 4 values are set to -1.0 as a "not visible" sentinel — callers
        (video overlay, aux loss) must check for this before using the box.
        """
        cam_pos_w  = self._robot.data.root_pos_w       # (N, 3)
        cam_quat_w = self._robot.data.root_quat_w      # (N, 4) wxyz

        marker_cam = quat_apply_inverse(
            cam_quat_w, self._marker_point_w - cam_pos_w   # (N, 3)
        )

        W  = self.cfg.camera.width    # 64
        H  = self.cfg.camera.height   # 64
        fx = W * 24.0 / 20.955        # ≈ 146.7 px

        MIN_DEPTH = 0.05  # below this, marker is behind/at the camera plane — projection is meaningless
        depth_raw = marker_cam[:, 0]            # X is forward (matches body_x_w in _get_rewards)
        depth     = depth_raw.clamp(min=MIN_DEPTH)
        MARKER_SIZE = 0.1  # meters, side length of the ArUco square
        cx = (-fx * marker_cam[:, 1] / depth + W / 2.0) / W
        cy = (-fx * marker_cam[:, 2] / depth + H / 2.0) / H
        apparent = (MARKER_SIZE * fx / depth / W).clamp(0.0, 1.0)

        # Visible if: in front of camera AND any part of the marker (using its
        # apparent half-width as margin) overlaps the image frame. Using the
        # raw center point with a hard [0,1] cutoff was rejecting frames where
        # the marker was clearly visible on-screen but its center point sat
        # just outside the boundary.
        in_front = depth_raw > MIN_DEPTH
        half     = apparent / 2.0
        in_frame = (
            (cx + half >= 0.0) & (cx - half <= 1.0) &
            (cy + half >= 0.0) & (cy - half <= 1.0)
        )
        visible = in_front & in_frame

        sentinel = torch.full_like(cx, -1.0)
        cx       = torch.where(visible, cx, sentinel)
        cy       = torch.where(visible, cy, sentinel)
        apparent = torch.where(visible, apparent, torch.zeros_like(apparent))

        vis_flag = visible.float()
        return torch.stack([cx, cy, apparent, apparent, vis_flag], dim=-1)  # (N, 5)
    
    def _compute_marker_corner_obs(self) -> torch.Tensor:
        """
        Return four normalized marker corners:

            [u1, v1, u2, v2, u3, v3, u4, v4]

        Invalid corners are represented by (-1, -1).
        """

        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w

        corners_w = (
            self._marker_point_w[:, None, :]
            + self._marker_corner_offsets[None, :, :]
        )

        relative_w = corners_w - pos_w[:, None, :]

        quat_expanded = (
            quat_w[:, None, :]
            .expand(-1, 4, -1)
            .reshape(-1, 4)
        )

        corners_b = quat_apply_inverse(
            quat_expanded,
            relative_w.reshape(-1, 3),
        ).reshape(self.num_envs, 4, 3)

        width = self.cfg.camera.width
        height = self.cfg.camera.height

        fx = width * 24.0 / 20.955
        fy = fx

        # Camera forward axis is body X.
        depth = corners_b[..., 0]
        safe_depth = depth.clamp(min=0.05)

        u = (
            -fx * corners_b[..., 1] / safe_depth
            + width / 2.0
        ) / width

        v = (
            -fy * corners_b[..., 2] / safe_depth
            + height / 2.0
        ) / height

        valid = (
            (depth > 0.05)
            & (u >= 0.0)
            & (u <= 1.0)
            & (v >= 0.0)
            & (v <= 1.0)
        )

        if not getattr(self.cfg, "eval_mode", False):
            if self.cfg.corner_noise_px > 0.0:
                u = (
                    u
                    + torch.randn_like(u)
                    * self.cfg.corner_noise_px
                    / width
                )

                v = (
                    v
                    + torch.randn_like(v)
                    * self.cfg.corner_noise_px
                    / height
                )

            if self.cfg.corner_dropout_prob > 0.0:
                missing = (
                    torch.rand_like(u)
                    < self.cfg.corner_dropout_prob
                )
                valid = valid & ~missing

        invalid = torch.full_like(u, -1.0)

        u = torch.where(valid, u, invalid)
        v = torch.where(valid, v, invalid)

        return torch.stack(
            [u, v],
            dim=-1,
        ).reshape(self.num_envs, 8)

    # --------------------------------------------------------------------------
    # Main observation API
    # --------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Return four RGB frames, previous action, and privileged training state."""

        # ------------------------------------------------------------------
        # 1. Current projected ArUco corners
        # ------------------------------------------------------------------
        corner_obs = self._compute_marker_corner_obs()

        was_initialized = self._corner_history_initialized.clone()
        not_initialized = ~was_initialized

        # On reset, fill all 32 timestamps using the first observation.
        if not_initialized.any():
            self._corner_history[not_initialized] = (
                corner_obs[not_initialized]
                .unsqueeze(1)
                .expand(
                    -1,
                    self.cfg.visual_history_length,
                    -1,
                )
            )

            self._corner_history_initialized[not_initialized] = True

        # Shift already-running histories by one timestamp.
        if was_initialized.any():
            history = self._corner_history[was_initialized].clone()

            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = corner_obs[was_initialized]

            self._corner_history[was_initialized] = history

        policy_obs = self._corner_history.reshape(
            self.num_envs,
            -1,
        )

        expected_policy_dim = (
            self.cfg.visual_history_length * 8
        )

        assert policy_obs.shape[-1] == expected_policy_dim, (
            f"Policy observation dimension is "
            f"{policy_obs.shape[-1]}, expected {expected_policy_dim}"
        )

        # ------------------------------------------------------------------
        # 3. Privileged teacher/critic state
        # ------------------------------------------------------------------
        teacher_obs = self._compute_teacher_obs()

        assert teacher_obs.shape[-1] == 20, (
            f"Teacher observation must be 20-D, got {teacher_obs.shape[-1]}"
        )

        # ------------------------------------------------------------------
        # 4. Full-state evaluation vector
        # ------------------------------------------------------------------
        pos_w = self._robot.data.root_pos_w
        ori_w = self._robot.data.root_quat_w
        lv_w = self._robot.data.root_lin_vel_w
        av_b = self._robot.data.root_ang_vel_b

        dist = torch.linalg.norm(
            self._hover_point_w - pos_w,
            dim=-1,
            keepdim=True,
        )

        full_state = torch.cat(
            [
                pos_w,
                ori_w,
                lv_w,
                av_b,
                self._previous_action,
                dist,
            ],
            dim=-1,
        )

        return {
            "policy": policy_obs,
            "teacher": teacher_obs,
            "critic": teacher_obs,
            "full_state": full_state,

            # # Leave this temporarily if AuxDistillation currently expects it.
            # # Its loss weight is zero, so it does not affect the student.
            # "bbox_gt": self._compute_bbox_gt(),
        }