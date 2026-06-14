# configs/quadrotor_asset.py
# Asset configuration for the quadrotor drone.
#
# ── PLACEHOLDERS ──────────────────────────────────────────────────────────────
# Before running, you need to fill in:
#   1. MODELS_PATH       — path to your models/ folder (see utils/assets.py)
#   2. usd_path          — path to your drone's .usd file
#   3. motor joint names — replace "m1_joint" etc. with your USD's actual joint names
#   4. init_state pos    — spawn height (currently 1.0 m)
#   5. Total drone mass  — update MassPropertiesCfg once you know your hardware weight
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from utils.assets import MODELS_PATH  # defined in utils/assets.py


# ── Main drone config ──────────────────────────────────────────────────────────
QUADROTOR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # TODO: Replace with the path to your drone's .usd file.
        # Example: usd_path=f"{MODELS_PATH}/my_drone.usd"
        usd_path=f"{MODELS_PATH}/crazyflie_brushless.usd", 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Drone spawns 1.0 m above ground, facing the ArUco wall at x = +1.5 m.
        # Adjust z to match your desired hover height.
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={
            # TODO: Replace these joint names with the actual motor joint names
            # from your USD file. Run:
            #   grep -i "joint" your_drone.urdf
            # to find them.
            "m1_joint":  200.0,   # front-left  (CCW)
            "m2_joint": -200.0,   # front-right (CW)
            "m3_joint":  200.0,   # back-right  (CCW)
            "m4_joint": -200.0,   # back-left   (CW)
        },
    ),
    actuators={
        # ImplicitActuator with zero stiffness/damping = Isaac Lab lets the
        # env apply forces/torques directly via external wrench (standard
        # approach for quadrotors — no joint-level motor simulation).
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

"""
Quadrotor drone config for the AerialCatching ArUco hover task.

Hardware this will represent (fill in as you finalise the build):
  - Frame:        TODO (e.g. 250mm carbon fibre X-frame)
  - Motors:       TODO (e.g. 2306 2400KV brushless)
  - Camera:       Intel RealSense D435i (RGB + depth + IMU)
  - Compute:      Nvidia Jetson Orin NX
  - Total mass:   TODO kg  (frame + motors + battery + Realsense + Orin NX)
"""


# ── ArUco marker wall config ───────────────────────────────────────────────────
ARUCO_WALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ArucoWall",
    spawn=sim_utils.CuboidCfg(
        size=(0.02, 0.6, 0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.1, 0.1),  # dark wall
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.5, 0.0, 1.0),
    ),
)
"""
Static wall carrying the ArUco marker.
The drone's goal is to hover at x = 0.0, facing this wall at x = 1.5 m.
"""