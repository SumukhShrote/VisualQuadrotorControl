"""
demo_env_walls.py

Demo environment with:
- Ground plane
- 50m x 50m enclosed arena
- ArUco marker USD
- Camera
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import sys

print("\n========== OMNI MODULES ==========")
mods = sorted([m for m in sys.modules.keys() if "replicator" in m.lower()])
for m in mods:
    print(m)
print("==================================\n")
import os
import torch
import numpy as np
import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

MODELS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models")
)

USD_MARKER_PATH = os.path.join(
    MODELS_PATH,
    "aruco_marker",
    "aruco_marker.usd",
)

ARENA_SIZE = 10.0
WALL_HEIGHT = 10.0
WALL_THICKNESS = 0.5


@configclass
class DemoSceneCfg(InteractiveSceneCfg):

    num_envs: int = args_cli.num_envs
    env_spacing: float = 4.0

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    # ------------------------------------------------------------------
    # Marker
    # ------------------------------------------------------------------

    marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Marker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_MARKER_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 1.0),
            rot=(0.7071, 0.0, 0.7071, 0.0),
        ),
    )

    # ------------------------------------------------------------------
    # +X Wall
    # ------------------------------------------------------------------

    wall_px = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallPX",
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, ARENA_SIZE, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(ARENA_SIZE / 2, 0.0, WALL_HEIGHT / 2),
        ),
    )

    # ------------------------------------------------------------------
    # -X Wall
    # ------------------------------------------------------------------

    wall_nx = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallNX",
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, ARENA_SIZE, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-ARENA_SIZE / 2, 0.0, WALL_HEIGHT / 2),
        ),
    )

    # ------------------------------------------------------------------
    # +Y Wall
    # ------------------------------------------------------------------

    wall_py = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallPY",
        spawn=sim_utils.CuboidCfg(
            size=(ARENA_SIZE, WALL_THICKNESS, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, ARENA_SIZE / 2, WALL_HEIGHT / 2),
        ),
    )

    # ------------------------------------------------------------------
    # -Y Wall
    # ------------------------------------------------------------------

    wall_ny = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallNY",
        spawn=sim_utils.CuboidCfg(
            size=(ARENA_SIZE, WALL_THICKNESS, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -ARENA_SIZE / 2, WALL_HEIGHT / 2),
        ),
    )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


@configclass
class DemoEnvCfg(DirectRLEnvCfg):

    episode_length_s: float = 10.0
    decimation: int = 1

    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=1,
    )

    scene: DemoSceneCfg = DemoSceneCfg()


class DemoEnv(DirectRLEnv):

    cfg: DemoEnvCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def _setup_scene(self):

        light_intensity = float(np.random.uniform(500, 5000))

        light_color = (
            float(np.random.uniform(0.85, 1.0)),
            float(np.random.uniform(0.85, 1.0)),
            float(np.random.uniform(0.85, 1.0)),
        )

        print(f"Light intensity: {light_intensity}")
        print(f"Light color: {light_color}")

        light = sim_utils.DomeLightCfg(
            intensity=light_intensity,
            color=light_color,
        )

        light.func("/World/Light", light)

        self.scene.clone_environments(copy_from_source=False)
        import omni.replicator.core as rep
        import random

        texture_dir = os.path.join(
            MODELS_PATH,
            "textures",
            "generated",
        )

        texture_files = [
            os.path.join(texture_dir, f)
            for f in os.listdir(texture_dir)
            if f.endswith(".png")
        ]

        wall_texture = random.choice(texture_files)
        ground_texture = random.choice(texture_files)
        while ground_texture == wall_texture:
            ground_texture = random.choice(texture_files)
        print("\n==============================")
        print("DOMAIN RANDOMIZATION")
        print("==============================")
        print("Wall texture  :", os.path.basename(wall_texture))
        print("Ground texture:", os.path.basename(ground_texture))
        print("==============================\n")
        
        def apply_texture(path_pattern, texture_path):

            stage = self.scene["wall_px"].stage

            prims_group = rep.functional.get.prims(
                path_pattern=path_pattern,
                stage=stage,
            )

            if len(prims_group) == 0:
                print(f"No prims found: {path_pattern}")
                return

            for prim in prims_group:
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl",
                bind_prims=prims_group,
                count=len(prims_group),
                project_uvw=True,
            )

            rep.functional.modify.attribute(
                material_prims,
                "diffuse_texture",
                [texture_path] * len(material_prims),
            )

            rep.functional.modify.attribute(
                material_prims,
                "texture_scale",
                [(10.0, 10.0)] * len(material_prims),
            )
        
        apply_texture(
            "/World/envs/env_.*/WallPX",
            wall_texture,
        )

        apply_texture(
            "/World/envs/env_.*/WallNX",
            wall_texture,
        )

        apply_texture(
            "/World/envs/env_.*/WallPY",
            wall_texture,
        )

        apply_texture(
            "/World/envs/env_.*/WallNY",
            wall_texture,
        )

        apply_texture(
            "/World/ground.*",
            ground_texture,
        )

        self.scene.filter_collisions(
            global_prim_paths=[
                self.cfg.scene.terrain.prim_path
            ]
        )

        self.camera = self.scene.sensors["camera"]

    def _get_observations(self):
        return {
            "policy": torch.zeros(
                self.num_envs,
                1,
                device=self.device,
            )
        }

    def _get_rewards(self):
        return torch.zeros(
            self.num_envs,
            device=self.device,
        )

    def _get_dones(self):
        return (
            torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            ),
            torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass


def main():

    cfg = DemoEnvCfg()

    out_dir = os.path.expanduser(
        "~/AerialCatching/videos/demo"
    )

    os.makedirs(out_dir, exist_ok=True)

    env = DemoEnv(cfg=cfg)

    print("Warming up...")

    for _ in range(30):
        simulation_app.update()
        env.sim.render()
        env.sim.step()

    print("Recording...")

    frames = []

    dummy_action = torch.zeros(
        args_cli.num_envs,
        1,
        device=env.device,
    )

    for _ in range(args_cli.video_length):

        env.step(dummy_action)

        rgb = env.camera.data.output["rgb"]

        if rgb is not None:
            frames.append(
                rgb[0]
                .cpu()
                .numpy()[..., :3]
                .astype(np.uint8)
            )

    if frames:
        import imageio

        imageio.mimwrite(
            os.path.join(
                out_dir,
                "env_demo.mp4",
            ),
            frames,
            fps=30,
        )

        print("Saved video.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()