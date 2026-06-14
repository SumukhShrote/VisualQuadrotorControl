"""
create_marker_usd.py — Synchronous USD Creation for ArUco Marker
Run: python create_marker_usd.py
"""

import os
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf 

MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
TEXTURE_PATH = os.path.join(MODELS_PATH, "textures", "aruco_6x6_marker.png")
OUT_USD_PATH = os.path.join(MODELS_PATH, "aruco_marker", "aruco_marker.usd")

os.makedirs(os.path.dirname(OUT_USD_PATH), exist_ok=True)
if os.path.exists(OUT_USD_PATH):
    os.remove(OUT_USD_PATH)

print(f"[INFO] Creating new stage synchronously at {OUT_USD_PATH}...")
stage = Usd.Stage.CreateNew(OUT_USD_PATH)

# --- THE FIX: Create a Root Xform and set it as Default Prim ---
root_path = "/Marker"
root_prim = UsdGeom.Xform.Define(stage, root_path)
stage.SetDefaultPrim(root_prim.GetPrim()) # Now Isaac Lab knows what to import!

# 4. Create the Mesh UNDER the root prim
mesh_path = f"{root_path}/geometry"
mesh = UsdGeom.Mesh.Define(stage, mesh_path)

MARKER_SIZE = 0.10  # meters (10 cm)
THICKNESS = 0.01    # meters (1 cm)

h = MARKER_SIZE / 2
t = THICKNESS / 2

pts = [
    (-h, -h, -t), (h, -h, -t), (h, h, -t), (-h, h, -t),
    (-h, -h,  t), (h, -h,  t), (h, h,  t), (-h, h,  t),
]

mesh.CreatePointsAttr(pts)

face_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7  
]
mesh.CreateFaceVertexIndicesAttr(face_indices)
mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])

uvs = [(0,0), (1,0), (1,1), (0,1)] * 6
primvars_api = UsdGeom.PrimvarsAPI(mesh)
st_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
st_primvar.Set(uvs)

# --- THE FIX: Add Rigid Body API natively to the root ---
UsdPhysics.RigidBodyAPI.Apply(root_prim.GetPrim())

# 5. Create Material
mat_path = f"{root_path}/Material"
material = UsdShade.Material.Define(stage, mat_path)

shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))

st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
st_reader.CreateIdAttr("UsdPrimvarReader_float2")
st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

texture_sampler = UsdShade.Shader.Define(stage, f"{mat_path}/Texture")
texture_sampler.CreateIdAttr("UsdUVTexture")
texture_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(TEXTURE_PATH)
texture_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")

shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture_sampler.ConnectableAPI(), "rgb")
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

UsdShade.MaterialBindingAPI(mesh).Bind(material)

stage.GetRootLayer().Save()
print(f"[SUCCESS] Successfully saved robust, pre-textured ArUco marker USD to {OUT_USD_PATH}")
simulation_app.close()