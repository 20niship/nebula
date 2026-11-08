"""demo_fluid_10m の .nbp フレームを Blender (Cycles) でレンダリングする。

uv run python render/render_fluid.py out/fluid --frames 0 120 --out out/fluid_png
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa: E402
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from nbp import load_nbp, surface_mesh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="out/fluid_png")
ap.add_argument("--frames", type=int, nargs=2, default=[0, 0], metavar=("START", "END"))
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--voxel", type=float, default=0.02, help="surface reconstruction voxel [m]")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

sc = C.reset_scene(tuple(a.res), a.samples)
C.sky_world(30, 160, 0.35)
C.camera((6.4, -4.6, 2.6), (1.55, -1.0, 0.45), lens=32)

floor = C.material("floor", (0.10, 0.11, 0.12, 1), 0.3)
sc.view_settings.exposure = -1.2
steel = C.material("steel", (0.8, 0.82, 0.85, 1), 0.25, Metallic=1.0)
water = C.material("water", (0.8, 0.93, 1.0, 1), 0.0, Transmission_Weight=1.0, IOR=1.33)
C.box("floor", (2.0, -0.5, 1.0), (30, 0.5, 30), floor)
wall = C.material("wall", (0.18, 0.2, 0.24, 1), 0.5)
C.box("wall_back", (2.0, 1.0, -0.05), (2.05, 1.0, 0.05), wall)   # z=0 側 (カメラの奥)
C.box("wall_end", (4.05, 1.0, 1.0), (0.05, 1.0, 1.05), wall)     # x=4 側
water.node_tree.links.new(
    water.node_tree.nodes.new("ShaderNodeVolumeAbsorption").outputs[0], water.node_tree.nodes["Material Output"].inputs["Volume"])
ab = [n for n in water.node_tree.nodes if n.bl_idname == "ShaderNodeVolumeAbsorption"][0]
ab.inputs["Color"].default_value = (0.35, 0.7, 0.95, 1)
ab.inputs["Density"].default_value = 3.0
for z in (0.55, 1.45):
    C.cylinder(f"post{z}", (2.6, 0.5, z), 0.14, 0.5, steel)
C.box("block", (3.3, 0.2, 1.0), (0.12, 0.2, 0.35), steel)

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.0
sun.rotation_euler = (0.9, 0.2, 0.6)
bpy.context.collection.objects.link(sun)

for f in range(a.frames[0], a.frames[1] + 1, a.step):
    if C.done(f"{a.out}/frame_{f:05d}.png"):
        continue
    t0 = time.time()
    pos, spd, lo, hi = load_nbp(f"{a.src}/frame_{f:05d}.nbp")
    v, fa = surface_mesh(pos, lo, hi, a.voxel)
    ob = C.mesh_object("water", v, fa, water)
    t1 = time.time()
    C.render_still(f"{a.out}/frame_{f:05d}.png")
    bpy.data.objects.remove(ob)
    print(f"frame {f}: {len(pos)} particles -> {len(fa)} tris, mesh {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
