"""demo_avalanche_mesh の .nbp を Blender (Cycles) でレンダリングする。

雪粒子の密度場から毎フレーム数百万三角形の雪面メッシュを再構成し、地形メッシュ・樹木・岩と合わせて描画する。
uv run python render/render_avalanche.py out/avalanche --frames 120 120 --out out/avalanche_png
"""
import argparse
import math
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa: E402
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from nbp import load_nbp, surface_mesh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="out/avalanche_png")
ap.add_argument("--frames", type=int, nargs=2, default=[0, 0], metavar=("START", "END"))
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--voxel", type=float, default=0.1, help="surface reconstruction voxel [m]")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

sc = C.reset_scene(tuple(a.res), a.samples)
sc.view_settings.exposure = -1.1
C.sky_world(22, 250, 0.8)
C.camera(C.sim_to_blender((82, 20, 64)), C.sim_to_blender((30, 5, 18)), lens=30)

# 地形: terrain.f32 (nx, nz, step, heights[nz][nx])
with open(f"{a.src}/terrain.f32", "rb") as f:
    nx, nz, step = struct.unpack("<iif", f.read(12))
    H = np.frombuffer(f.read(nx * nz * 4), np.float32).reshape(nz, nx)
X, Z = np.meshgrid(np.arange(nx) * step, np.arange(nz) * step)
V = np.stack([X, H, Z], -1).reshape(-1, 3)
idx = np.arange(nx * nz).reshape(nz, nx)
q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
F = np.concatenate([q[:, [0, 2, 1]], q[:, [0, 3, 2]]])  # 上向き法線になる巻き順
rock = C.material("terrain", (0.18, 0.15, 0.12, 1), 0.9)
C.mesh_object("terrain", V, F, rock)
# 地形の外側の地面
C.box("ground", (32, -0.6, 18), (200, 0.5, 200), C.material("ground", (0.12, 0.11, 0.1, 1), 0.9))

# 樹木と岩
bark = C.material("bark", (0.12, 0.08, 0.05, 1), 0.9)
leaf = C.material("leaf", (0.03, 0.16, 0.06, 1), 0.8)
stone = C.material("stone", (0.3, 0.3, 0.31, 1), 0.7)
for line in open(f"{a.src}/obstacles.txt"):
    t = line.split()
    v = list(map(float, t[1:]))
    if t[0] == "tree":
        x, y, z, r, h = v
        C.cylinder("trunk", (x, y + h * 0.18, z), r, h * 0.18, bark)
        for k in range(4):
            bpy.ops.mesh.primitive_cone_add(radius1=r * (4.6 - 0.8 * k), radius2=0, depth=h * 0.42, vertices=18, location=C.sim_to_blender((x, y + h * (0.30 + 0.2 * k), z)))
            bpy.context.object.data.materials.append(leaf)
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=v[3], location=C.sim_to_blender(v[:3]))
        bpy.context.object.scale = (1.0, 1.0, 0.75)
        bpy.context.object.data.materials.append(stone)

snow = C.material("snow", (0.95, 0.97, 1.0, 1), 0.55, Subsurface_Weight=0.35, Specular_IOR_Level=0.4)
snow.node_tree.nodes["Principled BSDF"].inputs["Subsurface Radius"].default_value = (0.8, 0.9, 1.0)
snow.node_tree.nodes["Principled BSDF"].inputs["Subsurface Scale"].default_value = 0.2

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.5
sun.rotation_euler = (math.radians(55), 0, math.radians(120))
bpy.context.collection.objects.link(sun)

for f in range(a.frames[0], a.frames[1] + 1, a.step):
    t0 = time.time()
    pos, spd, lo, hi = load_nbp(f"{a.src}/frame_{f:05d}.nbp")
    v, fa = surface_mesh(pos, lo, hi, a.voxel, sigma=1.5, level=0.5)
    ob = C.mesh_object("snow", v, fa, snow)
    t1 = time.time()
    C.render_still(f"{a.out}/frame_{f:05d}.png")
    bpy.data.objects.remove(ob)
    print(f"frame {f}: {len(pos)} particles -> {len(fa)} tris, mesh {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
