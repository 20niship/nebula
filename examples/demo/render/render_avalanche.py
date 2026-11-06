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
import bpy
import mathutils  # noqa: E402
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
C.camera(C.sim_to_blender((86, 36, -20)), C.sim_to_blender((26, 12, 22)), lens=30)

# 地形: terrain.f32 (nx, nz, step, heights[nz][nx])
with open(f"{a.src}/terrain.f32", "rb") as f:
    nx, nz, step = struct.unpack("<iif", f.read(12))
    H = np.frombuffer(f.read(nx * nz * 4), np.float32).reshape(nz, nx)
X, Z = np.meshgrid(np.arange(nx) * step, np.arange(nz) * step)
V = np.stack([X, H, Z], -1).reshape(-1, 3)
idx = np.arange(nx * nz).reshape(nz, nx)
q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
F = np.concatenate([q[:, [0, 2, 1]], q[:, [0, 3, 2]]])  # 上向き法線になる巻き順
rock = C.material("terrain", (0.07, 0.055, 0.04, 1), 0.95)
C.mesh_object("terrain", V, F, rock)
# 地形の側面 (スカート) と外側の地面。ジオラマ的に地形の縁を底面 y=0 まで閉じる
sk = []
for e, fix, axis in ((H[0], 0.0, 0), (H[-1], (nz - 1) * step, 0), (H[:, 0], 0.0, 1), (H[:, -1], (nx - 1) * step, 1)):
    n = len(e); t = np.arange(n) * step
    top = np.stack([t, e, np.full(n, fix)], -1) if axis == 0 else np.stack([np.full(n, fix), e, t], -1)
    bot = top.copy(); bot[:, 1] = 0.0
    vv = np.concatenate([top, bot]); ff = np.concatenate([np.stack([np.arange(n - 1), np.arange(1, n), n + np.arange(1, n)], -1), np.stack([np.arange(n - 1), n + np.arange(1, n), n + np.arange(n - 1)], -1)])
    C.mesh_object("skirt", vv, ff, rock)
C.box("ground", (32, -0.6, 18), (400, 0.5, 400), C.material("ground", (0.12, 0.11, 0.1, 1), 0.9))

# 樹木は virtual-tokyo の黒松メッシュ (USD) をインスタンス化。アセットはリポジトリに入れず、ローカルの VT_ASSETS から読む。
VT = os.environ.get("VT_ASSETS", "/home/owner/Desktop/programs/virtual-tokyo/assets/modules")
VARIANTS = ("veg_kuromatsu_m_0_summer", "veg_kuromatsu_m_1_summer", "veg_kuromatsu_l_0_summer")
src_trees = {}
for name in VARIANTS:
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=f"{VT}/{name}.usdc")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    top = max((o.matrix_world @ mathutils.Vector(c)).z for o in meshes for c in o.bound_box)
    src_trees[name] = (meshes, top)
    for o in new:
        o.hide_render = True
        o.hide_viewport = True

def plant(name, pos, height, yaw):
    meshes, top = src_trees[name]
    k = height / top
    M = mathutils.Matrix.Translation(C.sim_to_blender(pos)) @ mathutils.Matrix.Rotation(yaw, 4, "Z") @ mathutils.Matrix.Scale(k, 4)
    for o in meshes:
        c = o.copy()
        c.parent = None
        c.hide_render = False
        c.hide_viewport = False
        bpy.context.collection.objects.link(c)
        c.matrix_world = M @ o.matrix_world

stone = C.material("stone", (0.09, 0.085, 0.08, 1), 0.85)
rng = np.random.default_rng(5)
for line in open(f"{a.src}/obstacles.txt"):
    t = line.split()
    v = list(map(float, t[1:]))
    if t[0] == "tree":
        x, y, z, r, h = v
        name = VARIANTS[int(rng.integers(0, 2))]
        plant(name, (x, y, z), h, rng.uniform(0, 6.28))
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=v[3], location=C.sim_to_blender(v[:3]))
        ob = bpy.context.object
        co = np.empty(len(ob.data.vertices) * 3, np.float32)
        ob.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        n = np.sin(co[:, 0] * 2.3 + 1.0) * np.cos(co[:, 1] * 1.9) + 0.6 * np.sin(co[:, 2] * 4.1 + co[:, 0] * 3.0)
        co *= (1.0 + 0.07 * n / max(v[3], 0.5))[:, None] * np.array([1.0, 1.0, 0.75])
        ob.data.vertices.foreach_set("co", co.ravel())
        ob.data.update()
        ob.data.materials.append(stone)
        bpy.ops.object.shade_smooth()

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
