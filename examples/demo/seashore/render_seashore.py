"""demo_seashore の .nbp/.nbf を Blender (Cycles) でレンダリングする (横動画 16:9 既定)。

水: 粒子密度場の等値面メッシュ (透過 + 吸収)。泡: 点を Geometry Nodes で小球インスタンス化 (白)。
地形・岩: seashore/make_scene.py の scene.json + terrain.npz、岩は virtual-tokyo の usdc をそのまま取り込む (マテリアル込み)。
  uv run python seashore/render_seashore.py out/seashore --frames 150 150 --out out/seashore_png --samples 64
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "render"))  # common.py / nbp.py
import bpy
import common as C  # noqa: E402
import mathutils  # noqa: E402
import numpy as np  # noqa: E402
from nbp import load_nbf, load_nbp, surface_mesh  # noqa: E402

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets_local", "seashore")
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="out/seashore_png")
ap.add_argument("--frames", type=int, nargs=2, default=[0, 0], metavar=("START", "END"))
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--voxel", type=float, default=0.085, help="水面再構成のボクセル [m] (粒子間隔の 1.2 倍程度)")
ap.add_argument("--foam-radius", type=float, default=0.025, help="泡粒子の半径 [m]")
ap.add_argument("--cam", type=float, nargs=3, default=[3.0, 9.0, 6.0])
ap.add_argument("--target", type=float, nargs=3, default=[32.0, 2.0, 17.0])
ap.add_argument("--lens", type=float, default=24.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
scene = json.load(open(f"{ASSETS}/scene.json"))

sc = C.reset_scene(tuple(a.res), a.samples)
sc.view_settings.exposure = -1.0
C.sky_world(28, 200, 1.0)
C.camera(C.sim_to_blender(a.cam), C.sim_to_blender(a.target), a.lens)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.0
sun.rotation_euler = (math.radians(55), 0, math.radians(150))
bpy.context.collection.objects.link(sun)

# ── 砂浜 (ハイトフィールド) ──
T = np.load(f"{ASSETS}/terrain.npz")
H, step = T["h"], float(T["step"])
nz, nx = H.shape
X, Z = np.meshgrid(np.arange(nx) * step, np.arange(nz) * step)
idx = np.arange(nx * nz).reshape(nz, nx)
q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
sand = C.material("sand", (0.55, 0.45, 0.32, 1), 0.95)
C.mesh_object("beach", np.stack([X, H, Z], -1).reshape(-1, 3), np.concatenate([q[:, [0, 2, 1]], q[:, [0, 3, 2]]]), sand)

def rock_material():
    """usdc のテクスチャ (../textures/*.png) はリポジトリ外で未取得のため、手続き的な岩マテリアルに差し替える。"""
    m = C.material("rock", (0.3, 0.28, 0.25, 1), 0.8)
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value, tex.inputs["Detail"].default_value = 3.0, 12.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color, ramp.color_ramp.elements[1].color = (0.08, 0.075, 0.07, 1), (0.42, 0.38, 0.33, 1)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 60.0
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(fine.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


ROCK = rock_material()

# ── 岩: usdc を取り込み、make_scene.py と同じ変換 pos + Ry(yaw) @ scale*(v - pivot) で配置 ──
rocks_dir = f"{ASSETS}/rocks"
cache = {}
for r in scene["rocks"]:
    name = r["name"]
    if name not in cache:
        before = set(bpy.data.objects)
        bpy.ops.wm.usd_import(filepath=f"{rocks_dir}/{name}.usdc")
        new = [o for o in bpy.data.objects if o not in before]
        cache[name] = [o for o in new if o.type == "MESH"]
        for o in cache[name]:
            o.data.materials.clear()
            o.data.materials.append(ROCK)
        for o in new:
            o.hide_render = o.hide_viewport = True
    M = (
        mathutils.Matrix.Translation(C.sim_to_blender(r["pos"]))
        @ mathutils.Matrix.Rotation(r["yaw"], 4, "Z")
        @ mathutils.Matrix.Scale(r["scale"], 4)
        @ mathutils.Matrix.Translation(-mathutils.Vector(C.sim_to_blender(r["pivot"])))
    )
    for o in cache[name]:
        c = o.copy()
        c.hide_render = c.hide_viewport = False
        bpy.context.collection.objects.link(c)
        c.matrix_world = M @ o.matrix_world

# ── 水・泡 ──
water = C.material("water", (0.8, 0.95, 1.0, 1), 0.02, Transmission_Weight=1.0, IOR=1.33)
foam_mat = C.material("foam", (1, 1, 1, 1), 0.6)
foam_mat.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.0

# 泡用ジオメトリノード: 頂点に小球をインスタンス化 (半径はランダムに 0.5..1.2 倍)
gn = bpy.data.node_groups.new("foam_instances", "GeometryNodeTree")
gn.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
gn.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
n_in, n_out = gn.nodes.new("NodeGroupInput"), gn.nodes.new("NodeGroupOutput")
n_ico = gn.nodes.new("GeometryNodeMeshIcoSphere")
n_ico.inputs["Radius"].default_value = a.foam_radius
n_ico.inputs["Subdivisions"].default_value = 1
n_inst = gn.nodes.new("GeometryNodeInstanceOnPoints")
n_rnd = gn.nodes.new("FunctionNodeRandomValue")
n_rnd.data_type = "FLOAT"
n_rnd.inputs["Min"].default_value, n_rnd.inputs["Max"].default_value = 0.5, 1.2
n_mat = gn.nodes.new("GeometryNodeSetMaterial")
n_mat.inputs["Material"].default_value = foam_mat
gn.links.new(n_in.outputs[0], n_inst.inputs["Points"])
gn.links.new(n_ico.outputs["Mesh"], n_inst.inputs["Instance"])
gn.links.new(n_rnd.outputs[1], n_inst.inputs["Scale"])
gn.links.new(n_inst.outputs["Instances"], n_mat.inputs["Geometry"])
gn.links.new(n_mat.outputs[0], n_out.inputs[0])


def foam_object(P):
    P = C.to_blender(P)
    me = bpy.data.meshes.new("foam")
    me.vertices.add(len(P))
    me.vertices.foreach_set("co", P.ravel())
    ob = bpy.data.objects.new("foam", me)
    bpy.context.collection.objects.link(ob)
    ob.modifiers.new("gn", "NODES").node_group = gn
    return ob


for f in range(a.frames[0], a.frames[1] + 1, a.step):
    t0 = time.time()
    pos, spd, lo, hi = load_nbp(f"{a.src}/frame_{f:05d}.nbp")
    v, fa = surface_mesh(pos, lo, hi, a.voxel, sigma=1.2, level=0.5)
    w = C.mesh_object("water", v, fa, water)
    fp, _ = load_nbf(f"{a.src}/foam_{f:05d}.nbf") if os.path.exists(f"{a.src}/foam_{f:05d}.nbf") else (np.zeros((0, 3), np.float32), None)
    fo = foam_object(fp)
    t1 = time.time()
    C.render_still(f"{a.out}/frame_{f:05d}.png")
    bpy.data.objects.remove(w)
    bpy.data.objects.remove(fo)
    print(f"frame {f}: {len(pos)} water -> {len(fa)} tris, {len(fp)} foam, mesh {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
