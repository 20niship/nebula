"""demo_car_wind の tracers.bin を Blender (Cycles) でレンダリングする: 自転する車 (car/make_car.py の car.npz) + 風速で色分けしたトレーサの軌跡。
可視化は --vis で選ぶ:
  q           Q 基準 (速度勾配テンソルの渦度 - ひずみ、Q>0 が渦) の等値面を速度で色付け。FluidX3D の渦構造の可視化と同じ。車体に被らない (既定)
  streamlines 風速で色分けしたトレーサの軌跡。時間方向に連続して流れるのでちらつかないが、車体に被りやすい
  both        両方 (軌跡は少なめ)
車は角度 omega*dt*(sim step) で回す (エンジンの回転コライダーと同じ向き)。
  uv run python car/render_car.py out/car --frames 100 100 --out out/car_png --samples 48 --res 1920 1080
"""
import argparse
import math
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "render"))
import bpy  # noqa: E402
import common as C  # noqa: E402
import mathutils  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets_local", "car")


def load_tracers(path):
    """(pos[steps,M,3], gen[steps,M], dt, domain)。軌跡は pool の添字ごとに時間方向へ連続し、湧き直すと gen が変わる。"""
    with open(path, "rb") as f:
        assert f.read(4) == b"TRC1", path
        steps, m = struct.unpack("<2I", f.read(8))
        dt, *dom = struct.unpack("<4f", f.read(16))
        raw = np.frombuffer(f.read(steps * m * 16), np.dtype([("p", "<f4", 3), ("g", "<u4")])).reshape(steps, m)
    return raw["p"], raw["g"], dt, np.array(dom, np.float32)


def load_qvox(path, boost=1.0):
    """(present[nx,ny,nz] bool, speed01[nx,ny,nz] float32, cell, vmax)。Q が閾値以上のセルだけが書かれたスパース形式。"""
    with open(path, "rb") as f:
        assert f.read(4) == b"QVX1", path
        nx, ny, nz = struct.unpack("<3I", f.read(12))
        cell, vmax, n = struct.unpack("<ffI", f.read(12))
        d = np.frombuffer(f.read(n * 8), np.dtype([("p", "<u2", 3), ("s", "u1"), ("q", "u1")]))
    d = d[d["q"] >= int(np.log2(max(boost, 1.0)) / 3.0 * 255)]  # 書き出し時の閾値の boost 倍以上 (q は 1..8 倍を対数で 0..255)
    present = np.zeros((nx, ny, nz), bool)
    speed = np.zeros((nx, ny, nz), np.float32)
    p = d["p"].astype(np.int64)
    present[p[:, 0], p[:, 1], p[:, 2]] = True
    speed[p[:, 0], p[:, 1], p[:, 2]] = d["s"] / 255.0
    return present, speed, cell, vmax


def hue(t):
    """0 (遅い) -> 青、0.5 -> 緑、1 (速い) -> 赤 の虹色 (FluidX3D 風)。"""
    h = (1.0 - np.clip(t, 0, 1)) * (2.0 / 3.0)
    r = np.clip(np.abs(h * 6 - 3) - 1, 0, 1)
    g = np.clip(2 - np.abs(h * 6 - 2), 0, 1)
    b = np.clip(2 - np.abs(h * 6 - 4), 0, 1)
    return np.stack([r, g, b], -1)


def vortex_mesh(path, smooth, boost, name="vortex"):
    present, speed, cell, vmax = load_qvox(path, boost)
    f = gaussian_filter(present.astype(np.float32), smooth)
    v, fa, _, _ = marching_cubes(f, 0.35, spacing=(cell,) * 3)
    v = v + 0.5 * cell
    idx = np.clip(np.round(v / cell - 0.5).astype(int), 0, np.array(speed.shape) - 1)
    # 頂点の色: 近傍 (3x3x3) の最大速度ではなく、最近傍の存在セルの速度 (存在しなければ周囲の最大)
    sp = gaussian_filter(speed, smooth) / np.maximum(gaussian_filter(present.astype(np.float32), smooth), 1e-3)
    col = hue(np.clip(sp[idx[:, 0], idx[:, 1], idx[:, 2]] * 1.0, 0, 1))
    ob = C.mesh_object(name, v.astype(np.float32), fa.astype(np.int32), None, smooth=True)
    me = ob.data
    ca = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.concatenate([col, np.ones((len(col), 1))], 1).astype(np.float32).ravel())
    return ob, len(fa)


def curves(items, idx, radius, name):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 1
    for i in idx:
        pts = items[i]
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        co = np.zeros((len(pts), 4), np.float32)
        co[:, :3] = C.to_blender(pts)
        co[:, 3] = 1
        sp.points.foreach_set("co", co.ravel())
    ob = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(ob)
    return ob


ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="out/car_png")
ap.add_argument("--vis", choices=["q", "streamlines", "both"], default="q")
ap.add_argument("--q-every", type=int, default=None, help="q_*.qvox の間隔 [記録ステップ]。未指定なら存在するファイルから推定")
ap.add_argument("--q-boost", type=float, default=4.0, help="書き出し時の Q 閾値の何倍以上を面にするか (1..8)。大きいほど細い渦管だけになる")
ap.add_argument("--smooth", type=float, default=0.8, help="Q 等値面のガウス平滑 [セル]")
ap.add_argument("--frames", type=int, nargs=2, default=[0, 0], metavar=("START", "END"))
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--trail-sec", type=float, default=0.15, help="軌跡の長さ [s] (風速 20m/s なら約 4m)")
ap.add_argument("--radius", type=float, default=0.006)
ap.add_argument("--max-streams", type=int, default=5000)
ap.add_argument("--vmax", type=float, default=30.0)
ap.add_argument("--warmup", type=int, default=None, help="scene.txt に無い古い出力用")
ap.add_argument("--cam", type=float, nargs=3, default=[5.5, 3.4, -6.2], help="車の自転軸からの相対位置 [m]")
ap.add_argument("--target", type=float, nargs=3, default=[-0.4, 0.45, 0.0])
ap.add_argument("--lens", type=float, default=36.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

TP, TG, DT, DOM = load_tracers(f"{a.src}/tracers.bin")
sc_txt = np.loadtxt(f"{a.src}/scene.txt")
pivot = sc_txt[4:6]
omega = float(sc_txt[6])
warm = int(sc_txt[8]) if len(sc_txt) > 8 else (a.warmup or 0)
if a.warmup is not None:
    warm = a.warmup

sc = C.reset_scene(tuple(a.res), a.samples)
sc.view_settings.view_transform = "Standard"
if a.vis == "streamlines":
    sc.view_settings.exposure = -0.2
    C.sky_world(30, 220, 0.35)
else:  # 黒背景 + 発光する渦 (FluidX3D 風)
    sc.view_settings.exposure = 0.0
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.004, 0.006, 1)
P3 = np.array([pivot[0], 0.0, pivot[1]])
C.camera(C.sim_to_blender(P3 + a.cam), C.sim_to_blender(P3 + a.target), a.lens)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.5
sun.rotation_euler = (0.9, 0.0, 0.8)
bpy.context.collection.objects.link(sun)

ground = C.material("ground", (0.12, 0.125, 0.13, 1) if a.vis == "streamlines" else (0.015, 0.016, 0.02, 1), 0.55)
C.box("ground", (DOM[0] / 2, -0.5, DOM[2] / 2), (300, 0.5, 300), ground)

npz = np.load(f"{ASSETS}/car.npz")
paint = C.material("paint", (0.55, 0.02, 0.02, 1), 0.22, Metallic=0.35)
if a.vis != "streamlines":  # 渦が車体の後ろ側からも見えるよう、車は暗い半透明にする
    paint = C.material("paint", (0.05, 0.055, 0.07, 1), 0.3, Metallic=0.6)
    paint.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.55
car = C.mesh_object("car", npz["verts"] + np.array([pivot[0], 0.0, pivot[1]], np.float32), npz["faces"], paint)
car.rotation_mode = "AXIS_ANGLE"

vort = bpy.data.materials.new("vortex")
vort.use_nodes = True
nt = vort.node_tree
nt.nodes.clear()
at = nt.nodes.new("ShaderNodeAttribute")
at.attribute_name = "col"
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Strength"].default_value = 1.6
mix = nt.nodes.new("ShaderNodeMixShader")
dif = nt.nodes.new("ShaderNodeBsdfDiffuse")
nt.links.new(at.outputs["Color"], em.inputs["Color"])
nt.links.new(at.outputs["Color"], dif.inputs["Color"])
mix.inputs["Fac"].default_value = 0.45
nt.links.new(dif.outputs[0], mix.inputs[1])
nt.links.new(em.outputs[0], mix.inputs[2])
oo = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(mix.outputs[0], oo.inputs[0])

NB = 8
COLORS = [(0.05, 0.25, 1.0), (0.0, 0.6, 1.0), (0.0, 0.9, 0.8), (0.1, 1.0, 0.4), (0.7, 1.0, 0.1), (1.0, 0.8, 0.0), (1.0, 0.45, 0.05), (1.0, 0.1, 0.05)]
GLOWS = []
for i, col in enumerate(COLORS):
    m = bpy.data.materials.new(f"glow{i}")
    m.use_nodes = True
    n = m.node_tree
    n.nodes.clear()
    em = n.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*col, 1)
    em.inputs["Strength"].default_value = 2.5
    out = n.nodes.new("ShaderNodeOutputMaterial")
    n.links.new(em.outputs[0], out.inputs[0])
    GLOWS.append(m)

for f in range(a.frames[0], a.frames[1] + 1, a.step):
    if C.done(f"{a.out}/frame_{f:05d}.png"):
        continue
    t0 = time.time()
    f = min(f, len(TP) - 1)
    # 車: 回転コライダーの角度 omega*dt*(sim step)。ピボットまわりに y 軸 (Blender では Z 軸) 回転
    theta = omega * DT * (warm + f + 1)
    car.matrix_world = (
        mathutils.Matrix.Translation(C.sim_to_blender((pivot[0], 0.0, pivot[1])))
        @ mathutils.Matrix.Rotation(theta, 4, "Z")
        @ mathutils.Matrix.Translation(-mathutils.Vector(C.sim_to_blender((pivot[0], 0.0, pivot[1]))))
    )
    obs = []
    nq = 0
    if a.vis in ("q", "both"):
        qs = sorted(int(n[2:7]) for n in os.listdir(a.src) if n.startswith("q_") and n.endswith(".qvox"))
        qf = min(qs, key=lambda k: abs(k - f))  # 最も近い記録ステップの Q
        o, nq = vortex_mesh(f"{a.src}/q_{qf:05d}.qvox", a.smooth, a.q_boost)
        o.data.materials.append(vort)
        obs.append(o)
    lo = max(0, f - max(8, round(a.trail_sec / DT)) + 1)
    gen = TG[lo : f + 1]
    same = (gen == gen[-1]).astype(np.int8)[::-1].cumprod(axis=0)[::-1].astype(bool)
    P = TP[lo : f + 1]
    n = same.sum(0)
    sel = np.nonzero(n >= 6)[0]
    if a.vis == "q":
        sel = sel[:0]
    else:
        cap = a.max_streams if a.vis == "streamlines" else 600
        sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), cap)).astype(int)] if len(sel) else sel
    items, spd = {}, np.zeros(P.shape[1])
    for i in sel:
        pts = P[len(P) - n[i] :, i]
        items[i] = pts
        spd[i] = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean() / DT
    bins = np.clip((spd / a.vmax * NB).astype(int), 0, NB - 1)
    for k in range(NB):
        idx = [i for i in sel if bins[i] == k]
        if idx:
            o = curves(items, idx, a.radius, f"streams{k}")
            o.data.materials.append(GLOWS[k])
            obs.append(o)
    t1 = time.time()
    C.render_still(f"{a.out}/frame_{f:05d}.png")
    for o in obs:
        bpy.data.objects.remove(o)
    print(f"frame {f}: {len(sel)} trails, {nq} vortex tris, theta {math.degrees(theta):.1f}deg, build {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
